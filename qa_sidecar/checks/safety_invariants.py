"""
Check: Safety Invariants

Verifies that the safety system is internally consistent and that no module
bypasses safety limits. This is the most critical check for a robotic arm
control system — any safety bypass could damage hardware or injure people.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "safety_invariants"


def run(config: QAConfig) -> list[Issue]:
    """Run all safety invariant checks."""
    issues: list[Issue] = []

    issues.extend(_check_limits_consistency(config))
    issues.extend(_check_safety_monitor_coverage(config))
    issues.extend(_check_no_hardcoded_limits(config))
    issues.extend(_check_command_validation_usage(config))
    issues.extend(_check_estop_paths(config))
    issues.extend(_check_nan_guards(config))

    return issues


def _check_limits_consistency(config: QAConfig) -> list[Issue]:
    """Verify that limits.py values are physically sensible."""
    issues: list[Issue] = []
    limits_path = config.src_dir / "safety" / "limits.py"

    if not limits_path.exists():
        issues.append(Issue(
            title="Missing safety limits file",
            description="src/safety/limits.py does not exist. This is the single source of truth for all joint limits.",
            category=Category.SAFETY,
            severity=Severity.CRITICAL,
            suggestion="Restore the safety limits file immediately.",
            check_name=CHECK_NAME,
        ))
        return issues

    source = limits_path.read_text(encoding="utf-8")

    # Check that joint limits are symmetric and within hardware specs
    # Hardware spec: J1/J2/J4 = ±85°, others = ±135°
    hw_limits = {
        0: 135.0,  # J0 base yaw
        1: 85.0,   # J1 shoulder pitch
        2: 85.0,   # J2 elbow pitch
        3: 135.0,  # J3 elbow roll
        4: 85.0,   # J4 wrist pitch
        5: 135.0,  # J5 wrist roll
    }

    # Parse JOINT_LIMITS_DEG from source
    deg_match = re.search(
        r"JOINT_LIMITS_DEG\s*=\s*np\.array\(\s*\[([\s\S]*?)\]\s*\)",
        source,
    )
    if deg_match:
        rows_text = deg_match.group(1)
        row_pattern = re.compile(r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
        for idx, row_match in enumerate(row_pattern.finditer(rows_text)):
            low = float(row_match.group(1))
            high = float(row_match.group(2))

            # Check symmetry
            if abs(abs(low) - abs(high)) > 0.1:
                issues.append(Issue(
                    title=f"Joint {idx} limits are asymmetric",
                    description=(
                        f"Joint {idx} has limits [{low}, {high}]. "
                        f"Asymmetric limits are unusual and may indicate a mistake."
                    ),
                    category=Category.SAFETY,
                    severity=Severity.MEDIUM,
                    file_path="src/safety/limits.py",
                    suggestion="Verify this asymmetry is intentional.",
                    check_name=CHECK_NAME,
                ))

            # Check not exceeding hardware limits
            if idx in hw_limits:
                hw_max = hw_limits[idx]
                if abs(high) > hw_max:
                    issues.append(Issue(
                        title=f"Joint {idx} software limit exceeds hardware limit",
                        description=(
                            f"Joint {idx} software limit is ±{abs(high)}° but hardware "
                            f"limit is ±{hw_max}°. The software limit MUST be at or below "
                            f"the hardware limit to protect the arm."
                        ),
                        category=Category.SAFETY,
                        severity=Severity.CRITICAL,
                        file_path="src/safety/limits.py",
                        suggestion=f"Reduce J{idx} limit to at most ±{hw_max}° (ideally with a 5° margin).",
                        check_name=CHECK_NAME,
                    ))

    # Check velocity limits are positive
    if "VELOCITY_MAX_RAD" in source and "np.array([" in source:
        vel_match = re.search(r"VELOCITY_MAX_RAD\s*=\s*np\.array\(\[([\d., ]+)\]\)", source)
        if vel_match:
            vals = [float(v.strip()) for v in vel_match.group(1).split(",")]
            for i, v in enumerate(vals):
                if v <= 0:
                    issues.append(Issue(
                        title=f"Non-positive velocity limit for joint {i}",
                        description=f"Velocity limit for joint {i} is {v} rad/s which is not positive.",
                        category=Category.SAFETY,
                        severity=Severity.CRITICAL,
                        file_path="src/safety/limits.py",
                        suggestion="Velocity limits must be positive values.",
                        check_name=CHECK_NAME,
                    ))
                if v > 5.0:
                    issues.append(Issue(
                        title=f"Very high velocity limit for joint {i}",
                        description=(
                            f"Joint {i} velocity limit is {v} rad/s ({math.degrees(v):.0f}°/s). "
                            f"This is quite fast and may be unsafe for a collaborative arm."
                        ),
                        category=Category.SAFETY,
                        severity=Severity.MEDIUM,
                        file_path="src/safety/limits.py",
                        suggestion="Verify this velocity limit is safe for the operating environment.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _check_safety_monitor_coverage(config: QAConfig) -> list[Issue]:
    """Check that SafetyMonitor validates all required fields."""
    issues: list[Issue] = []
    monitor_path = config.src_dir / "safety" / "safety_monitor.py"

    if not monitor_path.exists():
        issues.append(Issue(
            title="Missing safety monitor",
            description="src/safety/safety_monitor.py does not exist.",
            category=Category.SAFETY,
            severity=Severity.CRITICAL,
            check_name=CHECK_NAME,
        ))
        return issues

    source = monitor_path.read_text(encoding="utf-8")

    required_checks = [
        ("position_min", "position limit (minimum)"),
        ("position_max", "position limit (maximum)"),
        ("velocity_max", "velocity limit"),
        ("torque_max", "torque limit"),
        ("gripper", "gripper bounds"),
        ("estop", "emergency stop"),
        ("NaN", "NaN/Inf rejection"),
    ]

    for keyword, description in required_checks:
        if keyword.lower() not in source.lower():
            issues.append(Issue(
                title=f"Safety monitor missing {description} check",
                description=(
                    f"The SafetyMonitor does not appear to check {description}. "
                    f"All safety constraints must be validated on every command."
                ),
                category=Category.SAFETY,
                severity=Severity.CRITICAL,
                file_path="src/safety/safety_monitor.py",
                suggestion=f"Add {description} validation to validate_command().",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_no_hardcoded_limits(config: QAConfig) -> list[Issue]:
    """Scan for hardcoded joint limits outside of limits.py."""
    issues: list[Issue] = []

    # Patterns that suggest hardcoded limits
    limit_patterns = [
        (re.compile(r"[-+]?(?:80|85|135)\s*(?:degrees|deg|\°)"), "degree limit"),
        (re.compile(r"(?:1\.39|1\.48|2\.35)\d*\s*(?:#.*rad|.*radian)"), "radian limit"),
        (re.compile(r"position_min|position_max|joint_limit", re.IGNORECASE), "limit variable"),
    ]

    # Files that are ALLOWED to reference limits
    allowed_files = {
        "src/safety/limits.py",
        "src/safety/safety_monitor.py",
        "tests/test_safety_monitor.py",
        "tests/test_safety_integration.py",
    }

    for py_file in config.src_dir.rglob("*.py"):
        rel = str(py_file.relative_to(config.project_root))
        if rel in allowed_files:
            continue

        source = py_file.read_text(encoding="utf-8")
        for pattern, desc in limit_patterns:
            matches = pattern.findall(source)
            if matches and "import" not in source.split(matches[0])[0][-100:]:
                # Check if the file properly imports from limits
                if "from src.safety.limits import" not in source:
                    for match_text in matches[:1]:  # Report once per pattern per file
                        issues.append(Issue(
                            title=f"Possible hardcoded {desc} in {rel}",
                            description=(
                                f"File {rel} appears to contain a hardcoded {desc} "
                                f"('{match_text.strip()}') without importing from "
                                f"src.safety.limits. All limits should come from the "
                                f"single source of truth."
                            ),
                            category=Category.SAFETY,
                            severity=Severity.MEDIUM,
                            file_path=rel,
                            suggestion="Import limits from src.safety.limits instead of hardcoding.",
                            check_name=CHECK_NAME,
                        ))

    return issues


def _check_command_validation_usage(config: QAConfig) -> list[Issue]:
    """Check that modules sending commands use SafetyMonitor.validate_command()."""
    issues: list[Issue] = []

    # Files that send commands (likely to construct D1Command)
    for py_file in config.src_dir.rglob("*.py"):
        rel = str(py_file.relative_to(config.project_root))
        source = py_file.read_text(encoding="utf-8")

        # If file creates D1Command objects, it should use safety validation
        if "D1Command(" in source and "safety" not in rel:
            if "validate_command" not in source and "safety_monitor" not in source.lower():
                if "clamp_command" not in source:
                    issues.append(Issue(
                        title=f"Commands created without safety validation in {rel}",
                        description=(
                            f"File {rel} constructs D1Command objects but does not appear "
                            f"to validate them through SafetyMonitor. All commands MUST be "
                            f"validated or clamped before being sent to the arm."
                        ),
                        category=Category.SAFETY,
                        severity=Severity.HIGH,
                        file_path=rel,
                        suggestion="Pass all D1Command objects through SafetyMonitor.validate_command() or clamp_command() before sending.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _check_estop_paths(config: QAConfig) -> list[Issue]:
    """Verify e-stop is accessible from all control paths."""
    issues: list[Issue] = []

    # The server should have an e-stop endpoint
    server_path = config.web_dir / "server.py"
    if server_path.exists():
        source = server_path.read_text(encoding="utf-8")
        if "stop" not in source.lower():
            issues.append(Issue(
                title="No stop/e-stop endpoint in web server",
                description="The web server does not appear to have a stop or e-stop endpoint.",
                category=Category.SAFETY,
                severity=Severity.CRITICAL,
                file_path="web/server.py",
                suggestion="Add a POST /api/command/stop endpoint that triggers SafetyMonitor.trigger_estop().",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_nan_guards(config: QAConfig) -> list[Issue]:
    """Check that numeric pipelines guard against NaN propagation."""
    issues: list[Issue] = []

    critical_modules = [
        "src/kinematics/kinematics.py",
        "src/control/joint_controller.py",
        "src/control/smooth_trajectory.py",
        "src/planning/motion_planner.py",
    ]

    for mod_rel in critical_modules:
        mod_path = config.project_root / mod_rel
        if not mod_path.exists():
            continue
        source = mod_path.read_text(encoding="utf-8")

        has_nan_check = any(kw in source for kw in ["isnan", "isfinite", "np.nan", "math.nan", "NaN"])
        if not has_nan_check:
            issues.append(Issue(
                title=f"No NaN guards in {mod_rel}",
                description=(
                    f"Module {mod_rel} performs numerical computations but has no "
                    f"NaN/Inf checking. In a robotic arm, NaN propagation through "
                    f"kinematics or trajectory planning can cause dangerous behavior."
                ),
                category=Category.SAFETY,
                severity=Severity.HIGH,
                file_path=mod_rel,
                suggestion="Add np.isfinite() checks on outputs before returning computed values.",
                check_name=CHECK_NAME,
            ))

    return issues
