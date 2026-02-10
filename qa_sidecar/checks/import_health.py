"""
Check: Import Health

Verifies that every Python module in the project can be imported without errors.
Catches broken imports, circular dependencies, and missing dependencies.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "import_health"

# Modules that require hardware or optional heavy deps — test in subprocess
SUBPROCESS_MODULES = {
    "src.interface.d1_dds_connection",  # needs cyclonedds
}

# Missing-dependency errors that are environment issues (not code bugs).
# When these are the root cause, downgrade from HIGH to MEDIUM.
KNOWN_OPTIONAL_DEPS = [
    "numpy", "scipy", "cv2", "opencv", "matplotlib", "transforms3d",
    "cyclonedds", "unitree", "rclpy", "google", "generativeai",
    "zmq", "pyzmq", "dotenv", "PIL", "pillow",
]


def run(config: QAConfig) -> list[Issue]:
    """Try importing every src module and report failures."""
    issues: list[Issue] = []

    # Collect all Python files under src/
    src_files = sorted(config.src_dir.rglob("*.py"))

    for py_file in src_files:
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue

        # Convert file path to module path
        rel = py_file.relative_to(config.project_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)

        if not module_name:
            continue

        # For modules needing special env, test in subprocess
        if module_name in SUBPROCESS_MODULES:
            issue = _check_import_subprocess(module_name, config)
        else:
            issue = _check_import_inline(module_name, str(rel))

        if issue:
            issues.append(issue)

    # Also check web/ modules
    web_files = sorted(config.web_dir.rglob("*.py"))
    for py_file in web_files:
        if py_file.name.startswith("_"):
            continue
        rel = py_file.relative_to(config.project_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)
        if not module_name:
            continue

        issue = _check_import_subprocess(module_name, config)
        if issue:
            issues.append(issue)

    if not issues:
        issues.append(Issue(
            title=f"All {len(src_files)} source modules import cleanly",
            description="Every module under src/ was imported without errors.",
            category=Category.IMPORT_HEALTH,
            severity=Severity.INFO,
            check_name=CHECK_NAME,
        ))

    return issues


def _check_import_inline(module_name: str, file_path: str) -> Issue | None:
    """Try importing a module in-process."""
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        return None
    except Exception as exc:
        tb = traceback.format_exc()
        # If the root cause is a known optional/heavy dependency not being
        # installed in this environment, downgrade to MEDIUM — it's an env
        # issue, not a code bug.
        is_env_issue = any(dep in tb for dep in KNOWN_OPTIONAL_DEPS)
        severity = Severity.MEDIUM if is_env_issue else Severity.HIGH
        desc_suffix = (
            " (Root cause appears to be a missing environment dependency, not a code bug.)"
            if is_env_issue else ""
        )
        return Issue(
            title=f"Import failure: {module_name}",
            description=(
                f"Module '{module_name}' cannot be imported. "
                f"Error type: {type(exc).__name__}. "
                f"This means any code depending on this module will also break."
                f"{desc_suffix}"
            ),
            category=Category.IMPORT_HEALTH,
            severity=severity,
            file_path=file_path,
            suggestion=(
                f"Fix the import error in {file_path}. "
                f"Common causes: missing dependency, circular import, syntax error."
            ),
            evidence=tb[-500:],
            check_name=CHECK_NAME,
        )


def _check_import_subprocess(module_name: str, config: QAConfig) -> Issue | None:
    """Try importing a module in a subprocess to avoid side effects."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(config.project_root),
            env={**__import__("os").environ, "PYTHONPATH": str(config.project_root)},
        )
        if result.returncode != 0:
            # Check if it's a known optional dependency issue
            stderr = result.stderr
            if any(dep in stderr for dep in ["cyclonedds", "unitree", "rclpy"]):
                return None  # Expected — optional hardware dependency
            return Issue(
                title=f"Import failure: {module_name}",
                description=(
                    f"Module '{module_name}' fails to import in a clean subprocess. "
                    f"This indicates a real import problem beyond optional dependencies."
                ),
                category=Category.IMPORT_HEALTH,
                severity=Severity.HIGH,
                suggestion="Check the traceback and fix the root import error.",
                evidence=stderr[-500:],
                check_name=CHECK_NAME,
            )
        return None
    except subprocess.TimeoutExpired:
        return Issue(
            title=f"Import hangs: {module_name}",
            description=(
                f"Importing '{module_name}' timed out after 15 seconds. "
                "This suggests the module runs heavy initialization at import time."
            ),
            category=Category.IMPORT_HEALTH,
            severity=Severity.MEDIUM,
            suggestion="Move heavy initialization out of module-level code into functions.",
            check_name=CHECK_NAME,
        )
