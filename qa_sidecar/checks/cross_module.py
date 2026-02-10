"""
Check: Cross-Module Integration

Verifies consistency across modules, detects interface mismatches,
and checks that the system components work together correctly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from collections import defaultdict

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "cross_module"


def run(config: QAConfig) -> list[Issue]:
    """Run all cross-module integration checks."""
    issues: list[Issue] = []

    issues.extend(_check_num_joints_consistency(config))
    issues.extend(_check_import_cycles(config))
    issues.extend(_check_interface_contracts(config))
    issues.extend(_check_test_coverage_gaps(config))
    issues.extend(_check_server_module_alignment(config))

    return issues


def _check_num_joints_consistency(config: QAConfig) -> list[Issue]:
    """Verify NUM_JOINTS is used consistently across the codebase."""
    issues: list[Issue] = []

    # The canonical values from limits.py
    canonical = {
        "NUM_JOINTS": 7,
        "NUM_ARM_JOINTS": 6,
    }

    # Search for hardcoded joint counts
    pattern_7 = re.compile(r"(?:range|zeros|ones|empty|array)\s*\(\s*7\s*\)")
    pattern_6 = re.compile(r"(?:range|zeros|ones|empty|array)\s*\(\s*6\s*\)")

    for py_file in config.src_dir.rglob("*.py"):
        rel = str(py_file.relative_to(config.project_root))
        source = py_file.read_text(encoding="utf-8")

        # Check for hardcoded 7 where NUM_JOINTS should be used
        for match in pattern_7.finditer(source):
            line_no = source[:match.start()].count("\n") + 1
            # Check if NUM_JOINTS is imported
            if "NUM_JOINTS" not in source:
                issues.append(Issue(
                    title=f"Hardcoded joint count (7) in {rel}:{line_no}",
                    description=(
                        f"File {rel} uses a hardcoded value of 7 where NUM_JOINTS "
                        f"should be used. If the joint count ever changes, this "
                        f"would be missed."
                    ),
                    category=Category.CROSS_MODULE,
                    severity=Severity.LOW,
                    file_path=rel,
                    line_number=line_no,
                    suggestion="Import NUM_JOINTS from src.safety.limits or src.interface.d1_connection.",
                    check_name=CHECK_NAME,
                ))

    return issues


def _check_import_cycles(config: QAConfig) -> list[Issue]:
    """Detect circular import dependencies."""
    issues: list[Issue] = []

    # Build import graph
    import_graph: dict[str, set[str]] = defaultdict(set)

    for py_file in config.src_dir.rglob("*.py"):
        rel = py_file.relative_to(config.project_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)

        source = py_file.read_text(encoding="utf-8")
        import_pattern = re.compile(
            r"^\s*(?:from\s+(src\.\S+)\s+import|import\s+(src\.\S+))",
            re.MULTILINE,
        )
        for match in import_pattern.finditer(source):
            imported = (match.group(1) or match.group(2)).split(".")[0:3]
            imported_mod = ".".join(imported)
            if imported_mod != module_name:
                import_graph[module_name].add(imported_mod)

    # Simple cycle detection (DFS)
    visited = set()
    rec_stack = set()

    def _dfs(node: str, path: list[str]) -> list[str] | None:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in import_graph.get(node, set()):
            if neighbor not in visited:
                result = _dfs(neighbor, path + [neighbor])
                if result:
                    return result
            elif neighbor in rec_stack:
                return path + [neighbor]
        rec_stack.discard(node)
        return None

    for module in import_graph:
        if module not in visited:
            cycle = _dfs(module, [module])
            if cycle:
                cycle_str = " -> ".join(cycle)
                issues.append(Issue(
                    title=f"Circular import detected",
                    description=(
                        f"Circular import chain: {cycle_str}. "
                        f"Circular imports can cause ImportError at runtime "
                        f"depending on the order modules are loaded."
                    ),
                    category=Category.CROSS_MODULE,
                    severity=Severity.HIGH,
                    suggestion="Break the cycle by moving shared types to a separate module or using deferred imports.",
                    check_name=CHECK_NAME,
                ))
                break  # One cycle report is enough

    return issues


def _check_interface_contracts(config: QAConfig) -> list[Issue]:
    """Check that D1State and D1Command are used with correct field names."""
    issues: list[Issue] = []

    # Canonical fields
    state_fields = {
        "joint_positions", "joint_velocities", "joint_torques",
        "gripper_position", "timestamp",
    }
    command_fields = {
        "mode", "joint_positions", "joint_velocities",
        "joint_torques", "gripper_position",
    }

    for py_file in config.src_dir.rglob("*.py"):
        rel = str(py_file.relative_to(config.project_root))
        source = py_file.read_text(encoding="utf-8")

        # Check for typos in state field access
        state_access = re.compile(r"state\.(\w+)")
        if "D1State" in source or "state" in source.lower():
            for match in state_access.finditer(source):
                field = match.group(1)
                if field.startswith("_"):
                    continue
                # Check for common misspellings
                typo_candidates = {
                    "positions": "joint_positions",
                    "velocities": "joint_velocities",
                    "torques": "joint_torques",
                    "gripper": "gripper_position",
                    "joints": "joint_positions",
                }
                if field in typo_candidates:
                    issues.append(Issue(
                        title=f"Possible field name error: state.{field}",
                        description=(
                            f"In {rel}, 'state.{field}' is accessed but the correct "
                            f"field name is '{typo_candidates[field]}'."
                        ),
                        category=Category.CROSS_MODULE,
                        severity=Severity.HIGH,
                        file_path=rel,
                        suggestion=f"Change 'state.{field}' to 'state.{typo_candidates[field]}'.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _check_test_coverage_gaps(config: QAConfig) -> list[Issue]:
    """Check that every src module has a corresponding test file."""
    issues: list[Issue] = []

    # Get all source modules (not __init__)
    src_modules = set()
    for py_file in config.src_dir.rglob("*.py"):
        if py_file.name == "__init__.py" or "__pycache__" in str(py_file):
            continue
        rel = py_file.relative_to(config.src_dir)
        module_name = str(rel.with_suffix("")).replace("/", "_")
        src_modules.add((module_name, str(py_file.relative_to(config.project_root))))

    # Get all test files
    test_files = set()
    for test_file in config.tests_dir.rglob("test_*.py"):
        name = test_file.stem  # e.g., test_kinematics
        test_files.add(name)

    # Check for missing tests
    for module_name, file_path in sorted(src_modules):
        # Generate expected test file names
        possible_test_names = [
            f"test_{module_name}",
            f"test_{module_name.split('_')[-1]}",  # last part
        ]
        # Also try just the filename
        simple_name = Path(file_path).stem
        possible_test_names.append(f"test_{simple_name}")

        if not any(tn in test_files for tn in possible_test_names):
            issues.append(Issue(
                title=f"No test file for {file_path}",
                description=(
                    f"Source module '{file_path}' has no corresponding test file. "
                    f"Expected one of: {', '.join(possible_test_names)}.py"
                ),
                category=Category.CROSS_MODULE,
                severity=Severity.LOW,
                file_path=file_path,
                suggestion=f"Create a test file (e.g., tests/{possible_test_names[0]}.py) with basic tests.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_server_module_alignment(config: QAConfig) -> list[Issue]:
    """Check that server files import modules that exist."""
    issues: list[Issue] = []

    for server_file in config.web_dir.glob("*.py"):
        rel = str(server_file.relative_to(config.project_root))
        source = server_file.read_text(encoding="utf-8")

        # Find all src.* imports
        import_pattern = re.compile(r"from\s+(src\.\S+)\s+import")
        for match in import_pattern.finditer(source):
            module_path = match.group(1)
            # Convert to file path
            file_path = config.project_root / module_path.replace(".", "/")
            py_path = file_path.with_suffix(".py")
            pkg_path = file_path / "__init__.py"

            if not py_path.exists() and not pkg_path.exists():
                line_no = source[:match.start()].count("\n") + 1
                issues.append(Issue(
                    title=f"Import of non-existent module: {module_path}",
                    description=(
                        f"Server file {rel} imports '{module_path}' but that "
                        f"module does not exist on disk."
                    ),
                    category=Category.CROSS_MODULE,
                    severity=Severity.HIGH,
                    file_path=rel,
                    line_number=line_no,
                    suggestion="Fix the import path or create the missing module.",
                    check_name=CHECK_NAME,
                ))

    return issues
