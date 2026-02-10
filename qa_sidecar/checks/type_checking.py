"""
Check: Type Checking

Runs mypy on the codebase and converts type errors into plain-language issues.
"""

from __future__ import annotations

import re
import subprocess
import sys

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "type_checking"


def run(config: QAConfig) -> list[Issue]:
    """Run mypy and parse results."""
    issues: list[Issue] = []

    cmd = [
        sys.executable, "-m", "mypy",
        str(config.src_dir),
        "--ignore-missing-imports",
        "--no-error-summary",
        "--show-column-numbers",
        "--no-color",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.mypy_timeout_seconds,
            cwd=str(config.project_root),
        )
    except subprocess.TimeoutExpired:
        issues.append(Issue(
            title="mypy timed out",
            description=f"mypy did not complete within {config.mypy_timeout_seconds} seconds.",
            category=Category.TYPE_ERROR,
            severity=Severity.MEDIUM,
            suggestion="Increase timeout or check for very large modules slowing mypy.",
            check_name=CHECK_NAME,
        ))
        return issues
    except FileNotFoundError:
        issues.append(Issue(
            title="mypy not installed",
            description="mypy is not installed. Cannot run type checking.",
            category=Category.TYPE_ERROR,
            severity=Severity.MEDIUM,
            suggestion="Install mypy: pip install mypy",
            check_name=CHECK_NAME,
        ))
        return issues

    output = result.stdout

    # Parse mypy output lines: path:line:col: severity: message
    error_pattern = re.compile(
        r"^(.+?):(\d+):(?:(\d+):)?\s*(error|warning|note):\s*(.+?)(?:\s*\[(.+?)\])?\s*$",
        re.MULTILINE,
    )

    error_count = 0
    for match in error_pattern.finditer(output):
        file_path = match.group(1)
        line_no = int(match.group(2))
        col = match.group(3)
        level = match.group(4)
        message = match.group(5).strip()
        error_code = match.group(6) or ""

        # Map mypy severity to our severity
        if level == "error":
            severity = Severity.HIGH
            error_count += 1
        elif level == "warning":
            severity = Severity.MEDIUM
        else:
            continue  # Skip notes

        # Make path relative
        try:
            rel_path = str(Path(file_path).relative_to(config.project_root))
        except (ValueError, TypeError):
            rel_path = file_path

        # Generate a plain-language description
        desc = _humanize_mypy_error(message, error_code)

        issues.append(Issue(
            title=f"Type error in {Path(rel_path).name}:{line_no}",
            description=desc,
            category=Category.TYPE_ERROR,
            severity=severity,
            file_path=rel_path,
            line_number=line_no,
            suggestion=_suggest_fix(message, error_code),
            evidence=f"mypy [{error_code}]: {message}" if error_code else f"mypy: {message}",
            check_name=CHECK_NAME,
        ))

    # Cap to avoid flooding
    if len(issues) > 50:
        count = len(issues)
        issues = issues[:50]
        issues.append(Issue(
            title=f"Type checking found {count} issues (showing first 50)",
            description=f"mypy reported {count} type errors/warnings. Only the first 50 are shown.",
            category=Category.TYPE_ERROR,
            severity=Severity.INFO,
            check_name=CHECK_NAME,
        ))

    if error_count == 0 and result.returncode == 0:
        issues.append(Issue(
            title="Type checking passed cleanly",
            description="mypy found no type errors in the source code.",
            category=Category.TYPE_ERROR,
            severity=Severity.INFO,
            check_name=CHECK_NAME,
        ))

    return issues


# Need to import Path for the relative path computation
from pathlib import Path


def _humanize_mypy_error(message: str, code: str) -> str:
    """Convert a mypy error message into plain language."""
    translations = {
        "arg-type": f"A function is being called with an argument of the wrong type. {message}",
        "return-value": f"A function returns a value that doesn't match its declared return type. {message}",
        "assignment": f"A variable is being assigned a value of an incompatible type. {message}",
        "name-defined": f"A variable or name is being used before it's defined. {message}",
        "attr-defined": f"Code is accessing an attribute that doesn't exist on the object. {message}",
        "override": f"A method override has an incompatible signature with the parent class. {message}",
        "union-attr": f"Code accesses an attribute on a union type, but not all types in the union have that attribute. {message}",
        "index": f"An invalid index type or out-of-bounds index is being used. {message}",
        "misc": message,
    }
    return translations.get(code, f"{message}")


def _suggest_fix(message: str, code: str) -> str:
    """Generate a fix suggestion based on the error type."""
    suggestions = {
        "arg-type": "Change the argument to match the expected type, or update the function signature.",
        "return-value": "Fix the return statement to match the declared return type.",
        "assignment": "Use the correct type for the assignment, or update the type annotation.",
        "name-defined": "Define the variable before using it, or fix the spelling.",
        "attr-defined": "Check if the attribute name is correct, or add it to the class.",
        "override": "Update the method signature to be compatible with the parent class.",
        "union-attr": "Add a type check (isinstance) before accessing the attribute.",
        "index": "Verify the index type and bounds.",
    }
    return suggestions.get(code, "Review the type annotation and fix the mismatch.")
