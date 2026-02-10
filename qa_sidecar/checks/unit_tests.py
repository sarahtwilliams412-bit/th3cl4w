"""
Check: Unit Tests

Runs the full pytest suite and converts failures/errors into plain-language issues.
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "unit_tests"


def run(config: QAConfig) -> list[Issue]:
    """Run pytest and parse results into issues."""
    issues: list[Issue] = []

    cmd = [
        "python", "-m", "pytest",
        str(config.tests_dir),
        "--tb=short",
        "-q",
        "--no-header",
        f"--timeout={config.pytest_timeout_seconds}",
        *config.pytest_extra_args,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.pytest_timeout_seconds + 30,
            cwd=str(config.project_root),
        )
    except subprocess.TimeoutExpired:
        issues.append(Issue(
            title="Pytest suite timed out",
            description=(
                f"The full test suite did not complete within "
                f"{config.pytest_timeout_seconds + 30} seconds. "
                "This suggests tests are hanging or extremely slow."
            ),
            category=Category.UNIT_TEST,
            severity=Severity.CRITICAL,
            suggestion="Investigate slow or hanging tests. Add timeouts to individual tests.",
            check_name=CHECK_NAME,
        ))
        return issues
    except FileNotFoundError:
        issues.append(Issue(
            title="pytest not found",
            description="The pytest command could not be found. It may not be installed.",
            category=Category.UNIT_TEST,
            severity=Severity.CRITICAL,
            suggestion="Install pytest: pip install pytest pytest-asyncio",
            check_name=CHECK_NAME,
        ))
        return issues

    output = result.stdout + "\n" + result.stderr

    # Parse the summary line (e.g., "5 failed, 42 passed, 3 errors")
    summary_match = re.search(
        r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error",
        output,
    )

    # Extract individual failures using FAILED pattern
    failure_pattern = re.compile(r"FAILED\s+([\w/\\.]+)::(\w+)(?:::(\w+))?\s*-\s*(.*)")
    for match in failure_pattern.finditer(output):
        file_path = match.group(1)
        test_class = match.group(2)
        test_func = match.group(3) or test_class
        reason = match.group(4).strip()

        issues.append(Issue(
            title=f"Test failure: {test_func}",
            description=(
                f"Test '{test_func}' in {file_path} failed. "
                f"Reason: {reason}"
            ),
            category=Category.UNIT_TEST,
            severity=Severity.HIGH,
            file_path=file_path,
            suggestion=f"Fix the failing test or the code it tests. Error: {reason}",
            evidence=_extract_failure_block(output, test_func),
            check_name=CHECK_NAME,
        ))

    # Extract errors (collection errors, import errors)
    error_pattern = re.compile(r"ERROR\s+([\w/\\.]+)(?:::(\w+))?\s*-\s*(.*)")
    for match in error_pattern.finditer(output):
        file_path = match.group(1)
        func = match.group(2) or ""
        reason = match.group(3).strip()

        issues.append(Issue(
            title=f"Test error: {file_path}",
            description=(
                f"Test collection or setup error in {file_path}. "
                f"This usually means an import failed or a fixture is broken. "
                f"Detail: {reason}"
            ),
            category=Category.UNIT_TEST,
            severity=Severity.HIGH,
            file_path=file_path,
            suggestion="Fix the import or fixture error so tests can collect properly.",
            evidence=reason,
            check_name=CHECK_NAME,
        ))

    # Extract warnings count
    warning_match = re.search(r"(\d+)\s+warning", output)
    if warning_match:
        count = int(warning_match.group(1))
        if count > 20:
            issues.append(Issue(
                title=f"Excessive test warnings ({count})",
                description=(
                    f"The test suite produced {count} warnings. "
                    "Excessive warnings can mask real problems and slow down test output."
                ),
                category=Category.UNIT_TEST,
                severity=Severity.LOW,
                suggestion="Address deprecation warnings and filter expected warnings in conftest.py.",
                check_name=CHECK_NAME,
            ))

    # If pytest returned non-zero but we didn't parse specific failures
    if result.returncode != 0 and not issues:
        issues.append(Issue(
            title="Pytest exited with non-zero status",
            description=(
                f"pytest exited with code {result.returncode} but no specific "
                f"failure patterns were parsed. This may indicate a configuration "
                f"error or crash."
            ),
            category=Category.UNIT_TEST,
            severity=Severity.HIGH,
            suggestion="Run pytest manually to see full output.",
            evidence=output[-1000:] if len(output) > 1000 else output,
            check_name=CHECK_NAME,
        ))

    # If everything passed, log as info
    if result.returncode == 0 and not issues:
        passed_match = re.search(r"(\d+)\s+passed", output)
        count = passed_match.group(1) if passed_match else "?"
        issues.append(Issue(
            title=f"All {count} tests passed",
            description=f"The full pytest suite passed with {count} tests.",
            category=Category.UNIT_TEST,
            severity=Severity.INFO,
            check_name=CHECK_NAME,
        ))

    return issues


def _extract_failure_block(output: str, test_name: str) -> str:
    """Extract the traceback block for a specific test failure."""
    lines = output.split("\n")
    block = []
    capturing = False
    for line in lines:
        if test_name in line and ("FAILED" in line or "___" in line):
            capturing = True
            block = [line]
            continue
        if capturing:
            block.append(line)
            if line.startswith("FAILED") or line.startswith("====") or len(block) > 30:
                break
    return "\n".join(block[:30]) if block else ""
