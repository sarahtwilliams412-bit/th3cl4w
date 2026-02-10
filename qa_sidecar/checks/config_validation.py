"""
Check: Configuration Validation

Validates that configuration files are well-formed, consistent with each other,
and that all referenced config values are used correctly in the codebase.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "config_validation"


def run(config: QAConfig) -> list[Issue]:
    """Run all configuration validation checks."""
    issues: list[Issue] = []

    issues.extend(_check_json_configs(config))
    issues.extend(_check_env_example(config))
    issues.extend(_check_pick_config(config))
    issues.extend(_check_pyproject_consistency(config))
    issues.extend(_check_gitignore(config))

    return issues


def _check_json_configs(config: QAConfig) -> list[Issue]:
    """Validate all JSON config files are parseable."""
    issues: list[Issue] = []
    data_dir = config.project_root / "data"

    if not data_dir.exists():
        return issues

    for json_file in data_dir.rglob("*.json"):
        rel = str(json_file.relative_to(config.project_root))
        try:
            content = json_file.read_text(encoding="utf-8")
            data = json.loads(content)

            # Check for empty config files
            if not data:
                issues.append(Issue(
                    title=f"Empty config file: {rel}",
                    description=f"Config file {rel} parses as empty (no keys/values).",
                    category=Category.CONFIG,
                    severity=Severity.LOW,
                    file_path=rel,
                    suggestion="Either populate the config or remove the empty file.",
                    check_name=CHECK_NAME,
                ))

        except json.JSONDecodeError as e:
            issues.append(Issue(
                title=f"Invalid JSON: {rel}",
                description=f"Config file {rel} contains invalid JSON: {e}",
                category=Category.CONFIG,
                severity=Severity.HIGH,
                file_path=rel,
                suggestion="Fix the JSON syntax error.",
                evidence=str(e),
                check_name=CHECK_NAME,
            ))

    return issues


def _check_env_example(config: QAConfig) -> list[Issue]:
    """Check .env.example is present and all keys are documented."""
    issues: list[Issue] = []

    env_example = config.project_root / ".env.example"
    if not env_example.exists():
        issues.append(Issue(
            title="Missing .env.example",
            description=(
                "No .env.example file found. New developers won't know which "
                "environment variables are needed."
            ),
            category=Category.CONFIG,
            severity=Severity.MEDIUM,
            suggestion="Create .env.example listing all required environment variables with placeholder values.",
            check_name=CHECK_NAME,
        ))
        return issues

    env_content = env_example.read_text(encoding="utf-8")
    env_keys = set()
    for line in env_content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key = line.split("=")[0].strip()
            env_keys.add(key)

    # Search source for os.getenv / os.environ calls
    getenv_pattern = re.compile(r'(?:os\.getenv|os\.environ(?:\.get)?)\s*\(\s*["\'](\w+)["\']')

    used_env_keys: set[str] = set()
    for py_file in list(config.src_dir.rglob("*.py")) + list(config.web_dir.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        for match in getenv_pattern.finditer(source):
            used_env_keys.add(match.group(1))

    # Keys used in code but not in .env.example
    for key in used_env_keys - env_keys:
        issues.append(Issue(
            title=f"Undocumented env var: {key}",
            description=(
                f"Environment variable '{key}' is used in source code but not "
                f"listed in .env.example."
            ),
            category=Category.CONFIG,
            severity=Severity.MEDIUM,
            file_path=".env.example",
            suggestion=f"Add '{key}=<placeholder>' to .env.example.",
            check_name=CHECK_NAME,
        ))

    return issues


def _check_pick_config(config: QAConfig) -> list[Issue]:
    """Validate pick_config.py values are sensible."""
    issues: list[Issue] = []
    pick_config_path = config.src_dir / "config" / "pick_config.py"

    if not pick_config_path.exists():
        return issues

    source = pick_config_path.read_text(encoding="utf-8")

    # Check for negative heights or distances
    number_pattern = re.compile(r"(\w+)\s*[:=]\s*([-\d.]+)")
    for match in number_pattern.finditer(source):
        name = match.group(1)
        value = match.group(2)
        try:
            val = float(value)
        except ValueError:
            continue

        if "height" in name.lower() or "distance" in name.lower() or "radius" in name.lower():
            if val < 0:
                issues.append(Issue(
                    title=f"Negative physical value: {name}={val}",
                    description=(
                        f"Configuration value '{name}' is {val}, which is negative. "
                        f"Heights, distances, and radii should be non-negative."
                    ),
                    category=Category.CONFIG,
                    severity=Severity.HIGH,
                    file_path="src/config/pick_config.py",
                    suggestion=f"Change '{name}' to a non-negative value.",
                    check_name=CHECK_NAME,
                ))

        if "timeout" in name.lower() and val <= 0:
            issues.append(Issue(
                title=f"Non-positive timeout: {name}={val}",
                description=f"Timeout '{name}' is {val}. Timeouts must be positive.",
                category=Category.CONFIG,
                severity=Severity.HIGH,
                file_path="src/config/pick_config.py",
                suggestion=f"Set '{name}' to a positive value.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_pyproject_consistency(config: QAConfig) -> list[Issue]:
    """Check pyproject.toml for common issues."""
    issues: list[Issue] = []
    pyproject_path = config.project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return issues

    content = pyproject_path.read_text(encoding="utf-8")

    # Check Python version constraint
    if "requires-python" not in content:
        issues.append(Issue(
            title="No requires-python in pyproject.toml",
            description="pyproject.toml does not specify a minimum Python version.",
            category=Category.CONFIG,
            severity=Severity.MEDIUM,
            file_path="pyproject.toml",
            suggestion="Add requires-python = '>=3.10' to [project].",
            check_name=CHECK_NAME,
        ))

    # Check test configuration
    if "[tool.pytest.ini_options]" not in content:
        issues.append(Issue(
            title="No pytest configuration in pyproject.toml",
            description="pyproject.toml has no [tool.pytest.ini_options] section.",
            category=Category.CONFIG,
            severity=Severity.LOW,
            file_path="pyproject.toml",
            suggestion="Add pytest configuration to standardize test execution.",
            check_name=CHECK_NAME,
        ))

    return issues


def _check_gitignore(config: QAConfig) -> list[Issue]:
    """Check .gitignore covers sensitive and generated files."""
    issues: list[Issue] = []
    gitignore_path = config.project_root / ".gitignore"

    if not gitignore_path.exists():
        issues.append(Issue(
            title="Missing .gitignore",
            description="No .gitignore file found. Sensitive files may be committed.",
            category=Category.CONFIG,
            severity=Severity.HIGH,
            suggestion="Create a .gitignore with common Python exclusions.",
            check_name=CHECK_NAME,
        ))
        return issues

    content = gitignore_path.read_text(encoding="utf-8")
    required_patterns = [
        (".env", "Environment variables with secrets"),
        ("__pycache__", "Python bytecode cache"),
        ("*.pyc", "Compiled Python files"),
        (".db", "Database files"),
    ]

    for pattern, description in required_patterns:
        if pattern not in content:
            issues.append(Issue(
                title=f".gitignore missing pattern: {pattern}",
                description=f".gitignore does not exclude '{pattern}' ({description}).",
                category=Category.CONFIG,
                severity=Severity.MEDIUM if pattern == ".env" else Severity.LOW,
                file_path=".gitignore",
                suggestion=f"Add '{pattern}' to .gitignore.",
                check_name=CHECK_NAME,
            ))

    return issues
