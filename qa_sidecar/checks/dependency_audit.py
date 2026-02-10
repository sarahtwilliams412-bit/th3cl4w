"""
Check: Dependency Audit

Validates that dependencies are consistent between pyproject.toml and
requirements.txt, checks for known issues, and verifies installed versions.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "dependency_audit"


def run(config: QAConfig) -> list[Issue]:
    """Run all dependency checks."""
    issues: list[Issue] = []

    issues.extend(_check_dependency_consistency(config))
    issues.extend(_check_installed_versions(config))
    issues.extend(_check_missing_deps(config))
    issues.extend(_check_unused_deps(config))

    return issues


def _parse_requirements(text: str) -> dict[str, str]:
    """Parse a requirements file into {name: version_spec}."""
    deps = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse name>=version, name==version, etc.
        match = re.match(r"^([a-zA-Z0-9_-]+)(?:\[.*?\])?\s*(.*)", line)
        if match:
            name = match.group(1).lower().replace("-", "_")
            version = match.group(2).strip()
            deps[name] = version
    return deps


def _parse_pyproject_deps(text: str) -> dict[str, str]:
    """Parse dependencies from pyproject.toml content."""
    deps = {}
    in_deps = False
    in_dev = False
    bracket_depth = 0

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("dependencies = ["):
            in_deps = True
            bracket_depth = 1
            continue
        if 'dev = [' in stripped:
            in_dev = True
            bracket_depth = 1
            continue

        if in_deps or in_dev:
            if "]" in stripped:
                in_deps = False
                in_dev = False
                continue
            # Parse "package>=version",
            match = re.match(r'"([a-zA-Z0-9_-]+)(?:\[.*?\])?\s*(.*?)"', stripped)
            if match:
                name = match.group(1).lower().replace("-", "_")
                version = match.group(2).strip().rstrip('",')
                deps[name] = version

    return deps


def _check_dependency_consistency(config: QAConfig) -> list[Issue]:
    """Check that pyproject.toml and requirements.txt are consistent."""
    issues: list[Issue] = []

    pyproject_path = config.project_root / "pyproject.toml"
    req_path = config.project_root / "requirements.txt"

    if not pyproject_path.exists() or not req_path.exists():
        if not pyproject_path.exists():
            issues.append(Issue(
                title="Missing pyproject.toml",
                description="No pyproject.toml found at project root.",
                category=Category.DEPENDENCY,
                severity=Severity.HIGH,
                check_name=CHECK_NAME,
            ))
        return issues

    pyproject_deps = _parse_pyproject_deps(pyproject_path.read_text(encoding="utf-8"))
    req_deps = _parse_requirements(req_path.read_text(encoding="utf-8"))

    # Check for deps in requirements.txt but not in pyproject.toml
    for name in req_deps:
        if name not in pyproject_deps and name not in ("pip", "setuptools", "wheel"):
            issues.append(Issue(
                title=f"Dependency '{name}' in requirements.txt but not pyproject.toml",
                description=(
                    f"Package '{name}' is listed in requirements.txt but not in "
                    f"pyproject.toml dependencies. These files should be kept in sync."
                ),
                category=Category.DEPENDENCY,
                severity=Severity.LOW,
                file_path="requirements.txt",
                suggestion="Add the dependency to pyproject.toml or remove from requirements.txt.",
                check_name=CHECK_NAME,
            ))

    # Check for version mismatches
    for name in pyproject_deps:
        if name in req_deps:
            if pyproject_deps[name] and req_deps[name]:
                if pyproject_deps[name] != req_deps[name]:
                    issues.append(Issue(
                        title=f"Version mismatch for '{name}'",
                        description=(
                            f"Package '{name}' has different version specs: "
                            f"pyproject.toml says '{pyproject_deps[name]}', "
                            f"requirements.txt says '{req_deps[name]}'."
                        ),
                        category=Category.DEPENDENCY,
                        severity=Severity.MEDIUM,
                        suggestion="Align the version specifications between both files.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _check_installed_versions(config: QAConfig) -> list[Issue]:
    """Check that installed packages match requirements."""
    issues: list[Issue] = []

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return issues

        import json
        installed = {}
        for pkg in json.loads(result.stdout):
            name = pkg["name"].lower().replace("-", "_")
            installed[name] = pkg["version"]

    except Exception:
        return issues

    req_path = config.project_root / "requirements.txt"
    if not req_path.exists():
        return issues

    req_deps = _parse_requirements(req_path.read_text(encoding="utf-8"))

    for name, spec in req_deps.items():
        if not spec:
            continue
        norm_name = name.lower().replace("-", "_")
        if norm_name not in installed:
            issues.append(Issue(
                title=f"Required package '{name}' not installed",
                description=(
                    f"Package '{name}' ({spec}) is listed in requirements.txt "
                    f"but is not installed in the current environment."
                ),
                category=Category.DEPENDENCY,
                severity=Severity.HIGH,
                suggestion=f"Install with: pip install '{name}{spec}'",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_missing_deps(config: QAConfig) -> list[Issue]:
    """Look for imports in source code that aren't listed as dependencies."""
    issues: list[Issue] = []

    # Get all declared deps
    pyproject_path = config.project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return issues

    all_deps = _parse_pyproject_deps(pyproject_path.read_text(encoding="utf-8"))
    declared_names = set(all_deps.keys())

    # Common stdlib modules to ignore
    stdlib = {
        "os", "sys", "re", "json", "time", "math", "logging", "pathlib",
        "typing", "dataclasses", "enum", "collections", "functools",
        "itertools", "abc", "copy", "io", "struct", "socket", "threading",
        "asyncio", "unittest", "subprocess", "tempfile", "hashlib",
        "base64", "datetime", "traceback", "contextlib", "signal",
        "importlib", "inspect", "textwrap", "argparse", "shutil",
        "sqlite3", "uuid", "secrets", "types", "array", "warnings",
        "concurrent", "queue", "statistics",
    }

    # Collect all third-party imports used in source code
    import_pattern = re.compile(r"^\s*(?:from|import)\s+(\w+)", re.MULTILINE)
    used_packages: set[str] = set()

    for py_file in config.src_dir.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        for match in import_pattern.finditer(source):
            pkg = match.group(1).lower()
            if pkg not in stdlib and not pkg.startswith("src") and not pkg.startswith("web"):
                used_packages.add(pkg)

    # Map common package import names to pip names
    import_to_pip = {
        "cv2": "opencv_python",
        "yaml": "pyyaml",
        "zmq": "pyzmq",
        "PIL": "pillow",
        "sklearn": "scikit_learn",
        "google": "google_generativeai",
        "dotenv": "python_dotenv",
    }

    for pkg in used_packages:
        pip_name = import_to_pip.get(pkg, pkg)
        if pip_name not in declared_names and pkg not in declared_names:
            issues.append(Issue(
                title=f"Undeclared dependency: {pkg}",
                description=(
                    f"Package '{pkg}' is imported in source code but not declared "
                    f"in pyproject.toml or requirements.txt."
                ),
                category=Category.DEPENDENCY,
                severity=Severity.MEDIUM,
                suggestion=f"Add '{pkg}' to the dependencies in pyproject.toml.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_unused_deps(config: QAConfig) -> list[Issue]:
    """Check for declared dependencies that aren't imported anywhere."""
    issues: list[Issue] = []

    pyproject_path = config.project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return issues

    all_deps = _parse_pyproject_deps(pyproject_path.read_text(encoding="utf-8"))

    # Map pip names to import names
    pip_to_import = {
        "opencv_python": ["cv2"],
        "pyyaml": ["yaml"],
        "pyzmq": ["zmq"],
        "pillow": ["PIL"],
        "scikit_learn": ["sklearn"],
        "google_generativeai": ["google"],
        "python_dotenv": ["dotenv"],
        "uvicorn": ["uvicorn"],
        "websockets": ["websockets"],
        "httpx": ["httpx"],
        "transforms3d": ["transforms3d"],
    }

    # Collect all imports from source
    all_source = ""
    for py_file in config.src_dir.rglob("*.py"):
        all_source += py_file.read_text(encoding="utf-8") + "\n"
    for py_file in config.web_dir.rglob("*.py"):
        all_source += py_file.read_text(encoding="utf-8") + "\n"

    for dep_name in all_deps:
        if dep_name in ("setuptools", "wheel", "pip"):
            continue
        import_names = pip_to_import.get(dep_name, [dep_name])
        found = any(name in all_source for name in import_names)
        if not found:
            issues.append(Issue(
                title=f"Possibly unused dependency: {dep_name}",
                description=(
                    f"Package '{dep_name}' is declared as a dependency but no "
                    f"import of it was found in the source code."
                ),
                category=Category.DEPENDENCY,
                severity=Severity.LOW,
                suggestion="Remove the dependency if it's truly unused, or add a comment explaining indirect usage.",
                check_name=CHECK_NAME,
            ))

    return issues
