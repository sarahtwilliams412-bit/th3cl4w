"""
Check: Code Quality

Analyzes code for complexity, long functions, dead code patterns,
and other quality issues that make the codebase harder to maintain.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "code_quality"


def run(config: QAConfig) -> list[Issue]:
    """Run all code quality checks."""
    issues: list[Issue] = []

    py_files = list(config.src_dir.rglob("*.py")) + list(config.web_dir.rglob("*.py"))

    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(config.project_root))
        source = py_file.read_text(encoding="utf-8")

        issues.extend(_check_file_length(source, rel, config))
        issues.extend(_check_function_complexity(source, rel, config))
        issues.extend(_check_dead_code_patterns(source, rel))
        issues.extend(_check_todo_fixme(source, rel))
        issues.extend(_check_print_statements(source, rel))
        issues.extend(_check_broad_exceptions(source, rel))

    return issues


def _check_file_length(source: str, file_path: str, config: QAConfig) -> list[Issue]:
    """Flag files that are too long."""
    lines = source.count("\n") + 1
    if lines > config.max_file_lines:
        return [Issue(
            title=f"Long file: {file_path} ({lines} lines)",
            description=(
                f"File {file_path} is {lines} lines long (threshold: {config.max_file_lines}). "
                f"Long files are harder to understand, test, and maintain."
            ),
            category=Category.CODE_QUALITY,
            severity=Severity.LOW,
            file_path=file_path,
            suggestion="Consider splitting into smaller, focused modules.",
            check_name=CHECK_NAME,
        )]
    return []


def _check_function_complexity(source: str, file_path: str, config: QAConfig) -> list[Issue]:
    """Check for overly complex or long functions."""
    issues: list[Issue] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name

            # Check function length
            if hasattr(node, "end_lineno") and node.end_lineno:
                func_lines = node.end_lineno - node.lineno
                if func_lines > config.max_function_lines:
                    issues.append(Issue(
                        title=f"Long function: {func_name} ({func_lines} lines)",
                        description=(
                            f"Function '{func_name}' in {file_path} is {func_lines} lines. "
                            f"Long functions are hard to test and reason about."
                        ),
                        category=Category.CODE_QUALITY,
                        severity=Severity.MEDIUM,
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Break this function into smaller helper functions.",
                        check_name=CHECK_NAME,
                    ))

            # Check cyclomatic complexity (count branches)
            complexity = _count_complexity(node)
            if complexity > config.max_function_complexity:
                issues.append(Issue(
                    title=f"Complex function: {func_name} (complexity={complexity})",
                    description=(
                        f"Function '{func_name}' in {file_path} has cyclomatic complexity "
                        f"of {complexity} (threshold: {config.max_function_complexity}). "
                        f"High complexity means many execution paths, making the function "
                        f"difficult to test exhaustively and prone to bugs."
                    ),
                    category=Category.CODE_QUALITY,
                    severity=Severity.MEDIUM,
                    file_path=file_path,
                    line_number=node.lineno,
                    suggestion="Refactor to reduce branching. Extract conditional logic into helper functions.",
                    check_name=CHECK_NAME,
                ))

            # Check too many parameters
            params = node.args
            num_params = (
                len(params.args)
                + len(params.posonlyargs)
                + len(params.kwonlyargs)
            )
            # Subtract 'self' or 'cls'
            if params.args and params.args[0].arg in ("self", "cls"):
                num_params -= 1
            if num_params > 8:
                issues.append(Issue(
                    title=f"Too many parameters: {func_name} ({num_params} params)",
                    description=(
                        f"Function '{func_name}' in {file_path} takes {num_params} parameters. "
                        f"Functions with many parameters are hard to call correctly."
                    ),
                    category=Category.CODE_QUALITY,
                    severity=Severity.LOW,
                    file_path=file_path,
                    line_number=node.lineno,
                    suggestion="Group related parameters into a dataclass or config object.",
                    check_name=CHECK_NAME,
                ))

    return issues


def _count_complexity(node: ast.AST) -> int:
    """Count cyclomatic complexity of a function node."""
    complexity = 1  # Base path
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each 'and'/'or' adds a path
            complexity += len(child.values) - 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            complexity += 1
    return complexity


def _check_dead_code_patterns(source: str, file_path: str) -> list[Issue]:
    """Detect obvious dead code patterns."""
    issues: list[Issue] = []
    lines = source.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Unreachable code after return/raise/break/continue
        if i < len(lines):
            if stripped in ("return", "raise", "break", "continue") or stripped.startswith(
                ("return ", "raise ", "break ", "continue ")
            ):
                # Check if next non-empty, non-comment line is at same indent
                for j in range(i, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line or next_line.startswith("#"):
                        continue
                    # Check indent level
                    curr_indent = len(line) - len(line.lstrip())
                    next_indent = len(lines[j]) - len(lines[j].lstrip())
                    if next_indent >= curr_indent and next_line not in ("", "else:", "elif", "except:", "finally:"):
                        if not next_line.startswith(("def ", "class ", "async def ", "@", "elif ", "else:", "except", "finally:")):
                            issues.append(Issue(
                                title=f"Possible unreachable code at {file_path}:{j+1}",
                                description=(
                                    f"Code at line {j+1} appears unreachable after "
                                    f"'{stripped}' at line {i}."
                                ),
                                category=Category.CODE_QUALITY,
                                severity=Severity.LOW,
                                file_path=file_path,
                                line_number=j + 1,
                                suggestion="Remove dead code or restructure the control flow.",
                                check_name=CHECK_NAME,
                            ))
                    break

    return issues


def _check_todo_fixme(source: str, file_path: str) -> list[Issue]:
    """Report TODO/FIXME/HACK/XXX comments."""
    issues: list[Issue] = []
    pattern = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BROKEN)\s*:?\s*(.*)", re.IGNORECASE)

    for i, line in enumerate(source.split("\n"), 1):
        match = pattern.search(line)
        if match:
            tag = match.group(1).upper()
            text = match.group(2).strip()
            sev = Severity.LOW
            if tag in ("FIXME", "BROKEN"):
                sev = Severity.MEDIUM
            if tag == "HACK":
                sev = Severity.MEDIUM

            issues.append(Issue(
                title=f"{tag} comment in {file_path}:{i}",
                description=f"{tag}: {text}" if text else f"Unmarked {tag} found.",
                category=Category.CODE_QUALITY,
                severity=sev,
                file_path=file_path,
                line_number=i,
                suggestion=f"Address the {tag} item or remove it if resolved.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_print_statements(source: str, file_path: str) -> list[Issue]:
    """Flag print() statements that should be logger calls."""
    issues: list[Issue] = []

    # Skip test files and scripts
    if "test" in file_path.lower() or "script" in file_path.lower():
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                issues.append(Issue(
                    title=f"print() in production code: {file_path}:{node.lineno}",
                    description=(
                        f"File {file_path} uses print() at line {node.lineno}. "
                        f"Production code should use the logging module instead."
                    ),
                    category=Category.CODE_QUALITY,
                    severity=Severity.LOW,
                    file_path=file_path,
                    line_number=node.lineno,
                    suggestion="Replace print() with logger.info() or logger.debug().",
                    check_name=CHECK_NAME,
                ))

    return issues


def _check_broad_exceptions(source: str, file_path: str) -> list[Issue]:
    """Flag overly broad exception handling."""
    issues: list[Issue] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(Issue(
                    title=f"Bare except in {file_path}:{node.lineno}",
                    description=(
                        f"Line {node.lineno} has a bare 'except:' that catches everything "
                        f"including KeyboardInterrupt and SystemExit."
                    ),
                    category=Category.CODE_QUALITY,
                    severity=Severity.MEDIUM,
                    file_path=file_path,
                    line_number=node.lineno,
                    suggestion="Catch specific exceptions like 'except ValueError:' or at minimum 'except Exception:'.",
                    check_name=CHECK_NAME,
                ))

    return issues
