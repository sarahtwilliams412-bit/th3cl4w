"""
Check: API Contracts

Validates that all FastAPI endpoints are properly defined, have correct
response models, and that route patterns are consistent. Also checks for
common API anti-patterns.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "api_contracts"

# Expected server files
SERVER_FILES = [
    "web/server.py",
    "web/v2_server.py",
    "web/camera_server.py",
    "web/ascii_server.py",
    "web/location_server.py",
    "web/map_server.py",
]


def run(config: QAConfig) -> list[Issue]:
    """Analyze API endpoint definitions for consistency and correctness."""
    issues: list[Issue] = []

    for server_rel in SERVER_FILES:
        server_path = config.project_root / server_rel
        if not server_path.exists():
            issues.append(Issue(
                title=f"Missing server file: {server_rel}",
                description=f"Expected server file '{server_rel}' does not exist.",
                category=Category.API_CONTRACT,
                severity=Severity.MEDIUM,
                file_path=server_rel,
                suggestion="Verify the file hasn't been moved or renamed.",
                check_name=CHECK_NAME,
            ))
            continue

        source = server_path.read_text(encoding="utf-8")
        file_issues = _analyze_server_file(source, server_rel)
        issues.extend(file_issues)

    # Cross-server checks
    issues.extend(_check_route_conflicts(config))
    issues.extend(_check_websocket_handlers(config))

    return issues


def _analyze_server_file(source: str, file_path: str) -> list[Issue]:
    """Analyze a single server file for API issues."""
    issues: list[Issue] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(Issue(
            title=f"Syntax error in {file_path}",
            description=f"Cannot parse {file_path}: {e}",
            category=Category.API_CONTRACT,
            severity=Severity.CRITICAL,
            file_path=file_path,
            line_number=e.lineno,
            suggestion="Fix the syntax error.",
            check_name=CHECK_NAME,
        ))
        return issues

    lines = source.split("\n")

    # Find route decorators
    route_pattern = re.compile(
        r'@\w+\.(get|post|put|delete|patch|websocket)\s*\(\s*["\']([^"\']+)["\']'
    )
    routes_found = []
    for i, line in enumerate(lines, 1):
        match = route_pattern.search(line)
        if match:
            method = match.group(1).upper()
            path = match.group(2)
            routes_found.append((method, path, i))

    # Check: routes without error handling
    for method, path, line_no in routes_found:
        # Look at the function body for try/except
        func_body = _get_function_body_after_line(lines, line_no)
        if func_body and "try" not in func_body and method in ("POST", "PUT", "DELETE"):
            issues.append(Issue(
                title=f"No error handling in {method} {path}",
                description=(
                    f"The endpoint {method} {path} in {file_path} has no try/except. "
                    f"Mutation endpoints should handle errors gracefully to return "
                    f"proper HTTP error responses instead of 500s."
                ),
                category=Category.API_CONTRACT,
                severity=Severity.MEDIUM,
                file_path=file_path,
                line_number=line_no,
                suggestion="Wrap the handler body in try/except and return appropriate HTTP status codes.",
                check_name=CHECK_NAME,
            ))

    # Check: inconsistent route prefixes
    prefixes = set()
    for _, path, _ in routes_found:
        parts = path.strip("/").split("/")
        if len(parts) >= 2:
            prefixes.add(f"/{parts[0]}/{parts[1]}")
    if len(prefixes) > 3:
        issues.append(Issue(
            title=f"Many route prefixes in {file_path}",
            description=(
                f"Found {len(prefixes)} distinct route prefixes in {file_path}: "
                f"{', '.join(sorted(prefixes)[:5])}... "
                f"This suggests the file handles too many concerns."
            ),
            category=Category.API_CONTRACT,
            severity=Severity.LOW,
            file_path=file_path,
            suggestion="Consider splitting into focused sub-routers using FastAPI's APIRouter.",
            check_name=CHECK_NAME,
        ))

    # Check: duplicate routes
    route_keys = [(m, p) for m, p, _ in routes_found]
    seen = set()
    for key in route_keys:
        if key in seen:
            issues.append(Issue(
                title=f"Duplicate route: {key[0]} {key[1]}",
                description=f"The route {key[0]} {key[1]} is defined more than once in {file_path}.",
                category=Category.API_CONTRACT,
                severity=Severity.HIGH,
                file_path=file_path,
                suggestion="Remove the duplicate route definition.",
                check_name=CHECK_NAME,
            ))
        seen.add(key)

    # Check: bare exception handlers
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "except:" or stripped == "except Exception:":
            issues.append(Issue(
                title=f"Bare except in {file_path}:{i}",
                description=(
                    f"Line {i} has a broad exception handler that catches all errors. "
                    f"This can silently swallow important errors and make debugging harder."
                ),
                category=Category.API_CONTRACT,
                severity=Severity.MEDIUM,
                file_path=file_path,
                line_number=i,
                suggestion="Catch specific exception types and log/re-raise unexpected ones.",
                check_name=CHECK_NAME,
            ))

    return issues


def _get_function_body_after_line(lines: list[str], decorator_line: int) -> str:
    """Extract the function body text following a decorator."""
    body_lines = []
    in_func = False
    indent = None
    for i in range(decorator_line, min(decorator_line + 50, len(lines))):
        line = lines[i]
        if not in_func:
            if line.strip().startswith("def ") or line.strip().startswith("async def "):
                in_func = True
            continue
        if in_func:
            if indent is None:
                # Find the indentation level of the function body
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    indent = len(line) - len(stripped)
            if indent is not None and line.strip() and not line.startswith(" " * indent):
                break
            body_lines.append(line)
    return "\n".join(body_lines)


def _check_route_conflicts(config: QAConfig) -> list[Issue]:
    """Check for route conflicts across multiple server files."""
    issues: list[Issue] = []
    all_routes: dict[tuple[str, str], list[str]] = {}

    route_pattern = re.compile(
        r'@\w+\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
    )

    for server_rel in SERVER_FILES:
        server_path = config.project_root / server_rel
        if not server_path.exists():
            continue
        source = server_path.read_text(encoding="utf-8")
        for match in route_pattern.finditer(source):
            method = match.group(1).upper()
            path = match.group(2)
            key = (method, path)
            all_routes.setdefault(key, []).append(server_rel)

    for (method, path), files in all_routes.items():
        if len(files) > 1:
            issues.append(Issue(
                title=f"Route conflict: {method} {path}",
                description=(
                    f"The route {method} {path} is defined in multiple files: "
                    f"{', '.join(files)}. This will cause unpredictable behavior "
                    f"depending on which server mounts first."
                ),
                category=Category.API_CONTRACT,
                severity=Severity.HIGH,
                suggestion="Consolidate the route to a single server file.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_websocket_handlers(config: QAConfig) -> list[Issue]:
    """Check WebSocket handlers for common issues."""
    issues: list[Issue] = []

    for server_rel in SERVER_FILES:
        server_path = config.project_root / server_rel
        if not server_path.exists():
            continue
        source = server_path.read_text(encoding="utf-8")

        # Check WebSocket handlers have disconnect handling
        ws_pattern = re.compile(r'@\w+\.websocket\s*\(\s*["\']([^"\']+)["\']')
        for match in ws_pattern.finditer(source):
            ws_path = match.group(1)
            # Look for WebSocketDisconnect handling near this route
            start = match.start()
            chunk = source[start:start + 2000]
            if "WebSocketDisconnect" not in chunk and "disconnect" not in chunk.lower():
                issues.append(Issue(
                    title=f"WebSocket {ws_path} may not handle disconnects",
                    description=(
                        f"The WebSocket handler for '{ws_path}' in {server_rel} "
                        f"does not appear to catch WebSocketDisconnect. "
                        f"This can cause unhandled exceptions when clients disconnect."
                    ),
                    category=Category.API_CONTRACT,
                    severity=Severity.MEDIUM,
                    file_path=server_rel,
                    suggestion="Add a try/except for WebSocketDisconnect in the handler.",
                    check_name=CHECK_NAME,
                ))

    return issues
