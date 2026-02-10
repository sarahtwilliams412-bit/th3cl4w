"""
Check: Frontend Checks

Validates HTML, JavaScript, and CSS files in the web static directory.
Checks for broken references, accessibility issues, and common web
development problems.
"""

from __future__ import annotations

import re
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import Issue, Severity, Category

CHECK_NAME = "frontend_checks"


def run(config: QAConfig) -> list[Issue]:
    """Run all frontend checks."""
    issues: list[Issue] = []

    if not config.static_dir.exists():
        issues.append(Issue(
            title="Static directory missing",
            description=f"Expected static directory at {config.static_dir} but it doesn't exist.",
            category=Category.FRONTEND,
            severity=Severity.HIGH,
            check_name=CHECK_NAME,
        ))
        return issues

    issues.extend(_check_html_files(config))
    issues.extend(_check_js_files(config))
    issues.extend(_check_broken_references(config))
    issues.extend(_check_api_endpoint_references(config))

    return issues


def _check_html_files(config: QAConfig) -> list[Issue]:
    """Validate HTML files for common problems."""
    issues: list[Issue] = []

    for html_file in config.static_dir.rglob("*.html"):
        rel = str(html_file.relative_to(config.project_root))
        content = html_file.read_text(encoding="utf-8")

        # Check for missing doctype
        if "<!DOCTYPE" not in content.upper() and "<!doctype" not in content:
            issues.append(Issue(
                title=f"Missing DOCTYPE: {rel}",
                description=f"HTML file {rel} is missing <!DOCTYPE html> declaration.",
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion="Add <!DOCTYPE html> as the first line.",
                check_name=CHECK_NAME,
            ))

        # Check for missing meta charset
        if '<meta charset' not in content.lower() and 'charset=' not in content.lower():
            issues.append(Issue(
                title=f"Missing charset: {rel}",
                description=f"HTML file {rel} doesn't declare character encoding.",
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion='Add <meta charset="utf-8"> to the <head>.',
                check_name=CHECK_NAME,
            ))

        # Check for inline event handlers (security concern)
        inline_events = re.findall(r'\bon\w+\s*=\s*["\']', content)
        if len(inline_events) > 5:
            issues.append(Issue(
                title=f"Excessive inline event handlers in {rel}",
                description=(
                    f"Found {len(inline_events)} inline event handlers in {rel}. "
                    f"Inline handlers make CSP policies difficult and code harder to maintain."
                ),
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion="Move event handlers to JavaScript with addEventListener().",
                check_name=CHECK_NAME,
            ))

        # Check for hardcoded localhost URLs
        localhost_urls = re.findall(r'(?:http://|ws://)(?:localhost|127\.0\.0\.1):\d+', content)
        for url in localhost_urls:
            issues.append(Issue(
                title=f"Hardcoded localhost URL in {rel}",
                description=(
                    f"Found hardcoded URL '{url}' in {rel}. "
                    f"This will break when deployed to a different host."
                ),
                category=Category.FRONTEND,
                severity=Severity.MEDIUM,
                file_path=rel,
                suggestion="Use relative URLs or compute the host dynamically from window.location.",
                check_name=CHECK_NAME,
            ))

        # Check for console.log in production HTML
        console_logs = re.findall(r'console\.log\s*\(', content)
        if len(console_logs) > 10:
            issues.append(Issue(
                title=f"Many console.log calls in {rel} ({len(console_logs)})",
                description=f"Found {len(console_logs)} console.log() calls in {rel}.",
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion="Remove or gate console.log calls behind a DEBUG flag.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_js_files(config: QAConfig) -> list[Issue]:
    """Check JavaScript files for common issues."""
    issues: list[Issue] = []

    for js_file in config.static_dir.rglob("*.js"):
        rel = str(js_file.relative_to(config.project_root))
        content = js_file.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Check for var usage (should use let/const)
        var_pattern = re.compile(r"^\s*var\s+\w+", re.MULTILINE)
        var_matches = var_pattern.findall(content)
        if len(var_matches) > 5:
            issues.append(Issue(
                title=f"Uses 'var' instead of let/const: {rel}",
                description=(
                    f"Found {len(var_matches)} 'var' declarations in {rel}. "
                    f"Modern JavaScript should use 'let' or 'const' for proper scoping."
                ),
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion="Replace 'var' with 'let' (mutable) or 'const' (immutable).",
                check_name=CHECK_NAME,
            ))

        # Check for unhandled fetch errors
        fetch_pattern = re.compile(r'fetch\s*\(')
        catch_pattern = re.compile(r'\.catch\s*\(|try\s*\{')
        fetches = fetch_pattern.findall(content)
        catches = catch_pattern.findall(content)
        if len(fetches) > len(catches) + 2:
            issues.append(Issue(
                title=f"Unhandled fetch errors in {rel}",
                description=(
                    f"Found {len(fetches)} fetch() calls but only {len(catches)} "
                    f"error handlers in {rel}. Network errors will go unhandled."
                ),
                category=Category.FRONTEND,
                severity=Severity.MEDIUM,
                file_path=rel,
                suggestion="Add .catch() or try/catch to all fetch() calls.",
                check_name=CHECK_NAME,
            ))

        # Check file size
        if len(content) > 50000:  # 50KB
            issues.append(Issue(
                title=f"Large JS file: {rel} ({len(content) // 1024}KB)",
                description=f"JavaScript file {rel} is {len(content) // 1024}KB. Large files slow page loads.",
                category=Category.FRONTEND,
                severity=Severity.LOW,
                file_path=rel,
                suggestion="Consider splitting into modules or minifying.",
                check_name=CHECK_NAME,
            ))

    return issues


def _check_broken_references(config: QAConfig) -> list[Issue]:
    """Check for references to files that don't exist."""
    issues: list[Issue] = []

    # Patterns for file references in HTML
    ref_patterns = [
        re.compile(r'src\s*=\s*["\']([^"\']+)["\']'),
        re.compile(r'href\s*=\s*["\']([^"\']+)["\']'),
    ]

    for html_file in config.static_dir.rglob("*.html"):
        rel = str(html_file.relative_to(config.project_root))
        content = html_file.read_text(encoding="utf-8")

        for pattern in ref_patterns:
            for match in pattern.finditer(content):
                ref = match.group(1)

                # Skip external URLs, data URIs, anchors, templates
                if any(ref.startswith(p) for p in ("http", "//", "data:", "#", "{{", "ws:", "wss:")):
                    continue
                if ref.startswith("/api/") or ref.startswith("/ws/"):
                    continue  # API endpoints, not files

                # Resolve relative to the HTML file's directory
                ref_path = html_file.parent / ref
                # Also try from static root
                ref_path_from_root = config.static_dir / ref.lstrip("/")

                if not ref_path.exists() and not ref_path_from_root.exists():
                    line_no = content[:match.start()].count("\n") + 1
                    issues.append(Issue(
                        title=f"Broken reference in {rel}: {ref}",
                        description=(
                            f"HTML file {rel} references '{ref}' but the file "
                            f"does not exist at the expected location."
                        ),
                        category=Category.FRONTEND,
                        severity=Severity.MEDIUM,
                        file_path=rel,
                        line_number=line_no,
                        suggestion=f"Fix the path to '{ref}' or create the missing file.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _check_api_endpoint_references(config: QAConfig) -> list[Issue]:
    """Check that frontend API calls match backend endpoints."""
    issues: list[Issue] = []

    # Collect backend endpoints
    backend_endpoints: set[str] = set()
    route_pattern = re.compile(
        r'@\w+\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
    )
    for server_file in config.web_dir.glob("*.py"):
        source = server_file.read_text(encoding="utf-8")
        for match in route_pattern.finditer(source):
            path = match.group(2)
            # Normalize: remove path parameters
            normalized = re.sub(r'\{[^}]+\}', '*', path)
            backend_endpoints.add(normalized)

    # Collect frontend API calls
    frontend_api_pattern = re.compile(r'fetch\s*\(\s*[`"\']([^`"\']+)[`"\']')

    for static_file in list(config.static_dir.rglob("*.html")) + list(config.static_dir.rglob("*.js")):
        rel = str(static_file.relative_to(config.project_root))
        content = static_file.read_text(encoding="utf-8")

        for match in frontend_api_pattern.finditer(content):
            url = match.group(1)
            # Only check /api/ calls
            if not url.startswith("/api/"):
                continue
            # Strip query parameters
            url_path = url.split("?")[0]
            # Normalize path parameters
            normalized = re.sub(r'/\d+', '/*', url_path)

            # Check against backend
            if normalized not in backend_endpoints:
                # Try with wildcard matching
                found = False
                for endpoint in backend_endpoints:
                    if _paths_match(endpoint, normalized):
                        found = True
                        break
                if not found:
                    line_no = content[:match.start()].count("\n") + 1
                    issues.append(Issue(
                        title=f"Frontend calls unknown API: {url_path}",
                        description=(
                            f"Frontend file {rel} calls '{url_path}' but no "
                            f"matching backend endpoint was found."
                        ),
                        category=Category.FRONTEND,
                        severity=Severity.MEDIUM,
                        file_path=rel,
                        line_number=line_no,
                        suggestion="Verify the API endpoint exists or update the URL.",
                        check_name=CHECK_NAME,
                    ))

    return issues


def _paths_match(pattern: str, path: str) -> bool:
    """Check if an endpoint pattern matches a concrete path."""
    pattern_parts = pattern.strip("/").split("/")
    path_parts = path.strip("/").split("/")

    if len(pattern_parts) != len(path_parts):
        return False

    for pp, cp in zip(pattern_parts, path_parts):
        if pp == "*" or cp == "*":
            continue
        if pp != cp:
            return False
    return True
