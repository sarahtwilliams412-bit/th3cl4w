"""
QA Sidecar Configuration

Central configuration for all checks, thresholds, and paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QAConfig:
    """Configuration for the continuous QA sidecar."""

    # --- Paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    issue_log_dir: Path = field(default=None)  # type: ignore[assignment]
    src_dir: Path = field(default=None)  # type: ignore[assignment]
    tests_dir: Path = field(default=None)  # type: ignore[assignment]
    web_dir: Path = field(default=None)  # type: ignore[assignment]
    static_dir: Path = field(default=None)  # type: ignore[assignment]

    # --- Timing ---
    cycle_interval_seconds: int = 300  # 5 minutes between full cycles
    max_issues_per_cycle: int = 200  # cap to avoid flooding

    # --- Check toggles ---
    run_unit_tests: bool = True
    run_import_health: bool = True
    run_api_contracts: bool = True
    run_safety_invariants: bool = True
    run_type_checking: bool = True
    run_code_quality: bool = True
    run_dependency_audit: bool = True
    run_config_validation: bool = True
    run_cross_module: bool = True
    run_frontend_checks: bool = True

    # --- Thresholds ---
    max_function_complexity: int = 25  # cyclomatic complexity
    max_file_lines: int = 800
    max_function_lines: int = 100
    max_import_depth: int = 8

    # --- Pytest ---
    pytest_timeout_seconds: int = 120
    pytest_extra_args: list[str] = field(default_factory=list)

    # --- Mypy ---
    mypy_timeout_seconds: int = 90

    def __post_init__(self):
        if self.issue_log_dir is None:
            self.issue_log_dir = self.project_root / "qa_sidecar" / "issues"
        if self.src_dir is None:
            self.src_dir = self.project_root / "src"
        if self.tests_dir is None:
            self.tests_dir = self.project_root / "tests"
        if self.web_dir is None:
            self.web_dir = self.project_root / "web"
        if self.static_dir is None:
            self.static_dir = self.project_root / "web" / "static"
        # ensure log directory exists
        self.issue_log_dir.mkdir(parents=True, exist_ok=True)
