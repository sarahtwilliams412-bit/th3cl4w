"""
Plain-Language Issue Logger

Writes QA findings as human-readable, agent-parseable issue files.
Each QA cycle produces a single timestamped log file plus a running
cumulative file that a coding agent can consume.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"  # Blocks functionality or safety risk
    HIGH = "high"  # Test failures, broken imports, type errors
    MEDIUM = "medium"  # Code quality, missing coverage, stale config
    LOW = "low"  # Style, minor improvements, suggestions
    INFO = "info"  # Observations, metrics, non-actionable notes


class Category(Enum):
    """Issue categories matching check modules."""

    UNIT_TEST = "unit_test"
    IMPORT_HEALTH = "import_health"
    API_CONTRACT = "api_contract"
    SAFETY = "safety"
    TYPE_ERROR = "type_error"
    CODE_QUALITY = "code_quality"
    DEPENDENCY = "dependency"
    CONFIG = "config"
    CROSS_MODULE = "cross_module"
    FRONTEND = "frontend"


@dataclass
class Issue:
    """A single QA issue in plain language."""

    title: str  # One-line summary
    description: str  # Plain language explanation
    category: Category
    severity: Severity
    file_path: Optional[str] = None  # Relative path from project root
    line_number: Optional[int] = None
    suggestion: Optional[str] = None  # What a coding agent should do
    evidence: Optional[str] = None  # Raw output / traceback snippet
    check_name: str = ""  # Which check found this

    def to_plain_text(self) -> str:
        """Render as a plain-text block a coding agent can read."""
        lines = []
        lines.append(f"[{self.severity.value.upper()}] {self.title}")
        lines.append(f"  Category : {self.category.value}")
        if self.file_path:
            loc = self.file_path
            if self.line_number:
                loc += f":{self.line_number}"
            lines.append(f"  Location : {loc}")
        lines.append(f"  Problem  : {self.description}")
        if self.suggestion:
            lines.append(f"  Fix      : {self.suggestion}")
        if self.evidence:
            # Truncate evidence to keep logs readable
            ev = self.evidence.strip()
            if len(ev) > 500:
                ev = ev[:500] + "... (truncated)"
            lines.append(f"  Evidence : {ev}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d


@dataclass
class CycleReport:
    """Summary of a full QA cycle."""

    cycle_id: str
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    total_issues: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    checks_run: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


class IssueLogger:
    """Writes issues to disk in both human-readable and machine-readable formats."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_issues: list[Issue] = []
        self._cycle_start: float = 0.0
        self._cycle_id: str = ""

    def begin_cycle(self) -> str:
        """Start a new QA cycle. Returns the cycle ID."""
        self._cycle_start = time.time()
        self._cycle_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_issues = []
        return self._cycle_id

    def log_issue(self, issue: Issue) -> None:
        """Add an issue to the current cycle."""
        self._current_issues.append(issue)

    def log_issues(self, issues: list[Issue]) -> None:
        """Add multiple issues."""
        self._current_issues.extend(issues)

    def end_cycle(self, checks_run: list[str], checks_passed: list[str]) -> CycleReport:
        """Finalize the cycle, write all files, return the report."""
        duration = time.time() - self._cycle_start
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        report = CycleReport(
            cycle_id=self._cycle_id,
            started_at=datetime.fromtimestamp(self._cycle_start, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            finished_at=now_str,
            duration_seconds=round(duration, 2),
            total_issues=len(self._current_issues),
            critical_count=sum(1 for i in self._current_issues if i.severity == Severity.CRITICAL),
            high_count=sum(1 for i in self._current_issues if i.severity == Severity.HIGH),
            medium_count=sum(1 for i in self._current_issues if i.severity == Severity.MEDIUM),
            low_count=sum(1 for i in self._current_issues if i.severity == Severity.LOW),
            info_count=sum(1 for i in self._current_issues if i.severity == Severity.INFO),
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=[c for c in checks_run if c not in checks_passed],
        )

        self._write_plain_text_report(report)
        self._write_json_report(report)
        self._write_cumulative_log()
        self._write_latest_symlink()

        return report

    def _write_plain_text_report(self, report: CycleReport) -> None:
        """Write a human-readable .txt report for this cycle."""
        path = self.log_dir / f"qa_cycle_{self._cycle_id}.txt"

        lines = []
        lines.append("=" * 72)
        lines.append(f"  th3cl4w QA SIDECAR — Cycle Report")
        lines.append(f"  Cycle    : {report.cycle_id}")
        lines.append(f"  Started  : {report.started_at}")
        lines.append(f"  Finished : {report.finished_at}")
        lines.append(f"  Duration : {report.duration_seconds}s")
        lines.append("=" * 72)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total issues   : {report.total_issues}")
        lines.append(f"  Critical       : {report.critical_count}")
        lines.append(f"  High           : {report.high_count}")
        lines.append(f"  Medium         : {report.medium_count}")
        lines.append(f"  Low            : {report.low_count}")
        lines.append(f"  Info           : {report.info_count}")
        lines.append(f"  Checks run     : {', '.join(report.checks_run)}")
        lines.append(f"  Checks passed  : {', '.join(report.checks_passed)}")
        if report.checks_failed:
            lines.append(f"  Checks FAILED  : {', '.join(report.checks_failed)}")
        lines.append("")

        # Group issues by severity
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            sev_issues = [i for i in self._current_issues if i.severity == sev]
            if not sev_issues:
                continue
            lines.append(f"{'=' * 72}")
            lines.append(f"  {sev.value.upper()} ISSUES ({len(sev_issues)})")
            lines.append(f"{'=' * 72}")
            for idx, issue in enumerate(sev_issues, 1):
                lines.append("")
                lines.append(f"  #{idx}")
                lines.append(f"  {issue.to_plain_text()}")
            lines.append("")

        # Footer
        lines.append("-" * 72)
        lines.append("END OF REPORT")
        lines.append(
            "A coding agent should address CRITICAL and HIGH issues first,"
        )
        lines.append(
            "then MEDIUM. LOW and INFO items are optional improvements."
        )
        lines.append("-" * 72)

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_json_report(self, report: CycleReport) -> None:
        """Write a machine-readable JSON report for programmatic consumption."""
        path = self.log_dir / f"qa_cycle_{self._cycle_id}.json"
        data = {
            "report": asdict(report),
            "issues": [i.to_dict() for i in self._current_issues],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _write_cumulative_log(self) -> None:
        """Append issues to a running cumulative log file."""
        path = self.log_dir / "cumulative_issues.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for issue in self._current_issues:
                entry = issue.to_dict()
                entry["cycle_id"] = self._cycle_id
                entry["timestamp"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(entry) + "\n")

    def _write_latest_symlink(self) -> None:
        """Create/update 'latest.txt' pointing to the most recent report."""
        latest = self.log_dir / "latest.txt"
        source = self.log_dir / f"qa_cycle_{self._cycle_id}.txt"
        # Just copy content for portability (symlinks can be fragile)
        if source.exists():
            latest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
