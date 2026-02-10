"""
QA Sidecar Runner — Continuous QA Orchestrator

This is the main engine that runs all checks in sequence, collects issues,
and produces plain-language reports. It can run once or in a continuous loop.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

from qa_sidecar.config import QAConfig
from qa_sidecar.issue_logger import CycleReport, Issue, IssueLogger, Severity

logger = logging.getLogger("qa_sidecar")

# Registry of all check modules
CHECK_MODULES = [
    ("unit_tests", "qa_sidecar.checks.unit_tests"),
    ("import_health", "qa_sidecar.checks.import_health"),
    ("api_contracts", "qa_sidecar.checks.api_contracts"),
    ("safety_invariants", "qa_sidecar.checks.safety_invariants"),
    ("type_checking", "qa_sidecar.checks.type_checking"),
    ("code_quality", "qa_sidecar.checks.code_quality"),
    ("dependency_audit", "qa_sidecar.checks.dependency_audit"),
    ("config_validation", "qa_sidecar.checks.config_validation"),
    ("cross_module", "qa_sidecar.checks.cross_module"),
    ("frontend_checks", "qa_sidecar.checks.frontend_checks"),
]


def _is_check_enabled(config: QAConfig, check_name: str) -> bool:
    """Check if a specific check is enabled in config."""
    toggle_map = {
        "unit_tests": config.run_unit_tests,
        "import_health": config.run_import_health,
        "api_contracts": config.run_api_contracts,
        "safety_invariants": config.run_safety_invariants,
        "type_checking": config.run_type_checking,
        "code_quality": config.run_code_quality,
        "dependency_audit": config.run_dependency_audit,
        "config_validation": config.run_config_validation,
        "cross_module": config.run_cross_module,
        "frontend_checks": config.run_frontend_checks,
    }
    return toggle_map.get(check_name, True)


def run_single_cycle(config: QAConfig) -> CycleReport:
    """Execute one full QA cycle: run all enabled checks and produce a report."""
    issue_logger = IssueLogger(config.issue_log_dir)
    cycle_id = issue_logger.begin_cycle()

    logger.info("=" * 60)
    logger.info("QA SIDECAR — Starting cycle %s", cycle_id)
    logger.info("=" * 60)

    checks_run: list[str] = []
    checks_passed: list[str] = []

    for check_name, module_path in CHECK_MODULES:
        if not _is_check_enabled(config, check_name):
            logger.info("  [SKIP] %s (disabled)", check_name)
            continue

        logger.info("  [RUN]  %s ...", check_name)
        checks_run.append(check_name)

        try:
            import importlib
            mod = importlib.import_module(module_path)
            issues: list[Issue] = mod.run(config)

            # Log issues
            issue_logger.log_issues(issues)

            # Determine if the check "passed" (no CRITICAL or HIGH issues)
            has_failures = any(
                i.severity in (Severity.CRITICAL, Severity.HIGH) for i in issues
            )
            if has_failures:
                high_count = sum(
                    1 for i in issues if i.severity in (Severity.CRITICAL, Severity.HIGH)
                )
                logger.info(
                    "  [FAIL] %s — %d issues (%d critical/high)",
                    check_name, len(issues), high_count,
                )
            else:
                checks_passed.append(check_name)
                logger.info("  [PASS] %s — %d issues (none critical/high)", check_name, len(issues))

        except Exception:
            tb = traceback.format_exc()
            logger.error("  [ERR]  %s crashed: %s", check_name, tb[-200:])
            from qa_sidecar.issue_logger import Category
            issue_logger.log_issue(Issue(
                title=f"Check '{check_name}' crashed",
                description=(
                    f"The QA check '{check_name}' threw an unhandled exception "
                    f"and could not complete. This is a bug in the QA sidecar itself."
                ),
                category=Category.UNIT_TEST,
                severity=Severity.HIGH,
                suggestion="Fix the crash in the QA check module.",
                evidence=tb[-500:],
                check_name=check_name,
            ))

    # Finalize
    report = issue_logger.end_cycle(checks_run, checks_passed)

    logger.info("")
    logger.info("=" * 60)
    logger.info("QA CYCLE COMPLETE — %s", cycle_id)
    logger.info("  Total issues : %d", report.total_issues)
    logger.info("  Critical     : %d", report.critical_count)
    logger.info("  High         : %d", report.high_count)
    logger.info("  Medium       : %d", report.medium_count)
    logger.info("  Low          : %d", report.low_count)
    logger.info("  Passed       : %d/%d checks", len(checks_passed), len(checks_run))
    logger.info(
        "  Report       : %s",
        config.issue_log_dir / f"qa_cycle_{cycle_id}.txt",
    )
    logger.info("=" * 60)

    return report


def run_continuous(config: QAConfig) -> None:
    """Run QA cycles continuously with a delay between each cycle."""
    logger.info("QA Sidecar starting in CONTINUOUS mode")
    logger.info("  Cycle interval: %ds", config.cycle_interval_seconds)
    logger.info("  Issue log dir : %s", config.issue_log_dir)
    logger.info("")

    cycle_count = 0
    while True:
        try:
            cycle_count += 1
            logger.info("--- Continuous cycle #%d ---", cycle_count)
            report = run_single_cycle(config)

            if report.critical_count > 0:
                logger.warning(
                    "!!! %d CRITICAL issues found — a coding agent should address these immediately !!!",
                    report.critical_count,
                )

            logger.info(
                "Next cycle in %d seconds (Ctrl+C to stop)...",
                config.cycle_interval_seconds,
            )
            time.sleep(config.cycle_interval_seconds)

        except KeyboardInterrupt:
            logger.info("QA Sidecar stopped by user.")
            break
        except Exception:
            logger.error("Unexpected error in continuous loop: %s", traceback.format_exc())
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)
