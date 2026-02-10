#!/usr/bin/env python3
"""
th3cl4w QA Sidecar — CLI Entrypoint

Usage:
    # Run a single QA cycle (default):
    python -m qa_sidecar.run_qa

    # Run continuously (every 5 minutes):
    python -m qa_sidecar.run_qa --continuous

    # Run only specific checks:
    python -m qa_sidecar.run_qa --only unit_tests,safety_invariants

    # Skip specific checks:
    python -m qa_sidecar.run_qa --skip type_checking,code_quality

    # Custom cycle interval (seconds):
    python -m qa_sidecar.run_qa --continuous --interval 600

    # Custom issue log directory:
    python -m qa_sidecar.run_qa --log-dir /path/to/logs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qa_sidecar.config import QAConfig
from qa_sidecar.runner import run_continuous, run_single_cycle


ALL_CHECKS = [
    "unit_tests",
    "import_health",
    "api_contracts",
    "safety_invariants",
    "type_checking",
    "code_quality",
    "dependency_audit",
    "config_validation",
    "cross_module",
    "frontend_checks",
]


def main():
    parser = argparse.ArgumentParser(
        description="th3cl4w Continuous QA Sidecar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m qa_sidecar.run_qa                          # Single cycle\n"
            "  python -m qa_sidecar.run_qa --continuous              # Continuous mode\n"
            "  python -m qa_sidecar.run_qa --only safety_invariants  # Safety checks only\n"
            "  python -m qa_sidecar.run_qa --skip unit_tests         # Skip slow tests\n"
        ),
    )

    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously with a delay between cycles",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="Seconds between cycles in continuous mode (default: 300)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of checks to run (others disabled)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated list of checks to skip",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for issue log files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List all available checks and exit",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.list_checks:
        print("Available QA checks:")
        for check in ALL_CHECKS:
            print(f"  - {check}")
        return

    # Build config
    config = QAConfig()
    config.cycle_interval_seconds = args.interval

    if args.log_dir:
        config.issue_log_dir = Path(args.log_dir)
        config.issue_log_dir.mkdir(parents=True, exist_ok=True)

    # Handle --only
    if args.only:
        enabled = set(args.only.split(","))
        for check in ALL_CHECKS:
            setattr(config, f"run_{check}", check in enabled)

    # Handle --skip
    if args.skip:
        disabled = set(args.skip.split(","))
        for check in disabled:
            if hasattr(config, f"run_{check}"):
                setattr(config, f"run_{check}", False)

    # Run
    if args.continuous:
        run_continuous(config)
    else:
        report = run_single_cycle(config)
        # Exit with non-zero if critical issues found
        if report.critical_count > 0:
            sys.exit(2)
        elif report.high_count > 0:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
