"""Centralized logging configuration for th3cl4w.

Usage:
    from shared.utils.logging_config import setup_logging

    # Simple — writes to stderr + logs/<server_name>.log:
    setup_logging(server_name="web")

    # With debug level:
    setup_logging(server_name="camera", debug=True)

    # Custom log directory:
    setup_logging(server_name="map", log_dir="/var/log/th3cl4w")

    # Legacy — manual log_file path:
    setup_logging(log_file="/tmp/custom.log")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Log rotation defaults
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Standard log directory — project_root/logs/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = DEFAULT_FORMAT,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    *,
    server_name: str | None = None,
    log_dir: str | Path | None = None,
    debug: bool = False,
) -> None:
    """Configure root logger with consistent format and file output.

    Call this once at the start of each entry point (server, script, tool).
    """
    if debug:
        level = logging.DEBUG

    # Resolve the log file path
    resolved_log_file = log_file
    if server_name and not log_file:
        target_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_log_file = str(target_dir / f"{server_name}.log")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if resolved_log_file:
        # Ensure parent directory exists
        Path(resolved_log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                resolved_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
        )
    logging.basicConfig(level=level, format=fmt, handlers=handlers)

    if resolved_log_file:
        logging.getLogger().info("Debug logging pipeline active — writing to %s", resolved_log_file)
