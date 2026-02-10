"""Centralized logging configuration for th3cl4w.

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()  # Call once at entry point
"""

import logging
import sys
from logging.handlers import RotatingFileHandler


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Log rotation defaults
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = DEFAULT_FORMAT,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> None:
    """Configure root logger with consistent format.

    Call this once at the start of each entry point (server, script, tool).
    Safe to call multiple times — subsequent calls are no-ops due to basicConfig behavior.

    When *log_file* is provided, a RotatingFileHandler is used instead of a
    plain FileHandler so logs are capped at *max_bytes* with *backup_count*
    rotated backups.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
        )
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
