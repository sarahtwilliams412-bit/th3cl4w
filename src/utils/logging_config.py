"""Centralized logging configuration for th3cl4w.

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()  # Call once at entry point
"""

import logging
import sys


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = DEFAULT_FORMAT,
) -> None:
    """Configure root logger with consistent format.

    Call this once at the start of each entry point (server, script, tool).
    Safe to call multiple times — subsequent calls are no-ops due to basicConfig behavior.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
