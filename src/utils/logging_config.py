"""Backward-compatible shim — canonical source is now shared.utils.logging_config.

All imports from this module will continue to work.
New code should import from shared.utils.logging_config instead.
"""

from shared.utils.logging_config import *  # noqa: F401,F403
