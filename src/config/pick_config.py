"""Backward-compatible shim — canonical source is now shared.config.pick_config.

All imports from this module will continue to work.
New code should import from shared.config.pick_config instead.
"""

from shared.config.pick_config import *  # noqa: F401,F403
