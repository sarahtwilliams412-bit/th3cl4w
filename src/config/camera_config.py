"""Backward-compatible shim — canonical source is now shared.config.camera_config.

All imports from this module will continue to work.
New code should import from shared.config.camera_config instead.
"""

from shared.config.camera_config import *  # noqa: F401,F403
