"""Backward-compatible shim — canonical source is now shared.safety.collision_detector.

All imports from this module will continue to work.
New code should import from shared.safety.collision_detector instead.
"""
from shared.safety.collision_detector import *  # noqa: F401,F403
