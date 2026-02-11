"""Backward-compatible shim — canonical source is now shared.safety.safety_monitor.

All imports from this module will continue to work.
New code should import from shared.safety.safety_monitor instead.
"""
from shared.safety.safety_monitor import *  # noqa: F401,F403
