"""Backward-compatible shim — canonical source is now shared.safety.limits.

All imports from this module will continue to work.
New code should import from shared.safety.limits instead.
"""

from shared.safety.limits import *  # noqa: F401,F403
