"""Backward-compatible shim — canonical source is now shared.utils.gemini_limiter.

All imports from this module will continue to work.
New code should import from shared.utils.gemini_limiter instead.
"""

from shared.utils.gemini_limiter import *  # noqa: F401,F403
