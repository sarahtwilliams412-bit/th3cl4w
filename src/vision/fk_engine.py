"""Backward-compatible shim — canonical source is now shared.kinematics.fk_engine.

All imports from this module will continue to work.
New code should import from shared.kinematics.fk_engine instead.
"""

from shared.kinematics.fk_engine import *  # noqa: F401,F403
