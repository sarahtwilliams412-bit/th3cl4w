"""Backward-compatible shim — canonical source is now shared.kinematics.kinematics.

All imports from this module will continue to work.
New code should import from shared.kinematics.kinematics instead.
"""

from shared.kinematics.kinematics import *  # noqa: F401,F403
from shared.kinematics.kinematics import _D1_DH, _dh_transform  # noqa: F401
