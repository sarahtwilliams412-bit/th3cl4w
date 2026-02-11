"""Backward-compatible shim — canonical source is now shared.arm_model.joint_service.

All imports from this module will continue to work.
New code should import from shared.arm_model.joint_service instead.
"""
from shared.arm_model.joint_service import *  # noqa: F401,F403
