"""
Unified Joint Limits for Unitree D1 Robotic Arm

This module re-exports all limit constants from the authoritative source:
    src.control.joint_service

For backward compatibility, all the old names are preserved.
Any module importing from here will get the same values as before.
"""

import logging

from src.control.joint_service import (
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    joint_limits_deg_array,
    joint_limits_rad_min_array,
    joint_limits_rad_max_array,
    VELOCITY_MAX_RAD,
    MAX_JOINT_SPEED_DEG as MAX_JOINT_SPEED_DEG,
    MAX_JOINT_ACCEL_DEG as MAX_JOINT_ACCEL_DEG,
    TORQUE_MAX_NM,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
    MAX_WORKSPACE_RADIUS_MM,
    MAX_WORKSPACE_RADIUS_M,
    MAX_STEP_DEG,
    FEEDBACK_MAX_AGE_S,
    get_joint,
)

logger = logging.getLogger(__name__)

# Backward-compatible names
JOINT_LIMITS_DEG = joint_limits_deg_array()
JOINT_LIMITS_RAD_MIN = joint_limits_rad_min_array(include_gripper=True)
JOINT_LIMITS_RAD_MAX = joint_limits_rad_max_array(include_gripper=True)

# Keep the old function signature
def get_joint_limits_deg():
    """Return runtime joint limits from pick_config if available, else defaults."""
    try:
        from src.config.pick_config import get_pick_config

        cfg = get_pick_config()
        limits = cfg.get("safety", "joint_limits_deg")
        if limits and len(limits) == 6:
            import numpy as np
            return np.array(limits)
    except Exception:
        pass
    return JOINT_LIMITS_DEG
