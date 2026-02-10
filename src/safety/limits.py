"""
Unified Joint Limits for Unitree D1 Robotic Arm

Single source of truth for all safety limits. Every module that needs
joint limits MUST import from here — not define its own.

Hardware limits: ±85° on J1/J2/J4, with 5° software margin → ±80° effective.
Other joints use the most conservative limits found across the codebase.
Gripper: 0–65 mm.
"""

import math

import numpy as np

# Number of joints (6 arm + 1 gripper in the DDS layer)

import logging

logger = logging.getLogger(__name__)


NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper (matches d1_connection.NUM_JOINTS)

# ---------------------------------------------------------------------------
# Joint position limits (degrees) — the authoritative table
# ---------------------------------------------------------------------------
# Hardware spec: J1/J2/J4 = ±85°, others = ±135°
# Software margin: 5° on J1/J2/J4 → effective ±80°
# J3/J5 also capped conservatively at ±135° (hardware)
# J0 (base yaw): ±135° (hardware)

JOINT_LIMITS_DEG = np.array(
    [
        [-135.0, 135.0],  # J0 — base yaw
        [-80.0, 80.0],  # J1 — shoulder pitch (±85° hw - 5° margin)
        [-80.0, 80.0],  # J2 — elbow pitch   (±85° hw - 5° margin)
        [-135.0, 135.0],  # J3 — elbow roll
        [-80.0, 80.0],  # J4 — wrist pitch   (±85° hw - 5° margin)
        [-135.0, 135.0],  # J5 — wrist roll
    ]
)


def get_joint_limits_deg():
    """Return runtime joint limits from pick_config if available, else defaults."""
    try:
        from src.config.pick_config import get_pick_config

        cfg = get_pick_config()
        limits = cfg.get("safety", "joint_limits_deg")
        if limits and len(limits) == 6:
            return np.array(limits)
    except Exception:
        pass
    return JOINT_LIMITS_DEG


# Same limits in radians for the SafetyMonitor / DDS layer (7 joints, last = gripper)
JOINT_LIMITS_RAD_MIN = np.array(
    [math.radians(JOINT_LIMITS_DEG[i, 0]) for i in range(NUM_ARM_JOINTS)] + [0.0]
)  # gripper min = 0.0 (normalized)

JOINT_LIMITS_RAD_MAX = np.array(
    [math.radians(JOINT_LIMITS_DEG[i, 1]) for i in range(NUM_ARM_JOINTS)] + [1.0]
)  # gripper max = 1.0 (normalized)

# ---------------------------------------------------------------------------
# Velocity limits
# ---------------------------------------------------------------------------
# Conservative: from safety_monitor defaults, in rad/s (7 joints)
VELOCITY_MAX_RAD = np.array([2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 2.0])

# For the motion planner (deg/s, 6 arm joints only)
MAX_JOINT_SPEED_DEG = np.array([90.0, 90.0, 120.0, 120.0, 150.0, 150.0])
MAX_JOINT_ACCEL_DEG = np.array([180.0, 180.0, 240.0, 240.0, 300.0, 300.0])

# ---------------------------------------------------------------------------
# Torque limits (Nm, 7 joints)
# ---------------------------------------------------------------------------
TORQUE_MAX_NM = np.array([20.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0])

# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------
GRIPPER_MIN_MM = 0.0
GRIPPER_MAX_MM = 65.0

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
MAX_WORKSPACE_RADIUS_MM = 550.0
MAX_WORKSPACE_RADIUS_M = MAX_WORKSPACE_RADIUS_MM / 1000.0

# ---------------------------------------------------------------------------
# Smoother safety parameters
# ---------------------------------------------------------------------------
MAX_STEP_DEG = 10.0  # Maximum angular change per smoother tick (≤10° rule)

# ---------------------------------------------------------------------------
# Feedback freshness
# ---------------------------------------------------------------------------
FEEDBACK_MAX_AGE_S = 0.5  # 500ms — refuse commands if feedback older than this
