"""
Joint Configuration Service for Unitree D1 Robotic Arm

THE single source of truth for all joint configuration: names, IDs, limits,
DH parameters, types, default positions, and safety margins.

Every module that needs joint info MUST import from here.

Usage:
    from src.control.joint_service import (
        JOINTS, GRIPPER, J0, J1, J2, J3, J4, J5, GRIPPER_ID,
        get_joint, get_limits, joint_count, joint_names, joint_ids,
        get_dh_params, HOME_POSITION, READY_POSITION,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Joint configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DHParams:
    """Standard DH parameters for a single revolute joint."""
    a: float           # link length (along x_i) [meters]
    d: float           # link offset (along z_{i-1}) [meters]
    alpha: float       # link twist (about x_i) [radians]
    theta_offset: float = 0.0  # fixed offset added to joint variable [radians]


@dataclass(frozen=True)
class JointConfig:
    """Complete configuration for a single joint."""
    id: int                    # 0-5 for arm joints, 6 for gripper
    name: str                  # "J0", "J1", ..., "J5", "gripper"
    label: str                 # Human-readable: "Base Yaw", "Shoulder Pitch", etc.
    role: str                  # Lookup key: "base_yaw", "shoulder_pitch", etc.
    joint_type: str            # "yaw", "pitch", "roll", "gripper"
    hw_limit_deg: float        # Hardware limit (±degrees or max for gripper)
    safety_margin_deg: float   # Margin from hardware limit
    dh: Optional[DHParams]     # DH parameters (None for gripper)
    axis_direction: int = 1    # +1 or -1 (sign convention)
    unit: str = "deg"          # "deg" for joints, "mm" for gripper

    @property
    def min_deg(self) -> float:
        if self.unit == "mm":
            return 0.0
        return -self.hw_limit_deg

    @property
    def max_deg(self) -> float:
        if self.unit == "mm":
            return self.hw_limit_deg
        return self.hw_limit_deg

    @property
    def safe_min_deg(self) -> float:
        if self.unit == "mm":
            return 0.0
        return -self.hw_limit_deg + self.safety_margin_deg

    @property
    def safe_max_deg(self) -> float:
        if self.unit == "mm":
            return self.hw_limit_deg
        return self.hw_limit_deg - self.safety_margin_deg

    @property
    def min_rad(self) -> float:
        return math.radians(self.min_deg)

    @property
    def max_rad(self) -> float:
        return math.radians(self.max_deg)

    @property
    def safe_min_rad(self) -> float:
        return math.radians(self.safe_min_deg)

    @property
    def safe_max_rad(self) -> float:
        return math.radians(self.safe_max_deg)

    def to_dict(self) -> dict:
        """Serialize for JSON API response."""
        d = {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "role": self.role,
            "type": self.joint_type,
            "unit": self.unit,
            "hw_limit": self.hw_limit_deg,
            "safety_margin": self.safety_margin_deg,
            "min": self.min_deg,
            "max": self.max_deg,
            "safe_min": self.safe_min_deg,
            "safe_max": self.safe_max_deg,
            "axis_direction": self.axis_direction,
        }
        if self.dh is not None:
            d["dh"] = {
                "a": self.dh.a,
                "d": self.dh.d,
                "alpha": self.dh.alpha,
                "theta_offset": self.dh.theta_offset,
            }
        return d


# ---------------------------------------------------------------------------
# Named constants for joint IDs
# ---------------------------------------------------------------------------

J0 = 0  # Base yaw
J1 = 1  # Shoulder pitch
J2 = 2  # Elbow pitch
J3 = 3  # Forearm roll (elbow roll)
J4 = 4  # Wrist pitch
J5 = 5  # Wrist roll (gripper roll)
GRIPPER_ID = 6

# ---------------------------------------------------------------------------
# DH Parameters for D1 arm (7-DOF: 6 arm + 1 wrist tool frame)
# ---------------------------------------------------------------------------
# Note: The kinematics module uses 7 DH frames (J0-J5 + tool frame).
# The 7th DH frame is the tool/end-effector, not a controllable joint.

_DH_PARAMS = [
    DHParams(a=0.0, d=0.1215, alpha=-math.pi / 2, theta_offset=0.0),   # J0
    DHParams(a=0.0, d=0.0,    alpha=math.pi / 2,  theta_offset=0.0),   # J1
    DHParams(a=0.0, d=0.2085, alpha=-math.pi / 2, theta_offset=0.0),   # J2
    DHParams(a=0.0, d=0.0,    alpha=math.pi / 2,  theta_offset=0.0),   # J3
    DHParams(a=0.0, d=0.2085, alpha=-math.pi / 2, theta_offset=0.0),   # J4
    DHParams(a=0.0, d=0.0,    alpha=math.pi / 2,  theta_offset=0.0),   # J5
    DHParams(a=0.0, d=0.1130, alpha=0.0,           theta_offset=0.0),   # Tool frame
]

# ---------------------------------------------------------------------------
# Joint definitions — THE authoritative configuration
# ---------------------------------------------------------------------------

_JOINT_CONFIGS: list[JointConfig] = [
    JointConfig(
        id=J0, name="J0", label="Base Yaw", role="base_yaw",
        joint_type="yaw", hw_limit_deg=135.0, safety_margin_deg=0.0,
        dh=_DH_PARAMS[0], axis_direction=1,
    ),
    JointConfig(
        id=J1, name="J1", label="Shoulder Pitch", role="shoulder_pitch",
        joint_type="pitch", hw_limit_deg=90.0, safety_margin_deg=5.0,
        dh=_DH_PARAMS[1], axis_direction=1,
    ),
    JointConfig(
        id=J2, name="J2", label="Elbow Pitch", role="elbow_pitch",
        joint_type="pitch", hw_limit_deg=90.0, safety_margin_deg=5.0,
        dh=_DH_PARAMS[2], axis_direction=1,
    ),
    JointConfig(
        id=J3, name="J3", label="Forearm Roll", role="forearm_roll",
        joint_type="roll", hw_limit_deg=135.0, safety_margin_deg=0.0,
        dh=_DH_PARAMS[3], axis_direction=1,
    ),
    JointConfig(
        id=J4, name="J4", label="Wrist Pitch", role="wrist_pitch",
        joint_type="pitch", hw_limit_deg=90.0, safety_margin_deg=5.0,
        dh=_DH_PARAMS[4], axis_direction=1,
    ),
    JointConfig(
        id=J5, name="J5", label="Wrist Roll", role="wrist_roll",
        joint_type="roll", hw_limit_deg=135.0, safety_margin_deg=0.0,
        dh=_DH_PARAMS[5], axis_direction=1,
    ),
]

GRIPPER = JointConfig(
    id=GRIPPER_ID, name="gripper", label="Gripper", role="gripper",
    joint_type="gripper", hw_limit_deg=65.0, safety_margin_deg=0.0,
    dh=None, axis_direction=1, unit="mm",
)

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

NUM_ARM_JOINTS = len(_JOINT_CONFIGS)  # 6
NUM_JOINTS = NUM_ARM_JOINTS + 1       # 7 (includes gripper in DDS layer)

# JOINTS dict: id -> JointConfig (arm joints only, excludes gripper)
JOINTS: dict[int, JointConfig] = {j.id: j for j in _JOINT_CONFIGS}

# All configs including gripper
ALL_JOINTS: dict[int, JointConfig] = {**JOINTS, GRIPPER_ID: GRIPPER}

# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

# Build lookup tables for flexible access
_LOOKUP: dict[str, JointConfig] = {}
for _jc in _JOINT_CONFIGS:
    _LOOKUP[_jc.name.upper()] = _jc        # "J0"
    _LOOKUP[_jc.name.lower()] = _jc        # "j0"
    _LOOKUP[f"joint_{_jc.id}"] = _jc       # "joint_0"
    _LOOKUP[_jc.role] = _jc                # "base_yaw"
    _LOOKUP[str(_jc.id)] = _jc             # "0"
_LOOKUP["gripper"] = GRIPPER
_LOOKUP["GRIPPER"] = GRIPPER
_LOOKUP[str(GRIPPER_ID)] = GRIPPER


def get_joint(name_or_id: Union[str, int]) -> JointConfig:
    """Look up a joint by name, ID, or role.

    Accepts: "J0", "j0", 0, "base_yaw", "joint_0", "gripper", 6
    Raises KeyError if not found.
    """
    if isinstance(name_or_id, int):
        name_or_id = str(name_or_id)
    result = _LOOKUP.get(name_or_id)
    if result is None:
        raise KeyError(f"Unknown joint: {name_or_id!r}")
    return result


def joint_count() -> int:
    """Number of arm joints (excludes gripper)."""
    return NUM_ARM_JOINTS


def joint_names() -> list[str]:
    """List of arm joint names: ['J0', 'J1', ..., 'J5']."""
    return [j.name for j in _JOINT_CONFIGS]


def joint_ids() -> list[int]:
    """List of arm joint IDs: [0, 1, 2, 3, 4, 5]."""
    return [j.id for j in _JOINT_CONFIGS]


def get_limits(joint_id: int, safe: bool = True) -> tuple[float, float]:
    """Return (min_deg, max_deg) for a joint.

    If safe=True (default), returns limits with safety margin applied.
    If safe=False, returns raw hardware limits.
    """
    jc = ALL_JOINTS[joint_id]
    if safe:
        return (jc.safe_min_deg, jc.safe_max_deg)
    return (jc.min_deg, jc.max_deg)


def get_limits_rad(joint_id: int, safe: bool = True) -> tuple[float, float]:
    """Return (min_rad, max_rad) for a joint."""
    jc = ALL_JOINTS[joint_id]
    if safe:
        return (jc.safe_min_rad, jc.safe_max_rad)
    return (jc.min_rad, jc.max_rad)


# ---------------------------------------------------------------------------
# DH parameter access
# ---------------------------------------------------------------------------

def get_dh_params() -> list[DHParams]:
    """Return the full DH parameter table (7 entries: 6 joints + tool frame)."""
    return list(_DH_PARAMS)


def get_dh_params_for_kinematics():
    """Return DH params as DHParameters objects compatible with kinematics module.

    This returns the same data in the format kinematics.py expects.
    """
    from src.kinematics.kinematics import DHParameters
    return [
        DHParameters(a=dh.a, d=dh.d, alpha=dh.alpha, theta_offset=dh.theta_offset)
        for dh in _DH_PARAMS
    ]


# ---------------------------------------------------------------------------
# Limits as numpy arrays (for safety_monitor, motion_planner, etc.)
# ---------------------------------------------------------------------------

def joint_limits_deg_array() -> np.ndarray:
    """Return (6, 2) array of [min, max] in degrees for arm joints (no safety margin)."""
    return np.array([[j.min_deg, j.max_deg] for j in _JOINT_CONFIGS])


def joint_limits_rad_min_array(include_gripper: bool = True) -> np.ndarray:
    """Return min position limits in radians. Length 7 if include_gripper, else 6."""
    mins = [math.radians(j.min_deg) for j in _JOINT_CONFIGS]
    if include_gripper:
        mins.append(0.0)  # gripper min (normalized)
    return np.array(mins)


def joint_limits_rad_max_array(include_gripper: bool = True) -> np.ndarray:
    """Return max position limits in radians. Length 7 if include_gripper, else 6."""
    maxs = [math.radians(j.max_deg) for j in _JOINT_CONFIGS]
    if include_gripper:
        maxs.append(1.0)  # gripper max (normalized)
    return np.array(maxs)


# ---------------------------------------------------------------------------
# Default positions (degrees, 6 arm joints)
# ---------------------------------------------------------------------------

HOME_POSITION = np.zeros(NUM_ARM_JOINTS)
READY_POSITION = np.array([0.0, -45.0, 0.0, 90.0, 0.0, -45.0])

# ---------------------------------------------------------------------------
# Velocity / torque limits (kept here for convenience, same values as before)
# ---------------------------------------------------------------------------

VELOCITY_MAX_RAD = np.array([2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 2.0])  # 7 joints
MAX_JOINT_SPEED_DEG = np.array([90.0, 90.0, 120.0, 120.0, 150.0, 150.0])  # 6 arm
MAX_JOINT_ACCEL_DEG = np.array([180.0, 180.0, 240.0, 240.0, 300.0, 300.0])  # 6 arm
TORQUE_MAX_NM = np.array([20.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0])  # 7 joints

# ---------------------------------------------------------------------------
# Workspace limits
# ---------------------------------------------------------------------------

MAX_WORKSPACE_RADIUS_MM = 550.0
MAX_WORKSPACE_RADIUS_M = MAX_WORKSPACE_RADIUS_MM / 1000.0

# ---------------------------------------------------------------------------
# Smoother / feedback constants
# ---------------------------------------------------------------------------

MAX_STEP_DEG = 10.0           # Max angular change per smoother tick
FEEDBACK_MAX_AGE_S = 0.5      # 500ms feedback freshness limit
GRIPPER_MIN_MM = 0.0
GRIPPER_MAX_MM = 65.0


# ---------------------------------------------------------------------------
# API serialization
# ---------------------------------------------------------------------------

def all_joints_dict() -> list[dict]:
    """Return all joint configs as list of dicts (for HTTP API)."""
    result = [j.to_dict() for j in _JOINT_CONFIGS]
    result.append(GRIPPER.to_dict())
    return result


def joint_dict(joint_id: int) -> dict:
    """Return single joint config as dict."""
    return ALL_JOINTS[joint_id].to_dict()
