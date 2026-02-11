"""
Safety Monitor for Unitree D1 Robotic Arm

This is the most critical module in th3cl4w. It enforces joint limits,
velocity limits, torque limits, and workspace bounds to protect hardware
and humans. An emergency stop flag blocks all commands until manually reset.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from shared.arm_model.d1_state import D1Command, D1State
from shared.arm_model.joint_service import (
    NUM_JOINTS,
    joint_limits_rad_min_array,
    joint_limits_rad_max_array,
    VELOCITY_MAX_RAD,
    TORQUE_MAX_NM,
    MAX_WORKSPACE_RADIUS_MM,
    MAX_WORKSPACE_RADIUS_M,
    FEEDBACK_MAX_AGE_S,
)

# Backward-compatible names
JOINT_LIMITS_RAD_MIN = joint_limits_rad_min_array(include_gripper=True)
JOINT_LIMITS_RAD_MAX = joint_limits_rad_max_array(include_gripper=True)

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Categories of safety violations."""

    POSITION_LIMIT = "position_limit"
    VELOCITY_LIMIT = "velocity_limit"
    TORQUE_LIMIT = "torque_limit"
    WORKSPACE_BOUND = "workspace_bound"
    ESTOP_ACTIVE = "estop_active"
    GRIPPER_LIMIT = "gripper_limit"


@dataclass(frozen=True)
class SafetyViolation:
    """A single safety violation with context."""

    violation_type: ViolationType
    joint_index: Optional[int]  # None for workspace / e-stop violations
    message: str
    actual_value: float
    limit_value: float


@dataclass(frozen=True)
class SafetyResult:
    """Result of validating a command."""

    is_safe: bool
    violations: tuple[SafetyViolation, ...] = ()

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def __bool__(self) -> bool:
        return self.is_safe


@dataclass
class JointLimits:
    """Per-joint safety limits for the D1 arm.

    Arrays are length NUM_JOINTS (7): joints 0-5 are arm, joint 6 is gripper.
    """

    # Position limits (radians)
    position_min: np.ndarray
    position_max: np.ndarray
    # Velocity limits (rad/s) — absolute value
    velocity_max: np.ndarray
    # Torque limits (Nm) — absolute value
    torque_max: np.ndarray

    def __post_init__(self):
        for name, arr in [
            ("position_min", self.position_min),
            ("position_max", self.position_max),
            ("velocity_max", self.velocity_max),
            ("torque_max", self.torque_max),
        ]:
            if len(arr) != NUM_JOINTS:
                raise ValueError(f"{name} must have {NUM_JOINTS} elements, got {len(arr)}")
        if np.any(self.position_min >= self.position_max):
            raise ValueError("position_min must be strictly less than position_max for all joints")
        if np.any(self.velocity_max <= 0):
            raise ValueError("velocity_max must be positive for all joints")
        if np.any(self.torque_max <= 0):
            raise ValueError("torque_max must be positive for all joints")


def d1_default_limits() -> JointLimits:
    """Default joint limits for the Unitree D1 arm."""
    return JointLimits(
        position_min=JOINT_LIMITS_RAD_MIN.copy(),
        position_max=JOINT_LIMITS_RAD_MAX.copy(),
        velocity_max=VELOCITY_MAX_RAD.copy(),
        torque_max=TORQUE_MAX_NM.copy(),
    )


# ---------------------------------------------------------------------------
# Forward kinematics stub for workspace checking
# ---------------------------------------------------------------------------
_LINK_LENGTHS_M = np.array([0.0, 0.15, 0.22, 0.0, 0.18, 0.0, 0.05])  # rough


def _estimate_reach(joint_positions: np.ndarray) -> float:
    """Estimate end-effector distance from base using simplified 2D planar FK."""
    q1 = float(joint_positions[1])  # shoulder pitch
    q2 = float(joint_positions[2])  # elbow pitch
    q4 = float(joint_positions[4]) if len(joint_positions) > 4 else 0.0  # wrist pitch

    L1 = _LINK_LENGTHS_M[1]
    L2 = _LINK_LENGTHS_M[2]
    L3 = _LINK_LENGTHS_M[4] + _LINK_LENGTHS_M[6]

    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(q1 + q2 + q4)
    z = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(q1 + q2 + q4)

    return math.sqrt(x * x + z * z)


class SafetyMonitor:
    """Monitors and enforces safety constraints for the D1 arm.

    Thread-safe: the e-stop flag is a simple bool guarded by the GIL.
    """

    def __init__(self, limits: Optional[JointLimits] = None):
        self._limits = limits or d1_default_limits()
        self._estop = False

    @property
    def limits(self) -> JointLimits:
        return self._limits

    @property
    def estop_active(self) -> bool:
        return self._estop

    def trigger_estop(self, reason: str = "manual") -> None:
        """Activate emergency stop. All commands blocked until reset."""
        if not self._estop:
            logger.critical("E-STOP TRIGGERED: %s", reason)
        self._estop = True

    def reset_estop(self) -> None:
        """Reset emergency stop, allowing commands again."""
        logger.info("E-STOP RESET")
        self._estop = False

    @staticmethod
    def _is_valid_float(val: float) -> bool:
        """Return False if value is NaN or Inf."""
        return not (math.isnan(val) or math.isinf(val))

    def is_feedback_fresh(self, state: D1State, max_age: float = 0.5) -> bool:
        """Return True if the state timestamp is recent enough."""
        return (time.time() - state.timestamp) < max_age

    def validate_command(self, cmd: D1Command) -> SafetyResult:
        """Validate a command against all safety constraints."""
        violations: list[SafetyViolation] = []

        if self._estop:
            violations.append(
                SafetyViolation(
                    violation_type=ViolationType.ESTOP_ACTIVE,
                    joint_index=None,
                    message="Emergency stop is active — all commands blocked",
                    actual_value=0.0,
                    limit_value=0.0,
                )
            )
            return SafetyResult(is_safe=False, violations=tuple(violations))

        lim = self._limits

        for field_name, arr in [
            ("position", cmd.joint_positions),
            ("velocity", cmd.joint_velocities),
            ("torque", cmd.joint_torques),
        ]:
            if arr is not None:
                for i in range(NUM_JOINTS):
                    val = float(arr[i])
                    if not self._is_valid_float(val):
                        violations.append(
                            SafetyViolation(
                                violation_type=ViolationType.POSITION_LIMIT,
                                joint_index=i,
                                message=f"Joint {i} {field_name} is NaN/Inf — rejected",
                                actual_value=0.0,
                                limit_value=0.0,
                            )
                        )
        if cmd.gripper_position is not None and not self._is_valid_float(cmd.gripper_position):
            violations.append(
                SafetyViolation(
                    violation_type=ViolationType.GRIPPER_LIMIT,
                    joint_index=6,
                    message="Gripper position is NaN/Inf — rejected",
                    actual_value=0.0,
                    limit_value=0.0,
                )
            )
        if violations:
            return SafetyResult(is_safe=False, violations=tuple(violations))

        if cmd.joint_positions is not None:
            for i in range(NUM_JOINTS):
                val = float(cmd.joint_positions[i])
                if val < lim.position_min[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.POSITION_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} position {val:.4f} rad below minimum {lim.position_min[i]:.4f} rad",
                            actual_value=val,
                            limit_value=float(lim.position_min[i]),
                        )
                    )
                elif val > lim.position_max[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.POSITION_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} position {val:.4f} rad above maximum {lim.position_max[i]:.4f} rad",
                            actual_value=val,
                            limit_value=float(lim.position_max[i]),
                        )
                    )

        if cmd.joint_velocities is not None:
            for i in range(NUM_JOINTS):
                val = float(cmd.joint_velocities[i])
                if abs(val) > lim.velocity_max[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.VELOCITY_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} velocity {val:.4f} rad/s exceeds limit +/-{lim.velocity_max[i]:.4f} rad/s",
                            actual_value=val,
                            limit_value=float(lim.velocity_max[i]),
                        )
                    )

        if cmd.joint_torques is not None:
            for i in range(NUM_JOINTS):
                val = float(cmd.joint_torques[i])
                if abs(val) > lim.torque_max[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.TORQUE_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} torque {val:.4f} Nm exceeds limit +/-{lim.torque_max[i]:.4f} Nm",
                            actual_value=val,
                            limit_value=float(lim.torque_max[i]),
                        )
                    )

        if cmd.gripper_position is not None:
            if cmd.gripper_position < 0.0 or cmd.gripper_position > 1.0:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.GRIPPER_LIMIT,
                        joint_index=6,
                        message=f"Gripper position {cmd.gripper_position:.4f} outside [0.0, 1.0]",
                        actual_value=cmd.gripper_position,
                        limit_value=1.0 if cmd.gripper_position > 1.0 else 0.0,
                    )
                )

        return SafetyResult(
            is_safe=len(violations) == 0,
            violations=tuple(violations),
        )

    def check_state(self, state: D1State) -> list[SafetyViolation]:
        """Check current arm state for safety violations."""
        violations: list[SafetyViolation] = []
        lim = self._limits

        for field_name, arr in [
            ("position", state.joint_positions),
            ("velocity", state.joint_velocities),
            ("torque", state.joint_torques),
        ]:
            for i in range(NUM_JOINTS):
                val = float(arr[i])
                if not self._is_valid_float(val):
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.POSITION_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} {field_name} is NaN/Inf in state",
                            actual_value=0.0,
                            limit_value=0.0,
                        )
                    )
        if violations:
            return violations

        for i in range(NUM_JOINTS):
            val = float(state.joint_positions[i])
            if val < lim.position_min[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.POSITION_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} position {val:.4f} rad below minimum {lim.position_min[i]:.4f} rad",
                        actual_value=val,
                        limit_value=float(lim.position_min[i]),
                    )
                )
            elif val > lim.position_max[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.POSITION_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} position {val:.4f} rad above maximum {lim.position_max[i]:.4f} rad",
                        actual_value=val,
                        limit_value=float(lim.position_max[i]),
                    )
                )

        for i in range(NUM_JOINTS):
            val = float(state.joint_velocities[i])
            if abs(val) > lim.velocity_max[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.VELOCITY_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} velocity {val:.4f} rad/s exceeds limit +/-{lim.velocity_max[i]:.4f} rad/s",
                        actual_value=val,
                        limit_value=float(lim.velocity_max[i]),
                    )
                )

        for i in range(NUM_JOINTS):
            val = float(state.joint_torques[i])
            if abs(val) > lim.torque_max[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.TORQUE_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} torque {val:.4f} Nm exceeds limit +/-{lim.torque_max[i]:.4f} Nm",
                        actual_value=val,
                        limit_value=float(lim.torque_max[i]),
                    )
                )

        return violations

    def clamp_command(self, cmd: D1Command) -> D1Command:
        """Return a new command with all values clamped to safe limits."""
        if self._estop:
            return D1Command(
                mode=0,
                joint_positions=np.zeros(NUM_JOINTS),
                joint_velocities=np.zeros(NUM_JOINTS),
                joint_torques=np.zeros(NUM_JOINTS),
                gripper_position=0.0,
            )

        lim = self._limits

        positions = None
        if cmd.joint_positions is not None:
            positions = np.clip(cmd.joint_positions.copy(), lim.position_min, lim.position_max)

        velocities = None
        if cmd.joint_velocities is not None:
            velocities = np.clip(cmd.joint_velocities.copy(), -lim.velocity_max, lim.velocity_max)

        torques = cmd.joint_torques

        gripper = None
        if cmd.gripper_position is not None:
            gripper = float(np.clip(cmd.gripper_position, 0.0, 1.0))

        return D1Command(
            mode=cmd.mode,
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=torques,
            gripper_position=gripper,
        )
