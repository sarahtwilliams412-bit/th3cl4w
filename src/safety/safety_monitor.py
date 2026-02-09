"""
Safety Monitor for Unitree D1 Robotic Arm

This is the most critical module in th3cl4w. It enforces joint limits,
velocity limits, torque limits, and workspace bounds to protect hardware
and humans. An emergency stop flag blocks all commands until manually reset.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.interface.d1_connection import D1Command, D1State, NUM_JOINTS
from src.safety.limits import (
    JOINT_LIMITS_RAD_MIN,
    JOINT_LIMITS_RAD_MAX,
    VELOCITY_MAX_RAD,
    TORQUE_MAX_NM,
    MAX_WORKSPACE_RADIUS_MM,
    MAX_WORKSPACE_RADIUS_M,
    FEEDBACK_MAX_AGE_S,
)

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
    """Default joint limits for the Unitree D1 arm.

    Limits are imported from src.safety.limits — the single source of truth.
    """
    return JointLimits(
        position_min=JOINT_LIMITS_RAD_MIN.copy(),
        position_max=JOINT_LIMITS_RAD_MAX.copy(),
        velocity_max=VELOCITY_MAX_RAD.copy(),
        torque_max=TORQUE_MAX_NM.copy(),
    )


# ---------------------------------------------------------------------------
# Forward kinematics stub for workspace checking
# ---------------------------------------------------------------------------
# D1 approximate DH parameters (meters). These are rough but sufficient for a
# conservative spherical workspace check.  We use the distance from the base
# to the end-effector computed via simplified FK.
_LINK_LENGTHS_M = np.array([0.0, 0.15, 0.22, 0.0, 0.18, 0.0, 0.05])  # rough


def _estimate_reach(joint_positions: np.ndarray) -> float:
    """Estimate end-effector distance from base using simplified 2D planar FK.

    Uses shoulder (J1), elbow (J2), and wrist (J4) pitch joints with the
    approximate link lengths to compute a rough but configuration-dependent
    reach estimate. This is more accurate than the old constant (sum of all
    links) while remaining fast and conservative enough for a safety check.

    Returns distance in meters.
    """
    # Use pitch joints (J1, J2, J4) for a planar arm approximation
    # J0/J3/J5 are yaw/roll and don't change reach distance from base
    q1 = float(joint_positions[1])  # shoulder pitch
    q2 = float(joint_positions[2])  # elbow pitch
    q4 = float(joint_positions[4]) if len(joint_positions) > 4 else 0.0  # wrist pitch

    L1 = _LINK_LENGTHS_M[1]  # shoulder-to-elbow
    L2 = _LINK_LENGTHS_M[2]  # elbow-to-wrist
    L3 = _LINK_LENGTHS_M[4] + _LINK_LENGTHS_M[6]  # wrist-to-EE

    # Cumulative angles in the pitch plane
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(q1 + q2 + q4)
    z = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(q1 + q2 + q4)

    return math.sqrt(x * x + z * z)


class SafetyMonitor:
    """Monitors and enforces safety constraints for the D1 arm.

    Thread-safe: the e-stop flag is a simple bool guarded by the GIL.
    For multi-threaded use, wrap in a lock externally.
    """

    def __init__(self, limits: Optional[JointLimits] = None):
        self._limits = limits or d1_default_limits()
        self._estop = False

    # -- Properties ----------------------------------------------------------

    @property
    def limits(self) -> JointLimits:
        return self._limits

    @property
    def estop_active(self) -> bool:
        return self._estop

    # -- Emergency Stop ------------------------------------------------------

    def trigger_estop(self, reason: str = "manual") -> None:
        """Activate emergency stop. All commands blocked until reset."""
        if not self._estop:
            logger.critical("E-STOP TRIGGERED: %s", reason)
        self._estop = True

    def reset_estop(self) -> None:
        """Reset emergency stop, allowing commands again."""
        logger.info("E-STOP RESET")
        self._estop = False

    # -- Feedback Freshness --------------------------------------------------

    def is_feedback_fresh(self, state: D1State) -> bool:
        """Check whether feedback is recent enough to trust.

        Returns False if the state timestamp is older than FEEDBACK_MAX_AGE_S.
        """
        import time as _time
        age = _time.time() - state.timestamp
        return age <= FEEDBACK_MAX_AGE_S

    # -- Command Validation --------------------------------------------------

    def validate_command(self, cmd: D1Command) -> SafetyResult:
        """Validate a command against all safety constraints.

        Returns SafetyResult indicating whether the command is safe to execute.
        If e-stop is active, the command is always rejected.
        """
        violations: list[SafetyViolation] = []

        # E-stop check first
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

        # Position limits
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

        # Velocity limits
        if cmd.joint_velocities is not None:
            for i in range(NUM_JOINTS):
                val = float(cmd.joint_velocities[i])
                if abs(val) > lim.velocity_max[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.VELOCITY_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} velocity {val:.4f} rad/s exceeds limit ±{lim.velocity_max[i]:.4f} rad/s",
                            actual_value=val,
                            limit_value=float(lim.velocity_max[i]),
                        )
                    )

        # Torque limits
        if cmd.joint_torques is not None:
            for i in range(NUM_JOINTS):
                val = float(cmd.joint_torques[i])
                if abs(val) > lim.torque_max[i]:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.TORQUE_LIMIT,
                            joint_index=i,
                            message=f"Joint {i} torque {val:.4f} Nm exceeds limit ±{lim.torque_max[i]:.4f} Nm",
                            actual_value=val,
                            limit_value=float(lim.torque_max[i]),
                        )
                    )

        # Gripper bounds
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

    # -- State Monitoring ----------------------------------------------------

    def check_state(self, state: D1State) -> list[SafetyViolation]:
        """Check current arm state for safety violations.

        Does NOT trigger e-stop automatically — caller decides response.
        Returns list of violations (empty if state is safe).
        """
        violations: list[SafetyViolation] = []
        lim = self._limits

        # Position
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

        # Velocity
        for i in range(NUM_JOINTS):
            val = float(state.joint_velocities[i])
            if abs(val) > lim.velocity_max[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.VELOCITY_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} velocity {val:.4f} rad/s exceeds limit ±{lim.velocity_max[i]:.4f} rad/s",
                        actual_value=val,
                        limit_value=float(lim.velocity_max[i]),
                    )
                )

        # Torque
        for i in range(NUM_JOINTS):
            val = float(state.joint_torques[i])
            if abs(val) > lim.torque_max[i]:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.TORQUE_LIMIT,
                        joint_index=i,
                        message=f"Joint {i} torque {val:.4f} Nm exceeds limit ±{lim.torque_max[i]:.4f} Nm",
                        actual_value=val,
                        limit_value=float(lim.torque_max[i]),
                    )
                )

        # Workspace bound (spherical)
        reach = _estimate_reach(state.joint_positions)
        if reach > MAX_WORKSPACE_RADIUS_M:
            violations.append(
                SafetyViolation(
                    violation_type=ViolationType.WORKSPACE_BOUND,
                    joint_index=None,
                    message=f"Estimated reach {reach * 1000:.1f} mm exceeds workspace limit {MAX_WORKSPACE_RADIUS_MM:.1f} mm",
                    actual_value=reach,
                    limit_value=MAX_WORKSPACE_RADIUS_M,
                )
            )

        return violations

    # -- Command Clamping ----------------------------------------------------

    def clamp_command(self, cmd: D1Command) -> D1Command:
        """Return a new command with all values clamped to safe limits.

        If e-stop is active, returns an idle command (mode=0, all zeros).
        """
        if self._estop:
            return D1Command(
                mode=0,
                joint_positions=np.zeros(NUM_JOINTS),
                joint_velocities=np.zeros(NUM_JOINTS),
                joint_torques=np.zeros(NUM_JOINTS),
                gripper_position=0.0,
            )

        lim = self._limits

        # Clamp positions
        positions = None
        if cmd.joint_positions is not None:
            positions = np.clip(cmd.joint_positions.copy(), lim.position_min, lim.position_max)

        # Clamp velocities
        velocities = None
        if cmd.joint_velocities is not None:
            velocities = np.clip(cmd.joint_velocities.copy(), -lim.velocity_max, lim.velocity_max)

        # Clamp torques
        torques = None
        if cmd.joint_torques is not None:
            torques = np.clip(cmd.joint_torques.copy(), -lim.torque_max, lim.torque_max)

        # Clamp gripper
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
