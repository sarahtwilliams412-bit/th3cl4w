"""
Joint limits, velocity caps, and torque limits for the Unitree D1 arm.

These limits are based on the D1 hardware specifications.  Adjust them
if your specific unit has different firmware limits or if you need
tighter operational constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.interface.d1_connection import D1Command, NUM_JOINTS

logger = logging.getLogger(__name__)


class SafetyViolation(Exception):
    """Raised when a command violates safety limits."""
    pass


@dataclass
class D1SafetyLimits:
    """Safety limits for the D1 arm.

    All angular values are in radians.  Torque values are in Nm.
    Gripper range is 0.0 (closed) to 1.0 (open).

    The defaults are conservative estimates for the Unitree D1.
    Override them with your actual hardware limits.
    """

    # Joint position limits (radians) — [min, max] per joint
    # Joints 0-5: arm joints, Joint 6: wrist
    joint_position_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [-3.1, -2.0, -3.1, -2.4, -3.1, -2.0, -3.1], dtype=np.float64
        )
    )
    joint_position_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.1, 2.0, 3.1, 2.4, 3.1, 2.0, 3.1], dtype=np.float64
        )
    )

    # Joint velocity limits (rad/s) — symmetric, per joint
    joint_velocity_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64
        )
    )

    # Joint torque limits (Nm) — symmetric, per joint
    joint_torque_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [50.0, 50.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float64
        )
    )

    # Gripper limits
    gripper_min: float = 0.0
    gripper_max: float = 1.0

    # Valid command modes
    valid_modes: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def check_positions(self, positions: np.ndarray) -> Optional[str]:
        """Return an error message if positions are out of bounds, else None."""
        if positions.shape != (NUM_JOINTS,):
            return f"Expected {NUM_JOINTS} joint positions, got shape {positions.shape}"
        violations = []
        for i in range(NUM_JOINTS):
            if positions[i] < self.joint_position_min[i]:
                violations.append(
                    f"joint {i}: {positions[i]:.4f} < min {self.joint_position_min[i]:.4f}"
                )
            elif positions[i] > self.joint_position_max[i]:
                violations.append(
                    f"joint {i}: {positions[i]:.4f} > max {self.joint_position_max[i]:.4f}"
                )
        if violations:
            return "Position limit violations: " + "; ".join(violations)
        return None

    def check_velocities(self, velocities: np.ndarray) -> Optional[str]:
        """Return an error message if velocities exceed limits, else None."""
        if velocities.shape != (NUM_JOINTS,):
            return f"Expected {NUM_JOINTS} joint velocities, got shape {velocities.shape}"
        violations = []
        for i in range(NUM_JOINTS):
            if abs(velocities[i]) > self.joint_velocity_max[i]:
                violations.append(
                    f"joint {i}: |{velocities[i]:.4f}| > max {self.joint_velocity_max[i]:.4f}"
                )
        if violations:
            return "Velocity limit violations: " + "; ".join(violations)
        return None

    def check_torques(self, torques: np.ndarray) -> Optional[str]:
        """Return an error message if torques exceed limits, else None."""
        if torques.shape != (NUM_JOINTS,):
            return f"Expected {NUM_JOINTS} joint torques, got shape {torques.shape}"
        violations = []
        for i in range(NUM_JOINTS):
            if abs(torques[i]) > self.joint_torque_max[i]:
                violations.append(
                    f"joint {i}: |{torques[i]:.4f}| > max {self.joint_torque_max[i]:.4f}"
                )
        if violations:
            return "Torque limit violations: " + "; ".join(violations)
        return None

    def clamp_positions(self, positions: np.ndarray) -> np.ndarray:
        """Clamp joint positions to within limits."""
        return np.clip(positions, self.joint_position_min, self.joint_position_max)

    def clamp_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Clamp joint velocities to within limits (symmetric)."""
        return np.clip(velocities, -self.joint_velocity_max, self.joint_velocity_max)

    def clamp_torques(self, torques: np.ndarray) -> np.ndarray:
        """Clamp joint torques to within limits (symmetric)."""
        return np.clip(torques, -self.joint_torque_max, self.joint_torque_max)

    def clamp_gripper(self, value: float) -> float:
        """Clamp gripper position to valid range."""
        return float(np.clip(value, self.gripper_min, self.gripper_max))


def validate_command(cmd: D1Command, limits: D1SafetyLimits) -> None:
    """Validate a command against safety limits.

    Raises SafetyViolation if any field is out of bounds.
    """
    if cmd.mode not in limits.valid_modes:
        raise SafetyViolation(f"Invalid mode {cmd.mode}, valid modes: {limits.valid_modes}")

    if cmd.joint_positions is not None:
        err = limits.check_positions(cmd.joint_positions)
        if err:
            raise SafetyViolation(err)

    if cmd.joint_velocities is not None:
        err = limits.check_velocities(cmd.joint_velocities)
        if err:
            raise SafetyViolation(err)

    if cmd.joint_torques is not None:
        err = limits.check_torques(cmd.joint_torques)
        if err:
            raise SafetyViolation(err)

    if cmd.gripper_position is not None:
        if not (limits.gripper_min <= cmd.gripper_position <= limits.gripper_max):
            raise SafetyViolation(
                f"Gripper position {cmd.gripper_position} outside "
                f"[{limits.gripper_min}, {limits.gripper_max}]"
            )


def clamp_command(cmd: D1Command, limits: D1SafetyLimits) -> D1Command:
    """Return a new command with all fields clamped to safety limits.

    This is the permissive alternative to validate_command — instead of
    raising on violations, it silently saturates values to the nearest
    safe bound.  Logs a warning when clamping changes a value.
    """
    positions = cmd.joint_positions
    velocities = cmd.joint_velocities
    torques = cmd.joint_torques
    gripper = cmd.gripper_position

    if positions is not None:
        clamped = limits.clamp_positions(positions)
        if not np.array_equal(clamped, positions):
            logger.warning("Clamped joint positions to safety limits")
        positions = clamped

    if velocities is not None:
        clamped = limits.clamp_velocities(velocities)
        if not np.array_equal(clamped, velocities):
            logger.warning("Clamped joint velocities to safety limits")
        velocities = clamped

    if torques is not None:
        clamped = limits.clamp_torques(torques)
        if not np.array_equal(clamped, torques):
            logger.warning("Clamped joint torques to safety limits")
        torques = clamped

    if gripper is not None:
        clamped_g = limits.clamp_gripper(gripper)
        if clamped_g != gripper:
            logger.warning("Clamped gripper position to safety limits")
        gripper = clamped_g

    return D1Command(
        mode=cmd.mode,
        joint_positions=positions,
        joint_velocities=velocities,
        joint_torques=torques,
        gripper_position=gripper,
    )
