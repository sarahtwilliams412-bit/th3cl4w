"""
Controller interfaces for the D1 arm.

Provides a base Controller ABC and a concrete JointPositionController
that commands the arm to track target joint positions with a simple
proportional-derivative (PD) control law.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.interface.d1_connection import D1Command, D1State, NUM_JOINTS

logger = logging.getLogger(__name__)


class Controller(ABC):
    """Abstract base class for arm controllers.

    Subclasses implement ``compute()`` which takes the current arm state
    and returns a command to send.
    """

    @abstractmethod
    def compute(self, state: D1State) -> D1Command:
        """Compute a command based on the current state.

        Args:
            state: Current arm state.

        Returns:
            Command to send to the arm.
        """
        ...

    def reset(self) -> None:
        """Reset controller internal state (e.g. integral terms)."""
        pass


class JointPositionController(Controller):
    """PD joint-space position controller.

    Computes torque commands to track a target joint configuration::

        tau = Kp * (q_target - q) - Kd * dq

    Operates in torque mode (mode=3) by default.
    """

    def __init__(
        self,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        # Default gains — conservative for a 2.2kg arm
        self.kp = kp if kp is not None else np.array(
            [40.0, 40.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float64
        )
        self.kd = kd if kd is not None else np.array(
            [5.0, 5.0, 4.0, 4.0, 2.0, 2.0, 2.0], dtype=np.float64
        )

        if self.kp.shape != (NUM_JOINTS,):
            raise ValueError(f"kp must have shape ({NUM_JOINTS},), got {self.kp.shape}")
        if self.kd.shape != (NUM_JOINTS,):
            raise ValueError(f"kd must have shape ({NUM_JOINTS},), got {self.kd.shape}")

        self._target_positions: Optional[np.ndarray] = None
        self._target_gripper: Optional[float] = None

    def set_target(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> None:
        """Set the target joint positions.

        Args:
            positions: Array of 7 target joint angles (radians).
            gripper: Optional gripper target (0.0–1.0).
        """
        if positions.shape != (NUM_JOINTS,):
            raise ValueError(f"positions must have shape ({NUM_JOINTS},), got {positions.shape}")
        self._target_positions = positions.copy()
        self._target_gripper = gripper

    @property
    def target_positions(self) -> Optional[np.ndarray]:
        return self._target_positions

    def compute(self, state: D1State) -> D1Command:
        """Compute PD torque command."""
        if self._target_positions is None:
            # No target set — send idle
            return D1Command(mode=0)

        error = self._target_positions - state.joint_positions
        torques = self.kp * error - self.kd * state.joint_velocities

        return D1Command(
            mode=3,  # torque mode
            joint_torques=torques,
            gripper_position=self._target_gripper,
        )

    def reset(self) -> None:
        self._target_positions = None
        self._target_gripper = None
