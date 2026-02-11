"""
D1 Arm State and Command data types.

Extracted from the D1 connection module so that any service can reference
these types without pulling in socket/hardware dependencies.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class D1State:
    """Current state of the D1 arm."""

    joint_positions: np.ndarray  # 7 joints (6 arm + 1 gripper)
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_position: float  # 0.0 (closed) to 1.0 (open)
    timestamp: float


@dataclass
class D1Command:
    """Command to send to the D1 arm."""

    mode: int  # 0=idle, 1=position, 2=velocity, 3=torque
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    gripper_position: Optional[float] = None
