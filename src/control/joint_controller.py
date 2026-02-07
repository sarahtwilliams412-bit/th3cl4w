"""
Joint Controller for Unitree D1 Arm

Provides high-level joint control with trajectory generation,
PID gains configuration, and multiple control modes.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.interface.d1_connection import D1Connection, D1State, D1Command, NUM_JOINTS

logger = logging.getLogger(__name__)


@dataclass
class PIDGains:
    """PID gains per joint (arrays of 7)."""
    kp: np.ndarray = field(default_factory=lambda: np.full(NUM_JOINTS, 100.0))
    ki: np.ndarray = field(default_factory=lambda: np.full(NUM_JOINTS, 0.0))
    kd: np.ndarray = field(default_factory=lambda: np.full(NUM_JOINTS, 10.0))

    def __post_init__(self):
        self.kp = np.asarray(self.kp, dtype=np.float64)
        self.ki = np.asarray(self.ki, dtype=np.float64)
        self.kd = np.asarray(self.kd, dtype=np.float64)
        for name, arr in [("kp", self.kp), ("ki", self.ki), ("kd", self.kd)]:
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(f"{name} must have shape ({NUM_JOINTS},), got {arr.shape}")


def cubic_interpolation(
    q0: np.ndarray,
    qf: np.ndarray,
    v0: np.ndarray,
    vf: np.ndarray,
    duration: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic polynomial trajectory between two joint configurations.

    Returns (position, velocity) at time t.
    Boundary conditions: q(0)=q0, q(T)=qf, v(0)=v0, v(T)=vf.
    """
    T = duration
    if T <= 0:
        return qf.copy(), vf.copy()
    t = np.clip(t, 0.0, T)

    a0 = q0
    a1 = v0
    a2 = (3.0 * (qf - q0) / T**2) - (2.0 * v0 / T) - (vf / T)
    a3 = (-2.0 * (qf - q0) / T**3) + ((vf + v0) / T**2)

    pos = a0 + a1 * t + a2 * t**2 + a3 * t**3
    vel = a1 + 2.0 * a2 * t + 3.0 * a3 * t**2
    return pos, vel


class JointController:
    """
    High-level joint controller for the Unitree D1 arm.

    Supports position (with cubic trajectory), velocity, and torque control modes.
    """

    CONTROL_RATE_HZ = 200  # 5ms control loop

    def __init__(self, connection: D1Connection, gains: PIDGains | None = None):
        self.connection = connection
        self.gains = gains or PIDGains()
        self._last_gripper: float = 0.0

    def get_state(self) -> D1State:
        """Read current arm state. Raises RuntimeError if unavailable."""
        state = self.connection.get_state()
        if state is None:
            raise RuntimeError("Failed to read arm state")
        return state

    def stop(self):
        """Immediate stop — send idle command."""
        cmd = D1Command(mode=0)
        self.connection.send_command(cmd)

    def set_velocity(self, velocities: np.ndarray):
        """Direct velocity mode command."""
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.shape != (NUM_JOINTS,):
            raise ValueError(f"Expected {NUM_JOINTS} velocities, got {velocities.shape}")
        cmd = D1Command(
            mode=2,
            joint_velocities=velocities,
            gripper_position=self._last_gripper,
        )
        self.connection.send_command(cmd)

    def set_torque(self, torques: np.ndarray):
        """Direct torque mode command."""
        torques = np.asarray(torques, dtype=np.float64)
        if torques.shape != (NUM_JOINTS,):
            raise ValueError(f"Expected {NUM_JOINTS} torques, got {torques.shape}")
        cmd = D1Command(
            mode=3,
            joint_torques=torques,
            gripper_position=self._last_gripper,
        )
        self.connection.send_command(cmd)

    def set_gripper(self, position: float):
        """Set gripper position. 0.0 = closed, 1.0 = open."""
        position = float(np.clip(position, 0.0, 1.0))
        self._last_gripper = position
        # Send current joint positions with updated gripper
        try:
            state = self.get_state()
            cmd = D1Command(
                mode=1,
                joint_positions=state.joint_positions,
                gripper_position=position,
            )
        except RuntimeError:
            cmd = D1Command(mode=0, gripper_position=position)
        self.connection.send_command(cmd)

    def move_to_position(self, target_positions: np.ndarray, duration: float):
        """
        Smooth position control using cubic interpolation.

        Blocks for `duration` seconds, sending position commands at CONTROL_RATE_HZ.
        """
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (NUM_JOINTS,):
            raise ValueError(f"Expected {NUM_JOINTS} positions, got {target_positions.shape}")
        if duration <= 0:
            raise ValueError("Duration must be positive")

        state = self.get_state()
        q0 = state.joint_positions.copy()
        v0 = state.joint_velocities.copy()
        vf = np.zeros(NUM_JOINTS)

        dt = 1.0 / self.CONTROL_RATE_HZ
        t_start = time.monotonic()

        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break

            pos, vel = cubic_interpolation(q0, target_positions, v0, vf, duration, elapsed)
            cmd = D1Command(
                mode=1,
                joint_positions=pos,
                joint_velocities=vel,
                gripper_position=self._last_gripper,
            )
            self.connection.send_command(cmd)

            # Sleep for remainder of control period
            sleep_time = dt - (time.monotonic() - t_start - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Send final position
        cmd = D1Command(
            mode=1,
            joint_positions=target_positions.copy(),
            joint_velocities=np.zeros(NUM_JOINTS),
            gripper_position=self._last_gripper,
        )
        self.connection.send_command(cmd)

    def go_home(self, duration: float = 3.0):
        """Move all joints to zero/home position safely."""
        home = np.zeros(NUM_JOINTS)
        self.move_to_position(home, duration)
