"""
Joint Controller for Unitree D1 Arm

Provides high-level joint control with trajectory generation,
PID gains configuration, and multiple control modes.

Supports two trajectory modes:
- Legacy cubic interpolation (backward compatible)
- Minimum-jerk trajectories with full smoothing pipeline (recommended)

The smooth motion pipeline chains:
    Minimum-jerk trajectory → Gravity compensation → Command filter → Hardware
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.interface.d1_connection import D1Connection, D1State, D1Command
from src.control.joint_service import NUM_JOINTS
from src.control.smooth_trajectory import (
    minimum_jerk_waypoint,
    compute_movement_duration,
)
from src.control.command_filter import SmoothCommandPipeline
from src.control.gravity_compensation import GravityCompensator, ThermalMonitor

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

    Supports position (with cubic or minimum-jerk trajectory), velocity,
    and torque control modes.

    When smooth_motion=True (default), uses the full smoothing pipeline:
    - Minimum-jerk trajectory generation (bell-shaped velocity profiles)
    - Fitts' Law duration estimation (human-like timing)
    - Gravity compensation feedforward (reduces servo oscillation)
    - Dual-EMA command smoothing (eliminates step discontinuities)
    - Jerk-limited rate limiting (hardware resonance protection)
    - Thermal monitoring (prevents servo overheating)
    """

    CONTROL_RATE_HZ = 200  # 5ms control loop

    def __init__(
        self,
        connection: D1Connection,
        gains: PIDGains | None = None,
        smooth_motion: bool = True,
        ema_alpha: float = 0.3,
        enable_gravity_comp: bool = True,
        enable_thermal_monitor: bool = True,
    ):
        self.connection = connection
        self.gains = gains or PIDGains()
        self._last_gripper: float = 0.0
        self.smooth_motion = smooth_motion

        dt = 1.0 / self.CONTROL_RATE_HZ

        # Smooth motion pipeline components
        self._command_pipeline: SmoothCommandPipeline | None = None
        self._gravity_comp: GravityCompensator | None = None
        self._thermal_monitor: ThermalMonitor | None = None
        self._pipeline_initialized = False

        if smooth_motion:
            from src.safety.limits import VELOCITY_MAX_RAD

            max_vel = VELOCITY_MAX_RAD.copy()
            max_acc = max_vel * 3.0  # reasonable accel limit
            max_jerk = max_acc * 8.0  # reasonable jerk limit

            self._command_pipeline = SmoothCommandPipeline(
                n_joints=NUM_JOINTS,
                ema_alpha=ema_alpha,
                max_velocity=max_vel,
                max_acceleration=max_acc,
                max_jerk=max_jerk,
                dt=dt,
            )

            if enable_gravity_comp:
                self._gravity_comp = GravityCompensator()

            if enable_thermal_monitor:
                self._thermal_monitor = ThermalMonitor(NUM_JOINTS)

    def _init_pipeline(self, state: D1State) -> None:
        """Initialize the smooth command pipeline from current arm state."""
        if self._command_pipeline is not None and not self._pipeline_initialized:
            self._command_pipeline.reset(state.joint_positions)
            self._pipeline_initialized = True

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
        self._pipeline_initialized = False

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

    def _send_smooth_position(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Send a position command through the smoothing pipeline."""
        pos = positions
        if self._command_pipeline is not None and self._pipeline_initialized:
            pos = self._command_pipeline.smooth_command(positions)

        cmd = D1Command(
            mode=1,
            joint_positions=pos,
            joint_velocities=velocities,
            gripper_position=self._last_gripper,
        )
        self.connection.send_command(cmd)

    def _send_position_direct(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Send a position command directly (no smoothing pipeline)."""
        cmd = D1Command(
            mode=1,
            joint_positions=positions,
            joint_velocities=velocities,
            gripper_position=self._last_gripper,
        )
        self.connection.send_command(cmd)

    def move_to_position(self, target_positions: np.ndarray, duration: float):
        """
        Smooth position control using cubic interpolation (legacy path).

        Blocks for `duration` seconds, sending position commands at CONTROL_RATE_HZ.
        Does NOT use the smooth motion pipeline — use move_smooth() for that.
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
            self._send_position_direct(pos, vel)

            # Sleep for remainder of control period
            sleep_time = dt - (time.monotonic() - t_start - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Send final position
        self._send_position_direct(target_positions.copy(), np.zeros(NUM_JOINTS))

    def move_smooth(
        self,
        target_positions: np.ndarray,
        duration: float | None = None,
        speed_factor: float = 1.0,
    ):
        """Smooth position control using minimum-jerk trajectory.

        Uses the full smooth motion pipeline:
        1. Minimum-jerk trajectory (bell-shaped velocity profile)
        2. Fitts' Law duration (if duration not specified)
        3. Gravity compensation feedforward
        4. Dual-EMA command smoothing
        5. Jerk-limited rate limiting

        Parameters
        ----------
        target_positions : (7,) target joint angles in radians
        duration : movement time in seconds (auto-computed via Fitts' Law if None)
        speed_factor : speed multiplier for Fitts' Law duration (>1 = faster)
        """
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.shape != (NUM_JOINTS,):
            raise ValueError(f"Expected {NUM_JOINTS} positions, got {target_positions.shape}")

        state = self.get_state()
        q0 = state.joint_positions.copy()
        v0 = state.joint_velocities.copy()
        vf = np.zeros(NUM_JOINTS)

        self._init_pipeline(state)

        # Compute duration via Fitts' Law if not provided
        if duration is None:
            duration = compute_movement_duration(q0, target_positions, speed_factor=speed_factor)
        if duration <= 0:
            duration = 0.3  # minimum duration

        dt = 1.0 / self.CONTROL_RATE_HZ
        t_start = time.monotonic()

        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break

            # Minimum-jerk interpolation
            pos, vel, acc = minimum_jerk_waypoint(q0, target_positions, v0, vf, duration, elapsed)

            # Apply gravity compensation at current configuration (not just start)
            if self._gravity_comp is not None:
                gravity_offset = self._gravity_comp.compute_position_offset(pos)
                pos = pos + gravity_offset

            # Send through smoothing pipeline
            self._send_smooth_position(pos, vel)

            # Update thermal monitor
            if self._thermal_monitor is not None and self._gravity_comp is not None:
                tau_g = self._gravity_comp.compute_gravity_torques(pos)
                self._thermal_monitor.update(tau_g, dt)

            # Sleep for remainder of control period
            sleep_time = dt - (time.monotonic() - t_start - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Send final position with gravity compensation
        if self._gravity_comp is not None:
            gravity_offset = self._gravity_comp.compute_position_offset(target_positions)
        else:
            gravity_offset = np.zeros(NUM_JOINTS)
        final_pos = target_positions + gravity_offset
        self._send_smooth_position(final_pos, np.zeros(NUM_JOINTS))

    def go_home(self, duration: float = 3.0):
        """Move all joints to zero/home position safely."""
        home = np.zeros(NUM_JOINTS)
        if self.smooth_motion:
            self.move_smooth(home, duration=duration)
        else:
            self.move_to_position(home, duration)

    @property
    def thermal_status(self) -> dict | None:
        """Get thermal monitoring status, or None if not enabled."""
        if self._thermal_monitor is None:
            return None
        return {
            "max_temperature": self._thermal_monitor.max_temperature,
            "is_warning": self._thermal_monitor.is_warning,
            "is_critical": self._thermal_monitor.is_critical,
            "speed_reduction": self._thermal_monitor.speed_reduction_factor(),
            "per_joint": self._thermal_monitor.thermal_loads.tolist(),
        }
