"""
Motion Planner for Unitree D1 Robotic Arm

Waypoint-based planning, joint-space and Cartesian-space interpolation,
collision-free trajectory generation, and speed/acceleration enforcement.

All angles are in DEGREES. Gripper in mm (0-65).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.kinematics.kinematics import D1Kinematics
from src.safety.safety_monitor import SafetyMonitor, d1_default_limits

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + gripper

# Default limits (degrees/s and degrees/s²)
DEFAULT_MAX_JOINT_SPEED = np.array([90.0, 90.0, 120.0, 120.0, 150.0, 150.0])  # deg/s
DEFAULT_MAX_JOINT_ACCEL = np.array([180.0, 180.0, 240.0, 240.0, 300.0, 300.0])  # deg/s²

# Joint limits in degrees
JOINT_LIMITS_DEG = np.array(
    [
        [-135.0, 135.0],  # J0
        [-90.0, 90.0],  # J1
        [-90.0, 90.0],  # J2
        [-135.0, 135.0],  # J3
        [-90.0, 90.0],  # J4
        [-135.0, 135.0],  # J5
    ]
)

GRIPPER_MIN_MM = 0.0
GRIPPER_MAX_MM = 65.0


@dataclass
class Waypoint:
    """A target configuration in joint space (degrees) with optional gripper (mm)."""

    joint_angles: np.ndarray  # shape (6,), degrees
    gripper_mm: float = 0.0
    max_speed_factor: float = 1.0  # 0-1 scale factor on speed limits
    label: str = ""

    def __post_init__(self):
        self.joint_angles = np.asarray(self.joint_angles, dtype=float)
        if self.joint_angles.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_angles must have {NUM_ARM_JOINTS} elements, got {self.joint_angles.shape}"
            )
        self.gripper_mm = float(np.clip(self.gripper_mm, GRIPPER_MIN_MM, GRIPPER_MAX_MM))
        self.max_speed_factor = float(np.clip(self.max_speed_factor, 0.01, 1.0))


@dataclass
class TrajectoryPoint:
    """A single point along a trajectory."""

    time: float  # seconds from trajectory start
    positions: np.ndarray  # (6,) degrees
    velocities: np.ndarray  # (6,) deg/s
    accelerations: np.ndarray  # (6,) deg/s²
    gripper_mm: float = 0.0


@dataclass
class Trajectory:
    """A complete trajectory: ordered list of TrajectoryPoints."""

    points: list[TrajectoryPoint] = field(default_factory=list)
    label: str = ""

    @property
    def duration(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].time - self.points[0].time

    @property
    def num_points(self) -> int:
        return len(self.points)

    def positions_array(self) -> np.ndarray:
        """Return (N, 6) array of positions."""
        return np.array([p.positions for p in self.points])

    def times_array(self) -> np.ndarray:
        """Return (N,) array of times."""
        return np.array([p.time for p in self.points])


class MotionPlanner:
    """Plans trajectories for the D1 arm in joint and Cartesian space."""

    def __init__(
        self,
        kinematics: D1Kinematics | None = None,
        max_joint_speed: np.ndarray | None = None,
        max_joint_accel: np.ndarray | None = None,
        dt: float = 0.01,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.max_joint_speed = (
            np.array(max_joint_speed)
            if max_joint_speed is not None
            else DEFAULT_MAX_JOINT_SPEED.copy()
        )
        self.max_joint_accel = (
            np.array(max_joint_accel)
            if max_joint_accel is not None
            else DEFAULT_MAX_JOINT_ACCEL.copy()
        )
        self.dt = dt

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_joint_angles(self, angles_deg: np.ndarray) -> bool:
        """Check if joint angles are within limits."""
        angles_deg = np.asarray(angles_deg)
        for i in range(NUM_ARM_JOINTS):
            if angles_deg[i] < JOINT_LIMITS_DEG[i, 0] or angles_deg[i] > JOINT_LIMITS_DEG[i, 1]:
                return False
        return True

    def clamp_joint_angles(self, angles_deg: np.ndarray) -> np.ndarray:
        """Clamp joint angles to within limits."""
        angles_deg = np.asarray(angles_deg, dtype=float).copy()
        for i in range(NUM_ARM_JOINTS):
            angles_deg[i] = np.clip(angles_deg[i], JOINT_LIMITS_DEG[i, 0], JOINT_LIMITS_DEG[i, 1])
        return angles_deg

    # ------------------------------------------------------------------
    # Joint-space linear interpolation
    # ------------------------------------------------------------------

    def linear_joint_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed_factor: float = 1.0,
        gripper_start: float = 0.0,
        gripper_end: float = 0.0,
    ) -> Trajectory:
        """Linear interpolation in joint space with trapezoidal velocity profile.

        Parameters
        ----------
        start, end : (6,) joint angles in degrees
        speed_factor : scale on max speed (0, 1]
        gripper_start, gripper_end : gripper in mm

        Returns
        -------
        Trajectory with points at self.dt intervals.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        gripper_delta = gripper_end - gripper_start

        # Determine duration from slowest joint using trapezoidal profile
        max_speed = self.max_joint_speed * speed_factor
        max_accel = self.max_joint_accel * speed_factor

        durations = []
        for i in range(NUM_ARM_JOINTS):
            d = abs(delta[i])
            if d < 1e-9:
                durations.append(0.0)
                continue
            # Trapezoidal: check if we reach max speed
            t_accel = max_speed[i] / max_accel[i]
            d_accel = 0.5 * max_accel[i] * t_accel**2
            if 2 * d_accel <= d:
                # Trapezoidal: accel + cruise + decel
                t_cruise = (d - 2 * d_accel) / max_speed[i]
                durations.append(2 * t_accel + t_cruise)
            else:
                # Triangular: no cruise phase
                durations.append(2.0 * math.sqrt(d / max_accel[i]))

        total_time = max(durations) if durations else 0.0
        if total_time < self.dt:
            # Start == end (or very close)
            pt = TrajectoryPoint(
                time=0.0,
                positions=start.copy(),
                velocities=np.zeros(NUM_ARM_JOINTS),
                accelerations=np.zeros(NUM_ARM_JOINTS),
                gripper_mm=gripper_start,
            )
            return Trajectory(points=[pt])

        n_points = max(2, int(math.ceil(total_time / self.dt)) + 1)
        times = np.linspace(0.0, total_time, n_points)
        points: list[TrajectoryPoint] = []

        for t in times:
            s = t / total_time  # normalized position 0..1
            # Use smooth s-curve (cubic hermite: 3s²-2s³)
            s_smooth = 3 * s**2 - 2 * s**3
            ds = (6 * s - 6 * s**2) / total_time
            dds = (6 - 12 * s) / (total_time**2)

            pos = start + delta * s_smooth
            vel = delta * ds
            acc = delta * dds
            g = gripper_start + gripper_delta * s_smooth

            points.append(
                TrajectoryPoint(
                    time=t,
                    positions=pos,
                    velocities=vel,
                    accelerations=acc,
                    gripper_mm=g,
                )
            )

        return Trajectory(points=points)

    # ------------------------------------------------------------------
    # Waypoint-based planning
    # ------------------------------------------------------------------

    def plan_waypoints(self, waypoints: list[Waypoint]) -> Trajectory:
        """Plan a trajectory through a sequence of waypoints.

        Concatenates linear joint trajectories between consecutive waypoints.
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        combined = Trajectory(label="waypoint_trajectory")
        time_offset = 0.0

        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]
            seg = self.linear_joint_trajectory(
                start=wp_start.joint_angles,
                end=wp_end.joint_angles,
                speed_factor=wp_end.max_speed_factor,
                gripper_start=wp_start.gripper_mm,
                gripper_end=wp_end.gripper_mm,
            )
            for j, pt in enumerate(seg.points):
                # Skip first point of subsequent segments (overlap)
                if i > 0 and j == 0:
                    continue
                combined.points.append(
                    TrajectoryPoint(
                        time=pt.time + time_offset,
                        positions=pt.positions.copy(),
                        velocities=pt.velocities.copy(),
                        accelerations=pt.accelerations.copy(),
                        gripper_mm=pt.gripper_mm,
                    )
                )
            time_offset += seg.duration

        return combined

    # ------------------------------------------------------------------
    # Cartesian path planning
    # ------------------------------------------------------------------

    def cartesian_linear_path(
        self,
        start_angles_deg: np.ndarray,
        target_pose: np.ndarray,
        n_cartesian_steps: int = 20,
        speed_factor: float = 1.0,
        gripper_start: float = 0.0,
        gripper_end: float = 0.0,
    ) -> Trajectory:
        """Plan a straight-line path in Cartesian space using IK at each step.

        Parameters
        ----------
        start_angles_deg : (6,) current joint angles in degrees
        target_pose : (4,4) desired end-effector pose
        n_cartesian_steps : number of interpolation steps in Cartesian space
        speed_factor : speed scaling
        gripper_start, gripper_end : gripper positions in mm

        Returns
        -------
        Trajectory in joint space
        """
        start_angles_deg = np.asarray(start_angles_deg, dtype=float)
        start_rad = np.deg2rad(start_angles_deg)

        # Pad to 7-DOF for kinematics (gripper = 0)
        q_start_7 = np.zeros(7)
        q_start_7[:6] = start_rad

        start_pose = self.kinematics.forward_kinematics(q_start_7)

        # Interpolate position and rotation
        start_pos = start_pose[:3, 3]
        end_pos = target_pose[:3, 3]
        start_R = start_pose[:3, :3]
        end_R = target_pose[:3, :3]

        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_matrix(np.stack([start_R, end_R]))
        slerp = Slerp([0.0, 1.0], rots)

        waypoints: list[Waypoint] = []
        q_prev = q_start_7.copy()

        for i in range(n_cartesian_steps + 1):
            s = i / n_cartesian_steps
            pos = start_pos + s * (end_pos - start_pos)
            rot = slerp(s).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = pos

            q_sol = self.kinematics.inverse_kinematics(T, q_init=q_prev)
            q_prev = q_sol.copy()

            angles_deg = np.rad2deg(q_sol[:6])
            angles_deg = self.clamp_joint_angles(angles_deg)
            g = gripper_start + s * (gripper_end - gripper_start)

            waypoints.append(
                Waypoint(
                    joint_angles=angles_deg,
                    gripper_mm=g,
                    max_speed_factor=speed_factor,
                )
            )

        if len(waypoints) < 2:
            waypoints.append(waypoints[0])

        return self.plan_waypoints(waypoints)

    # ------------------------------------------------------------------
    # Collision-free trajectory (simple self-collision check)
    # ------------------------------------------------------------------

    def check_self_collision(self, angles_deg: np.ndarray, min_clearance: float = 0.03) -> bool:
        """Simple self-collision check using joint positions from FK.

        Returns True if configuration is collision-free.
        """
        angles_rad = np.deg2rad(np.asarray(angles_deg))
        q7 = np.zeros(7)
        q7[:6] = angles_rad

        positions = self.kinematics.get_joint_positions_3d(q7)

        # Check distance between non-adjacent link endpoints
        n = len(positions)
        for i in range(n):
            for j in range(i + 2, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_clearance:
                    return False
        return True

    def plan_collision_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed_factor: float = 1.0,
        gripper_start: float = 0.0,
        gripper_end: float = 0.0,
        min_clearance: float = 0.03,
    ) -> Trajectory:
        """Plan a joint-space trajectory and verify it's collision-free.

        Raises ValueError if any point along the trajectory is in collision.
        """
        traj = self.linear_joint_trajectory(
            start,
            end,
            speed_factor,
            gripper_start,
            gripper_end,
        )

        for pt in traj.points:
            if not self.check_self_collision(pt.positions, min_clearance):
                raise ValueError(
                    f"Self-collision detected at t={pt.time:.3f}s, " f"angles={pt.positions}"
                )

        return traj

    # ------------------------------------------------------------------
    # Speed/acceleration limit enforcement
    # ------------------------------------------------------------------

    def enforce_limits(self, trajectory: Trajectory) -> Trajectory:
        """Scale trajectory timing to respect speed and acceleration limits.

        Returns a new trajectory with adjusted timing if needed.
        """
        if len(trajectory.points) < 2:
            return trajectory

        max_speed_ratio = 0.0
        max_accel_ratio = 0.0

        for pt in trajectory.points:
            for i in range(NUM_ARM_JOINTS):
                if self.max_joint_speed[i] > 0:
                    r = abs(pt.velocities[i]) / self.max_joint_speed[i]
                    max_speed_ratio = max(max_speed_ratio, r)
                if self.max_joint_accel[i] > 0:
                    r = abs(pt.accelerations[i]) / self.max_joint_accel[i]
                    max_accel_ratio = max(max_accel_ratio, r)

        # Scale factor: if any limit exceeded, slow down
        scale = max(1.0, max_speed_ratio, math.sqrt(max_accel_ratio))

        if scale <= 1.0:
            return trajectory

        new_points = []
        for pt in trajectory.points:
            new_points.append(
                TrajectoryPoint(
                    time=pt.time * scale,
                    positions=pt.positions.copy(),
                    velocities=pt.velocities / scale,
                    accelerations=pt.accelerations / (scale**2),
                    gripper_mm=pt.gripper_mm,
                )
            )

        return Trajectory(points=new_points, label=trajectory.label)
