"""
Path Optimizer for Unitree D1 Robotic Arm

Trajectory smoothing and time-optimal parameterization.
All angles in DEGREES.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.interpolate import CubicSpline

from .motion_planner import (
    Trajectory,
    TrajectoryPoint,
    NUM_ARM_JOINTS,
    DEFAULT_MAX_JOINT_SPEED,
    DEFAULT_MAX_JOINT_ACCEL,
)

logger = logging.getLogger(__name__)


class PathOptimizer:
    """Optimizes trajectories via smoothing and time-optimal parameterization."""

    def __init__(
        self,
        max_joint_speed: np.ndarray | None = None,
        max_joint_accel: np.ndarray | None = None,
    ):
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

    # ------------------------------------------------------------------
    # Trajectory smoothing via cubic spline
    # ------------------------------------------------------------------

    def smooth(self, trajectory: Trajectory, dt: float = 0.01) -> Trajectory:
        """Smooth a trajectory using cubic spline interpolation.

        Fits a cubic spline through the trajectory positions and
        resamples at uniform dt intervals.

        Parameters
        ----------
        trajectory : input trajectory (may have non-smooth junctions)
        dt : output sample period in seconds

        Returns
        -------
        Smoothed trajectory with continuous velocity and acceleration.
        """
        if len(trajectory.points) < 3:
            return trajectory

        times = trajectory.times_array()
        positions = trajectory.positions_array()  # (N, 6)
        grippers = np.array([p.gripper_mm for p in trajectory.points])

        # Fit cubic spline per joint
        splines = []
        for j in range(NUM_ARM_JOINTS):
            cs = CubicSpline(times, positions[:, j], bc_type="clamped")
            splines.append(cs)

        # Gripper spline
        gripper_spline = CubicSpline(times, grippers, bc_type="clamped")

        # Resample
        duration = times[-1] - times[0]
        n_points = max(2, int(math.ceil(duration / dt)) + 1)
        new_times = np.linspace(times[0], times[-1], n_points)

        new_points = []
        for t in new_times:
            pos = np.array([splines[j](t) for j in range(NUM_ARM_JOINTS)])
            vel = np.array([splines[j](t, 1) for j in range(NUM_ARM_JOINTS)])
            acc = np.array([splines[j](t, 2) for j in range(NUM_ARM_JOINTS)])
            g = float(gripper_spline(t))

            new_points.append(
                TrajectoryPoint(
                    time=float(t),
                    positions=pos,
                    velocities=vel,
                    accelerations=acc,
                    gripper_mm=g,
                )
            )

        return Trajectory(points=new_points, label=trajectory.label + "_smoothed")

    # ------------------------------------------------------------------
    # Time-optimal parameterization
    # ------------------------------------------------------------------

    def time_optimal_parameterize(
        self,
        trajectory: Trajectory,
        dt: float = 0.01,
    ) -> Trajectory:
        """Reparameterize a trajectory for minimum time while respecting limits.

        Uses a simple approach: compute the maximum allowable speed at each
        point based on joint velocity and acceleration limits, then integrate
        to find the time-optimal timing.

        The geometric path is preserved; only timing changes.
        """
        if len(trajectory.points) < 2:
            return trajectory

        positions = trajectory.positions_array()
        grippers = np.array([p.gripper_mm for p in trajectory.points])
        n = len(positions)

        # Compute path parameter (cumulative arc length in joint space)
        ds = np.zeros(n)
        for i in range(1, n):
            ds[i] = np.linalg.norm(positions[i] - positions[i - 1])

        s = np.cumsum(ds)
        total_s = s[-1]
        if total_s < 1e-12:
            return trajectory

        # Compute max speed (ds/dt) at each point
        # dq/dt = (dq/ds) * (ds/dt), so ds/dt_max = min_j(v_max_j / |dq_j/ds|)
        max_sdot = np.full(n, 1e6)
        for i in range(1, n):
            dq_ds = (positions[i] - positions[i - 1]) / max(ds[i], 1e-12)
            for j in range(NUM_ARM_JOINTS):
                if abs(dq_ds[j]) > 1e-9:
                    max_sdot[i] = min(max_sdot[i], self.max_joint_speed[j] / abs(dq_ds[j]))

        # Forward pass: limit acceleration
        sdot = np.zeros(n)
        sdot[0] = 0.0  # start from rest
        for i in range(1, n):
            delta_s = max(ds[i], 1e-12)
            dq_ds = (positions[i] - positions[i - 1]) / delta_s
            # Max acceleration: min_j(a_max_j / |dq_j/ds|)
            max_sddot = 1e6
            for j in range(NUM_ARM_JOINTS):
                if abs(dq_ds[j]) > 1e-9:
                    max_sddot = min(max_sddot, self.max_joint_accel[j] / abs(dq_ds[j]))
            # v² = v0² + 2*a*ds
            sdot_sq = sdot[i - 1] ** 2 + 2 * max_sddot * delta_s
            sdot[i] = min(math.sqrt(max(sdot_sq, 0.0)), max_sdot[i])

        # Backward pass: limit deceleration
        sdot[-1] = 0.0  # end at rest
        for i in range(n - 2, -1, -1):
            delta_s = max(ds[i + 1], 1e-12)
            dq_ds = (positions[i + 1] - positions[i]) / delta_s
            max_sddot = 1e6
            for j in range(NUM_ARM_JOINTS):
                if abs(dq_ds[j]) > 1e-9:
                    max_sddot = min(max_sddot, self.max_joint_accel[j] / abs(dq_ds[j]))
            sdot_sq = sdot[i + 1] ** 2 + 2 * max_sddot * delta_s
            sdot[i] = min(sdot[i], math.sqrt(max(sdot_sq, 0.0)))

        # Integrate time
        new_times = np.zeros(n)
        for i in range(1, n):
            avg_sdot = 0.5 * (sdot[i - 1] + sdot[i])
            if avg_sdot < 1e-12:
                new_times[i] = new_times[i - 1] + 0.001  # small step for zero-speed
            else:
                new_times[i] = new_times[i - 1] + ds[i] / avg_sdot

        # Compute velocities and accelerations with new timing
        new_points = []
        for i in range(n):
            if i == 0:
                vel = np.zeros(NUM_ARM_JOINTS)
                acc = np.zeros(NUM_ARM_JOINTS)
            elif i == n - 1:
                dt_seg = new_times[i] - new_times[i - 1]
                vel = np.zeros(NUM_ARM_JOINTS)
                acc = np.zeros(NUM_ARM_JOINTS)
            else:
                dt_prev = new_times[i] - new_times[i - 1]
                dt_next = new_times[i + 1] - new_times[i]
                if dt_prev > 1e-12:
                    vel = (positions[i] - positions[i - 1]) / dt_prev
                else:
                    vel = np.zeros(NUM_ARM_JOINTS)
                if dt_prev + dt_next > 1e-12:
                    v_prev = (positions[i] - positions[i - 1]) / max(dt_prev, 1e-12)
                    v_next = (positions[i + 1] - positions[i]) / max(dt_next, 1e-12)
                    acc = (v_next - v_prev) / (0.5 * (dt_prev + dt_next))
                else:
                    acc = np.zeros(NUM_ARM_JOINTS)

            new_points.append(
                TrajectoryPoint(
                    time=new_times[i],
                    positions=positions[i].copy(),
                    velocities=vel,
                    accelerations=acc,
                    gripper_mm=grippers[i],
                )
            )

        return Trajectory(points=new_points, label=trajectory.label + "_timeopt")

    # ------------------------------------------------------------------
    # Combined: smooth then time-optimize
    # ------------------------------------------------------------------

    def optimize(self, trajectory: Trajectory, dt: float = 0.01) -> Trajectory:
        """Full optimization: smooth then time-optimal parameterize."""
        smoothed = self.smooth(trajectory, dt)
        return self.time_optimal_parameterize(smoothed, dt)
