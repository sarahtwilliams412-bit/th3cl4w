"""
Joint-space trajectory generation for the D1 arm.

Provides linear and cubic interpolation between waypoints in joint space.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.interface.d1_connection import NUM_JOINTS

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """A single waypoint in a joint-space trajectory."""
    positions: np.ndarray   # Joint angles (radians), shape (NUM_JOINTS,)
    time: float             # Time at which this point should be reached (seconds)
    velocities: Optional[np.ndarray] = None  # Optional joint velocities at this point


class JointTrajectory:
    """Joint-space trajectory with interpolation.

    Build a trajectory from waypoints and sample it at arbitrary times::

        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=mid, time=1.0))
        traj.add_point(TrajectoryPoint(positions=end, time=2.0))

        positions = traj.sample(0.5)  # interpolated positions at t=0.5s
    """

    def __init__(self):
        self._points: List[TrajectoryPoint] = []

    def add_point(self, point: TrajectoryPoint) -> None:
        """Add a waypoint to the trajectory.

        Points must be added in order of increasing time.
        """
        if point.positions.shape != (NUM_JOINTS,):
            raise ValueError(
                f"positions must have shape ({NUM_JOINTS},), got {point.positions.shape}"
            )
        if self._points and point.time <= self._points[-1].time:
            raise ValueError(
                f"Points must be in increasing time order: "
                f"{point.time} <= {self._points[-1].time}"
            )
        self._points.append(point)

    @property
    def points(self) -> List[TrajectoryPoint]:
        return list(self._points)

    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        if len(self._points) < 2:
            return 0.0
        return self._points[-1].time - self._points[0].time

    @property
    def start_time(self) -> float:
        if not self._points:
            return 0.0
        return self._points[0].time

    @property
    def end_time(self) -> float:
        if not self._points:
            return 0.0
        return self._points[-1].time

    def is_empty(self) -> bool:
        return len(self._points) < 2

    def sample(self, t: float) -> np.ndarray:
        """Sample the trajectory at time t.

        Uses cubic interpolation when velocities are provided at both
        endpoints of a segment, otherwise falls back to linear interpolation.

        Returns the first/last waypoint positions if t is outside the
        trajectory time range (no extrapolation).

        Args:
            t: Time to sample at (seconds).

        Returns:
            Joint positions array of shape (NUM_JOINTS,).
        """
        if len(self._points) == 0:
            raise ValueError("Trajectory has no points")
        if len(self._points) == 1:
            return self._points[0].positions.copy()

        # Clamp to trajectory bounds
        if t <= self._points[0].time:
            return self._points[0].positions.copy()
        if t >= self._points[-1].time:
            return self._points[-1].positions.copy()

        # Find the segment containing t
        for i in range(len(self._points) - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            if p0.time <= t <= p1.time:
                return self._interpolate_segment(p0, p1, t)

        # Shouldn't reach here, but return last point as fallback
        return self._points[-1].positions.copy()

    def sample_velocity(self, t: float) -> np.ndarray:
        """Sample the trajectory velocity at time t.

        Returns zeros if outside trajectory range or if only linear
        interpolation is used without velocity data.

        Args:
            t: Time to sample at (seconds).

        Returns:
            Joint velocities array of shape (NUM_JOINTS,).
        """
        if len(self._points) < 2:
            return np.zeros(NUM_JOINTS, dtype=np.float64)

        if t <= self._points[0].time or t >= self._points[-1].time:
            return np.zeros(NUM_JOINTS, dtype=np.float64)

        for i in range(len(self._points) - 1):
            p0 = self._points[i]
            p1 = self._points[i + 1]
            if p0.time <= t <= p1.time:
                return self._interpolate_segment_velocity(p0, p1, t)

        return np.zeros(NUM_JOINTS, dtype=np.float64)

    def _interpolate_segment(
        self, p0: TrajectoryPoint, p1: TrajectoryPoint, t: float
    ) -> np.ndarray:
        """Interpolate within a single segment."""
        dt = p1.time - p0.time
        if dt <= 0:
            return p0.positions.copy()

        s = (t - p0.time) / dt  # normalized time [0, 1]

        if p0.velocities is not None and p1.velocities is not None:
            return self._cubic_interp(p0, p1, s, dt)
        else:
            return self._linear_interp(p0, p1, s)

    def _interpolate_segment_velocity(
        self, p0: TrajectoryPoint, p1: TrajectoryPoint, t: float
    ) -> np.ndarray:
        """Interpolate velocity within a single segment."""
        dt = p1.time - p0.time
        if dt <= 0:
            return np.zeros(NUM_JOINTS, dtype=np.float64)

        s = (t - p0.time) / dt

        if p0.velocities is not None and p1.velocities is not None:
            return self._cubic_interp_velocity(p0, p1, s, dt)
        else:
            # Linear interpolation: constant velocity
            return (p1.positions - p0.positions) / dt

    @staticmethod
    def _linear_interp(
        p0: TrajectoryPoint, p1: TrajectoryPoint, s: float
    ) -> np.ndarray:
        """Linear interpolation between two points."""
        return p0.positions + s * (p1.positions - p0.positions)

    @staticmethod
    def _cubic_interp(
        p0: TrajectoryPoint, p1: TrajectoryPoint, s: float, dt: float
    ) -> np.ndarray:
        """Cubic Hermite interpolation between two points.

        Uses positions and velocities at both endpoints to produce a
        smooth C1-continuous trajectory.
        """
        q0 = p0.positions
        q1 = p1.positions
        v0 = p0.velocities * dt  # scale to [0,1] interval
        v1 = p1.velocities * dt

        s2 = s * s
        s3 = s2 * s

        # Hermite basis functions
        h00 = 2*s3 - 3*s2 + 1
        h10 = s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 = s3 - s2

        return h00 * q0 + h10 * v0 + h01 * q1 + h11 * v1

    @staticmethod
    def _cubic_interp_velocity(
        p0: TrajectoryPoint, p1: TrajectoryPoint, s: float, dt: float
    ) -> np.ndarray:
        """Derivative of cubic Hermite interpolation."""
        q0 = p0.positions
        q1 = p1.positions
        v0 = p0.velocities * dt
        v1 = p1.velocities * dt

        s2 = s * s

        # Derivatives of Hermite basis functions (w.r.t. s)
        dh00 = 6*s2 - 6*s
        dh10 = 3*s2 - 4*s + 1
        dh01 = -6*s2 + 6*s
        dh11 = 3*s2 - 2*s

        # ds/dt = 1/dt, so dq/dt = (dq/ds) / dt
        dq_ds = dh00 * q0 + dh10 * v0 + dh01 * q1 + dh11 * v1
        return dq_ds / dt


def create_linear_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    start_time: float = 0.0,
) -> JointTrajectory:
    """Create a simple linear trajectory between two joint configurations.

    Args:
        start: Starting joint positions.
        end: Ending joint positions.
        duration: Time to travel (seconds).
        start_time: Start time offset (seconds).

    Returns:
        A JointTrajectory with two points (linear interpolation).
    """
    traj = JointTrajectory()
    traj.add_point(TrajectoryPoint(positions=start.copy(), time=start_time))
    traj.add_point(TrajectoryPoint(positions=end.copy(), time=start_time + duration))
    return traj


def create_smooth_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    start_time: float = 0.0,
) -> JointTrajectory:
    """Create a smooth (cubic) trajectory that starts and ends at rest.

    Uses zero velocities at both endpoints so the motion accelerates
    smoothly from rest and decelerates to rest.

    Args:
        start: Starting joint positions.
        end: Ending joint positions.
        duration: Time to travel (seconds).
        start_time: Start time offset (seconds).

    Returns:
        A JointTrajectory with cubic interpolation.
    """
    traj = JointTrajectory()
    zeros = np.zeros(NUM_JOINTS, dtype=np.float64)
    traj.add_point(TrajectoryPoint(
        positions=start.copy(), time=start_time, velocities=zeros.copy()
    ))
    traj.add_point(TrajectoryPoint(
        positions=end.copy(), time=start_time + duration, velocities=zeros.copy()
    ))
    return traj
