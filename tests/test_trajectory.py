"""Tests for trajectory planning."""

import numpy as np
import pytest

from src.interface.d1_connection import NUM_JOINTS
from src.planning.trajectory import (
    JointTrajectory,
    TrajectoryPoint,
    create_linear_trajectory,
    create_smooth_trajectory,
)


class TestTrajectoryPoint:
    def test_creation(self):
        pt = TrajectoryPoint(positions=np.zeros(NUM_JOINTS), time=0.0)
        assert pt.time == 0.0
        assert pt.velocities is None


class TestJointTrajectory:
    def test_empty_trajectory(self):
        traj = JointTrajectory()
        assert traj.is_empty()
        assert traj.duration == 0.0

    def test_single_point(self):
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=np.zeros(NUM_JOINTS), time=0.0))
        assert traj.is_empty()  # need at least 2 points
        result = traj.sample(0.0)
        np.testing.assert_array_equal(result, np.zeros(NUM_JOINTS))

    def test_wrong_shape_raises(self):
        traj = JointTrajectory()
        with pytest.raises(ValueError, match="shape"):
            traj.add_point(TrajectoryPoint(positions=np.zeros(3), time=0.0))

    def test_non_increasing_time_raises(self):
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=np.zeros(NUM_JOINTS), time=1.0))
        with pytest.raises(ValueError, match="increasing"):
            traj.add_point(TrajectoryPoint(positions=np.zeros(NUM_JOINTS), time=0.5))

    def test_linear_interpolation_midpoint(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0))

        mid = traj.sample(0.5)
        np.testing.assert_array_almost_equal(mid, np.full(NUM_JOINTS, 0.5))

    def test_linear_interpolation_endpoints(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS) * 2.0
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0))

        np.testing.assert_array_almost_equal(traj.sample(0.0), start)
        np.testing.assert_array_almost_equal(traj.sample(1.0), end)

    def test_clamps_before_start(self):
        start = np.ones(NUM_JOINTS)
        end = np.ones(NUM_JOINTS) * 2.0
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=1.0))
        traj.add_point(TrajectoryPoint(positions=end, time=2.0))

        np.testing.assert_array_almost_equal(traj.sample(0.0), start)

    def test_clamps_after_end(self):
        start = np.ones(NUM_JOINTS)
        end = np.ones(NUM_JOINTS) * 2.0
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0))

        np.testing.assert_array_almost_equal(traj.sample(5.0), end)

    def test_three_waypoints(self):
        p0 = np.zeros(NUM_JOINTS)
        p1 = np.ones(NUM_JOINTS)
        p2 = np.ones(NUM_JOINTS) * 3.0

        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=p0, time=0.0))
        traj.add_point(TrajectoryPoint(positions=p1, time=1.0))
        traj.add_point(TrajectoryPoint(positions=p2, time=2.0))

        assert traj.duration == 2.0
        np.testing.assert_array_almost_equal(traj.sample(0.5), np.full(NUM_JOINTS, 0.5))
        np.testing.assert_array_almost_equal(traj.sample(1.0), p1)
        np.testing.assert_array_almost_equal(traj.sample(1.5), np.full(NUM_JOINTS, 2.0))

    def test_duration(self):
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=np.zeros(NUM_JOINTS), time=1.0))
        traj.add_point(TrajectoryPoint(positions=np.ones(NUM_JOINTS), time=3.5))
        assert traj.duration == 2.5

    def test_sample_velocity_linear(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS) * 2.0
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0))

        vel = traj.sample_velocity(0.5)
        # Constant velocity = (end - start) / duration = 2.0
        np.testing.assert_array_almost_equal(vel, np.full(NUM_JOINTS, 2.0))

    def test_sample_velocity_at_boundaries(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0))

        # Velocity at or beyond boundaries should be zero
        np.testing.assert_array_equal(
            traj.sample_velocity(-1.0), np.zeros(NUM_JOINTS)
        )
        np.testing.assert_array_equal(
            traj.sample_velocity(2.0), np.zeros(NUM_JOINTS)
        )

    def test_empty_trajectory_sample_raises(self):
        traj = JointTrajectory()
        with pytest.raises(ValueError, match="no points"):
            traj.sample(0.0)


class TestCubicInterpolation:
    def test_smooth_starts_and_ends_at_rest(self):
        """Cubic trajectory with zero endpoint velocities should start/end at rest."""
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        zeros = np.zeros(NUM_JOINTS)

        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0, velocities=zeros))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0, velocities=zeros))

        # At t=0, velocity should be 0
        v0 = traj.sample_velocity(0.001)
        assert np.all(np.abs(v0) < 0.1)  # near zero at start

    def test_cubic_hits_endpoints(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS) * 2.0
        zeros = np.zeros(NUM_JOINTS)

        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0, velocities=zeros))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0, velocities=zeros))

        np.testing.assert_array_almost_equal(traj.sample(0.0), start)
        np.testing.assert_array_almost_equal(traj.sample(1.0), end)

    def test_cubic_midpoint_reasonable(self):
        """Midpoint of cubic with zero-velocity endpoints should be near linear midpoint."""
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        zeros = np.zeros(NUM_JOINTS)

        traj = JointTrajectory()
        traj.add_point(TrajectoryPoint(positions=start, time=0.0, velocities=zeros))
        traj.add_point(TrajectoryPoint(positions=end, time=1.0, velocities=zeros))

        mid = traj.sample(0.5)
        # Cubic Hermite with zero boundary velocities: at s=0.5, h00=0.5, h01=0.5
        np.testing.assert_array_almost_equal(mid, np.full(NUM_JOINTS, 0.5))


class TestConvenienceFunctions:
    def test_create_linear_trajectory(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        traj = create_linear_trajectory(start, end, duration=2.0)

        assert not traj.is_empty()
        assert traj.duration == 2.0
        np.testing.assert_array_almost_equal(traj.sample(1.0), np.full(NUM_JOINTS, 0.5))

    def test_create_smooth_trajectory(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        traj = create_smooth_trajectory(start, end, duration=2.0)

        assert not traj.is_empty()
        assert traj.duration == 2.0
        np.testing.assert_array_almost_equal(traj.sample(0.0), start)
        np.testing.assert_array_almost_equal(traj.sample(2.0), end)

    def test_create_linear_with_offset(self):
        start = np.zeros(NUM_JOINTS)
        end = np.ones(NUM_JOINTS)
        traj = create_linear_trajectory(start, end, duration=1.0, start_time=5.0)

        assert traj.start_time == 5.0
        assert traj.end_time == 6.0
        np.testing.assert_array_almost_equal(traj.sample(5.5), np.full(NUM_JOINTS, 0.5))
