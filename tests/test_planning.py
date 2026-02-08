"""
Comprehensive tests for the planning module.

Tests motion_planner, task_planner, and path_optimizer.
"""

import math
import sys
import os

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.planning.motion_planner import (
    MotionPlanner, Waypoint, Trajectory, TrajectoryPoint,
    NUM_ARM_JOINTS, JOINT_LIMITS_DEG, GRIPPER_MIN_MM, GRIPPER_MAX_MM,
)
from src.planning.task_planner import TaskPlanner, TaskResult, TaskStatus, HOME_POSE, READY_POSE
from src.planning.path_optimizer import PathOptimizer


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def planner():
    return MotionPlanner(dt=0.01)


@pytest.fixture
def task_planner():
    return TaskPlanner()


@pytest.fixture
def optimizer():
    return PathOptimizer()


@pytest.fixture
def home():
    return np.zeros(NUM_ARM_JOINTS)


@pytest.fixture
def pose_a():
    return np.array([30.0, -20.0, 15.0, 45.0, -10.0, 60.0])


@pytest.fixture
def pose_b():
    return np.array([-20.0, 40.0, -30.0, -60.0, 20.0, -45.0])


# ======================================================================
# Waypoint Tests
# ======================================================================

class TestWaypoint:
    def test_creation(self):
        wp = Waypoint(np.zeros(6))
        assert wp.joint_angles.shape == (6,)
        assert wp.gripper_mm == 0.0

    def test_gripper_clamping(self):
        wp = Waypoint(np.zeros(6), gripper_mm=100.0)
        assert wp.gripper_mm == GRIPPER_MAX_MM

        wp2 = Waypoint(np.zeros(6), gripper_mm=-10.0)
        assert wp2.gripper_mm == GRIPPER_MIN_MM

    def test_speed_factor_clamping(self):
        wp = Waypoint(np.zeros(6), max_speed_factor=2.0)
        assert wp.max_speed_factor == 1.0

        wp2 = Waypoint(np.zeros(6), max_speed_factor=0.0)
        assert wp2.max_speed_factor == 0.01

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            Waypoint(np.zeros(5))

    def test_label(self):
        wp = Waypoint(np.zeros(6), label="test")
        assert wp.label == "test"


# ======================================================================
# Trajectory Tests
# ======================================================================

class TestTrajectory:
    def test_empty(self):
        t = Trajectory()
        assert t.duration == 0.0
        assert t.num_points == 0

    def test_single_point(self):
        pt = TrajectoryPoint(0.0, np.zeros(6), np.zeros(6), np.zeros(6))
        t = Trajectory(points=[pt])
        assert t.duration == 0.0
        assert t.num_points == 1

    def test_positions_array(self):
        pts = [
            TrajectoryPoint(0.0, np.ones(6) * i, np.zeros(6), np.zeros(6))
            for i in range(5)
        ]
        t = Trajectory(points=pts)
        arr = t.positions_array()
        assert arr.shape == (5, 6)
        np.testing.assert_allclose(arr[3], np.ones(6) * 3)

    def test_times_array(self):
        pts = [
            TrajectoryPoint(i * 0.1, np.zeros(6), np.zeros(6), np.zeros(6))
            for i in range(10)
        ]
        t = Trajectory(points=pts)
        times = t.times_array()
        assert len(times) == 10
        assert abs(times[-1] - 0.9) < 1e-10


# ======================================================================
# MotionPlanner Tests
# ======================================================================

class TestMotionPlanner:
    def test_validate_joint_angles_valid(self, planner, home):
        assert planner.validate_joint_angles(home)
        assert planner.validate_joint_angles(np.array([130.0, 85.0, 85.0, 130.0, 85.0, 130.0]))

    def test_validate_joint_angles_invalid(self, planner):
        # J0 out of range
        assert not planner.validate_joint_angles(np.array([140.0, 0, 0, 0, 0, 0]))
        # J1 out of range
        assert not planner.validate_joint_angles(np.array([0, -95.0, 0, 0, 0, 0]))

    def test_clamp_joint_angles(self, planner):
        angles = np.array([200.0, -100.0, 50.0, -200.0, 0.0, 150.0])
        clamped = planner.clamp_joint_angles(angles)
        assert clamped[0] == 135.0
        assert clamped[1] == -90.0
        assert clamped[2] == 50.0
        assert clamped[3] == -135.0
        assert clamped[5] == 135.0

    def test_linear_trajectory_basic(self, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a)
        assert traj.num_points >= 2
        assert traj.duration > 0
        # Start and end positions correct
        np.testing.assert_allclose(traj.points[0].positions, home, atol=1e-6)
        np.testing.assert_allclose(traj.points[-1].positions, pose_a, atol=1e-6)

    def test_linear_trajectory_same_start_end(self, planner, home):
        traj = planner.linear_joint_trajectory(home, home)
        assert traj.num_points >= 1

    def test_linear_trajectory_gripper(self, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a, gripper_start=0.0, gripper_end=65.0)
        assert traj.points[0].gripper_mm == pytest.approx(0.0, abs=1e-6)
        assert traj.points[-1].gripper_mm == pytest.approx(65.0, abs=1e-6)

    def test_linear_trajectory_speed_factor(self, planner, home, pose_a):
        traj_fast = planner.linear_joint_trajectory(home, pose_a, speed_factor=1.0)
        traj_slow = planner.linear_joint_trajectory(home, pose_a, speed_factor=0.3)
        # Slower should take longer
        assert traj_slow.duration > traj_fast.duration

    def test_linear_trajectory_monotonic_time(self, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a)
        times = traj.times_array()
        assert np.all(np.diff(times) >= 0)

    def test_plan_waypoints(self, planner, home, pose_a, pose_b):
        wps = [
            Waypoint(home),
            Waypoint(pose_a, gripper_mm=30.0),
            Waypoint(pose_b, gripper_mm=60.0),
            Waypoint(home),
        ]
        traj = planner.plan_waypoints(wps)
        assert traj.num_points > 4
        assert traj.duration > 0
        np.testing.assert_allclose(traj.points[0].positions, home, atol=1e-6)
        np.testing.assert_allclose(traj.points[-1].positions, home, atol=1e-6)

    def test_plan_waypoints_too_few(self, planner, home):
        with pytest.raises(ValueError, match="at least 2"):
            planner.plan_waypoints([Waypoint(home)])

    def test_check_self_collision_home(self, planner, home):
        assert planner.check_self_collision(home)

    def test_plan_collision_free(self, planner, home, pose_a):
        traj = planner.plan_collision_free(home, pose_a)
        assert traj.num_points >= 2

    def test_enforce_limits_no_change(self, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a, speed_factor=0.3)
        enforced = planner.enforce_limits(traj)
        # Should not be slower than original (already slow)
        assert enforced.duration >= traj.duration - 1e-6

    def test_enforce_limits_respects_speed(self, planner):
        # Create a trajectory with excessive velocity
        pts = [
            TrajectoryPoint(0.0, np.zeros(6), np.zeros(6), np.zeros(6)),
            TrajectoryPoint(0.001, np.ones(6) * 50.0, np.ones(6) * 5000.0, np.zeros(6)),
        ]
        traj = Trajectory(points=pts)
        enforced = planner.enforce_limits(traj)
        # Should be slowed down
        assert enforced.duration >= traj.duration

    def test_cartesian_path(self, planner, home):
        """Test Cartesian path planning from home to a known pose."""
        target = np.eye(4)
        target[:3, 3] = [0.2, 0.0, 0.3]
        traj = planner.cartesian_linear_path(
            home, target, n_cartesian_steps=5, speed_factor=0.5,
        )
        assert traj.num_points >= 2
        assert traj.duration > 0


# ======================================================================
# TaskPlanner Tests
# ======================================================================

class TestTaskPlanner:
    def test_go_home(self, task_planner, pose_a):
        result = task_planner.go_home(pose_a)
        assert result.status == TaskStatus.SUCCESS
        assert result.trajectory.num_points >= 2
        np.testing.assert_allclose(
            result.trajectory.points[-1].positions, HOME_POSE, atol=1e-6,
        )

    def test_go_ready(self, task_planner, home):
        result = task_planner.go_ready(home)
        assert result.status == TaskStatus.SUCCESS
        np.testing.assert_allclose(
            result.trajectory.points[-1].positions, READY_POSE, atol=1e-6,
        )

    def test_pick_and_place(self, task_planner, home):
        pick = np.array([30.0, -30.0, 20.0, 60.0, 0.0, 0.0])
        place = np.array([-30.0, -30.0, -20.0, 60.0, 0.0, 0.0])
        result = task_planner.pick_and_place(home, pick, place)
        assert result.status == TaskStatus.SUCCESS
        assert result.trajectory.num_points > 10
        assert result.trajectory.duration > 0
        assert len(result.sub_trajectories) == 8

    def test_pick_and_place_with_retreat(self, task_planner, home):
        pick = np.array([20.0, -20.0, 10.0, 40.0, 0.0, 0.0])
        place = np.array([-20.0, -20.0, -10.0, 40.0, 0.0, 0.0])
        retreat = np.array([0.0, -45.0, 0.0, 45.0, 0.0, 0.0])
        result = task_planner.pick_and_place(home, pick, place, retreat_pose=retreat)
        assert result.status == TaskStatus.SUCCESS

    def test_pour(self, task_planner, home):
        pour_pos = np.array([0.0, -45.0, 0.0, 90.0, 0.0, 0.0])
        result = task_planner.pour(home, pour_pos, pour_angle=80.0)
        assert result.status == TaskStatus.SUCCESS
        assert result.trajectory.num_points > 5
        assert len(result.sub_trajectories) == 4

    def test_wave(self, task_planner, home):
        result = task_planner.wave(home, n_waves=2)
        assert result.status == TaskStatus.SUCCESS
        assert result.trajectory.num_points > 5
        assert "2 cycles" in result.message

    def test_custom_sequence(self, task_planner, home, pose_a, pose_b):
        wps = [
            Waypoint(home, gripper_mm=0.0),
            Waypoint(pose_a, gripper_mm=30.0),
            Waypoint(pose_b, gripper_mm=60.0),
        ]
        result = task_planner.custom_sequence(wps, label="my_task")
        assert result.status == TaskStatus.SUCCESS
        assert "my_task" in result.message

    def test_custom_sequence_too_few(self, task_planner, home):
        result = task_planner.custom_sequence([Waypoint(home)])
        assert result.status == TaskStatus.FAILED


# ======================================================================
# PathOptimizer Tests
# ======================================================================

class TestPathOptimizer:
    def test_smooth_preserves_endpoints(self, optimizer, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a)
        smoothed = optimizer.smooth(traj)
        np.testing.assert_allclose(
            smoothed.points[0].positions, home, atol=1.0,
        )
        np.testing.assert_allclose(
            smoothed.points[-1].positions, pose_a, atol=1.0,
        )

    def test_smooth_short_trajectory(self, optimizer):
        pts = [
            TrajectoryPoint(0.0, np.zeros(6), np.zeros(6), np.zeros(6)),
            TrajectoryPoint(0.1, np.ones(6), np.zeros(6), np.zeros(6)),
        ]
        traj = Trajectory(points=pts)
        smoothed = optimizer.smooth(traj)
        # Too few points to smooth, returns as-is
        assert smoothed.num_points == 2

    def test_smooth_continuous_velocity(self, optimizer, planner, home, pose_a, pose_b):
        wps = [Waypoint(home), Waypoint(pose_a), Waypoint(pose_b), Waypoint(home)]
        traj = planner.plan_waypoints(wps)
        smoothed = optimizer.smooth(traj, dt=0.02)
        # Velocities should be smooth (no huge jumps between adjacent points)
        vels = np.array([p.velocities for p in smoothed.points])
        diffs = np.diff(vels, axis=0)
        # Check no single velocity jump > 500 deg/s between adjacent samples
        assert np.all(np.abs(diffs) < 500.0)

    def test_time_optimal_faster(self, optimizer, planner, home, pose_a):
        # A slow trajectory should be made faster
        traj = planner.linear_joint_trajectory(home, pose_a, speed_factor=0.1)
        optimized = optimizer.time_optimal_parameterize(traj)
        # Time-optimal should be shorter or equal
        assert optimized.duration <= traj.duration + 1e-6

    def test_time_optimal_endpoints(self, optimizer, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a)
        optimized = optimizer.time_optimal_parameterize(traj)
        np.testing.assert_allclose(
            optimized.points[0].positions, home, atol=1e-6,
        )
        np.testing.assert_allclose(
            optimized.points[-1].positions, pose_a, atol=1e-6,
        )

    def test_time_optimal_single_point(self, optimizer):
        pt = TrajectoryPoint(0.0, np.zeros(6), np.zeros(6), np.zeros(6))
        traj = Trajectory(points=[pt])
        result = optimizer.time_optimal_parameterize(traj)
        assert result.num_points == 1

    def test_optimize_full_pipeline(self, optimizer, planner, home, pose_a, pose_b):
        wps = [Waypoint(home), Waypoint(pose_a), Waypoint(pose_b), Waypoint(home)]
        traj = planner.plan_waypoints(wps)
        optimized = optimizer.optimize(traj)
        assert optimized.num_points >= 2
        assert optimized.duration > 0

    def test_time_optimal_respects_speed_limits(self, optimizer, planner, home, pose_a):
        traj = planner.linear_joint_trajectory(home, pose_a)
        optimized = optimizer.time_optimal_parameterize(traj)
        for pt in optimized.points:
            for j in range(NUM_ARM_JOINTS):
                assert abs(pt.velocities[j]) <= optimizer.max_joint_speed[j] + 5.0  # small tolerance


# ======================================================================
# Integration Tests
# ======================================================================

class TestIntegration:
    def test_full_pick_and_place_optimized(self):
        mp = MotionPlanner()
        tp = TaskPlanner(mp)
        opt = PathOptimizer()

        home = np.zeros(6)
        pick = np.array([30.0, -30.0, 20.0, 60.0, 0.0, 0.0])
        place = np.array([-30.0, -30.0, -20.0, 60.0, 0.0, 0.0])

        result = tp.pick_and_place(home, pick, place)
        assert result.status == TaskStatus.SUCCESS

        optimized = opt.optimize(result.trajectory)
        assert optimized.num_points >= 2
        assert optimized.duration > 0

    def test_wave_then_home(self):
        mp = MotionPlanner()
        tp = TaskPlanner(mp)

        current = np.array([10.0, -10.0, 5.0, 20.0, 0.0, 10.0])
        wave_result = tp.wave(current, n_waves=2)
        assert wave_result.status == TaskStatus.SUCCESS

        final_pos = wave_result.trajectory.points[-1].positions
        home_result = tp.go_home(final_pos)
        assert home_result.status == TaskStatus.SUCCESS

    def test_collision_free_with_optimizer(self):
        mp = MotionPlanner()
        opt = PathOptimizer()

        home = np.zeros(6)
        target = np.array([45.0, -30.0, 30.0, 60.0, -20.0, 40.0])

        traj = mp.plan_collision_free(home, target)
        optimized = opt.optimize(traj)
        assert optimized.num_points >= 2

    def test_enforce_then_optimize(self):
        mp = MotionPlanner()
        opt = PathOptimizer()

        home = np.zeros(6)
        target = np.array([80.0, -60.0, 50.0, 100.0, -40.0, 80.0])

        traj = mp.linear_joint_trajectory(home, target, speed_factor=1.0)
        enforced = mp.enforce_limits(traj)
        optimized = opt.optimize(enforced)
        assert optimized.duration > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
