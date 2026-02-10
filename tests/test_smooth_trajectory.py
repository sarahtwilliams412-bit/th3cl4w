"""Tests for the minimum-jerk trajectory generator and Fitts' Law duration."""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.smooth_trajectory import (
    minimum_jerk_scalar,
    minimum_jerk_waypoint,
    MinJerkProfile,
    fitts_law_duration,
    compute_movement_duration,
    generate_minimum_jerk_trajectory,
)


class TestMinimumJerkScalar:
    """Test the normalized minimum-jerk polynomial."""

    def test_start_position(self):
        """s(0) = 0."""
        p = minimum_jerk_scalar(0.0)
        assert p.s == pytest.approx(0.0, abs=1e-10)

    def test_end_position(self):
        """s(1) = 1."""
        p = minimum_jerk_scalar(1.0)
        assert p.s == pytest.approx(1.0, abs=1e-10)

    def test_midpoint_position(self):
        """s(0.5) = 0.5 (symmetry)."""
        p = minimum_jerk_scalar(0.5)
        assert p.s == pytest.approx(0.5, abs=1e-10)

    def test_zero_velocity_at_start(self):
        """s'(0) = 0."""
        p = minimum_jerk_scalar(0.0)
        assert p.ds == pytest.approx(0.0, abs=1e-10)

    def test_zero_velocity_at_end(self):
        """s'(1) = 0."""
        p = minimum_jerk_scalar(1.0)
        assert p.ds == pytest.approx(0.0, abs=1e-10)

    def test_zero_acceleration_at_start(self):
        """s''(0) = 0."""
        p = minimum_jerk_scalar(0.0)
        assert p.dds == pytest.approx(0.0, abs=1e-10)

    def test_zero_acceleration_at_end(self):
        """s''(1) = 0."""
        p = minimum_jerk_scalar(1.0)
        assert p.dds == pytest.approx(0.0, abs=1e-10)

    def test_peak_velocity_at_midpoint(self):
        """Velocity is maximum at tau=0.5 (bell-shaped profile)."""
        velocities = [minimum_jerk_scalar(t).ds for t in np.linspace(0, 1, 101)]
        peak_idx = np.argmax(velocities)
        # Peak should be at or very near midpoint (index 50)
        assert abs(peak_idx - 50) <= 1

    def test_bell_shaped_velocity(self):
        """Velocity profile is symmetric and bell-shaped."""
        n = 101
        taus = np.linspace(0, 1, n)
        velocities = np.array([minimum_jerk_scalar(t).ds for t in taus])

        # Check symmetry: v(tau) ≈ v(1-tau)
        for i in range(n // 2):
            assert velocities[i] == pytest.approx(velocities[n - 1 - i], abs=1e-8)

        # Check bell shape: velocity increases then decreases
        mid = n // 2
        for i in range(1, mid):
            assert velocities[i] >= velocities[i - 1] - 1e-10

    def test_monotonic_position(self):
        """Position monotonically increases from 0 to 1."""
        taus = np.linspace(0, 1, 100)
        positions = [minimum_jerk_scalar(t).s for t in taus]
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i - 1] - 1e-10

    def test_clamp_negative_tau(self):
        """Negative tau clamped to 0."""
        p = minimum_jerk_scalar(-0.5)
        assert p.s == pytest.approx(0.0, abs=1e-10)

    def test_clamp_tau_above_one(self):
        """tau > 1 clamped to 1."""
        p = minimum_jerk_scalar(1.5)
        assert p.s == pytest.approx(1.0, abs=1e-10)


class TestMinimumJerkWaypoint:
    """Test minimum-jerk interpolation between joint configurations."""

    def test_start_position(self):
        q0 = np.array([0.0, 1.0, 2.0])
        qf = np.array([1.0, 2.0, 3.0])
        v0 = np.zeros(3)
        vf = np.zeros(3)
        pos, vel, acc = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 0.0)
        np.testing.assert_allclose(pos, q0, atol=1e-10)

    def test_end_position(self):
        q0 = np.array([0.0, 1.0, 2.0])
        qf = np.array([1.0, 2.0, 3.0])
        v0 = np.zeros(3)
        vf = np.zeros(3)
        pos, vel, acc = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 1.0)
        np.testing.assert_allclose(pos, qf, atol=1e-10)

    def test_zero_velocity_at_endpoints(self):
        q0 = np.array([0.0, 0.0])
        qf = np.array([1.0, 1.0])
        v0 = np.zeros(2)
        vf = np.zeros(2)

        _, vel_start, _ = minimum_jerk_waypoint(q0, qf, v0, vf, 2.0, 0.0)
        _, vel_end, _ = minimum_jerk_waypoint(q0, qf, v0, vf, 2.0, 2.0)

        np.testing.assert_allclose(vel_start, 0.0, atol=1e-10)
        np.testing.assert_allclose(vel_end, 0.0, atol=1e-10)

    def test_zero_acceleration_at_endpoints(self):
        q0 = np.array([0.0])
        qf = np.array([1.0])
        v0 = np.zeros(1)
        vf = np.zeros(1)

        _, _, acc_start = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 0.0)
        _, _, acc_end = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 1.0)

        np.testing.assert_allclose(acc_start, 0.0, atol=1e-10)
        np.testing.assert_allclose(acc_end, 0.0, atol=1e-10)

    def test_midpoint(self):
        q0 = np.array([0.0])
        qf = np.array([2.0])
        v0 = np.zeros(1)
        vf = np.zeros(1)
        pos, _, _ = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 0.5)
        assert pos[0] == pytest.approx(1.0, abs=1e-10)

    def test_nonzero_initial_velocity(self):
        """When v0 != 0, should still reach qf with vf at t=T."""
        q0 = np.array([0.0])
        qf = np.array([1.0])
        v0 = np.array([0.5])
        vf = np.array([0.0])
        pos, vel, _ = minimum_jerk_waypoint(q0, qf, v0, vf, 1.0, 1.0)
        assert pos[0] == pytest.approx(1.0, abs=1e-8)
        assert vel[0] == pytest.approx(0.0, abs=1e-8)

    def test_zero_duration_returns_final(self):
        q0 = np.array([0.0, 1.0])
        qf = np.array([2.0, 3.0])
        v0 = np.zeros(2)
        vf = np.zeros(2)
        pos, vel, acc = minimum_jerk_waypoint(q0, qf, v0, vf, 0.0, 0.0)
        np.testing.assert_allclose(pos, qf)

    def test_multi_joint(self):
        """7-joint trajectory works correctly."""
        q0 = np.zeros(7)
        qf = np.ones(7) * np.pi / 4
        v0 = np.zeros(7)
        vf = np.zeros(7)
        pos, vel, acc = minimum_jerk_waypoint(q0, qf, v0, vf, 2.0, 1.0)
        # At midpoint, should be halfway
        np.testing.assert_allclose(pos, qf / 2, atol=1e-10)


class TestFittsLaw:
    """Test Fitts' Law duration estimation."""

    def test_zero_distance(self):
        """Zero distance should give minimum duration."""
        d = fitts_law_duration(0.0)
        assert d == pytest.approx(0.3, abs=0.01)  # min duration

    def test_short_distance(self):
        """Short distance (~10cm in rad) should give ~0.3-0.6s."""
        d = fitts_law_duration(0.15)  # ~8.6 degrees
        assert 0.3 <= d <= 0.8

    def test_long_distance(self):
        """Full workspace reach should give ~0.8-1.5s."""
        d = fitts_law_duration(2.0)  # ~115 degrees
        assert 0.5 <= d <= 3.0

    def test_increases_with_distance(self):
        """Duration increases with distance."""
        d1 = fitts_law_duration(0.1)
        d2 = fitts_law_duration(0.5)
        d3 = fitts_law_duration(2.0)
        assert d1 <= d2 <= d3

    def test_increases_with_precision(self):
        """Smaller target width (higher precision) → longer duration."""
        d_coarse = fitts_law_duration(1.0, target_width=0.1)
        d_fine = fitts_law_duration(1.0, target_width=0.01)
        assert d_fine > d_coarse

    def test_clamped_to_max(self):
        """Very large distances should clamp to max duration."""
        d = fitts_law_duration(1000.0)
        assert d <= 3.0

    def test_compute_movement_duration(self):
        """compute_movement_duration wraps fitts_law_duration correctly."""
        q_start = np.zeros(7)
        q_end = np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.05, 0.0])
        d = compute_movement_duration(q_start, q_end)
        assert 0.3 <= d <= 3.0

    def test_speed_factor(self):
        """speed_factor > 1 should reduce duration."""
        q_start = np.zeros(7)
        q_end = np.ones(7) * 0.5
        d_normal = compute_movement_duration(q_start, q_end, speed_factor=1.0)
        d_fast = compute_movement_duration(q_start, q_end, speed_factor=2.0)
        assert d_fast < d_normal


class TestGenerateTrajectory:
    """Test full trajectory generation."""

    def test_correct_shape(self):
        q0 = np.zeros(3)
        qf = np.ones(3)
        times, pos, vel, acc = generate_minimum_jerk_trajectory(q0, qf, 1.0, dt=0.01)
        assert pos.shape[1] == 3
        assert vel.shape[1] == 3
        assert acc.shape[1] == 3
        assert len(times) == pos.shape[0]

    def test_start_end_match(self):
        q0 = np.array([1.0, 2.0])
        qf = np.array([3.0, 4.0])
        times, pos, vel, acc = generate_minimum_jerk_trajectory(q0, qf, 2.0)
        np.testing.assert_allclose(pos[0], q0, atol=1e-10)
        np.testing.assert_allclose(pos[-1], qf, atol=1e-10)

    def test_zero_velocity_endpoints(self):
        q0 = np.zeros(4)
        qf = np.ones(4)
        times, pos, vel, acc = generate_minimum_jerk_trajectory(q0, qf, 1.0)
        np.testing.assert_allclose(vel[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(vel[-1], 0.0, atol=1e-10)

    def test_adequate_points(self):
        q0 = np.zeros(2)
        qf = np.ones(2)
        times, pos, vel, acc = generate_minimum_jerk_trajectory(q0, qf, 1.0, dt=0.01)
        assert len(times) >= 100

    def test_times_monotonic(self):
        q0 = np.zeros(3)
        qf = np.ones(3)
        times, _, _, _ = generate_minimum_jerk_trajectory(q0, qf, 2.0)
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_velocity_bell_shaped(self):
        """Peak velocity should be near the midpoint for single-joint move."""
        q0 = np.array([0.0])
        qf = np.array([1.0])
        times, pos, vel, acc = generate_minimum_jerk_trajectory(q0, qf, 1.0, dt=0.001)

        # Find peak velocity index
        vel_magnitudes = np.abs(vel[:, 0])
        peak_idx = np.argmax(vel_magnitudes)
        mid_idx = len(times) // 2

        # Peak should be within 5% of midpoint
        assert abs(peak_idx - mid_idx) / len(times) < 0.05
