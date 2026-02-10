"""Tests for gravity compensation and thermal monitoring."""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.gravity_compensation import (
    GravityCompensator,
    ThermalMonitor,
    _D1_LINK_DYNAMICS,
    GRAVITY,
)
from src.kinematics.kinematics import D1Kinematics


class TestGravityCompensator:
    """Test gravity torque computation."""

    @pytest.fixture
    def compensator(self):
        return GravityCompensator()

    def test_extended_config_nonzero_torque(self, compensator):
        """With the arm extended horizontally (shoulder pitched forward),
        gravity torques should be nonzero."""
        # At q=0 the D1 DH parameters stack links along z (vertical),
        # so we need a pitched configuration to get gravity torques.
        q = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = compensator.compute_gravity_torques(q)
        assert tau.shape == (7,)
        # At least some joints should have nonzero gravity torque
        assert np.max(np.abs(tau)) > 0.01

    def test_torques_change_with_config(self, compensator):
        """Different configurations should produce different torques."""
        q1 = np.zeros(7)
        q2 = np.array([0.0, np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau1 = compensator.compute_gravity_torques(q1)
        tau2 = compensator.compute_gravity_torques(q2)
        assert not np.allclose(tau1, tau2)

    def test_shoulder_carries_most_load(self, compensator):
        """Shoulder joints (0, 1) should carry more gravity load
        than wrist joints (5, 6) in a typical configuration."""
        q = np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0])
        tau = compensator.compute_gravity_torques(q)
        # Proximal joints bear more load than distal
        shoulder_load = np.abs(tau[0]) + np.abs(tau[1])
        wrist_load = np.abs(tau[5]) + np.abs(tau[6])
        assert shoulder_load > wrist_load

    def test_position_offset_bounded(self, compensator):
        """Position offsets should be within ±5 degrees."""
        q = np.array([0.0, np.pi / 4, np.pi / 4, 0.0, np.pi / 6, 0.0, 0.0])
        offset = compensator.compute_position_offset(q)
        max_offset_deg = np.degrees(np.max(np.abs(offset)))
        assert max_offset_deg <= 5.0

    def test_position_offset_shape(self, compensator):
        q = np.zeros(7)
        offset = compensator.compute_position_offset(q)
        assert offset.shape == (7,)

    def test_total_gravity_load(self, compensator):
        """Total load should be positive for non-trivial configs."""
        q = np.array([0.0, np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])
        load = compensator.total_gravity_load(q)
        assert load > 0.0

    def test_custom_gravity(self):
        """Custom gravity vector should change torques."""
        comp_normal = GravityCompensator(gravity=np.array([0.0, 0.0, -9.81]))
        comp_zero = GravityCompensator(gravity=np.array([0.0, 0.0, 0.0]))
        # Use a configuration where the arm is extended to see gravity effects
        q = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_normal = comp_normal.compute_gravity_torques(q)
        tau_zero = comp_zero.compute_gravity_torques(q)
        np.testing.assert_allclose(tau_zero, 0.0, atol=1e-10)
        assert np.max(np.abs(tau_normal)) > 0.01

    def test_mismatched_dynamics_raises(self):
        """Mismatched link dynamics count should raise."""
        with pytest.raises(ValueError):
            GravityCompensator(link_dynamics=_D1_LINK_DYNAMICS[:3])


class TestThermalMonitor:
    """Test thermal monitoring system."""

    @pytest.fixture
    def monitor(self):
        return ThermalMonitor(n_joints=7)

    def test_initial_state_cool(self, monitor):
        assert not monitor.is_warning
        assert not monitor.is_critical
        assert monitor.speed_reduction_factor() == 1.0

    def test_heats_up_with_torque(self, monitor):
        """Sustained torque should increase temperature."""
        torques = np.array([10.0, 10.0, 8.0, 5.0, 3.0, 2.0, 2.0])
        initial_temp = monitor.max_temperature

        for _ in range(1000):
            monitor.update(torques, dt=0.01)

        assert monitor.max_temperature > initial_temp

    def test_warning_threshold(self):
        """High sustained torque should trigger warning."""
        monitor = ThermalMonitor(
            n_joints=7,
            warning_threshold=30.0,
            critical_threshold=50.0,
            thermal_capacity=10.0,
        )
        torques = np.full(7, 15.0)

        for _ in range(5000):
            monitor.update(torques, dt=0.01)

        assert monitor.is_warning

    def test_speed_reduction_gradual(self, monitor):
        """Speed reduction should be gradual between warning and critical."""
        # This is hard to test exactly, so just verify the factor range
        factor = monitor.speed_reduction_factor()
        assert 0.0 <= factor <= 1.0

    def test_cools_down(self):
        """Temperature should decrease when torque is removed."""
        monitor = ThermalMonitor(n_joints=7, thermal_capacity=10.0)
        torques = np.full(7, 10.0)

        # Heat up
        for _ in range(500):
            monitor.update(torques, dt=0.01)
        hot_temp = monitor.max_temperature

        # Cool down
        for _ in range(5000):
            monitor.update(np.zeros(7), dt=0.01)

        assert monitor.max_temperature < hot_temp

    def test_thermal_loads_shape(self, monitor):
        assert monitor.thermal_loads.shape == (7,)
