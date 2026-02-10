"""Tests for impedance/admittance controller."""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.impedance_controller import (
    AdmittanceController,
    ImpedanceParams,
    ComplianceMode,
    StiffnessScheduler,
    _STIFFNESS_PROFILES,
    _DAMPING_PROFILES,
)


class TestImpedanceParams:
    """Test impedance parameter configuration."""

    def test_from_stiff_mode(self):
        params = ImpedanceParams.from_mode(ComplianceMode.STIFF)
        assert params.stiffness.shape == (7,)
        assert params.damping.shape == (7,)
        assert np.all(params.stiffness > 0)

    def test_from_compliant_mode(self):
        params = ImpedanceParams.from_mode(ComplianceMode.COMPLIANT)
        stiff_params = ImpedanceParams.from_mode(ComplianceMode.STIFF)
        # Compliant should have lower stiffness
        assert np.all(params.stiffness < stiff_params.stiffness)

    def test_damping_ratio(self):
        params = ImpedanceParams.from_mode(ComplianceMode.MEDIUM)
        ratio = params.damping_ratio
        assert ratio.shape == (7,)
        # Damping ratios should be positive and reasonable
        assert np.all(ratio > 0)
        assert np.all(ratio < 5.0)

    def test_all_modes_defined(self):
        for mode in ComplianceMode:
            params = ImpedanceParams.from_mode(mode)
            assert params.stiffness.shape == (7,)
            assert params.damping.shape == (7,)


class TestAdmittanceController:
    """Test admittance-mode compliance controller."""

    @pytest.fixture
    def controller(self):
        return AdmittanceController(
            n_joints=7,
            mode=ComplianceMode.MEDIUM,
            dt=0.01,
        )

    def test_no_external_torque_no_compliance(self, controller):
        """Zero external torque should produce no compliance offset."""
        q_des = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0, 0.0])
        tau_ext = np.zeros(7)

        result = controller.compute_compliance(q_des, tau_ext)
        np.testing.assert_allclose(result, q_des, atol=1e-10)

    def test_external_torque_produces_compliance(self, controller):
        """External torque should cause position offset."""
        q_des = np.zeros(7)
        tau_ext = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Apply for several steps
        for _ in range(50):
            result = controller.compute_compliance(q_des, tau_ext)

        # Joint 0 should have moved in the direction of the torque
        offset = controller.compliance_offset
        assert offset[0] > 0.001, "Should yield to external torque"

    def test_compliance_bounded(self, controller):
        """Compliance offset should respect maximum bounds."""
        q_des = np.zeros(7)
        tau_ext = np.full(7, 100.0)  # very large torque

        for _ in range(10000):
            controller.compute_compliance(q_des, tau_ext)

        offset = controller.compliance_offset
        assert np.all(np.abs(offset) <= controller.max_compliance + 1e-6)

    def test_returns_to_desired_position(self, controller):
        """After removing external torque, should return near desired position."""
        q_des = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_ext = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Apply torque
        for _ in range(100):
            controller.compute_compliance(q_des, tau_ext)

        # Remove torque — should decay back
        for _ in range(1000):
            result = controller.compute_compliance(q_des, np.zeros(7))

        np.testing.assert_allclose(result, q_des, atol=0.02)

    def test_set_mode(self, controller):
        """Mode switching should change stiffness."""
        controller.set_mode(ComplianceMode.STIFF)
        stiff_k = controller.params.stiffness.copy()

        controller.set_mode(ComplianceMode.COMPLIANT)
        compliant_k = controller.params.stiffness.copy()

        assert np.all(stiff_k > compliant_k)

    def test_reset(self, controller):
        """Reset should clear compliance offset."""
        tau_ext = np.full(7, 5.0)
        for _ in range(50):
            controller.compute_compliance(np.zeros(7), tau_ext)

        assert np.max(np.abs(controller.compliance_offset)) > 0
        controller.reset()
        np.testing.assert_allclose(controller.compliance_offset, 0.0)

    def test_is_in_contact(self, controller):
        """Contact detection based on compliance offset magnitude."""
        assert not controller.is_in_contact

        tau_ext = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            controller.compute_compliance(np.zeros(7), tau_ext)

        assert controller.is_in_contact

    def test_estimate_external_torque(self, controller):
        measured = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1])
        gravity = np.array([4.0, 2.0, 1.5, 0.8, 0.3, 0.1, 0.0])
        ext = controller.estimate_external_torque(measured, gravity)
        expected = measured - gravity
        np.testing.assert_allclose(ext, expected)

    def test_compliant_mode_yields_more(self):
        """More compliant mode should yield more for same torque."""
        ctrl_stiff = AdmittanceController(n_joints=7, mode=ComplianceMode.STIFF, dt=0.01)
        ctrl_soft = AdmittanceController(n_joints=7, mode=ComplianceMode.COMPLIANT, dt=0.01)

        tau_ext = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(200):
            ctrl_stiff.compute_compliance(np.zeros(7), tau_ext)
            ctrl_soft.compute_compliance(np.zeros(7), tau_ext)

        assert abs(ctrl_soft.compliance_offset[0]) > abs(ctrl_stiff.compliance_offset[0])


class TestStiffnessScheduler:
    """Test smooth stiffness transitions."""

    @pytest.fixture
    def scheduler(self):
        return StiffnessScheduler(n_joints=7, transition_time=0.5, dt=0.01)

    def test_initial_not_transitioning(self, scheduler):
        assert not scheduler.is_transitioning

    def test_transition_starts(self, scheduler):
        scheduler.set_target_mode(ComplianceMode.STIFF)
        assert scheduler.is_transitioning

    def test_transition_completes(self, scheduler):
        scheduler.set_target_mode(ComplianceMode.STIFF)

        for _ in range(100):  # 100 * 0.01 = 1.0s > transition_time=0.5s
            params = scheduler.update()

        assert not scheduler.is_transitioning
        np.testing.assert_allclose(
            params.stiffness,
            _STIFFNESS_PROFILES[ComplianceMode.STIFF],
            atol=0.1,
        )

    def test_smooth_transition(self, scheduler):
        """Stiffness should change smoothly (no jumps)."""
        scheduler.set_target_mode(ComplianceMode.COMPLIANT)

        stiffness_values = []
        for _ in range(60):
            params = scheduler.update()
            stiffness_values.append(params.stiffness[0])

        # Check no large jumps
        for i in range(1, len(stiffness_values)):
            diff = abs(stiffness_values[i] - stiffness_values[i - 1])
            assert diff < 5.0, f"Stiffness jump of {diff} is too large"

    def test_custom_params_transition(self, scheduler):
        custom_k = np.full(7, 50.0)
        custom_d = np.full(7, 5.0)
        scheduler.set_target_params(custom_k, custom_d)

        for _ in range(100):
            params = scheduler.update()

        np.testing.assert_allclose(params.stiffness, custom_k, atol=0.1)
        np.testing.assert_allclose(params.damping, custom_d, atol=0.1)
