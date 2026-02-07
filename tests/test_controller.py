"""Tests for controllers."""

import numpy as np
import pytest

from src.control.controller import Controller, JointPositionController
from src.interface.d1_connection import D1Command, D1State, NUM_JOINTS


def _make_state(positions=None, velocities=None, torques=None):
    """Helper to create a D1State with default values."""
    return D1State(
        joint_positions=positions if positions is not None else np.zeros(NUM_JOINTS),
        joint_velocities=velocities if velocities is not None else np.zeros(NUM_JOINTS),
        joint_torques=torques if torques is not None else np.zeros(NUM_JOINTS),
        gripper_position=0.0,
        timestamp=0.0,
    )


class TestJointPositionController:
    def test_idle_when_no_target(self):
        ctrl = JointPositionController()
        state = _make_state()
        cmd = ctrl.compute(state)
        assert cmd.mode == 0  # idle

    def test_computes_torque_command(self):
        ctrl = JointPositionController()
        target = np.ones(NUM_JOINTS) * 0.5
        ctrl.set_target(target)

        state = _make_state()  # at zero
        cmd = ctrl.compute(state)
        assert cmd.mode == 3  # torque mode
        assert cmd.joint_torques is not None
        # Torques should be positive (pushing toward target)
        assert np.all(cmd.joint_torques > 0)

    def test_zero_torque_at_target(self):
        ctrl = JointPositionController()
        target = np.zeros(NUM_JOINTS)
        ctrl.set_target(target)

        state = _make_state(positions=np.zeros(NUM_JOINTS))
        cmd = ctrl.compute(state)
        np.testing.assert_array_almost_equal(cmd.joint_torques, np.zeros(NUM_JOINTS))

    def test_damping_opposes_velocity(self):
        ctrl = JointPositionController()
        target = np.zeros(NUM_JOINTS)
        ctrl.set_target(target)

        # At target position but with positive velocity -> torque should be negative (damping)
        state = _make_state(
            positions=np.zeros(NUM_JOINTS),
            velocities=np.ones(NUM_JOINTS),
        )
        cmd = ctrl.compute(state)
        assert np.all(cmd.joint_torques < 0)

    def test_set_target_wrong_shape(self):
        ctrl = JointPositionController()
        with pytest.raises(ValueError, match="shape"):
            ctrl.set_target(np.zeros(3))

    def test_custom_gains(self):
        kp = np.ones(NUM_JOINTS) * 10.0
        kd = np.ones(NUM_JOINTS) * 1.0
        ctrl = JointPositionController(kp=kp, kd=kd)
        assert np.array_equal(ctrl.kp, kp)
        assert np.array_equal(ctrl.kd, kd)

    def test_invalid_gains_shape(self):
        with pytest.raises(ValueError, match="shape"):
            JointPositionController(kp=np.ones(3))

    def test_reset(self):
        ctrl = JointPositionController()
        ctrl.set_target(np.zeros(NUM_JOINTS))
        assert ctrl.target_positions is not None
        ctrl.reset()
        assert ctrl.target_positions is None

    def test_gripper_target(self):
        ctrl = JointPositionController()
        ctrl.set_target(np.zeros(NUM_JOINTS), gripper=0.8)
        state = _make_state()
        cmd = ctrl.compute(state)
        assert cmd.gripper_position == 0.8
