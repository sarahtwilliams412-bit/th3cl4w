"""Tests for JointController with mocked D1Connection."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.interface.d1_connection import D1Connection, D1State, D1Command, NUM_JOINTS
from src.control.joint_controller import JointController, PIDGains, cubic_interpolation


@pytest.fixture
def mock_connection():
    conn = MagicMock(spec=D1Connection)
    conn.send_command.return_value = True
    conn.get_state.return_value = D1State(
        joint_positions=np.zeros(NUM_JOINTS),
        joint_velocities=np.zeros(NUM_JOINTS),
        joint_torques=np.zeros(NUM_JOINTS),
        gripper_position=0.0,
        timestamp=time.time(),
    )
    return conn


@pytest.fixture
def controller(mock_connection):
    return JointController(mock_connection)


# --- PIDGains ---


class TestPIDGains:
    def test_defaults(self):
        g = PIDGains()
        assert g.kp.shape == (NUM_JOINTS,)
        assert g.ki.shape == (NUM_JOINTS,)
        assert g.kd.shape == (NUM_JOINTS,)
        np.testing.assert_array_equal(g.kp, 100.0)
        np.testing.assert_array_equal(g.ki, 0.0)
        np.testing.assert_array_equal(g.kd, 10.0)

    def test_custom_gains(self):
        kp = np.ones(NUM_JOINTS) * 50
        g = PIDGains(kp=kp)
        np.testing.assert_array_equal(g.kp, 50.0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            PIDGains(kp=np.ones(3))


# --- cubic_interpolation ---


class TestCubicInterpolation:
    def test_boundary_start(self):
        q0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        qf = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        v0 = np.zeros(NUM_JOINTS)
        vf = np.zeros(NUM_JOINTS)
        pos, vel = cubic_interpolation(q0, qf, v0, vf, 2.0, 0.0)
        np.testing.assert_allclose(pos, q0)
        np.testing.assert_allclose(vel, v0)

    def test_boundary_end(self):
        q0 = np.zeros(NUM_JOINTS)
        qf = np.ones(NUM_JOINTS)
        v0 = np.zeros(NUM_JOINTS)
        vf = np.zeros(NUM_JOINTS)
        pos, vel = cubic_interpolation(q0, qf, v0, vf, 1.0, 1.0)
        np.testing.assert_allclose(pos, qf, atol=1e-10)
        np.testing.assert_allclose(vel, vf, atol=1e-10)

    def test_midpoint_between(self):
        q0 = np.zeros(NUM_JOINTS)
        qf = np.ones(NUM_JOINTS) * 2.0
        v0 = np.zeros(NUM_JOINTS)
        vf = np.zeros(NUM_JOINTS)
        pos, _ = cubic_interpolation(q0, qf, v0, vf, 2.0, 1.0)
        # At midpoint with zero velocities, should be at midpoint value
        np.testing.assert_allclose(pos, 1.0, atol=1e-10)

    def test_zero_duration(self):
        q0 = np.zeros(NUM_JOINTS)
        qf = np.ones(NUM_JOINTS)
        pos, _ = cubic_interpolation(q0, qf, np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), 0.0, 0.0)
        np.testing.assert_allclose(pos, qf)


# --- JointController ---


class TestJointController:
    def test_get_state(self, controller, mock_connection):
        state = controller.get_state()
        assert isinstance(state, D1State)
        mock_connection.get_state.assert_called_once()

    def test_get_state_raises_on_none(self, mock_connection):
        mock_connection.get_state.return_value = None
        ctrl = JointController(mock_connection)
        with pytest.raises(RuntimeError):
            ctrl.get_state()

    def test_stop(self, controller, mock_connection):
        controller.stop()
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.mode == 0

    def test_set_velocity(self, controller, mock_connection):
        vels = np.ones(NUM_JOINTS) * 0.5
        controller.set_velocity(vels)
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.mode == 2
        np.testing.assert_array_equal(cmd.joint_velocities, vels)

    def test_set_velocity_wrong_shape(self, controller):
        with pytest.raises(ValueError):
            controller.set_velocity(np.ones(3))

    def test_set_torque(self, controller, mock_connection):
        torques = np.ones(NUM_JOINTS) * 1.5
        controller.set_torque(torques)
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.mode == 3
        np.testing.assert_array_equal(cmd.joint_torques, torques)

    def test_set_torque_wrong_shape(self, controller):
        with pytest.raises(ValueError):
            controller.set_torque(np.ones(5))

    def test_set_gripper_clamps(self, controller, mock_connection):
        controller.set_gripper(1.5)
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.gripper_position == 1.0

        controller.set_gripper(-0.5)
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.gripper_position == 0.0

    def test_set_gripper_normal(self, controller, mock_connection):
        controller.set_gripper(0.5)
        cmd = mock_connection.send_command.call_args[0][0]
        assert cmd.gripper_position == 0.5

    @patch("src.control.joint_controller.time")
    def test_move_to_position(self, mock_time, mock_connection):
        # Simulate time progressing past duration immediately
        call_count = [0]

        def monotonic_side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                return 0.0
            return 10.0  # past duration

        mock_time.monotonic.side_effect = monotonic_side_effect
        mock_time.sleep = MagicMock()

        ctrl = JointController(mock_connection)
        target = np.ones(NUM_JOINTS)
        ctrl.move_to_position(target, duration=1.0)

        # Should have sent at least the final command
        assert mock_connection.send_command.called
        last_cmd = mock_connection.send_command.call_args[0][0]
        assert last_cmd.mode == 1
        np.testing.assert_allclose(last_cmd.joint_positions, target)

    def test_move_to_position_bad_duration(self, controller):
        with pytest.raises(ValueError):
            controller.move_to_position(np.zeros(NUM_JOINTS), duration=-1.0)

    def test_move_to_position_wrong_shape(self, controller):
        with pytest.raises(ValueError):
            controller.move_to_position(np.zeros(3), duration=1.0)

    @patch("src.control.joint_controller.time")
    def test_go_home(self, mock_time, mock_connection):
        call_count = [0]

        def monotonic_side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                return 0.0
            return 10.0

        mock_time.monotonic.side_effect = monotonic_side_effect
        mock_time.sleep = MagicMock()

        ctrl = JointController(mock_connection)
        ctrl.go_home()

        last_cmd = mock_connection.send_command.call_args[0][0]
        assert last_cmd.mode == 1
        np.testing.assert_allclose(last_cmd.joint_positions, 0.0)

    def test_custom_gains(self, mock_connection):
        gains = PIDGains(kp=np.full(NUM_JOINTS, 200.0))
        ctrl = JointController(mock_connection, gains=gains)
        np.testing.assert_array_equal(ctrl.gains.kp, 200.0)
