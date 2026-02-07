"""Tests for the control loop."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.control.controller import JointPositionController
from src.control.loop import ControlLoop
from src.interface.d1_connection import D1Command, D1Connection, D1State, NUM_JOINTS


def _make_mock_connection(connected=True):
    """Create a mock D1Connection."""
    conn = MagicMock(spec=D1Connection)
    conn.is_connected = connected
    conn.send_command.return_value = True
    conn.get_state.return_value = D1State(
        joint_positions=np.zeros(NUM_JOINTS),
        joint_velocities=np.zeros(NUM_JOINTS),
        joint_torques=np.zeros(NUM_JOINTS),
        gripper_position=0.0,
        timestamp=time.time(),
    )
    return conn


class TestControlLoop:
    def test_start_requires_connection(self):
        conn = _make_mock_connection(connected=False)
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl, frequency=100)
        with pytest.raises(RuntimeError, match="not established"):
            loop.start()

    def test_start_stop(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)

        loop.start()
        assert loop.is_running
        time.sleep(0.05)  # let a few cycles run
        loop.stop()
        assert not loop.is_running
        assert loop.cycle_count > 0

    def test_sends_commands(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        ctrl.set_target(np.ones(NUM_JOINTS) * 0.5)
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)

        loop.start()
        time.sleep(0.1)
        loop.stop()

        assert conn.send_command.call_count > 0

    def test_on_state_callback(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        states = []
        loop = ControlLoop(
            conn, ctrl, frequency=100, watchdog_timeout=0,
            on_state=lambda s: states.append(s),
        )

        loop.start()
        time.sleep(0.1)
        loop.stop()

        assert len(states) > 0

    def test_stop_sends_idle(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)

        loop.start()
        time.sleep(0.05)
        loop.stop()

        # Last call to send_command should be idle (mode=0)
        last_call = conn.send_command.call_args_list[-1]
        last_cmd = last_call[0][0]
        assert last_cmd.mode == 0

    def test_start_is_idempotent(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)
        loop.start()
        loop.start()  # should not crash
        loop.stop()

    def test_stop_is_idempotent(self):
        conn = _make_mock_connection()
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)
        loop.stop()  # not started — should not crash

    def test_safety_limits_applied(self):
        conn = _make_mock_connection()
        # Controller that produces huge torques
        ctrl = MagicMock()
        ctrl.compute.return_value = D1Command(
            mode=3, joint_torques=np.full(NUM_JOINTS, 999.0)
        )
        loop = ControlLoop(conn, ctrl, frequency=100, watchdog_timeout=0)

        loop.start()
        time.sleep(0.1)
        loop.stop()

        # Commands sent should have clamped torques
        for call in conn.send_command.call_args_list:
            cmd = call[0][0]
            if cmd.joint_torques is not None:
                assert np.all(np.abs(cmd.joint_torques) <= 50.0)
