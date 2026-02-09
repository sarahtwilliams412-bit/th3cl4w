"""
Tests for error recovery scenarios.

Covers: e-stop recovery, overcurrent handling, disable/enable cycle,
power off/on recovery, safety monitor state after errors.
"""

import numpy as np
import pytest

from src.safety.safety_monitor import SafetyMonitor, ViolationType
from src.interface.d1_connection import D1Command, D1State, NUM_JOINTS


def make_cmd(positions=None, mode=1):
    return D1Command(
        mode=mode,
        joint_positions=np.array(positions or [0.0] * NUM_JOINTS, dtype=np.float64),
    )


def make_state(**kwargs):
    return D1State(
        joint_positions=np.array(kwargs.get("positions", [0.0] * NUM_JOINTS), dtype=np.float64),
        joint_velocities=np.array(kwargs.get("velocities", [0.0] * NUM_JOINTS), dtype=np.float64),
        joint_torques=np.array(kwargs.get("torques", [0.0] * NUM_JOINTS), dtype=np.float64),
        gripper_position=kwargs.get("gripper", 0.0),
        timestamp=kwargs.get("timestamp", 0.0),
    )


class TestEstopRecovery:
    """Test that the system recovers correctly after e-stop."""

    def test_estop_then_reset_then_command(self):
        m = SafetyMonitor()
        m.trigger_estop("overcurrent")
        assert not m.validate_command(make_cmd()).is_safe
        m.reset_estop()
        assert m.validate_command(make_cmd()).is_safe

    def test_estop_clamp_returns_idle(self):
        m = SafetyMonitor()
        m.trigger_estop("collision")
        clamped = m.clamp_command(make_cmd(positions=[1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.5]))
        assert clamped.mode == 0
        np.testing.assert_array_equal(clamped.joint_positions, np.zeros(NUM_JOINTS))

    def test_estop_does_not_clear_on_validate(self):
        m = SafetyMonitor()
        m.trigger_estop("test")
        # Multiple validates should all fail
        for _ in range(10):
            assert not m.validate_command(make_cmd()).is_safe
        assert m.estop_active

    def test_state_check_after_estop_reset(self):
        m = SafetyMonitor()
        m.trigger_estop("test")
        m.reset_estop()
        # State check should work normally
        violations = m.check_state(make_state())
        pos_violations = [v for v in violations if v.violation_type == ViolationType.POSITION_LIMIT]
        assert len(pos_violations) == 0


class TestOvercurrentRecovery:
    """Simulates overcurrent → e-stop → reset cycle."""

    def test_overcurrent_triggers_estop_then_recover(self):
        m = SafetyMonitor()
        # Detect overcurrent via state check
        state = make_state(torques=[25.0, 0, 0, 0, 0, 0, 0])  # exceeds 20 Nm limit
        violations = m.check_state(state)
        torque_violations = [
            v for v in violations if v.violation_type == ViolationType.TORQUE_LIMIT
        ]
        assert len(torque_violations) > 0
        # Caller triggers e-stop
        m.trigger_estop("overcurrent detected")
        assert not m.validate_command(make_cmd()).is_safe
        # After fixing, reset
        m.reset_estop()
        assert m.validate_command(make_cmd()).is_safe


class TestRapidCommands:
    """Test monitor handles rapid sequential commands correctly."""

    def test_many_safe_commands(self):
        m = SafetyMonitor()
        for i in range(100):
            pos = [float(i % 3) * 0.1] * NUM_JOINTS
            pos[6] = min(abs(pos[6]), 1.0)
            result = m.validate_command(make_cmd(positions=pos))
            assert result.is_safe

    def test_alternating_safe_unsafe(self):
        m = SafetyMonitor()
        safe = make_cmd(positions=[0.0] * NUM_JOINTS)
        unsafe = make_cmd(positions=[100.0] * NUM_JOINTS)
        for _ in range(50):
            assert m.validate_command(safe).is_safe
            assert not m.validate_command(unsafe).is_safe
