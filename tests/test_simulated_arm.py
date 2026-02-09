"""Tests for the SimulatedArm class."""

import time

import numpy as np
import pytest

from src.interface.simulated_arm import SimulatedArm


class TestSimulatedArmBasic:
    """Basic lifecycle and state tests."""

    def test_creation_defaults(self):
        arm = SimulatedArm()
        assert arm.is_connected is True
        angles = arm.get_joint_angles()
        assert angles is not None
        assert len(angles) == 7  # 6 arm + gripper
        np.testing.assert_allclose(angles, 0.0)

    def test_connect_disconnect(self):
        arm = SimulatedArm()
        assert arm.connect() is True
        assert arm.is_connected is True
        arm.disconnect()
        assert arm.is_connected is False

    def test_connect_with_interface(self):
        arm = SimulatedArm()
        assert arm.connect(interface_name="eth0", domain_id=42) is True


class TestSimulatedArmPowerEnable:
    """Power and enable state machine."""

    def test_power_on_off(self):
        arm = SimulatedArm()
        assert arm.get_status()["power_status"] == 0
        arm.power_on()
        assert arm.get_status()["power_status"] == 1
        arm.power_off()
        assert arm.get_status()["power_status"] == 0

    def test_enable_requires_power(self):
        arm = SimulatedArm()
        assert arm.enable_motors() is False
        assert arm.get_status()["enable_status"] == 0

    def test_enable_after_power(self):
        arm = SimulatedArm()
        arm.power_on()
        assert arm.enable_motors() is True
        assert arm.get_status()["enable_status"] == 1

    def test_power_off_disables(self):
        arm = SimulatedArm()
        arm.power_on()
        arm.enable_motors()
        arm.power_off()
        assert arm.get_status()["enable_status"] == 0
        assert arm.get_status()["power_status"] == 0

    def test_disable_motors(self):
        arm = SimulatedArm()
        arm.power_on()
        arm.enable_motors()
        arm.disable_motors()
        assert arm.get_status()["enable_status"] == 0
        assert arm.get_status()["power_status"] == 1


class TestSimulatedArmMotion:
    """Joint movement and interpolation."""

    def test_set_joint_moves_toward_target(self):
        arm = SimulatedArm()
        arm.set_joint(0, 90.0)
        # After several reads, should approach target
        for _ in range(50):
            angles = arm.get_joint_angles()
        assert abs(angles[0] - 90.0) < 1.0

    def test_set_all_joints(self):
        arm = SimulatedArm()
        target = [10.0, 20.0, -30.0, 40.0, -50.0, 60.0]
        arm.set_all_joints(target)
        for _ in range(100):
            angles = arm.get_joint_angles()
        for i in range(6):
            assert abs(angles[i] - target[i]) < 1.0

    def test_gripper(self):
        arm = SimulatedArm()
        arm.set_gripper(30.0)
        for _ in range(50):
            arm.get_joint_angles()  # triggers interpolation
        assert abs(arm.get_gripper_position() - 30.0) < 1.0

    def test_joint_limits_clamped(self):
        arm = SimulatedArm()
        # Joint 0 limit is [-135, 135]
        arm.set_joint(0, 999.0)
        for _ in range(100):
            angles = arm.get_joint_angles()
        assert angles[0] <= 136.0  # within tolerance of 135

    def test_gripper_limits_clamped(self):
        arm = SimulatedArm()
        arm.set_gripper(999.0)
        for _ in range(50):
            arm.get_joint_angles()
        assert arm.get_gripper_position() <= 65.5

    def test_invalid_joint_id(self):
        arm = SimulatedArm()
        assert arm.set_joint(99, 10.0) is False
        assert arm.set_joint(-1, 10.0) is False

    def test_reset_to_zero(self):
        arm = SimulatedArm()
        arm.set_joint(0, 45.0)
        arm.set_gripper(20.0)
        for _ in range(50):
            arm.get_joint_angles()
        arm.reset_to_zero()
        for _ in range(100):
            angles = arm.get_joint_angles()
        np.testing.assert_allclose(angles[:6], 0.0, atol=1.0)
        assert abs(arm.get_gripper_position()) < 1.0


class TestSimulatedArmState:
    """D1State and status interfaces."""

    def test_get_state(self):
        arm = SimulatedArm()
        state = arm.get_state()
        assert state is not None
        assert len(state.joint_positions) == 7
        assert state.gripper_position == 0.0
        assert state.timestamp > 0

    def test_get_reliable_state(self):
        arm = SimulatedArm()
        state = arm.get_reliable_state()
        assert state is not None

    def test_get_reliable_joint_angles(self):
        arm = SimulatedArm()
        angles = arm.get_reliable_joint_angles()
        assert angles is not None
        assert len(angles) == 7

    def test_feedback_always_fresh(self):
        arm = SimulatedArm()
        assert arm.is_feedback_fresh() is True
        assert arm.is_feedback_fresh(max_age_s=0.001) is True

    def test_joint_freshness(self):
        arm = SimulatedArm()
        freshness = arm.get_joint_freshness()
        assert len(freshness) == 7
        assert all(v == 0.0 for v in freshness.values())

    def test_feedback_health(self):
        arm = SimulatedArm()
        health = arm.get_feedback_health()
        assert health["zero_rate"] == 0.0
        assert health["stale"] is False

    def test_feedback_monitor(self):
        arm = SimulatedArm()
        mon = arm.feedback_monitor
        assert mon.is_feedback_fresh() is True


class TestSimulatedArmRawCommand:
    """send_command with raw dicts."""

    def test_single_joint_command(self):
        arm = SimulatedArm()
        assert arm.send_command({"funcode": 1, "data": {"id": 0, "angle": 45.0}}) is True

    def test_all_joints_command(self):
        arm = SimulatedArm()
        data = {"mode": 0}
        for i in range(6):
            data[f"angle{i}"] = float(i * 10)
        assert arm.send_command({"funcode": 2, "data": data}) is True

    def test_enable_disable_command(self):
        arm = SimulatedArm()
        arm.power_on()
        assert arm.send_command({"funcode": 5, "data": {"mode": 1}}) is True
        assert arm.get_status()["enable_status"] == 1

    def test_power_command(self):
        arm = SimulatedArm()
        assert arm.send_command({"funcode": 6, "data": {"power": 1}}) is True
        assert arm.get_status()["power_status"] == 1

    def test_reset_command(self):
        arm = SimulatedArm()
        assert arm.send_command({"funcode": 7}) is True


class TestSimulatedArmFeedbackLoop:
    """Background feedback thread."""

    def test_feedback_loop_interpolates(self):
        arm = SimulatedArm()
        arm.set_joint(0, 90.0)
        arm.start_feedback_loop(rate_hz=100.0)
        time.sleep(0.5)
        angles = arm.get_joint_angles()
        arm.disconnect()
        assert abs(angles[0] - 90.0) < 5.0

    def test_feedback_loop_stops_on_disconnect(self):
        arm = SimulatedArm()
        arm.start_feedback_loop(rate_hz=10.0)
        arm.disconnect()
        assert arm.is_connected is False


class TestSimulatedArmCorrelationId:
    """Ensure correlation_id kwarg doesn't break anything."""

    def test_power_on_with_cid(self):
        arm = SimulatedArm()
        assert arm.power_on(_correlation_id="test-123") is True

    def test_set_joint_with_cid(self):
        arm = SimulatedArm()
        assert arm.set_joint(0, 45.0, _correlation_id="test-456") is True

    def test_set_all_with_cid(self):
        arm = SimulatedArm()
        assert arm.set_all_joints([0.0] * 6, _correlation_id="test-789") is True
