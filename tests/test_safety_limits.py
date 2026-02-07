"""Tests for safety limits and command validation."""

import numpy as np
import pytest

from src.interface.d1_connection import D1Command, NUM_JOINTS
from src.safety.limits import (
    D1SafetyLimits,
    SafetyViolation,
    clamp_command,
    validate_command,
)


class TestD1SafetyLimits:
    def test_default_limits(self):
        limits = D1SafetyLimits()
        assert limits.joint_position_min.shape == (NUM_JOINTS,)
        assert limits.joint_position_max.shape == (NUM_JOINTS,)
        assert limits.joint_velocity_max.shape == (NUM_JOINTS,)
        assert limits.joint_torque_max.shape == (NUM_JOINTS,)
        assert limits.gripper_min == 0.0
        assert limits.gripper_max == 1.0

    def test_check_positions_valid(self):
        limits = D1SafetyLimits()
        assert limits.check_positions(np.zeros(NUM_JOINTS)) is None

    def test_check_positions_out_of_range(self):
        limits = D1SafetyLimits()
        positions = np.zeros(NUM_JOINTS)
        positions[0] = 5.0  # way beyond limit
        err = limits.check_positions(positions)
        assert err is not None
        assert "joint 0" in err

    def test_check_positions_wrong_shape(self):
        limits = D1SafetyLimits()
        err = limits.check_positions(np.zeros(3))
        assert err is not None
        assert "shape" in err

    def test_check_velocities_valid(self):
        limits = D1SafetyLimits()
        assert limits.check_velocities(np.zeros(NUM_JOINTS)) is None

    def test_check_velocities_exceeded(self):
        limits = D1SafetyLimits()
        vels = np.zeros(NUM_JOINTS)
        vels[2] = 100.0
        err = limits.check_velocities(vels)
        assert err is not None
        assert "joint 2" in err

    def test_check_torques_valid(self):
        limits = D1SafetyLimits()
        assert limits.check_torques(np.zeros(NUM_JOINTS)) is None

    def test_check_torques_exceeded(self):
        limits = D1SafetyLimits()
        torques = np.zeros(NUM_JOINTS)
        torques[0] = -100.0
        err = limits.check_torques(torques)
        assert err is not None
        assert "joint 0" in err

    def test_clamp_positions(self):
        limits = D1SafetyLimits()
        positions = np.full(NUM_JOINTS, 10.0)
        clamped = limits.clamp_positions(positions)
        np.testing.assert_array_less(clamped, positions)
        for i in range(NUM_JOINTS):
            assert clamped[i] <= limits.joint_position_max[i]

    def test_clamp_velocities(self):
        limits = D1SafetyLimits()
        velocities = np.full(NUM_JOINTS, 50.0)
        clamped = limits.clamp_velocities(velocities)
        for i in range(NUM_JOINTS):
            assert abs(clamped[i]) <= limits.joint_velocity_max[i]

    def test_clamp_torques(self):
        limits = D1SafetyLimits()
        torques = np.full(NUM_JOINTS, -200.0)
        clamped = limits.clamp_torques(torques)
        for i in range(NUM_JOINTS):
            assert abs(clamped[i]) <= limits.joint_torque_max[i]

    def test_clamp_gripper(self):
        limits = D1SafetyLimits()
        assert limits.clamp_gripper(-0.5) == 0.0
        assert limits.clamp_gripper(1.5) == 1.0
        assert limits.clamp_gripper(0.5) == 0.5


class TestValidateCommand:
    def test_valid_idle_command(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=0)
        validate_command(cmd, limits)  # should not raise

    def test_invalid_mode(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=99)
        with pytest.raises(SafetyViolation, match="Invalid mode"):
            validate_command(cmd, limits)

    def test_position_violation(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=1, joint_positions=np.full(NUM_JOINTS, 10.0))
        with pytest.raises(SafetyViolation, match="Position limit"):
            validate_command(cmd, limits)

    def test_velocity_violation(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=2, joint_velocities=np.full(NUM_JOINTS, 100.0))
        with pytest.raises(SafetyViolation, match="Velocity limit"):
            validate_command(cmd, limits)

    def test_torque_violation(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=3, joint_torques=np.full(NUM_JOINTS, 200.0))
        with pytest.raises(SafetyViolation, match="Torque limit"):
            validate_command(cmd, limits)

    def test_gripper_violation(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=1, gripper_position=2.0)
        with pytest.raises(SafetyViolation, match="Gripper"):
            validate_command(cmd, limits)

    def test_valid_full_command(self):
        limits = D1SafetyLimits()
        cmd = D1Command(
            mode=1,
            joint_positions=np.zeros(NUM_JOINTS),
            joint_velocities=np.zeros(NUM_JOINTS),
            joint_torques=np.zeros(NUM_JOINTS),
            gripper_position=0.5,
        )
        validate_command(cmd, limits)  # should not raise


class TestClampCommand:
    def test_clamp_no_change_needed(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=0)
        result = clamp_command(cmd, limits)
        assert result.mode == 0

    def test_clamp_positions(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=1, joint_positions=np.full(NUM_JOINTS, 10.0))
        result = clamp_command(cmd, limits)
        for i in range(NUM_JOINTS):
            assert result.joint_positions[i] <= limits.joint_position_max[i]

    def test_clamp_preserves_mode(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=2, joint_velocities=np.full(NUM_JOINTS, 100.0))
        result = clamp_command(cmd, limits)
        assert result.mode == 2

    def test_clamp_gripper(self):
        limits = D1SafetyLimits()
        cmd = D1Command(mode=1, gripper_position=-1.0)
        result = clamp_command(cmd, limits)
        assert result.gripper_position == 0.0
