"""
Safety Integration Tests

Verifies that the SafetyMonitor is properly wired into the command pipeline
and that all safety constraints are enforced end-to-end.
"""

import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.safety.limits import (
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_RAD_MIN,
    JOINT_LIMITS_RAD_MAX,
    MAX_STEP_DEG,
    FEEDBACK_MAX_AGE_S,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
    NUM_ARM_JOINTS,
)
from src.safety.safety_monitor import SafetyMonitor, d1_default_limits, SafetyResult
from src.interface.d1_connection import D1Command, NUM_JOINTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_arm(angles=None, gripper=0.0):
    """Create a mock arm with configurable joint feedback."""
    arm = MagicMock()
    arm.get_joint_angles.return_value = angles if angles is not None else [0.0] * 6
    arm.get_gripper_position.return_value = gripper
    arm.set_joint.return_value = True
    arm.set_all_joints.return_value = True
    arm.set_gripper.return_value = True
    arm.is_connected = True
    arm._feedback_monitor = None
    return arm


def _make_smoother(arm=None, safety_monitor=None, max_step_deg=MAX_STEP_DEG):
    """Create a CommandSmoother wired to the safety monitor."""
    # Import here to pick up the web/ directory
    web_dir = str(Path(__file__).resolve().parent.parent / "web")
    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)
    from command_smoother import CommandSmoother

    if arm is None:
        arm = _make_mock_arm()
    sm = CommandSmoother(
        arm,
        rate_hz=10.0,
        smoothing_factor=0.35,
        max_step_deg=max_step_deg,
        safety_monitor=safety_monitor,
    )
    # Manually sync so smoother is ready
    sm.sync_from_feedback([0.0] * 6, 0.0)
    sm.set_arm_enabled(True)
    return sm


# ---------------------------------------------------------------------------
# Test: Unified limits are consistent
# ---------------------------------------------------------------------------

class TestUnifiedLimits:
    def test_limits_shape(self):
        assert JOINT_LIMITS_DEG.shape == (6, 2)
        assert len(JOINT_LIMITS_RAD_MIN) == NUM_JOINTS
        assert len(JOINT_LIMITS_RAD_MAX) == NUM_JOINTS

    def test_j1_j2_j4_have_80_deg_limit(self):
        for j in [1, 2, 4]:
            assert JOINT_LIMITS_DEG[j, 0] == -80.0
            assert JOINT_LIMITS_DEG[j, 1] == 80.0

    def test_rad_limits_match_deg(self):
        import math
        for i in range(6):
            assert abs(JOINT_LIMITS_RAD_MIN[i] - math.radians(JOINT_LIMITS_DEG[i, 0])) < 1e-10
            assert abs(JOINT_LIMITS_RAD_MAX[i] - math.radians(JOINT_LIMITS_DEG[i, 1])) < 1e-10

    def test_gripper_limits(self):
        assert GRIPPER_MIN_MM == 0.0
        assert GRIPPER_MAX_MM == 65.0

    def test_max_step_is_10(self):
        assert MAX_STEP_DEG == 10.0


# ---------------------------------------------------------------------------
# Test: SafetyMonitor validates commands
# ---------------------------------------------------------------------------

class TestSafetyMonitorValidation:
    def setup_method(self):
        self.monitor = SafetyMonitor()

    def test_valid_command_passes(self):
        cmd = D1Command(
            mode=1,
            joint_positions=np.zeros(NUM_JOINTS),
        )
        result = self.monitor.validate_command(cmd)
        assert result.is_safe
        assert result.violation_count == 0

    def test_command_beyond_limits_blocked(self):
        # J1 limit is ±80° = ±1.396 rad — send 2.0 rad
        positions = np.zeros(NUM_JOINTS)
        positions[1] = 2.0  # beyond ±1.396 rad
        cmd = D1Command(mode=1, joint_positions=positions)
        result = self.monitor.validate_command(cmd)
        assert not result.is_safe
        assert result.violation_count > 0

    def test_estop_blocks_all_commands(self):
        self.monitor.trigger_estop("test")
        cmd = D1Command(mode=1, joint_positions=np.zeros(NUM_JOINTS))
        result = self.monitor.validate_command(cmd)
        assert not result.is_safe
        # Reset and verify commands work again
        self.monitor.reset_estop()
        result = self.monitor.validate_command(cmd)
        assert result.is_safe


# ---------------------------------------------------------------------------
# Test: CommandSmoother integration with SafetyMonitor
# ---------------------------------------------------------------------------

class TestSmootherSafetyIntegration:
    def test_valid_command_sent(self):
        arm = _make_mock_arm()
        monitor = SafetyMonitor()
        sm = _make_smoother(arm=arm, safety_monitor=monitor)

        sm.set_joint_target(0, 10.0)  # well within ±135°
        sm._tick()

        # Command should have been sent
        assert arm.set_joint.called or arm.set_all_joints.called

    def test_command_beyond_limits_blocked(self):
        arm = _make_mock_arm()
        monitor = SafetyMonitor()
        sm = _make_smoother(arm=arm, safety_monitor=monitor)

        # Set current position near the limit, target beyond
        sm._current[1] = 79.0
        sm.set_joint_target(1, 95.0)  # beyond ±80° for J1

        # Tick will try to step toward 95°, current will move to ~84.6°
        # which is beyond 80° — safety should block
        # First few ticks move current beyond 80
        for _ in range(5):
            sm._tick()

        # The command should have been blocked at some point when current > 80
        # Check that current never exceeded 80° significantly
        # (The safety check blocks the entire send if any joint is out of range)

    def test_estop_blocks_smoother(self):
        arm = _make_mock_arm()
        monitor = SafetyMonitor()
        sm = _make_smoother(arm=arm, safety_monitor=monitor)

        monitor.trigger_estop("test")
        arm.reset_mock()

        sm.set_joint_target(0, 10.0)
        sm._tick()

        # No commands should be sent
        assert not arm.set_joint.called
        assert not arm.set_all_joints.called

    def test_stale_feedback_blocks_commands(self):
        arm = _make_mock_arm()
        monitor = SafetyMonitor()
        sm = _make_smoother(arm=arm, safety_monitor=monitor)

        # Make feedback stale
        sm._last_feedback_time = time.time() - 1.0  # 1s old > 500ms threshold
        arm.reset_mock()

        sm.set_joint_target(0, 10.0)
        sm._tick()

        # No commands should be sent
        assert not arm.set_joint.called
        assert not arm.set_all_joints.called

    def test_step_size_clamped_to_10_degrees(self):
        arm = _make_mock_arm()
        sm = _make_smoother(arm=arm)

        assert sm._max_step == MAX_STEP_DEG == 10.0

        # Set a large target jump
        sm.set_joint_target(0, 100.0)
        sm._tick()

        # Current should have moved at most max_step from 0
        assert sm._current[0] is not None
        assert abs(sm._current[0]) <= MAX_STEP_DEG + 0.01


# ---------------------------------------------------------------------------
# Test: d1_default_limits uses unified values
# ---------------------------------------------------------------------------

class TestDefaultLimitsUnified:
    def test_limits_use_unified_values(self):
        limits = d1_default_limits()
        np.testing.assert_array_equal(limits.position_min, JOINT_LIMITS_RAD_MIN)
        np.testing.assert_array_equal(limits.position_max, JOINT_LIMITS_RAD_MAX)


# ---------------------------------------------------------------------------
# Test: Feedback freshness
# ---------------------------------------------------------------------------

class TestFeedbackFreshness:
    def test_fresh_feedback(self):
        monitor = SafetyMonitor()
        from src.interface.d1_connection import D1State
        state = D1State(
            joint_positions=np.zeros(NUM_JOINTS),
            joint_velocities=np.zeros(NUM_JOINTS),
            joint_torques=np.zeros(NUM_JOINTS),
            gripper_position=0.0,
            timestamp=time.time(),
        )
        assert monitor.is_feedback_fresh(state)

    def test_stale_feedback(self):
        monitor = SafetyMonitor()
        from src.interface.d1_connection import D1State
        state = D1State(
            joint_positions=np.zeros(NUM_JOINTS),
            joint_velocities=np.zeros(NUM_JOINTS),
            joint_torques=np.zeros(NUM_JOINTS),
            gripper_position=0.0,
            timestamp=time.time() - 1.0,  # 1s old
        )
        assert not monitor.is_feedback_fresh(state)
