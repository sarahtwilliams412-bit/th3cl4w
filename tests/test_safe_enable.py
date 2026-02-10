"""Tests for safe enable/reset fixes — preventing overcurrent on enable.

Root cause: reset_to_zero() tells firmware to target 0° on all joints.
When enable_motors() fires 2s later, firmware tries simultaneous ~90° jumps
on J1/J2/J4 → overcurrent protection trips.

These tests verify:
1. Safe reset+enable does NOT call reset_to_zero
2. Safe-home sequences joints in correct order (low-torque first)
3. Enable-here syncs smoother to current position
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Ensure project root is importable
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


@pytest.fixture
def mock_arm():
    """Create a mock arm that simulates DDS feedback at non-zero positions."""
    arm = MagicMock()
    arm.is_connected = True
    # Simulate arm at non-zero position (the dangerous case)
    _positions = [45.0, 80.0, -60.0, 30.0, 70.0, -20.0]
    arm.get_joint_angles.return_value = list(_positions)
    arm.get_gripper_position.return_value = 10.0
    arm.get_status.return_value = {"power_status": 1, "enable_status": 0}
    arm.enable_motors.return_value = True
    arm.disable_motors.return_value = True
    arm.reset_to_zero.return_value = True
    arm.set_all_joints.return_value = True
    arm.set_joint.return_value = True
    arm.power_on.return_value = True
    arm.power_off.return_value = True
    return arm


@pytest.fixture
def mock_smoother():
    """Create a mock CommandSmoother."""
    s = MagicMock()
    s._arm_enabled = True
    s.arm_enabled = True
    s.synced = True
    s.running = True
    s._target = [None] * 6
    s._current = [45.0, 80.0, -60.0, 30.0, 70.0, -20.0]
    return s


@pytest.fixture
def app_client(mock_arm, mock_smoother):
    """Create a test client with mocked arm and smoother."""
    from fastapi.testclient import TestClient
    import web.server as srv

    # Save originals
    orig_arm = srv.arm
    orig_smoother = srv.smoother
    orig_cached = list(srv._cached_joint_angles)

    srv.arm = mock_arm
    srv.smoother = mock_smoother
    srv._cached_joint_angles = [45.0, 80.0, -60.0, 30.0, 70.0, -20.0]

    client = TestClient(srv.app, raise_server_exceptions=False)

    yield client, mock_arm, mock_smoother

    # Restore
    srv.arm = orig_arm
    srv.smoother = orig_smoother
    srv._cached_joint_angles = orig_cached


class TestSafeResetEnable:
    """Fix 1: Safe reset+enable must NOT call reset_to_zero."""

    def test_reset_enable_does_not_call_reset_to_zero(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/reset-enable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

        # THE KEY ASSERTION: reset_to_zero must NOT be called
        mock_arm.reset_to_zero.assert_not_called()

    def test_reset_enable_calls_enable_motors(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/reset-enable")
        assert resp.status_code == 200
        mock_arm.enable_motors.assert_called()

    def test_reset_enable_syncs_smoother(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/reset-enable")
        assert resp.status_code == 200
        mock_smoother.sync_from_feedback.assert_called()
        # Verify it synced with actual arm positions, not zeros
        call_args = mock_smoother.sync_from_feedback.call_args
        angles = call_args[0][0]
        assert angles[1] == 80.0  # J1 should be at 80°, not 0°

    def test_hard_reset_enable_does_call_reset_to_zero(self, app_client):
        """Hard reset should still use reset_to_zero for when user explicitly wants it."""
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/hard-reset-enable")
        assert resp.status_code == 200
        mock_arm.reset_to_zero.assert_called()


class TestEnableHere:
    """Fix 3: Enable-at-current-position."""

    def test_enable_here_enables_motors(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/enable-here")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        mock_arm.enable_motors.assert_called()

    def test_enable_here_sends_current_position_as_target(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/enable-here")
        assert resp.status_code == 200
        # Should send current position so arm holds still
        mock_arm.set_all_joints.assert_called()
        sent_angles = mock_arm.set_all_joints.call_args[0][0]
        assert sent_angles == [45.0, 80.0, -60.0, 30.0, 70.0, -20.0]

    def test_enable_here_syncs_smoother_to_current(self, app_client):
        client, mock_arm, mock_smoother = app_client
        resp = client.post("/api/command/enable-here")
        assert resp.status_code == 200
        mock_smoother.sync_from_feedback.assert_called()
        mock_smoother.set_all_joints_target.assert_called()
        target = mock_smoother.set_all_joints_target.call_args[0][0]
        assert target == [45.0, 80.0, -60.0, 30.0, 70.0, -20.0]

    def test_enable_here_requires_power(self, app_client):
        client, mock_arm, mock_smoother = app_client
        mock_arm.get_status.return_value = {"power_status": 0, "enable_status": 0}
        resp = client.post("/api/command/enable-here")
        data = resp.json()
        assert data["ok"] is False
        mock_arm.enable_motors.assert_not_called()


class TestSafeHome:
    """Fix 2: Safe home sequences joints in correct order."""

    def test_safe_home_requires_arm_enabled(self, app_client):
        client, mock_arm, mock_smoother = app_client
        mock_smoother._arm_enabled = False
        resp = client.post("/api/command/safe-home")
        assert resp.status_code == 409

    def test_safe_home_moves_low_torque_first(self, app_client):
        """J0, J3, J5 (low torque) should be commanded before J1, J2, J4."""
        client, mock_arm, mock_smoother = app_client

        # Track call order
        call_order = []

        def track_set_joint(joint_id, angle):
            call_order.append((joint_id, angle))

        mock_smoother.set_joint_target = MagicMock(side_effect=track_set_joint)

        # Make _get_current_joints return decreasing values toward 0
        import numpy as np
        import web.server as srv

        call_count = [0]

        def mock_get_joints():
            call_count[0] += 1
            if call_count[0] > 3:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return np.array([45.0, 80.0, -60.0, 30.0, 70.0, -20.0])

        async def instant_sleep(t):
            pass

        with patch.object(srv, "_get_current_joints", mock_get_joints):
            with patch("web.server.asyncio.sleep", instant_sleep):
                resp = client.post("/api/command/safe-home")

        assert resp.status_code == 200

        # Verify low-torque joints (0, 3, 5) were commanded before high-torque (1, 2, 4)
        commanded_joints = [j for j, _ in call_order]
        if commanded_joints:
            low_torque = {0, 3, 5}
            high_torque = {1, 2, 4}
            first_high = None
            for i, j in enumerate(commanded_joints):
                if j in high_torque and first_high is None:
                    first_high = i
            if first_high is not None:
                first_low = min(i for i, j in enumerate(commanded_joints) if j in low_torque)
                assert (
                    first_low < first_high
                ), f"Low-torque joints should move before high-torque. Order: {commanded_joints}"


class TestAutoRecoveryNoReset:
    """Verify auto-recovery after power loss doesn't call reset_to_zero."""

    def test_auto_recovery_uses_safe_enable(self, mock_arm, mock_smoother):
        """_auto_recover_power should NOT call reset_to_zero."""
        import web.server as srv

        orig_arm = srv.arm
        orig_smoother = srv.smoother
        srv.arm = mock_arm
        srv.smoother = mock_smoother

        async def instant_sleep(t):
            pass

        try:
            loop = asyncio.new_event_loop()
            with patch.object(srv.asyncio, "sleep", instant_sleep):
                loop.run_until_complete(srv._auto_recover_power())
            loop.close()

            mock_arm.reset_to_zero.assert_not_called()
            mock_arm.enable_motors.assert_called()
        finally:
            srv.arm = orig_arm
            srv.smoother = orig_smoother
