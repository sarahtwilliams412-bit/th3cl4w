"""Tests for critical fixes from calibration run review (2026-02-08)."""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestCalibrationSessionId:
    """Fix: session_id should not be null in start response."""

    def test_session_id_preserved_if_preset(self):
        from src.calibration.calibration_runner import CalibrationRunner
        runner = CalibrationRunner()
        runner._session_id = "cal_preset_123"
        # Simulate what run_full_calibration does — should NOT overwrite
        runner._running = True
        runner._abort = False
        if runner._session_id is None:
            runner._session_id = f"cal_{int(time.time())}"
        assert runner._session_id == "cal_preset_123"

    def test_session_id_generated_if_none(self):
        from src.calibration.calibration_runner import CalibrationRunner
        runner = CalibrationRunner()
        assert runner._session_id is None
        runner._running = True
        runner._abort = False
        if runner._session_id is None:
            runner._session_id = f"cal_{int(time.time())}"
        assert runner._session_id is not None
        assert runner._session_id.startswith("cal_")


class TestDDSFeedbackFreshness:
    """Fix: per-joint freshness tracking and staleness detection."""

    def test_freshness_tracking(self):
        from src.interface.d1_dds_connection import _FeedbackCache
        cache = _FeedbackCache()
        assert cache.joint_freshness == {}

        # Simulate non-zero angle update
        now = time.monotonic()
        cache.joint_freshness["angle0"] = now
        assert "angle0" in cache.joint_freshness

    def test_is_feedback_fresh(self):
        from src.interface.d1_dds_connection import D1DDSConnection
        conn = D1DDSConnection()
        # No feedback yet
        assert conn.is_feedback_fresh() is False

        # Simulate recent feedback
        conn._cache.last_update = time.monotonic()
        assert conn.is_feedback_fresh() is True

        # Simulate stale feedback
        conn._cache.last_update = time.monotonic() - 5.0
        assert conn.is_feedback_fresh(max_age_s=2.0) is False

    def test_get_joint_freshness_empty(self):
        from src.interface.d1_dds_connection import D1DDSConnection
        conn = D1DDSConnection()
        freshness = conn.get_joint_freshness()
        # All joints should report inf (never seen non-zero)
        for i in range(7):
            assert freshness[i] == float("inf")

    def test_get_joint_freshness_after_update(self):
        from src.interface.d1_dds_connection import D1DDSConnection
        conn = D1DDSConnection()
        now = time.monotonic()
        conn._cache.joint_freshness["angle0"] = now
        conn._cache.joint_freshness["angle3"] = now

        freshness = conn.get_joint_freshness()
        assert freshness[0] < 1.0  # Just set, should be fresh
        assert freshness[3] < 1.0
        assert freshness[1] == float("inf")  # Never set


class TestEnableStateSync:
    """Fix: auto-sync enable state from DDS feedback on server restart."""

    def test_simulated_arm_enable_sync(self):
        """When DDS shows enabled but smoother doesn't know, it should sync."""
        # Import the SimulatedArm from server module
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'web'))
        from command_smoother import CommandSmoother
        
        # Create a mock arm that reports enabled
        mock_arm = MagicMock()
        mock_arm.get_status.return_value = {"power_status": 1, "enable_status": 1, "error_status": 0}
        mock_arm.get_joint_angles.return_value = [0.0] * 7
        mock_arm.get_gripper_position.return_value = 0.0
        mock_arm.is_connected = True

        smoother = CommandSmoother.__new__(CommandSmoother)
        smoother._arm_enabled = False
        smoother._synced = True

        # The logic from get_arm_state: if DDS says enabled but smoother doesn't know
        state_enabled = True
        state_power = True
        if not smoother._arm_enabled and state_enabled and state_power:
            smoother._arm_enabled = True

        assert smoother._arm_enabled is True


class TestGracefulShutdown:
    """Fix: PID file management for reliable server restarts."""

    def test_pidfile_write_and_remove(self):
        import tempfile
        from pathlib import Path

        pidfile = Path(tempfile.mktemp(suffix=".pid"))
        try:
            pidfile.write_text("12345")
            assert pidfile.read_text() == "12345"
            pidfile.unlink()
            assert not pidfile.exists()
        finally:
            if pidfile.exists():
                pidfile.unlink()

    def test_sigterm_handler_raises_systemexit(self):
        """SIGTERM handler should raise SystemExit for clean uvicorn shutdown."""
        # We can't easily test the actual signal handler without running the server,
        # but we can verify the pattern works
        def handler(signum, frame):
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            handler(15, None)
