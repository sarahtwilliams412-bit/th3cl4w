"""Tests for SimTelemetryBridge."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.telemetry.sim_bridge import SimTelemetryBridge


def _make_mock_arm(angles=None, powered=False, enabled=False, error=0):
    arm = MagicMock()
    arm._lock = threading.Lock()
    arm._powered = powered
    arm._enabled = enabled
    arm._error = error
    if angles is None:
        angles = [0.0] * 7  # 6 joints + gripper
    arm.get_joint_angles.return_value = np.array(angles, dtype=np.float64)
    return arm


def _make_mock_collector():
    collector = MagicMock()
    collector._enabled = True
    collector.emit = MagicMock()
    return collector


class TestSimTelemetryBridge:
    """SimTelemetryBridge unit tests."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        arm = _make_mock_arm()
        bridge = SimTelemetryBridge(arm, None)
        await bridge.start()
        assert bridge.stats["running"] is True
        await bridge.stop()
        assert bridge.stats["running"] is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        arm = _make_mock_arm()
        bridge = SimTelemetryBridge(arm, None)
        await bridge.start()
        task1 = bridge._task
        await bridge.start()  # second start is no-op
        assert bridge._task is task1
        await bridge.stop()

    def test_emit_feedback_funcode1_and_3(self):
        arm = _make_mock_arm(angles=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0], powered=True, enabled=True)
        collector = _make_mock_collector()
        bridge = SimTelemetryBridge(arm, collector)

        bridge._emit_feedback()

        calls = collector.emit.call_args_list
        # Should have 3 calls: funcode=1, funcode=3, state_update (first call)
        assert len(calls) == 3

        # funcode=1 (joint angles)
        _, kw1 = calls[0]
        assert kw1["payload"]["funcode"] == 1
        assert kw1["payload"]["angles"]["angle0"] == 10.0
        assert kw1["payload"]["sim"] is True

        # funcode=3 (status)
        _, kw3 = calls[1]
        assert kw3["payload"]["funcode"] == 3
        assert kw3["payload"]["status"]["power_status"] == 1
        assert kw3["payload"]["status"]["enable_status"] == 1
        assert kw3["payload"]["sim"] is True

    def test_state_update_on_angle_change(self):
        arm = _make_mock_arm(angles=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0])
        collector = _make_mock_collector()
        bridge = SimTelemetryBridge(arm, collector)

        # First call — always emits state_update
        bridge._emit_feedback()
        assert len(collector.emit.call_args_list) == 3  # funcode1 + funcode3 + state_update

        collector.emit.reset_mock()

        # Same angles — no state_update
        bridge._emit_feedback()
        assert len(collector.emit.call_args_list) == 2  # funcode1 + funcode3 only

        collector.emit.reset_mock()

        # Change angle beyond threshold
        arm.get_joint_angles.return_value = np.array([10.1, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0])
        bridge._emit_feedback()
        assert len(collector.emit.call_args_list) == 3  # state_update fires

        # Verify last call is state_update
        _, kw = collector.emit.call_args_list[2]
        assert kw["payload"]["sim"] is True
        assert "angles" in kw["payload"]

    def test_sim_flag_present_in_all_events(self):
        arm = _make_mock_arm()
        collector = _make_mock_collector()
        bridge = SimTelemetryBridge(arm, collector)
        bridge._emit_feedback()

        for call in collector.emit.call_args_list:
            _, kw = call
            assert kw["payload"]["sim"] is True

    def test_stats_counts(self):
        arm = _make_mock_arm()
        bridge = SimTelemetryBridge(arm, None)

        assert bridge.stats == {"running": False, "event_count": 0, "seq": 0}

        bridge._emit_feedback()
        # 3 events (funcode1 + funcode3 + state_update first time), 1 seq
        assert bridge.stats["event_count"] == 3
        assert bridge.stats["seq"] == 1

        bridge._emit_feedback()
        # +2 events (no state_update), seq=2
        assert bridge.stats["event_count"] == 5
        assert bridge.stats["seq"] == 2

    def test_no_collector_still_counts(self):
        """Events are counted even without a collector."""
        arm = _make_mock_arm()
        bridge = SimTelemetryBridge(arm, None)
        bridge._emit_feedback()
        assert bridge.stats["event_count"] == 3

    def test_none_angles_skips(self):
        arm = _make_mock_arm()
        arm.get_joint_angles.return_value = None
        bridge = SimTelemetryBridge(arm, None)
        bridge._emit_feedback()
        assert bridge.stats["event_count"] == 0
