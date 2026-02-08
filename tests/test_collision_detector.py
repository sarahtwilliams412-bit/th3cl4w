"""Tests for CollisionDetector — stall detection with simulated sequences."""

import time
from unittest.mock import MagicMock

import pytest

from src.safety.collision_detector import CollisionDetector, StallEvent


class TestCollisionDetector:
    def test_no_stall_when_positions_match(self):
        cd = CollisionDetector()
        result = cd.update([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        assert result is None

    def test_no_stall_below_threshold(self):
        cd = CollisionDetector(position_error_deg=3.0)
        result = cd.update([2.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        assert result is None

    def test_no_stall_before_duration(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5)
        # Error starts at t=0
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        # Not enough time at t=0.3
        result = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.3)
        assert result is None

    def test_stall_detected_after_duration(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.3)
        result = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.6)
        assert result is not None
        assert result.joint_id == 0
        assert result.commanded_deg == 10
        assert result.actual_deg == 0
        assert result.error_deg == 10

    def test_stall_callback_fired(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5, cooldown_s=1.0)
        cb = MagicMock()
        cd.on_stall(cb)

        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.6)
        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert isinstance(event, StallEvent)
        assert event.joint_id == 0

    def test_cooldown_prevents_repeated_trigger(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5, cooldown_s=5.0)
        # First stall
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        r1 = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.6)
        assert r1 is not None

        # Still in error at t=1.0, but cooldown active
        r2 = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=1.0)
        assert r2 is None

    def test_cooldown_expires_allows_retrigger(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5, cooldown_s=2.0)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        r1 = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.6)
        assert r1 is not None

        # Error clears then re-enters after cooldown
        cd.update([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=1.0)  # clear
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=3.0)  # re-enter
        r2 = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=3.6)
        assert r2 is not None

    def test_last_good_position_tracked(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5)
        # Normal operation — joint 0 at 20°
        cd.update([20, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0], now=0.0)
        assert cd.last_good_positions[0] == 20.0

        # Now command to 40 but stuck at 20
        cd.update([40, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0], now=0.1)
        result = cd.update([40, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0], now=0.7)
        assert result is not None
        assert result.last_good_position == 20.0

    def test_disabled_detector_returns_none(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.0)
        cd.enabled = False
        result = cd.update([50, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        assert result is None

    def test_multiple_joints_independent(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5, cooldown_s=1.0)
        # Joint 0 stalls
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        r1 = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.6)
        assert r1 is not None
        assert r1.joint_id == 0

        # Joint 2 stalls independently (joint 0 in cooldown)
        cd.update([10, 0, 50, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.7)
        r2 = cd.update([10, 0, 50, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=1.3)
        assert r2 is not None
        assert r2.joint_id == 2

    def test_reset_clears_state(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        cd.reset()
        # After reset, need full duration again
        result = cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.3)
        assert result is None

    def test_error_clears_when_position_recovered(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.5)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        # Position recovers before duration
        cd.update([10, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0], now=0.3)
        # New error — needs full duration from scratch
        cd.update([20, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.4)
        result = cd.update([20, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.7)
        assert result is None  # Only 0.3s, not 0.5

    def test_recent_stalls_stored(self):
        cd = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.0, cooldown_s=0.0)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.0)
        cd.update([10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], now=0.1)
        assert len(cd.recent_stalls) >= 1
