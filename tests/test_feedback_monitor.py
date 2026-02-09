"""Tests for the DDS feedback quality monitor."""

import math
import time
import threading

import numpy as np
import pytest

from src.interface.feedback_monitor import (
    FeedbackMonitor,
    FeedbackSample,
    _is_zero_reading,
    _has_nan_or_inf,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angles(vals=None):
    """Build an angle dict. vals is a list of 7 floats or None for zeros."""
    if vals is None:
        vals = [0.0] * 7
    return {f"angle{i}": v for i, v in enumerate(vals)}


def _nonzero_angles():
    return _angles([10.0, -20.0, 30.0, -15.0, 25.0, -5.0, 40.0])


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_zero_reading_all_zeros(self):
        assert _is_zero_reading(_angles()) is True

    def test_zero_reading_small_values(self):
        assert _is_zero_reading(_angles([0.0005] * 7)) is True

    def test_nonzero_reading(self):
        assert _is_zero_reading(_nonzero_angles()) is False

    def test_nan_detection(self):
        a = _angles([1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0])
        assert _has_nan_or_inf(a) is True

    def test_inf_detection(self):
        a = _angles([0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0])
        assert _has_nan_or_inf(a) is True

    def test_neg_inf_detection(self):
        a = _angles([0.0, 0.0, 0.0, float("-inf"), 0.0, 0.0, 0.0])
        assert _has_nan_or_inf(a) is True

    def test_normal_values_no_nan(self):
        assert _has_nan_or_inf(_nonzero_angles()) is False


# ---------------------------------------------------------------------------
# FeedbackMonitor tests
# ---------------------------------------------------------------------------

class TestFeedbackMonitor:
    def test_empty_monitor_returns_none(self):
        m = FeedbackMonitor()
        assert m.get_reliable_state() is None
        assert m.get_reliable_angles() is None

    def test_nonzero_sample_is_reliable(self):
        m = FeedbackMonitor()
        angles = _nonzero_angles()
        ok = m.record_sample(angles)
        assert ok is True
        state = m.get_reliable_state()
        assert state is not None
        assert state["angle0"] == 10.0

    def test_zero_sample_not_reliable(self):
        m = FeedbackMonitor()
        ok = m.record_sample(_angles())
        assert ok is False
        assert m.get_reliable_state() is None

    def test_zero_filtered_returns_last_good(self):
        m = FeedbackMonitor()
        good = _nonzero_angles()
        m.record_sample(good)
        # Now send zeros — should still return the good reading
        m.record_sample(_angles())
        m.record_sample(_angles())
        state = m.get_reliable_state()
        assert state is not None
        assert state["angle0"] == 10.0

    def test_nan_rejected(self):
        m = FeedbackMonitor()
        bad = _angles([1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0])
        ok = m.record_sample(bad)
        assert ok is False
        assert m.get_reliable_state() is None

    def test_inf_rejected(self):
        m = FeedbackMonitor()
        bad = _angles([float("inf")] * 7)
        ok = m.record_sample(bad)
        assert ok is False

    def test_health_no_samples(self):
        m = FeedbackMonitor()
        h = m.get_health()
        assert h.is_healthy is False
        assert h.degraded_reason == "no_samples"

    def test_health_all_good(self):
        m = FeedbackMonitor()
        for _ in range(20):
            m.record_sample(_nonzero_angles())
        h = m.get_health()
        assert h.is_healthy is True
        assert h.zero_rate == 0.0

    def test_health_mixed_zeros(self):
        m = FeedbackMonitor()
        for i in range(20):
            if i % 2 == 0:
                m.record_sample(_nonzero_angles())
            else:
                m.record_sample(_angles())
        h = m.get_health()
        assert h.zero_rate == pytest.approx(0.5, abs=0.1)

    def test_reliable_angles_numpy(self):
        m = FeedbackMonitor()
        m.record_sample(_nonzero_angles())
        arr = m.get_reliable_angles()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (7,)
        assert arr[0] == 10.0

    def test_reliable_gripper(self):
        m = FeedbackMonitor()
        m.record_sample(_nonzero_angles())
        g = m.get_reliable_gripper()
        assert g == 40.0  # angle6

    def test_is_feedback_fresh(self):
        m = FeedbackMonitor()
        assert m.is_feedback_fresh() is False
        m.record_sample(_nonzero_angles())
        assert m.is_feedback_fresh() is True

    def test_recent_samples(self):
        m = FeedbackMonitor()
        m.record_sample(_nonzero_angles(), seq=1)
        m.record_sample(_angles(), seq=2)
        samples = m.get_recent_samples(10)
        assert len(samples) == 2
        assert samples[0]["seq"] == 1
        assert samples[0]["is_zero"] is False
        assert samples[1]["seq"] == 2
        assert samples[1]["is_zero"] is True

    def test_degradation_callback(self):
        called = []
        m = FeedbackMonitor()
        m.set_degradation_callback(lambda h: called.append(h))
        # Send 10 zero samples to trigger check
        for i in range(10):
            m.record_sample(_angles(), seq=i)
        assert len(called) >= 1
        assert called[0].is_healthy is False

    def test_thread_safety(self):
        """Concurrent record + read shouldn't crash."""
        m = FeedbackMonitor()
        errors = []

        def writer():
            try:
                for i in range(100):
                    m.record_sample(_nonzero_angles(), seq=i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    m.get_reliable_state()
                    m.get_health()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Safety monitor NaN/Inf tests
# ---------------------------------------------------------------------------

class TestSafetyMonitorNaN:
    """Test that safety monitor rejects NaN/Inf values."""

    def test_nan_position_rejected(self):
        from src.safety.safety_monitor import SafetyMonitor
        from src.interface.d1_connection import D1Command
        sm = SafetyMonitor()
        cmd = D1Command(
            mode=1,
            joint_positions=np.array([float("nan"), 0, 0, 0, 0, 0, 0]),
        )
        result = sm.validate_command(cmd)
        assert not result.is_safe
        assert any("NaN/Inf" in v.message for v in result.violations)

    def test_inf_velocity_rejected(self):
        from src.safety.safety_monitor import SafetyMonitor
        from src.interface.d1_connection import D1Command
        sm = SafetyMonitor()
        cmd = D1Command(
            mode=1,
            joint_velocities=np.array([0, float("inf"), 0, 0, 0, 0, 0]),
        )
        result = sm.validate_command(cmd)
        assert not result.is_safe

    def test_nan_gripper_rejected(self):
        from src.safety.safety_monitor import SafetyMonitor
        from src.interface.d1_connection import D1Command
        sm = SafetyMonitor()
        cmd = D1Command(mode=1, gripper_position=float("nan"))
        result = sm.validate_command(cmd)
        assert not result.is_safe

    def test_valid_command_passes(self):
        from src.safety.safety_monitor import SafetyMonitor
        from src.interface.d1_connection import D1Command
        sm = SafetyMonitor()
        cmd = D1Command(
            mode=1,
            joint_positions=np.zeros(7),
        )
        result = sm.validate_command(cmd)
        assert result.is_safe

    def test_check_state_nan_detected(self):
        from src.safety.safety_monitor import SafetyMonitor
        from src.interface.d1_connection import D1State
        sm = SafetyMonitor()
        state = D1State(
            joint_positions=np.array([float("nan"), 0, 0, 0, 0, 0, 0]),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
            gripper_position=0.0,
            timestamp=time.time(),
        )
        violations = sm.check_state(state)
        assert len(violations) > 0
        assert any("NaN/Inf" in v.message for v in violations)
