"""Tests for the telemetry module."""

import threading
import time

import numpy as np
import pytest

from src.telemetry.collector import EventType, TelemetryCollector, TelemetryEvent
from src.telemetry.camera_monitor import CameraHealthMonitor


# ── TelemetryCollector ──────────────────────────────────────────────


class TestCollectorDisabled:
    def test_emit_when_disabled_is_noop(self):
        c = TelemetryCollector()
        result = c.emit("src", EventType.CMD_SENT, {"k": "v"})
        assert result is None
        assert c.get_events() == []

    def test_enabled_property_default_false(self):
        c = TelemetryCollector()
        assert c.enabled is False


class TestCollectorEnabled:
    def test_emit_stores_event(self):
        c = TelemetryCollector()
        c.enable()
        ev = c.emit("nav", EventType.CMD_SENT, {"cmd": "go"})
        assert ev is not None
        assert ev.source == "nav"
        assert ev.event_type == EventType.CMD_SENT
        assert ev.payload == {"cmd": "go"}

    def test_emit_returns_telemetry_event(self):
        c = TelemetryCollector()
        c.enable()
        ev = c.emit("x", EventType.ERROR, {})
        assert isinstance(ev, TelemetryEvent)
        assert ev.timestamp_ms > 0
        assert ev.wall_time_ms > 0

    def test_disable_stops_collection(self):
        c = TelemetryCollector()
        c.enable()
        c.emit("a", EventType.CMD_SENT, {})
        c.disable()
        c.emit("b", EventType.CMD_SENT, {})
        assert len(c.get_events()) == 1

    def test_ring_buffer_evicts_oldest(self):
        c = TelemetryCollector(maxlen=5)
        c.enable()
        for i in range(10):
            c.emit("s", EventType.CMD_SENT, {"i": i})
        events = c.get_events(limit=10)
        assert len(events) == 5
        assert events[0].payload["i"] == 5


class TestGetEvents:
    def test_filter_by_event_type(self):
        c = TelemetryCollector()
        c.enable()
        c.emit("a", EventType.CMD_SENT, {})
        c.emit("a", EventType.ERROR, {})
        c.emit("a", EventType.CMD_SENT, {})
        assert len(c.get_events(event_type=EventType.CMD_SENT)) == 2

    def test_filter_by_source(self):
        c = TelemetryCollector()
        c.enable()
        c.emit("nav", EventType.CMD_SENT, {})
        c.emit("cam", EventType.CMD_SENT, {})
        assert len(c.get_events(source="cam")) == 1

    def test_limit(self):
        c = TelemetryCollector()
        c.enable()
        for _ in range(20):
            c.emit("s", EventType.CMD_SENT, {})
        assert len(c.get_events(limit=5)) == 5


class TestCorrelation:
    def test_new_correlation_id_format(self):
        cid = TelemetryCollector.new_correlation_id()
        assert len(cid) == 12
        int(cid, 16)  # must be valid hex

    def test_get_pipeline_tracks_events(self):
        c = TelemetryCollector()
        c.enable()
        cid = c.new_correlation_id()
        c.emit("a", EventType.CMD_SENT, {}, correlation_id=cid)
        c.emit("b", EventType.CMD_ACK, {}, correlation_id=cid)
        pipeline = c.get_pipeline(cid)
        assert len(pipeline) == 2
        assert pipeline[0]["latency_ms"] == 0.0
        assert pipeline[1]["latency_ms"] >= 0.0

    def test_pipeline_missing_correlation(self):
        c = TelemetryCollector()
        assert c.get_pipeline("nonexistent") == []

    def test_correlation_eviction(self):
        c = TelemetryCollector(maxlen=500)
        c.enable()
        # fill 500 correlations then add one more
        for i in range(501):
            c.emit("s", EventType.CMD_SENT, {}, correlation_id=f"cid-{i:04d}")
        # first should be evicted from the ring buffer
        assert c.get_pipeline("cid-0000") == []
        assert len(c.get_pipeline("cid-0500")) == 1


class TestGetStats:
    def test_stats_structure(self):
        c = TelemetryCollector()
        c.enable()
        c.emit("s", EventType.CMD_SENT, {})
        stats = c.get_stats()
        assert "total_events" in stats
        assert "rates_per_sec" in stats
        assert "staleness_ms" in stats
        assert stats["total_events"] == 1

    def test_rates_reflect_emitted(self):
        c = TelemetryCollector()
        c.enable()
        for _ in range(5):
            c.emit("s", EventType.WS_SEND, {})
        stats = c.get_stats()
        assert stats["rates_per_sec"]["ws_send"] > 0


class TestThreadSafety:
    def test_concurrent_emits(self):
        c = TelemetryCollector()
        c.enable()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(200):
                    c.emit("w", EventType.CMD_SENT, {"t": 1})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # ring buffer maxlen=1000, so at most 1000
        assert len(c.get_events(limit=2000)) <= 1000


# ── CameraHealthMonitor ────────────────────────────────────────────


class TestCameraMonitor:
    def test_fps_tracking(self):
        mon = CameraHealthMonitor("cam0", target_fps=30.0)
        # simulate frames at ~100fps for a short burst
        for _ in range(20):
            mon.on_frame((1920, 1080))
            time.sleep(0.01)
        fps = mon.actual_fps
        assert fps > 0

    def test_drop_counting(self):
        mon = CameraHealthMonitor("cam0")
        mon.on_drop()
        mon.on_drop()
        assert mon.stats["drop_count"] == 2

    def test_stall_detection_no_frames(self):
        mon = CameraHealthMonitor("cam0")
        assert mon.stats["stalled"] is True

    def test_stall_detection_old_frame(self):
        mon = CameraHealthMonitor("cam0")
        mon.on_frame((640, 480))
        # hack last frame time to simulate staleness
        mon._last_frame_time = time.monotonic() - 3.0
        assert mon.stats["stalled"] is True

    def test_not_stalled_after_recent_frame(self):
        mon = CameraHealthMonitor("cam0")
        mon.on_frame((640, 480))
        assert mon.stats["stalled"] is False

    def test_motion_detection_no_motion(self):
        mon = CameraHealthMonitor("cam0")
        frame = np.zeros((100, 100), dtype=np.uint8)
        score1 = mon.compute_motion(frame)
        score2 = mon.compute_motion(frame)
        assert score2 == 0.0

    def test_motion_detection_with_motion(self):
        mon = CameraHealthMonitor("cam0")
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.full((100, 100), 128, dtype=np.uint8)
        mon.compute_motion(frame_a)
        score = mon.compute_motion(frame_b)
        assert 0.0 < score <= 1.0

    def test_stats_keys(self):
        mon = CameraHealthMonitor("cam0", target_fps=15.0)
        mon.on_frame((1280, 720))
        s = mon.stats
        assert s["camera_id"] == "cam0"
        assert s["target_fps"] == 15.0
        assert s["resolution"] == (1280, 720)
        assert s["connected"] is True
