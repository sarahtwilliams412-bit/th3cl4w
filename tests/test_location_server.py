"""Tests for the location server components."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.location.reachability import (
    ReachStatus,
    classify_reach,
    is_reachable,
    ARM_MAX_REACH_MM,
    ARM_MIN_REACH_MM,
)
from src.location.world_model import (
    LocationWorldModel,
    TrackedObject,
    ObjectCategory,
    STALE_TIMEOUT_S,
    MATCH_DISTANCE_MM,
)
from src.location.detector import UnifiedDetector, DetectionResult


# ============================================================
# Reachability tests
# ============================================================


class TestReachability:
    def test_reachable_position(self):
        pos = np.array([200.0, 100.0, 0.0])
        status, dist = classify_reach(pos)
        assert status == ReachStatus.REACHABLE
        assert dist == pytest.approx(np.sqrt(200**2 + 100**2), abs=0.1)

    def test_out_of_range(self):
        pos = np.array([600.0, 0.0, 0.0])
        status, _ = classify_reach(pos)
        assert status == ReachStatus.OUT_OF_RANGE

    def test_too_close(self):
        pos = np.array([10.0, 10.0, 0.0])
        status, _ = classify_reach(pos)
        assert status == ReachStatus.TOO_CLOSE

    def test_marginal(self):
        pos = np.array([520.0, 0.0, 0.0])
        status, _ = classify_reach(pos)
        assert status == ReachStatus.MARGINAL

    def test_is_reachable_true(self):
        assert is_reachable(np.array([300.0, 0.0, 0.0]))

    def test_is_reachable_false(self):
        assert not is_reachable(np.array([700.0, 0.0, 0.0]))

    def test_z_ignored_for_xy_distance(self):
        pos = np.array([200.0, 0.0, 500.0])
        status, dist = classify_reach(pos)
        assert status == ReachStatus.REACHABLE
        assert dist == pytest.approx(200.0, abs=0.1)


# ============================================================
# World model tests
# ============================================================


class TestWorldModel:
    def setup_method(self):
        self.model = LocationWorldModel()

    def test_upsert_new_object(self):
        obj = self.model.upsert(
            label="redbull",
            position_mm=np.array([200.0, 100.0, 0.0]),
            confidence=0.8,
        )
        assert obj.label == "redbull"
        assert obj.id == "redbull_0"
        assert obj.observation_count == 1
        assert not obj.stable

    def test_upsert_updates_existing(self):
        self.model.upsert(
            label="redbull",
            position_mm=np.array([200.0, 100.0, 0.0]),
            confidence=0.7,
        )
        obj = self.model.upsert(
            label="redbull",
            position_mm=np.array([205.0, 102.0, 0.0]),
            confidence=0.8,
        )
        assert obj.observation_count == 2
        assert obj.confidence == 0.8  # keeps best

    def test_upsert_creates_new_if_far(self):
        self.model.upsert(
            label="redbull",
            position_mm=np.array([200.0, 100.0, 0.0]),
        )
        self.model.upsert(
            label="redbull",
            position_mm=np.array([500.0, 100.0, 0.0]),
        )
        assert self.model.object_count == 2

    def test_get_all_objects(self):
        self.model.upsert(label="a", position_mm=np.array([100.0, 0.0, 0.0]))
        self.model.upsert(label="b", position_mm=np.array([200.0, 0.0, 0.0]))
        assert len(self.model.get_all_objects()) == 2

    def test_get_reachable(self):
        self.model.upsert(label="near", position_mm=np.array([200.0, 0.0, 0.0]))
        self.model.upsert(label="far", position_mm=np.array([700.0, 0.0, 0.0]))
        reachable = self.model.get_reachable()
        assert len(reachable) == 1
        assert reachable[0].label == "near"

    def test_get_object_by_id(self):
        self.model.upsert(label="test", position_mm=np.array([100.0, 0.0, 0.0]))
        obj = self.model.get_object("test_0")
        assert obj is not None
        assert obj.label == "test"

    def test_get_object_not_found(self):
        assert self.model.get_object("nonexistent") is None

    def test_stability_after_observations(self):
        for _ in range(3):
            obj = self.model.upsert(
                label="stable_obj",
                position_mm=np.array([200.0, 0.0, 0.0]),
            )
        assert obj.stable

    def test_sweep_stale(self):
        obj = self.model.upsert(
            label="old_obj",
            position_mm=np.array([200.0, 0.0, 0.0]),
        )
        # Manually backdate last_seen
        obj.last_seen = time.time() - STALE_TIMEOUT_S - 1
        self.model.sweep_stale()
        assert obj.stale

    def test_to_dict(self):
        obj = self.model.upsert(
            label="test",
            position_mm=np.array([200.0, 100.0, 50.0]),
            confidence=0.75,
        )
        d = obj.to_dict()
        assert d["label"] == "test"
        assert d["confidence"] == 0.75
        assert len(d["position_mm"]) == 3
        assert "reach_status" in d

    def test_clear(self):
        self.model.upsert(label="a", position_mm=np.array([100.0, 0.0, 0.0]))
        self.model.clear()
        assert self.model.object_count == 0

    def test_scan_count(self):
        assert self.model.scan_count == 0
        self.model.mark_scan()
        assert self.model.scan_count == 1

    def test_callback_called(self):
        callback = MagicMock()
        self.model.register_callback(callback)
        self.model.upsert(label="x", position_mm=np.array([100.0, 0.0, 0.0]))
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert len(args) == 1
        assert args[0].label == "x"


# ============================================================
# Detector tests
# ============================================================


class TestDetector:
    def test_detect_fast_empty_frame(self):
        detector = UnifiedDetector()
        results = detector.detect_fast(np.array([]), camera_id=0)
        assert results == []

    def test_detect_fast_none_frame(self):
        detector = UnifiedDetector()
        results = detector.detect_fast(None, camera_id=0)
        assert results == []

    def test_detect_fast_red_object(self):
        """Create a synthetic frame with a red blob and verify detection."""
        detector = UnifiedDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a red rectangle
        cv2.rectangle(frame, (200, 150), (300, 250), (0, 0, 255), -1)
        results = detector.detect_fast(frame, camera_id=0, targets=["red"])
        assert len(results) >= 1
        assert results[0].label == "red"
        assert results[0].source == "hsv"
        # Centroid should be near center of rectangle
        cx, cy = results[0].centroid_px
        assert 200 <= cx <= 300
        assert 150 <= cy <= 250

    def test_detect_cv_blue_object(self):
        """Create a frame with a blue blob and verify CV detection."""
        detector = UnifiedDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 100, 0), -1)
        results = detector.detect_cv(frame, camera_id=1)
        # Should detect something blue
        blue_results = [r for r in results if r.label == "blue"]
        assert len(blue_results) >= 1


# ============================================================
# Integration: tracker position conversion
# ============================================================


class TestTrackerPositionConversion:
    def test_overhead_center_maps_to_origin(self):
        from src.location.tracker import _pixel_to_position_overhead
        pos = _pixel_to_position_overhead(960, 540, 1920, 1080)
        assert abs(pos[0]) < 1.0
        assert abs(pos[1]) < 1.0

    def test_overhead_offset(self):
        from src.location.tracker import _pixel_to_position_overhead
        pos = _pixel_to_position_overhead(1920, 540, 1920, 1080)
        assert pos[0] > 0  # right side = positive X

    def test_side_bottom_maps_to_ground(self):
        from src.location.tracker import _pixel_to_position_side
        pos = _pixel_to_position_side(960, 1080, 1920, 1080)
        assert abs(pos[2]) < 1.0  # bottom pixel = ground level


try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is None:
    pytest.skip("OpenCV not available", allow_module_level=True)
