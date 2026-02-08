"""Tests for ObjectDimensionEstimator, WorldModel, and StartupScanner."""

import time
import threading
from typing import Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.vision.dimension_estimator import (
    ObjectDimensionEstimator,
    DimensionEstimate,
    _EstimateHistory,
)
from src.vision.world_model import (
    WorldModel,
    WorldObject,
    WorldModelSnapshot,
    ObjectCategory,
    ReachStatus,
)
from src.vision.startup_scanner import (
    StartupScanner,
    ScanPhase,
    StartupScanReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_color_image(width=640, height=480, color=(200, 200, 200)):
    """Create a solid-color BGR image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img


def make_image_with_red_object(width=640, height=480, obj_x=200, obj_y=200, obj_w=60, obj_h=80):
    """Create a test image with a red rectangle (detectable by HSV segmentation)."""
    img = make_color_image(width, height, color=(200, 200, 200))
    # Draw a bright red rectangle
    # OpenCV BGR: red is (0, 0, 255)
    cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (0, 0, 255), -1)
    return img


def make_image_with_blue_object(width=640, height=480, obj_x=300, obj_y=150, obj_w=50, obj_h=70):
    """Create a test image with a blue rectangle."""
    img = make_color_image(width, height, color=(200, 200, 200))
    cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (255, 0, 0), -1)
    return img


class MockFrameProvider:
    """Mock camera that returns pre-set frames."""

    def __init__(self, frame: Optional[np.ndarray] = None, connected: bool = True):
        self._frame = frame
        self._connected = connected

    def get_raw_frame(self) -> Optional[np.ndarray]:
        return self._frame

    @property
    def connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# ObjectDimensionEstimator tests
# ---------------------------------------------------------------------------


class TestObjectDimensionEstimator:
    def test_init_defaults(self):
        est = ObjectDimensionEstimator()
        assert est.overhead_scale > 0
        assert est.front_scale > 0
        assert est.consistency_frames == 3
        assert est.max_history == 10

    def test_estimate_empty_frames(self):
        est = ObjectDimensionEstimator()
        results = est.estimate_from_frames(None, None)
        assert results == []

    def test_estimate_blank_image(self):
        est = ObjectDimensionEstimator()
        blank = make_color_image()
        results = est.estimate_from_frames(cam0_frame=blank, cam1_frame=blank)
        # No colored objects in a gray image
        assert results == []

    def test_estimate_single_camera_overhead(self):
        est = ObjectDimensionEstimator()
        img = make_image_with_red_object()
        results = est.estimate_from_frames(cam1_frame=img)
        # Should detect the red object from overhead
        assert len(results) >= 1
        red_est = [r for r in results if r.label == "red"]
        assert len(red_est) >= 1
        assert red_est[0].source == "cam1"
        assert red_est[0].width_mm > 0
        assert red_est[0].depth_mm > 0
        # Single camera => lower confidence
        assert red_est[0].confidence < 0.6

    def test_estimate_single_camera_front(self):
        est = ObjectDimensionEstimator()
        img = make_image_with_red_object()
        results = est.estimate_from_frames(cam0_frame=img)
        assert len(results) >= 1
        red_est = [r for r in results if r.label == "red"]
        assert len(red_est) >= 1
        assert red_est[0].source == "cam0"
        # Front-only is weakest
        assert red_est[0].confidence < 0.4

    def test_estimate_dual_camera(self):
        est = ObjectDimensionEstimator()
        img0 = make_image_with_red_object(obj_x=200, obj_y=200, obj_w=60, obj_h=80)
        img1 = make_image_with_red_object(obj_x=300, obj_y=250, obj_w=50, obj_h=50)
        results = est.estimate_from_frames(cam0_frame=img0, cam1_frame=img1)
        assert len(results) >= 1
        red_est = [r for r in results if r.label == "red"]
        assert len(red_est) >= 1
        # Dual camera should have higher base confidence
        assert red_est[0].source == "both"
        assert red_est[0].height_mm > 0

    def test_grading_letter_grades(self):
        est = ObjectDimensionEstimator()
        # Manually create an estimate and grade it
        dim_est = DimensionEstimate(
            label="red",
            width_mm=40.0,
            height_mm=100.0,
            depth_mm=40.0,
            confidence=0.55,
            source="both",
        )
        est._grade(dim_est)
        # Within gripper range, both cameras, plausible dims => decent grade
        assert dim_est.grade in ("A", "B", "C")
        assert dim_est.confidence > 0

    def test_grading_penalizes_implausible(self):
        est = ObjectDimensionEstimator()
        dim_est = DimensionEstimate(
            label="red",
            width_mm=1.0,  # too small
            height_mm=500.0,  # too tall
            depth_mm=40.0,
            confidence=0.55,
            source="both",
        )
        est._grade(dim_est)
        assert dim_est.confidence < 0.2
        assert dim_est.grade in ("D", "F")

    def test_grading_penalizes_no_height(self):
        est = ObjectDimensionEstimator()
        dim_est = DimensionEstimate(
            label="red",
            width_mm=40.0,
            height_mm=0.0,  # no height
            depth_mm=40.0,
            confidence=0.3,
            source="cam1",
        )
        est._grade(dim_est)
        assert "no height estimate" in " ".join(dim_est.grade_reasons)

    def test_multiframe_consistency_boost(self):
        est = ObjectDimensionEstimator()
        img0 = make_image_with_red_object(obj_x=200, obj_y=200, obj_w=60, obj_h=80)
        img1 = make_image_with_red_object(obj_x=300, obj_y=250, obj_w=50, obj_h=50)

        # Run multiple frames
        for _ in range(4):
            est.estimate_from_frames(cam0_frame=img0, cam1_frame=img1)

        best = est.get_best_estimates()
        assert len(best) >= 1
        red_best = [e for e in best if e.label == "red"]
        assert len(red_best) >= 1
        assert red_best[0].frame_count >= 4
        # Consistent measurements should have lower variance
        assert red_best[0].variance_mm >= 0

    def test_get_best_estimates_sorted_by_confidence(self):
        est = ObjectDimensionEstimator()
        img0 = make_image_with_red_object()
        img1 = make_image_with_blue_object()
        est.estimate_from_frames(cam0_frame=img0, cam1_frame=img1)

        best = est.get_best_estimates()
        if len(best) >= 2:
            assert best[0].confidence >= best[1].confidence

    def test_clear_history(self):
        est = ObjectDimensionEstimator()
        img = make_image_with_red_object()
        est.estimate_from_frames(cam1_frame=img)
        assert len(est._history) > 0
        est.clear_history()
        assert len(est._history) == 0

    def test_dimension_estimate_to_dict(self):
        dim_est = DimensionEstimate(
            label="red",
            width_mm=40.0,
            height_mm=100.0,
            depth_mm=40.0,
            confidence=0.7,
            grade="B",
            source="both",
        )
        d = dim_est.to_dict()
        assert d["label"] == "red"
        assert d["width_mm"] == 40.0
        assert d["graspable"] is True
        assert "volume_mm3" in d

    def test_graspable_property(self):
        small = DimensionEstimate(label="x", width_mm=30.0, height_mm=50.0, depth_mm=30.0)
        assert small.graspable is True

        big = DimensionEstimate(label="x", width_mm=100.0, height_mm=50.0, depth_mm=100.0)
        assert big.graspable is False

        tiny = DimensionEstimate(label="x", width_mm=1.0, height_mm=50.0, depth_mm=1.0)
        assert tiny.graspable is False

    def test_history_summary(self):
        est = ObjectDimensionEstimator()
        img = make_image_with_red_object()
        est.estimate_from_frames(cam1_frame=img)
        summary = est.get_history_summary()
        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# WorldModel tests
# ---------------------------------------------------------------------------


class TestWorldModel:
    def _make_estimate(self, label="red", w=40, h=100, d=40, conf=0.5, grade="C", source="both"):
        return DimensionEstimate(
            label=label,
            width_mm=w,
            height_mm=h,
            depth_mm=d,
            confidence=conf,
            grade=grade,
            source=source,
            bbox_cam1=(100, 100, 50, 50),
        )

    def test_empty_model(self):
        model = WorldModel()
        snap = model.snapshot()
        assert snap.objects == []
        assert snap.model_confidence == 0.0
        assert snap.free_zone_pct == 100.0

    def test_update_adds_objects(self):
        model = WorldModel()
        estimates = [self._make_estimate("red"), self._make_estimate("blue")]
        model.update(estimates)
        snap = model.snapshot()
        assert len(snap.objects) == 2

    def test_update_filters_low_grade(self):
        model = WorldModel()
        bad_est = self._make_estimate("red", conf=0.01, grade="F")
        model.update([bad_est])
        snap = model.snapshot()
        assert len(snap.objects) == 0

    def test_reachability_classification(self):
        model = WorldModel()
        assert model._classify_reach(300.0) == ReachStatus.REACHABLE
        assert model._classify_reach(50.0) == ReachStatus.TOO_CLOSE
        assert model._classify_reach(600.0) == ReachStatus.OUT_OF_RANGE
        assert model._classify_reach(500.0) == ReachStatus.MARGINAL

    def test_category_classification(self):
        model = WorldModel()
        small_graspable = self._make_estimate("red", w=40, h=100, d=40)
        cat = model._classify_category(small_graspable, ReachStatus.REACHABLE)
        assert cat == ObjectCategory.TARGET

        big_obj = self._make_estimate("wall", w=300, h=200, d=300)
        cat = model._classify_category(big_obj, ReachStatus.REACHABLE)
        assert cat == ObjectCategory.BOUNDARY

    def test_collision_check(self):
        model = WorldModel()
        est = self._make_estimate("red", w=40, h=100, d=40, conf=0.6, grade="C")
        positions = {"red": np.array([200.0, 0.0, 50.0])}
        model.update([est], positions)

        # Point right on the object should collide
        hits = model.check_collision(np.array([200.0, 0.0, 50.0]))
        assert len(hits) >= 1

        # Point far away should not
        hits = model.check_collision(np.array([-300.0, -300.0, 0.0]))
        assert len(hits) == 0

    def test_path_collision_check(self):
        model = WorldModel()
        est = self._make_estimate("red", conf=0.6, grade="C")
        positions = {"red": np.array([200.0, 0.0, 50.0])}
        model.update([est], positions)

        path = [
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 50.0]),  # hits the object
            np.array([400.0, 0.0, 0.0]),
        ]
        events = model.check_path_collisions(path)
        assert len(events) >= 1
        assert events[0]["path_index"] == 1

    def test_get_reachable_targets(self):
        model = WorldModel()
        est = self._make_estimate("red", conf=0.6, grade="C")
        positions = {"red": np.array([200.0, 0.0, 50.0])}
        model.update([est], positions)

        targets = model.get_reachable_targets()
        assert len(targets) >= 1
        assert targets[0].label == "red"

    def test_get_obstacles(self):
        model = WorldModel()
        big = self._make_estimate("wall", w=300, h=200, d=300, conf=0.6, grade="C")
        positions = {"wall": np.array([200.0, 0.0, 100.0])}
        model.update([big], positions)

        obstacles = model.get_obstacles()
        assert len(obstacles) >= 1

    def test_snapshot_to_dict(self):
        model = WorldModel()
        est = self._make_estimate("red", conf=0.6, grade="C")
        model.update([est])
        snap = model.snapshot()
        d = snap.to_dict()
        assert "objects" in d
        assert "model_confidence" in d
        assert "model_grade" in d

    def test_clear_resets_everything(self):
        model = WorldModel()
        est = self._make_estimate("red", conf=0.6, grade="C")
        model.update([est])
        model.clear()
        snap = model.snapshot()
        assert len(snap.objects) == 0
        assert snap.scan_count == 0

    def test_smoothed_position_updates(self):
        model = WorldModel()
        est = self._make_estimate("red", conf=0.6, grade="C")
        pos1 = {"red": np.array([200.0, 0.0, 50.0])}
        model.update([est], pos1)

        pos2 = {"red": np.array([210.0, 5.0, 52.0])}
        model.update([est], pos2)

        snap = model.snapshot()
        obj = [o for o in snap.objects if o.label == "red"][0]
        # Position should be smoothed between pos1 and pos2
        assert 200.0 <= obj.position_mm[0] <= 210.0

    def test_world_object_to_dict(self):
        obj = WorldObject(
            object_id="red_0",
            label="red",
            position_mm=np.array([200.0, 0.0, 50.0]),
            dimensions_mm=np.array([40.0, 100.0, 40.0]),
            confidence=0.6,
            grade="C",
            category=ObjectCategory.TARGET,
            reach_status=ReachStatus.REACHABLE,
            reach_distance_mm=200.0,
            graspable=True,
            safety_min_mm=np.array([150.0, -50.0, 0.0]),
            safety_max_mm=np.array([250.0, 50.0, 100.0]),
        )
        d = obj.to_dict()
        assert d["object_id"] == "red_0"
        assert d["category"] == "target"
        assert d["reach_status"] == "reachable"


# ---------------------------------------------------------------------------
# StartupScanner tests
# ---------------------------------------------------------------------------


class TestStartupScanner:
    def test_init_defaults(self):
        scanner = StartupScanner()
        assert scanner.phase == ScanPhase.WAITING_FOR_CAMERAS
        assert not scanner.is_running

    def test_scan_with_mock_cameras(self):
        img0 = make_image_with_red_object(obj_x=200, obj_y=200, obj_w=60, obj_h=80)
        img1 = make_image_with_red_object(obj_x=300, obj_y=250, obj_w=50, obj_h=50)

        cam0 = MockFrameProvider(frame=img0, connected=True)
        cam1 = MockFrameProvider(frame=img1, connected=True)

        scanner = StartupScanner(cam0=cam0, cam1=cam1)
        scanner.INITIAL_BURST_COUNT = 2
        scanner.BURST_FRAME_DELAY_S = 0.05
        scanner.REFINEMENT_INTERVAL_S = 0.1
        scanner.MAX_REFINEMENT_SCANS = 2
        scanner.CAMERA_TIMEOUT_S = 2.0

        model_ready = threading.Event()

        def on_ready(snap):
            model_ready.set()

        scanner.on_model_ready(on_ready)
        scanner.start()

        # Wait for completion
        model_ready.wait(timeout=10.0)
        assert model_ready.is_set(), "Model ready callback was not fired"

        # Wait for scanner to finish
        deadline = time.monotonic() + 15.0
        while scanner.is_running and time.monotonic() < deadline:
            time.sleep(0.1)

        report = scanner.get_report()
        assert report.scans_completed >= 2
        assert report.phase in (ScanPhase.COMPLETE, ScanPhase.CONTINUOUS_REFINEMENT)

        model = scanner.get_world_model()
        snap = model.snapshot()
        assert snap.scan_count >= 1

    def test_scan_no_cameras(self):
        scanner = StartupScanner(cam0=None, cam1=None)
        scanner.CAMERA_TIMEOUT_S = 1.0
        scanner.start()

        deadline = time.monotonic() + 5.0
        while scanner.is_running and time.monotonic() < deadline:
            time.sleep(0.1)

        assert scanner.phase == ScanPhase.ERROR

    def test_scan_one_camera_only(self):
        img1 = make_image_with_red_object()
        cam1 = MockFrameProvider(frame=img1, connected=True)

        scanner = StartupScanner(cam0=None, cam1=cam1)
        scanner.INITIAL_BURST_COUNT = 1
        scanner.BURST_FRAME_DELAY_S = 0.01
        scanner.REFINEMENT_INTERVAL_S = 0.1
        scanner.MAX_REFINEMENT_SCANS = 1
        scanner.CAMERA_TIMEOUT_S = 2.0

        scanner.start()
        deadline = time.monotonic() + 10.0
        while scanner.is_running and time.monotonic() < deadline:
            time.sleep(0.1)

        report = scanner.get_report()
        assert report.scans_completed >= 1

    def test_report_to_dict(self):
        scanner = StartupScanner()
        report = scanner.get_report()
        d = report.to_dict()
        assert "phase" in d
        assert "scans_completed" in d
        assert "world_model" in d

    def test_stop_while_running(self):
        cam0 = MockFrameProvider(frame=make_image_with_red_object(), connected=True)
        cam1 = MockFrameProvider(frame=make_image_with_red_object(), connected=True)

        scanner = StartupScanner(cam0=cam0, cam1=cam1)
        scanner.REFINEMENT_INTERVAL_S = 0.5
        scanner.MAX_REFINEMENT_SCANS = 100  # lots of scans
        scanner.start()
        time.sleep(0.5)
        scanner.stop()
        assert not scanner.is_running
