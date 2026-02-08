"""Tests for DualCameraArmTracker with independent cameras."""

import time

import cv2
import numpy as np
import pytest

from src.vision.arm_tracker import DualCameraArmTracker, TrackedObject, ArmTrackingResult
from src.vision.calibration import CameraCalibration


def make_red_circle_image(
    cx=320, cy=240, radius=30, image_size=(640, 480)
) -> np.ndarray:
    """Create a test image with a red circle (detectable as 'red')."""
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 180
    cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)  # BGR red
    return img


def make_blue_circle_image(
    cx=320, cy=240, radius=30, image_size=(640, 480)
) -> np.ndarray:
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 180
    cv2.circle(img, (cx, cy), radius, (255, 0, 0), -1)  # BGR blue
    return img


class TestTrackedObject:
    def test_creation(self):
        obj = TrackedObject(
            label="test",
            position_mm=np.array([100.0, 200.0, 50.0]),
            confidence=0.9,
        )
        assert obj.label == "test"
        assert obj.source == "both"
        assert obj.timestamp > 0

    def test_defaults(self):
        obj = TrackedObject(
            label="x",
            position_mm=np.zeros(3),
            confidence=0.5,
        )
        assert obj.bbox_cam0 is None
        assert obj.bbox_cam1 is None


class TestDualCameraArmTracker:
    def test_init_no_args(self):
        tracker = DualCameraArmTracker()
        assert tracker.tracking_count == 0
        assert tracker.get_last_result() is None

    def test_init_with_calibration(self):
        cal0 = CameraCalibration(camera_id="cam0")
        cal1 = CameraCalibration(camera_id="cam1")
        tracker = DualCameraArmTracker(cal_cam0=cal0, cal_cam1=cal1)
        assert tracker.cal_cam0 is cal0
        assert tracker.cal_cam1 is cal1

    def test_set_calibration(self):
        tracker = DualCameraArmTracker()
        cal = CameraCalibration(camera_id="cam0")
        tracker.set_calibration(cal_cam0=cal)
        assert tracker.cal_cam0 is cal

    def test_track_blank_images(self):
        tracker = DualCameraArmTracker()
        cam0 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cam1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = tracker.track(cam0, cam1, target_label="red")
        assert isinstance(result, ArmTrackingResult)
        assert result.status == "ok"
        assert len(result.objects) == 0
        assert tracker.tracking_count == 1

    def test_track_red_object_both_cameras(self):
        tracker = DualCameraArmTracker()
        cam0 = make_red_circle_image(cx=300, cy=200, radius=40)
        cam1 = make_red_circle_image(cx=350, cy=250, radius=35)
        result = tracker.track(cam0, cam1, target_label="red")
        assert result.status == "ok"
        # Should detect red in both views
        if len(result.objects) > 0:
            obj = result.objects[0]
            assert obj.label in ("red_low", "red_high", "red")
            assert obj.confidence > 0

    def test_track_returns_annotated_frames(self):
        tracker = DualCameraArmTracker()
        cam0 = make_red_circle_image()
        cam1 = make_red_circle_image()
        result = tracker.track(cam0, cam1, target_label="red", annotate=True)
        assert result.cam0_frame is not None
        assert result.cam1_frame is not None

    def test_track_no_annotate(self):
        tracker = DualCameraArmTracker()
        cam0 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cam1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = tracker.track(cam0, cam1, annotate=False)
        assert result.cam0_frame is None
        assert result.cam1_frame is None

    def test_tracking_count_increments(self):
        tracker = DualCameraArmTracker()
        cam = np.ones((480, 640, 3), dtype=np.uint8) * 128
        tracker.track(cam, cam)
        tracker.track(cam, cam)
        assert tracker.tracking_count == 2

    def test_elapsed_ms(self):
        tracker = DualCameraArmTracker()
        cam = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = tracker.track(cam, cam)
        assert result.elapsed_ms >= 0

    def test_different_detectors(self):
        tracker = DualCameraArmTracker()
        cam = np.ones((480, 640, 3), dtype=np.uint8) * 128
        for label in ["red", "blue", "green", "redbull", "all"]:
            result = tracker.track(cam, cam, target_label=label)
            assert result.status == "ok"
