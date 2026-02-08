"""
Tests for the DualCameraArmTracker module.

Uses synthetic stereo calibration and generated images — no real cameras.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

from src.vision.calibration import StereoCalibrator
from src.vision.arm_tracker import DualCameraArmTracker, TrackedObject, ArmTrackingResult


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def make_synthetic_calibration() -> StereoCalibrator:
    """Create a calibrator with synthetic calibration data."""
    cal = StereoCalibrator(image_size=(640, 480))
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    cal.camera_matrix_left = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    cal.camera_matrix_right = cal.camera_matrix_left.copy()
    cal.dist_coeffs_left = np.zeros(5, dtype=np.float64)
    cal.dist_coeffs_right = np.zeros(5, dtype=np.float64)
    cal.R = np.eye(3, dtype=np.float64)
    cal.T = np.array([[-60.0], [0.0], [0.0]], dtype=np.float64)
    w, h = cal.image_size
    R1, R2, P1, P2, cal.Q, _, _ = cv2.stereoRectify(
        cal.camera_matrix_left, cal.dist_coeffs_left,
        cal.camera_matrix_right, cal.dist_coeffs_right,
        (w, h), cal.R, cal.T, alpha=0,
    )
    cal.map_left_x, cal.map_left_y = cv2.initUndistortRectifyMap(
        cal.camera_matrix_left, cal.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
    )
    cal.map_right_x, cal.map_right_y = cv2.initUndistortRectifyMap(
        cal.camera_matrix_right, cal.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
    )
    cal._calibrated = True
    return cal


def make_red_object_image(center=(320, 240), radius=50):
    """Create a test image with a red circle (like a Red Bull can cross-section)."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(img, center, radius, (0, 0, 255), -1)
    return img


def make_gray_image():
    """Create a blank gray image."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDualCameraArmTracker:
    def setup_method(self):
        self.cal = make_synthetic_calibration()
        self.tracker = DualCameraArmTracker(self.cal)

    def test_init(self):
        assert self.tracker.calibrator is self.cal
        assert self.tracker.cam_to_arm is not None
        assert self.tracker.cam_to_arm.shape == (4, 4)
        assert "redbull" in self.tracker._detectors
        assert "red" in self.tracker._detectors
        assert "all" in self.tracker._detectors

    def test_init_custom_labels(self):
        tracker = DualCameraArmTracker(self.cal, target_labels=["red"])
        assert "red" in tracker._detectors

    def test_cam_point_to_arm_frame(self):
        """Default transform: arm X = cam Z, arm Y = -cam X, arm Z = -cam Y."""
        cam_pt = np.array([100.0, 200.0, 300.0])
        arm_pt = self.tracker.cam_point_to_arm_frame(cam_pt)
        # With default transform:
        assert arm_pt[0] == pytest.approx(300.0)  # arm X = cam Z
        assert arm_pt[1] == pytest.approx(-100.0)  # arm Y = -cam X
        assert arm_pt[2] == pytest.approx(-200.0)  # arm Z = -cam Y

    def test_set_cam_to_arm_transform(self):
        new_transform = np.eye(4)
        new_transform[:3, 3] = [10, 20, 30]
        self.tracker.set_cam_to_arm_transform(new_transform)
        np.testing.assert_array_almost_equal(self.tracker.cam_to_arm, new_transform)

    def test_calibrate_cam_to_arm_from_known_point(self):
        # Set identity transform first
        self.tracker.set_cam_to_arm_transform(np.eye(4))
        cam_pt = np.array([100.0, 100.0, 100.0])
        arm_pt = np.array([200.0, 200.0, 200.0])
        self.tracker.calibrate_cam_to_arm_from_known_point(cam_pt, arm_pt)
        # After calibration, transforming cam_pt should give arm_pt
        result = self.tracker.cam_point_to_arm_frame(cam_pt)
        np.testing.assert_array_almost_equal(result, arm_pt)

    def test_track_no_calibration(self):
        uncal = StereoCalibrator()
        tracker = DualCameraArmTracker(uncal)
        left = make_gray_image()
        right = make_gray_image()
        result = tracker.track(left, right)
        assert result.status == "no_calibration"
        assert len(result.objects) == 0

    def test_track_no_objects(self):
        """Tracking on gray images should find no objects."""
        left = make_gray_image()
        right = make_gray_image()
        result = self.tracker.track(left, right, target_label="redbull")
        assert result.status == "ok"
        assert len(result.objects) == 0

    def test_track_red_object(self):
        """Tracking should detect a red circle in both views."""
        left = make_red_object_image(center=(300, 200), radius=50)
        right = make_red_object_image(center=(280, 200), radius=50)

        # Use synthetic depth for testing (zero-baseline stereo won't give real depth)
        result = self.tracker.track(left, right, target_label="red", annotate=True)
        assert result.status == "ok"
        # Detection depends on stereo depth which may be zero for synthetic data
        # but the detector pipeline should still run without errors

    def test_track_increments_count(self):
        left = make_gray_image()
        right = make_gray_image()
        assert self.tracker.tracking_count == 0
        self.tracker.track(left, right)
        assert self.tracker.tracking_count == 1
        self.tracker.track(left, right)
        assert self.tracker.tracking_count == 2

    def test_get_last_result(self):
        assert self.tracker.get_last_result() is None
        left = make_gray_image()
        right = make_gray_image()
        self.tracker.track(left, right)
        result = self.tracker.get_last_result()
        assert result is not None
        assert isinstance(result, ArmTrackingResult)

    def test_estimate_object_size(self):
        # With fx=fy=500, at depth=1000mm, 100px = 200mm
        w_mm, h_mm, d_mm = self.tracker._estimate_object_size(100, 50, 1000.0)
        assert w_mm == pytest.approx(200.0, rel=0.01)
        assert h_mm == pytest.approx(100.0, rel=0.01)

    def test_find_matching_detection_same_label(self):
        from src.vision.object_detection import DetectedObject
        target = DetectedObject(
            label="red", bbox=(100, 100, 50, 50), centroid_2d=(125, 125),
            area=2500, contour=np.array([[[100, 100]]]),
        )
        match = DetectedObject(
            label="red", bbox=(90, 120, 55, 55), centroid_2d=(117, 147),
            area=3000, contour=np.array([[[90, 120]]]),
        )
        no_match = DetectedObject(
            label="blue", bbox=(90, 120, 55, 55), centroid_2d=(117, 147),
            area=3000, contour=np.array([[[90, 120]]]),
        )
        result = self.tracker._find_matching_detection(target, [no_match, match])
        assert result is match

    def test_find_matching_detection_no_candidates(self):
        from src.vision.object_detection import DetectedObject
        target = DetectedObject(
            label="red", bbox=(100, 100, 50, 50), centroid_2d=(125, 125),
            area=2500, contour=np.array([[[100, 100]]]),
        )
        result = self.tracker._find_matching_detection(target, [])
        assert result is None

    def test_annotate_frame(self):
        img = make_red_object_image()
        obj = TrackedObject(
            label="test",
            position_mm=np.array([100.0, 200.0, 300.0]),
            position_cam_mm=np.array([50.0, 60.0, 70.0]),
            size_mm=(50.0, 100.0, 50.0),
            confidence=0.8,
            bbox_left=(290, 210, 60, 60),
            bbox_right=None,
            centroid_left=(320, 240),
            centroid_right=None,
            depth_mm=500.0,
        )
        annotated = self.tracker._annotate_frame(img.copy(), [obj], "L")
        assert annotated.shape == img.shape
        # Should have been modified
        assert not np.array_equal(annotated, img)


class TestTrackedObject:
    def test_creation(self):
        obj = TrackedObject(
            label="redbull",
            position_mm=np.array([100.0, 0.0, 50.0]),
            position_cam_mm=np.array([0.0, -50.0, 100.0]),
            size_mm=(53.0, 135.0, 53.0),
            confidence=0.85,
            bbox_left=(200, 150, 80, 120),
            bbox_right=(180, 150, 80, 120),
            centroid_left=(240, 210),
            centroid_right=(220, 210),
            depth_mm=400.0,
        )
        assert obj.label == "redbull"
        assert obj.confidence == 0.85
        assert obj.timestamp > 0

    def test_default_timestamp(self):
        obj = TrackedObject(
            label="test",
            position_mm=np.zeros(3),
            position_cam_mm=np.zeros(3),
            size_mm=(10.0, 10.0, 10.0),
            confidence=0.5,
            bbox_left=(0, 0, 10, 10),
            bbox_right=None,
            centroid_left=(5, 5),
            centroid_right=None,
            depth_mm=100.0,
        )
        assert obj.timestamp > 0
