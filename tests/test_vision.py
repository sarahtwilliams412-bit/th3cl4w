"""
Tests for the stereo vision module (Phase 1).

Uses synthetic data and mocks — no real cameras required.
Requires opencv-python (cv2) — skipped if not installed.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Skip entire module if cv2 is not available (MagicMock doesn't count)
cv2 = pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

from src.vision.calibration import StereoCalibrator
from src.vision.stereo_depth import StereoDepthEstimator
from src.vision.object_detection import ObjectDetector, DetectedObject, ColorRange


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_checkerboard_image(
    board_size: tuple[int, int] = (9, 6),
    square_px: int = 40,
    img_size: tuple[int, int] = (640, 480),
    offset: tuple[int, int] = (50, 50),
) -> np.ndarray:
    """Render a synthetic checkerboard image."""
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
    cols, rows = board_size
    for r in range(rows + 1):
        for c in range(cols + 1):
            x = offset[0] + c * square_px
            y = offset[1] + r * square_px
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + square_px, y + square_px), (0, 0, 0), -1)
            else:
                cv2.rectangle(img, (x, y), (x + square_px, y + square_px), (255, 255, 255), -1)
    return img


def make_synthetic_calibration() -> StereoCalibrator:
    """Create a calibrator with synthetic calibration data (no real images)."""
    cal = StereoCalibrator(image_size=(640, 480))

    # Synthetic intrinsics (typical for 640x480)
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    cal.camera_matrix_left = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    cal.camera_matrix_right = cal.camera_matrix_left.copy()
    cal.dist_coeffs_left = np.zeros(5, dtype=np.float64)
    cal.dist_coeffs_right = np.zeros(5, dtype=np.float64)

    # 60mm baseline, cameras parallel
    cal.R = np.eye(3, dtype=np.float64)
    cal.T = np.array([[-60.0], [0.0], [0.0]], dtype=np.float64)

    # Compute rectification
    w, h = cal.image_size
    R1, R2, P1, P2, cal.Q, _, _ = cv2.stereoRectify(
        cal.camera_matrix_left, cal.dist_coeffs_left,
        cal.camera_matrix_right, cal.dist_coeffs_right,
        (w, h), cal.R, cal.T,
        alpha=0,
    )
    cal.map_left_x, cal.map_left_y = cv2.initUndistortRectifyMap(
        cal.camera_matrix_left, cal.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
    )
    cal.map_right_x, cal.map_right_y = cv2.initUndistortRectifyMap(
        cal.camera_matrix_right, cal.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
    )
    cal._calibrated = True
    return cal


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStereoCalibrator:
    def test_init_defaults(self):
        cal = StereoCalibrator()
        assert cal.board_size == (9, 6)
        assert cal.square_size == 25.0
        assert cal.image_size == (640, 480)
        assert not cal.is_calibrated

    def test_make_object_points(self):
        cal = StereoCalibrator(board_size=(4, 3), square_size=10.0)
        pts = cal._make_object_points()
        assert pts.shape == (12, 3)
        assert pts[0, 2] == 0.0  # all Z = 0
        assert np.allclose(pts[-1, :2], [30.0, 20.0])  # (3*10, 2*10)

    def test_find_corners_synthetic(self):
        cal = StereoCalibrator(board_size=(9, 6))
        img = make_checkerboard_image(board_size=(9, 6), square_px=40)
        corners = cal.find_corners(img)
        assert corners is not None
        assert corners.shape[0] == 54  # 9*6

    def test_find_corners_no_pattern(self):
        cal = StereoCalibrator()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        assert cal.find_corners(blank) is None

    def test_find_corners_grayscale(self):
        cal = StereoCalibrator(board_size=(9, 6))
        img = make_checkerboard_image(board_size=(9, 6))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cal.find_corners(gray)
        assert corners is not None

    def test_calibrate_insufficient_pairs(self):
        cal = StereoCalibrator()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        with pytest.raises(ValueError, match="at least 3"):
            cal.calibrate([(blank, blank)])

    def test_calibrate_synthetic(self):
        """Full calibration with synthetic checkerboard images at different offsets."""
        cal = StereoCalibrator(board_size=(9, 6), square_size=25.0)
        pairs = []
        offsets = [
            (50, 50), (80, 30), (30, 80), (60, 60), (40, 40),
            (70, 50), (50, 70), (90, 40), (45, 55), (65, 35),
        ]
        for ox, oy in offsets:
            left = make_checkerboard_image(board_size=(9, 6), offset=(ox, oy))
            # Same image for both (zero-baseline, but tests the math pipeline)
            right = make_checkerboard_image(board_size=(9, 6), offset=(ox, oy))
            pairs.append((left, right))

        rms = cal.calibrate(pairs)
        assert rms >= 0
        assert cal.is_calibrated
        assert cal.camera_matrix_left is not None
        assert cal.Q is not None

    def test_save_load_roundtrip(self):
        cal = make_synthetic_calibration()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_cal.npz"
            cal.save(path)
            assert path.exists()

            cal2 = StereoCalibrator()
            cal2.load(path)
            assert cal2.is_calibrated
            np.testing.assert_array_almost_equal(cal.camera_matrix_left, cal2.camera_matrix_left)
            np.testing.assert_array_almost_equal(cal.Q, cal2.Q)
            np.testing.assert_array_almost_equal(cal.T, cal2.T)
            assert cal2.image_size == cal.image_size

    def test_save_without_calibration_raises(self):
        cal = StereoCalibrator()
        with pytest.raises(RuntimeError):
            cal.save("/tmp/nope.npz")

    def test_rectify_without_calibration_raises(self):
        cal = StereoCalibrator()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            cal.rectify(img, img)

    def test_rectify_produces_output(self):
        cal = make_synthetic_calibration()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        left, right = cal.rectify(img, img)
        assert left.shape == (480, 640, 3)
        assert right.shape == (480, 640, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Stereo Depth Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStereoDepthEstimator:
    def setup_method(self):
        self.cal = make_synthetic_calibration()
        self.depth_est = StereoDepthEstimator(self.cal)

    def test_init(self):
        assert self.depth_est.stereo is not None
        assert self.depth_est.num_disparities == 128

    def test_compute_disparity_shape(self):
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        disp = self.depth_est.compute_disparity(left, right, rectify=True)
        assert disp.shape == (480, 640)
        assert disp.dtype == np.float32

    def test_compute_disparity_no_rectify(self):
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        disp = self.depth_est.compute_disparity(left, right, rectify=False)
        assert disp.shape == (480, 640)

    def test_disparity_to_depth(self):
        # Synthetic disparity: uniform 10 pixels
        disp = np.full((480, 640), 10.0, dtype=np.float32)
        depth = self.depth_est.disparity_to_depth(disp)
        assert depth.shape == (480, 640)
        # With our synthetic cal: focal=500, baseline=60mm
        # depth = focal * baseline / disparity = 500 * 60 / 10 = 3000mm
        # Via Q: depth = -Q[2,3] / (Q[3,2] * disp)
        expected = self.cal.Q[2, 3] / (self.cal.Q[3, 2] * 10.0)
        assert depth[240, 320] == pytest.approx(expected, rel=0.01)

    def test_disparity_to_depth_invalid(self):
        disp = np.full((480, 640), -1.0, dtype=np.float32)
        depth = self.depth_est.disparity_to_depth(disp)
        assert np.all(depth == 0)

    def test_disparity_to_depth_no_Q(self):
        self.cal.Q = None
        with pytest.raises(RuntimeError):
            self.depth_est.disparity_to_depth(np.zeros((480, 640), dtype=np.float32))

    def test_compute_depth_returns_both(self):
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        disp, depth = self.depth_est.compute_depth(left, right)
        assert disp.shape == (480, 640)
        assert depth.shape == (480, 640)

    def test_get_depth_at(self):
        depth_map = np.full((480, 640), 1500.0, dtype=np.float32)
        depth_map[100, 100] = 0  # one invalid pixel
        assert self.depth_est.get_depth_at(depth_map, 320, 240) == pytest.approx(1500.0)

    def test_get_depth_at_all_invalid(self):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        assert self.depth_est.get_depth_at(depth_map, 320, 240) == 0.0

    def test_get_depth_at_edge(self):
        depth_map = np.full((480, 640), 500.0, dtype=np.float32)
        assert self.depth_est.get_depth_at(depth_map, 0, 0, window=3) == pytest.approx(500.0)

    def test_compute_point_cloud(self):
        disp = np.full((480, 640), 10.0, dtype=np.float32)
        points = self.depth_est.compute_point_cloud(disp)
        assert points.ndim == 2
        assert points.shape[1] == 3

    def test_compute_point_cloud_with_color(self):
        disp = np.full((480, 640), 10.0, dtype=np.float32)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        points = self.depth_est.compute_point_cloud(disp, left_image=img)
        assert points.shape[1] == 6  # XYZ + BGR

    def test_compute_point_cloud_no_Q(self):
        self.cal.Q = None
        with pytest.raises(RuntimeError):
            self.depth_est.compute_point_cloud(np.zeros((480, 640), dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# Object Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

def make_colored_object_image(
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    center: tuple[int, int] = (320, 240),
    radius: int = 50,
) -> np.ndarray:
    """Create a test image with a colored circle on gray background."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(img, center, radius, color_bgr, -1)
    return img


class TestObjectDetector:
    def test_init_defaults(self):
        det = ObjectDetector()
        assert len(det.color_ranges) == 4  # red_low, red_high, green, blue
        assert det.min_area == 500.0

    def test_detect_red_circle(self):
        img = make_colored_object_image(color_bgr=(0, 0, 255), center=(300, 200), radius=60)
        det = ObjectDetector(min_area=100)
        results = det.detect(img)
        assert len(results) >= 1
        obj = results[0]
        assert obj.label == "red"
        # Centroid should be near (300, 200)
        assert abs(obj.centroid_2d[0] - 300) < 15
        assert abs(obj.centroid_2d[1] - 200) < 15

    def test_detect_green_circle(self):
        img = make_colored_object_image(color_bgr=(0, 255, 0), center=(400, 300), radius=40)
        det = ObjectDetector(min_area=100)
        results = det.detect(img)
        assert len(results) >= 1
        assert results[0].label == "green"

    def test_detect_blue_circle(self):
        img = make_colored_object_image(color_bgr=(255, 0, 0), center=(200, 150), radius=45)
        det = ObjectDetector(min_area=100)
        results = det.detect(img)
        assert len(results) >= 1
        assert results[0].label == "blue"

    def test_detect_nothing_on_gray(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        det = ObjectDetector()
        results = det.detect(img)
        assert len(results) == 0

    def test_detect_with_custom_color(self):
        # Yellow object
        img = make_colored_object_image(color_bgr=(0, 255, 255), center=(320, 240), radius=50)
        yellow = ColorRange("yellow", [20, 100, 100], [35, 255, 255])
        det = ObjectDetector(color_ranges=[yellow], min_area=100)
        results = det.detect(img)
        assert len(results) >= 1
        assert results[0].label == "yellow"

    def test_detect_filters_small_objects(self):
        img = make_colored_object_image(color_bgr=(0, 0, 255), radius=5)  # tiny
        det = ObjectDetector(min_area=500)
        results = det.detect(img)
        assert len(results) == 0

    def test_detect_with_depth(self):
        img = make_colored_object_image(color_bgr=(0, 0, 255), center=(320, 240), radius=50)
        depth = np.full((480, 640), 1000.0, dtype=np.float32)
        cal = make_synthetic_calibration()

        det = ObjectDetector(min_area=100)
        results = det.detect(img, depth_map=depth, Q=cal.Q)
        assert len(results) >= 1
        obj = results[0]
        assert obj.depth_mm == pytest.approx(1000.0, rel=0.1)
        assert obj.centroid_3d is not None
        assert obj.centroid_3d[2] == pytest.approx(1000.0, rel=0.1)

    def test_detect_sorted_by_area(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.circle(img, (200, 200), 30, (0, 0, 255), -1)  # small red
        cv2.circle(img, (400, 300), 80, (0, 0, 255), -1)  # big red
        det = ObjectDetector(min_area=100)
        results = det.detect(img)
        assert len(results) >= 2
        assert results[0].area >= results[1].area

    def test_draw_detections(self):
        img = make_colored_object_image(color_bgr=(0, 0, 255))
        det = ObjectDetector(min_area=100)
        results = det.detect(img)
        vis = det.draw_detections(img, results)
        assert vis.shape == img.shape
        # Original should be unchanged
        assert not np.array_equal(vis, img) or len(results) == 0

    def test_draw_detections_empty(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        det = ObjectDetector()
        vis = det.draw_detections(img, [])
        np.testing.assert_array_equal(vis, img)

    def test_pixel_to_3d(self):
        cal = make_synthetic_calibration()
        det = ObjectDetector()
        # With synthetic Q, cx=320, cy=240, f=500
        x3d = det._pixel_to_3d(320, 240, 1000.0, cal.Q)
        # At principal point, X and Y should be ~0
        assert abs(x3d[0]) < 1.0
        assert abs(x3d[1]) < 1.0
        assert x3d[2] == pytest.approx(1000.0)

    def test_get_depth_median(self):
        det = ObjectDetector()
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[238:243, 318:323] = 500.0
        d = det._get_depth(depth, 320, 240, window=5)
        assert d == pytest.approx(500.0)


# ═══════════════════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_synthetic(self):
        """End-to-end: calibration → depth → detection."""
        cal = make_synthetic_calibration()
        depth_est = StereoDepthEstimator(cal)

        # Create a stereo pair (identical for zero-baseline)
        img = make_colored_object_image(color_bgr=(0, 0, 255), center=(320, 240), radius=50)
        left = img.copy()
        right = img.copy()

        # Compute disparity and depth
        disp, depth = depth_est.compute_depth(left, right)
        assert disp.shape == (480, 640)

        # Detect objects
        detector = ObjectDetector(min_area=100)
        # Use a synthetic depth map since zero-baseline gives no real disparity
        fake_depth = np.full((480, 640), 800.0, dtype=np.float32)
        results = detector.detect(left, depth_map=fake_depth, Q=cal.Q)
        assert len(results) >= 1
        assert results[0].label == "red"
        assert results[0].depth_mm > 0
        assert results[0].centroid_3d is not None
