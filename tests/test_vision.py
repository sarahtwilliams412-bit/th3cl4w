"""Tests for independent camera calibration and workspace mapper."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from src.vision.calibration import (
    CameraCalibration,
    IndependentCalibrator,
    DEFAULT_BOARD_SIZE,
    DEFAULT_SQUARE_SIZE_MM,
)
from src.vision.workspace_mapper import WorkspaceMapper


# --- Helpers ---

def make_synthetic_checkerboard(
    board_size=(7, 5),
    square_size_px=30,
    image_size=(640, 480),
    offset=(100, 80),
) -> np.ndarray:
    """Create a synthetic checkerboard image for testing."""
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 200
    ox, oy = offset
    cols, rows = board_size[0] + 1, board_size[1] + 1
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x1 = ox + c * square_size_px
                y1 = oy + r * square_size_px
                x2 = x1 + square_size_px
                y2 = y1 + square_size_px
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return img


def make_varied_checkerboard_images(n=5, board_size=(7, 5)):
    """Generate n slightly different checkerboard images."""
    images = []
    for i in range(n):
        offset = (80 + i * 15, 60 + i * 10)
        sq_size = 28 + i * 2
        img = make_synthetic_checkerboard(
            board_size=board_size, square_size_px=sq_size, offset=offset
        )
        images.append(img)
    return images


# --- CameraCalibration tests ---

class TestCameraCalibration:
    def test_defaults(self):
        cal = CameraCalibration(camera_id="test")
        assert not cal.is_calibrated
        assert cal.fx == 500.0
        assert cal.image_size == (640, 480)

    def test_calibrated(self):
        cal = CameraCalibration(
            camera_id="test",
            camera_matrix=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.zeros(5, dtype=np.float64),
        )
        assert cal.is_calibrated
        assert cal.fx == 600.0
        assert cal.fy == 600.0
        assert cal.cx == 320.0
        assert cal.cy == 240.0

    def test_pixel_to_ray(self):
        cal = CameraCalibration(
            camera_id="test",
            camera_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.zeros(5),
        )
        # Center pixel should give forward ray
        ray = cal.pixel_to_ray(320.0, 240.0)
        assert ray.shape == (3,)
        assert abs(ray[0]) < 0.01
        assert abs(ray[1]) < 0.01
        assert ray[2] > 0.99

    def test_pixel_to_workspace_no_extrinsic(self):
        cal = CameraCalibration(camera_id="test")
        assert cal.pixel_to_workspace(320, 240) is None

    def test_pixel_to_workspace_with_extrinsic(self):
        cal = CameraCalibration(
            camera_id="test",
            camera_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.zeros(5),
        )
        # Camera looking straight down from 500mm height
        cal.cam_to_workspace = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # flip Y
            [0, 0, -1, 500],  # Z points up, camera at 500mm
            [0, 0, 0, 1],
        ], dtype=np.float64)
        pos = cal.pixel_to_workspace(320.0, 240.0, known_z=0.0)
        assert pos is not None
        # Center pixel looking straight down should hit near origin
        assert abs(pos[0]) < 1.0
        assert abs(pos[1]) < 1.0

    def test_save_load(self, tmp_path):
        cal = CameraCalibration(
            camera_id="cam0",
            camera_matrix=np.eye(3, dtype=np.float64) * 500,
            dist_coeffs=np.zeros(5, dtype=np.float64),
            reprojection_error=0.5,
        )
        path = str(tmp_path / "cal.json")
        cal.save(path)
        loaded = CameraCalibration.load(path)
        assert loaded.camera_id == "cam0"
        assert loaded.is_calibrated
        assert loaded.reprojection_error == 0.5

    def test_undistort_passthrough(self):
        cal = CameraCalibration(camera_id="test")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = cal.undistort(img)
        assert result.shape == img.shape


# --- IndependentCalibrator tests ---

class TestIndependentCalibrator:
    def test_find_corners_synthetic(self):
        calibrator = IndependentCalibrator(board_size=(7, 5))
        img = make_synthetic_checkerboard(board_size=(7, 5))
        corners = calibrator.find_corners(img)
        assert corners is not None
        assert corners.shape[0] == 7 * 5

    def test_find_corners_no_board(self):
        calibrator = IndependentCalibrator()
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        corners = calibrator.find_corners(img)
        assert corners is None

    def test_add_and_count(self):
        calibrator = IndependentCalibrator(board_size=(7, 5))
        img = make_synthetic_checkerboard(board_size=(7, 5))
        result = calibrator.add_calibration_image("cam0", img)
        assert result is not None
        assert calibrator.image_count("cam0") == 1
        assert calibrator.image_count("cam1") == 0

    def test_calibrate_insufficient_images(self):
        calibrator = IndependentCalibrator(board_size=(7, 5))
        img = make_synthetic_checkerboard(board_size=(7, 5))
        calibrator.add_calibration_image("cam0", img)
        result = calibrator.calibrate_camera("cam0", min_images=3)
        assert result is None

    def test_calibrate_camera(self):
        calibrator = IndependentCalibrator(board_size=(7, 5))
        images = make_varied_checkerboard_images(n=5, board_size=(7, 5))
        for img in images:
            calibrator.add_calibration_image("cam0", img)
        cal = calibrator.calibrate_camera("cam0", min_images=3)
        assert cal is not None
        assert cal.is_calibrated
        assert cal.camera_id == "cam0"
        assert cal.reprojection_error >= 0
        assert cal.fx > 0
        assert cal.fy > 0


# --- WorkspaceMapper tests ---

class TestWorkspaceMapper:
    def test_init_defaults(self):
        mapper = WorkspaceMapper()
        assert not mapper.enabled
        assert not mapper.scale_calibrated
        status = mapper.get_status()
        assert status["total_cells"] > 0

    def test_enable_disable(self):
        mapper = WorkspaceMapper()
        mapper.enable()
        assert mapper.enabled
        mapper.disable()
        assert not mapper.enabled
        result = mapper.toggle()
        assert result is True

    def test_check_point_unknown(self):
        mapper = WorkspaceMapper()
        assert mapper.check_point(0.0, 0.0) == "unknown"

    def test_check_point_out_of_bounds(self):
        mapper = WorkspaceMapper()
        assert mapper.check_point(9999.0, 9999.0) == "out_of_bounds"

    def test_clear(self):
        mapper = WorkspaceMapper()
        mapper.clear()
        summary = mapper.get_occupancy_summary()
        assert summary["occupied"] == 0
        assert summary["free"] == 0

    def test_get_occupied_cells_empty(self):
        mapper = WorkspaceMapper()
        assert mapper.get_occupied_cells() == []

    def test_update_disabled(self):
        mapper = WorkspaceMapper()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mapper.update_from_overhead(frame)
        assert result["status"] == "disabled"

    def test_update_no_calibration(self):
        mapper = WorkspaceMapper()
        mapper.enable()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mapper.update_from_overhead(frame)
        assert result["status"] == "not_calibrated"
