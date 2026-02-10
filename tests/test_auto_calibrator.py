"""Tests for AutoCalibrator — uses synthetic checkerboard images."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.calibration.auto_calibrator import AutoCalibrator, CalibrationResult


def generate_checkerboard_image(
    board_size=(8, 5),
    square_px=40,
    image_size=(640, 480),
    offset=(80, 60),
):
    """Generate a synthetic checkerboard image with known geometry."""
    img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200
    cols, rows = board_size[0] + 1, board_size[1] + 1  # outer squares
    ox, oy = offset
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x1 = ox + c * square_px
                y1 = oy + r * square_px
                x2 = x1 + square_px
                y2 = y1 + square_px
                cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def generate_varied_checkerboards(board_size=(8, 5), n=10, image_size=(640, 480)):
    """Generate multiple checkerboard images with slight position variation."""
    frames = []
    rng = np.random.RandomState(42)
    for i in range(n):
        ox = 40 + rng.randint(0, 80)
        oy = 30 + rng.randint(0, 60)
        sq = 30 + rng.randint(0, 15)
        frames.append(generate_checkerboard_image(board_size, sq, image_size, (ox, oy)))
    return frames


class TestDetection:
    def test_detect_default_board(self):
        cal = AutoCalibrator(board_size=(8, 5))
        img = generate_checkerboard_image((8, 5))
        corners = cal.detect_checkerboard(img)
        assert corners is not None
        assert corners.shape[0] == 8 * 5

    def test_detect_different_board(self):
        cal = AutoCalibrator(board_size=(7, 4))
        img = generate_checkerboard_image((7, 4))
        corners = cal.detect_checkerboard(img, board_size=(7, 4))
        assert corners is not None
        assert corners.shape[0] == 7 * 4

    def test_detect_auto_fallback(self):
        """If primary board size fails, auto-detect should try fallbacks."""
        cal = AutoCalibrator(board_size=(9, 6), auto_detect_board=True)
        # Image has a (7,4) board — should be found via fallback
        img = generate_checkerboard_image((7, 4), square_px=35)
        corners = cal.detect_checkerboard(img)
        assert corners is not None
        assert cal._last_detected_size == (7, 4)

    def test_detect_no_board(self):
        cal = AutoCalibrator()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        corners = cal.detect_checkerboard(blank)
        assert corners is None


class TestCalibration:
    def test_calibrate_synthetic(self):
        """Synthetic front-parallel boards calibrate (structure check, not accuracy).

        Note: Front-parallel synthetic boards produce degenerate focal lengths.
        Real calibration needs boards at varied angles. We just verify the
        pipeline runs and returns the right structure.
        """
        cal = AutoCalibrator(board_size=(8, 5), square_size_mm=19.0)
        cal.progress.camera_id = 0
        frames = generate_varied_checkerboards((8, 5), n=8)
        result = cal.calibrate(frames)

        assert isinstance(result, CalibrationResult)
        assert result.fx > 0
        assert result.fy > 0
        assert result.image_size == (640, 480)
        assert result.num_detected >= 3
        assert result.camera_matrix is not None
        assert result.dist_coeffs is not None

    def test_calibrate_insufficient_frames(self):
        cal = AutoCalibrator(board_size=(8, 5))
        cal.progress.camera_id = 0
        # Provide blank frames — no corners
        frames = [np.ones((480, 640, 3), dtype=np.uint8) * 128] * 5
        with pytest.raises(ValueError, match="need ≥3"):
            cal.calibrate(frames)


class TestQuality:
    def test_quality_warnings_fx_fy(self):
        cal = AutoCalibrator()
        result = CalibrationResult(
            camera_id=0,
            rms=0.1,
            fx=1400,
            fy=1000,
            cx=320,
            cy=240,
            k1=0,
            k2=0,
            p1=0,
            p2=0,
            k3=0,
            image_size=(640, 480),
            fov_h=60,
            fov_v=40,
            fov_d=70,
            board_size=(8, 5),
            square_size_mm=19,
            num_frames=10,
            num_detected=10,
        )
        corners = [np.random.rand(40, 1, 2).astype(np.float32) * 300 + 100]
        warnings = cal._assess_quality(result, corners, (640, 480))
        assert any("fx/fy" in w for w in warnings)

    def test_quality_warnings_distortion(self):
        cal = AutoCalibrator()
        result = CalibrationResult(
            camera_id=0,
            rms=0.1,
            fx=500,
            fy=500,
            cx=320,
            cy=240,
            k1=0,
            k2=15,
            p1=0,
            p2=0,
            k3=-20,
            image_size=(640, 480),
            fov_h=60,
            fov_v=40,
            fov_d=70,
            board_size=(8, 5),
            square_size_mm=19,
            num_frames=10,
            num_detected=10,
        )
        corners = [np.random.rand(40, 1, 2).astype(np.float32) * 300 + 100]
        warnings = cal._assess_quality(result, corners, (640, 480))
        assert any("k2" in w for w in warnings)
        assert any("k3" in w for w in warnings)


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        cal = AutoCalibrator(board_size=(8, 5), square_size_mm=19.0)
        cal.progress.camera_id = 0
        frames = generate_varied_checkerboards((8, 5), n=8)
        result = cal.calibrate(frames)

        # Patch paths to use tmp_path
        with (
            patch("src.calibration.auto_calibrator.CALIBRATION_RESULTS_DIR", tmp_path / "cal"),
            patch("src.calibration.auto_calibrator.DATA_DIR", tmp_path / "data"),
        ):
            saved = cal.save_results(result)

        # Check per-camera file
        cam_file = tmp_path / "cal" / "cam0_checkerboard_calibration.json"
        assert cam_file.exists()
        loaded = json.loads(cam_file.read_text())
        assert loaded["camera_matrix"]["fx"] == saved["camera_matrix"]["fx"]

        # Check intrinsics file
        intr_file = tmp_path / "cal" / "camera_intrinsics.json"
        assert intr_file.exists()
        intr = json.loads(intr_file.read_text())
        assert "cam0" in intr["cameras"]
        assert intr["cameras"]["cam0"]["camera_matrix"]["fx"] == saved["camera_matrix"]["fx"]

    def test_to_dict(self):
        r = CalibrationResult(
            camera_id=1,
            rms=0.22,
            fx=1400,
            fy=1500,
            cx=950,
            cy=540,
            k1=-0.5,
            k2=12,
            p1=-0.1,
            p2=-0.02,
            k3=-60,
            image_size=(1920, 1080),
            fov_h=69,
            fov_v=39,
            fov_d=74,
            board_size=(8, 5),
            square_size_mm=19,
            num_frames=10,
            num_detected=10,
        )
        d = r.to_dict()
        assert d["camera"] == "cam1"
        assert d["camera_matrix"]["fx"] == 1400
        assert d["distortion"]["k1"] == -0.5
        assert d["fov_deg"]["horizontal"] == 69


class TestProgress:
    def test_progress_to_dict(self):
        from src.calibration.auto_calibrator import CalibrationProgress

        p = CalibrationProgress(camera_id=0, state="capturing", frames_captured=3, frames_total=10)
        d = p.to_dict()
        assert d["state"] == "capturing"
        assert d["frames_captured"] == 3
