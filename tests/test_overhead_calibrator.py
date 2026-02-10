"""Tests for overhead_calibrator.py — pixel ↔ workspace mapping."""

import json
import numpy as np
import pytest
from pathlib import Path

from src.vision.overhead_calibrator import OverheadCalibrator


@pytest.fixture
def tmp_calibration(tmp_path):
    """OverheadCalibrator with a temp calibration file."""
    return OverheadCalibrator(calibration_path=tmp_path / "cal.json")


# Simple known homography: identity-like mapping
# Pixel (0,0) → world (0,0), pixel (100,0) → world (100,0), etc.
PIXEL_POINTS = [[0, 0], [100, 0], [100, 100], [0, 100]]
WORLD_POINTS = [[0, 0], [300, 0], [300, 200], [0, 200]]


class TestCalibration:
    def test_calibrate_basic(self, tmp_calibration):
        cal = tmp_calibration
        assert not cal.is_calibrated

        result = cal.calibrate(PIXEL_POINTS, WORLD_POINTS)
        assert result["calibrated"]
        assert result["num_points"] == 4
        assert result["reprojection_error_mm"] < 1.0
        assert cal.is_calibrated

    def test_pixel_to_workspace(self, tmp_calibration):
        cal = tmp_calibration
        cal.calibrate(PIXEL_POINTS, WORLD_POINTS)

        # Known corners should map correctly
        x, y = cal.pixel_to_workspace(0, 0)
        assert abs(x) < 2.0 and abs(y) < 2.0

        x, y = cal.pixel_to_workspace(100, 0)
        assert abs(x - 300) < 2.0 and abs(y) < 2.0

        x, y = cal.pixel_to_workspace(100, 100)
        assert abs(x - 300) < 2.0 and abs(y - 200) < 2.0

    def test_workspace_to_pixel(self, tmp_calibration):
        cal = tmp_calibration
        cal.calibrate(PIXEL_POINTS, WORLD_POINTS)

        u, v = cal.workspace_to_pixel(0, 0)
        assert abs(u) < 2.0 and abs(v) < 2.0

        u, v = cal.workspace_to_pixel(300, 200)
        assert abs(u - 100) < 2.0 and abs(v - 100) < 2.0

    def test_roundtrip(self, tmp_calibration):
        cal = tmp_calibration
        cal.calibrate(PIXEL_POINTS, WORLD_POINTS)

        # pixel → world → pixel should be identity
        for px, wd in zip(PIXEL_POINTS, WORLD_POINTS):
            wx, wy = cal.pixel_to_workspace(px[0], px[1])
            u, v = cal.workspace_to_pixel(wx, wy)
            assert abs(u - px[0]) < 2.0
            assert abs(v - px[1]) < 2.0

    def test_persistence(self, tmp_path):
        path = tmp_path / "cal.json"
        cal1 = OverheadCalibrator(calibration_path=path)
        cal1.calibrate(PIXEL_POINTS, WORLD_POINTS)
        assert path.exists()

        # Load in new instance
        cal2 = OverheadCalibrator(calibration_path=path)
        assert cal2.is_calibrated
        x, y = cal2.pixel_to_workspace(50, 50)
        x1, y1 = cal1.pixel_to_workspace(50, 50)
        assert abs(x - x1) < 0.01 and abs(y - y1) < 0.01

    def test_too_few_points(self, tmp_calibration):
        with pytest.raises(ValueError, match="at least 4"):
            tmp_calibration.calibrate([[0, 0], [1, 1]], [[0, 0], [1, 1]])

    def test_mismatched_points(self, tmp_calibration):
        with pytest.raises(ValueError, match="same length"):
            tmp_calibration.calibrate([[0, 0]] * 4, [[0, 0]] * 5)

    def test_uncalibrated_raises(self, tmp_calibration):
        with pytest.raises(RuntimeError):
            tmp_calibration.pixel_to_workspace(0, 0)

    def test_get_status(self, tmp_calibration):
        status = tmp_calibration.get_status()
        assert not status["calibrated"]

        tmp_calibration.calibrate(PIXEL_POINTS, WORLD_POINTS)
        status = tmp_calibration.get_status()
        assert status["calibrated"]
        assert status["num_points"] == 4
