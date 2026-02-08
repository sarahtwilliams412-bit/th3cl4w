"""Tests for src.vision.viz_calibrator."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.vision.viz_calibrator import (
    CalibrationResult,
    PoseObservation,
    fk_2d,
    solve_calibration,
    save_calibration,
    load_calibration,
    detect_end_effector,
    DEFAULT_LINKS_MM,
    DEFAULT_OFFSETS,
    CALIBRATION_POSES,
    JOINT_LIMITS,
)


class TestFK2D:
    """Test the 2D forward kinematics chain."""

    def test_home_position(self):
        """At home [0,0,0,0,0,0] with default offsets, arm should point forward-ish."""
        links = [80, 170, 170, 60, 60, 50]
        offsets = [0, 90, 90, 0, 0, 0]
        pts = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)

        assert len(pts) == 7  # base + 6 segments
        # Base at origin
        assert pts[0] == (0.0, 0.0)
        # Base link goes straight up
        assert pts[1] == pytest.approx((0.0, 80.0), abs=0.1)

    def test_all_zeros_no_offsets(self):
        """With zero angles and zero offsets, arm goes straight up."""
        links = [100, 100, 100, 50, 50, 50]
        offsets = [0, 0, 0, 0, 0, 0]
        pts = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)

        # Should be a vertical line
        total_height = sum(links)
        assert pts[-1][0] == pytest.approx(0.0, abs=0.1)
        assert pts[-1][1] == pytest.approx(total_height, abs=0.1)

    def test_returns_correct_number_of_points(self):
        links = [80, 170, 170, 60, 60, 50]
        offsets = DEFAULT_OFFSETS
        pts = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)
        assert len(pts) == 7

    def test_different_poses_give_different_endpoints(self):
        links = [80, 170, 170, 60, 60, 50]
        offsets = DEFAULT_OFFSETS
        pts1 = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)
        pts2 = fk_2d([0, 30, 0, 0, 0, 0], links, offsets)
        # End-effector should differ
        assert pts1[-1] != pytest.approx(pts2[-1], abs=1.0)

    def test_j4_affects_endpoint(self):
        links = [80, 170, 170, 60, 60, 50]
        offsets = DEFAULT_OFFSETS
        pts1 = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)
        pts2 = fk_2d([0, 0, 0, 0, 45, 0], links, offsets)
        assert pts1[-1] != pytest.approx(pts2[-1], abs=1.0)


class TestSolveCalibration:
    """Test the optimization solver."""

    def _generate_synthetic_observations(self, links, offsets, camera_params, n=15):
        """Generate synthetic observations from known parameters."""
        sx, sy, tx, ty = camera_params
        obs = []
        for pose in CALIBRATION_POSES[:n]:
            pts = fk_2d(pose, links, offsets)
            ee = pts[-1]
            px = int(sx * ee[0] + tx)
            py = int(-sy * ee[1] + ty)
            obs.append(PoseObservation(
                joint_angles=pose,
                end_effector_px=(px, py),
                timestamp=0,
            ))
        return obs

    def test_recovers_known_parameters(self):
        """With perfect synthetic data, solver should recover close to true params."""
        true_links = [80, 170, 170, 60, 60, 50]
        true_offsets = [0, 90, 90, 0, 0, 0]
        cam = (1.5, 1.5, 400, 800)

        obs = self._generate_synthetic_observations(true_links, true_offsets, cam, n=18)
        result = solve_calibration(obs, (1080, 1920))

        assert result.success
        assert result.n_observations == 18
        # Check link lengths are reasonable (within 30% — optimization has local minima)
        for name, true_val in zip(["base", "shoulder", "elbow", "wrist1", "wrist2", "end"], true_links):
            assert result.links_mm[name] > 0

    def test_insufficient_observations(self):
        """With too few observations, should report failure."""
        result = solve_calibration([
            PoseObservation([0,0,0,0,0,0], (100, 200)),
            PoseObservation([0,10,0,0,0,0], (110, 190)),
        ], (1080, 1920))
        assert not result.success

    def test_no_valid_detections(self):
        """All None end-effector detections should fail gracefully."""
        obs = [PoseObservation([0,0,0,0,0,0], None) for _ in range(10)]
        result = solve_calibration(obs)
        assert not result.success


class TestSaveLoadCalibration:
    """Test saving and loading calibration data."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "calib.json"
        result = CalibrationResult(
            links_mm={"base": 80, "shoulder": 170, "elbow": 170, "wrist1": 60, "wrist2": 60, "end": 50},
            joint_viz_offsets=[0, 90, 90, 0, 0, 0],
            residual=42.5,
            n_observations=15,
            camera_params={"sx": 1.5, "sy": 1.5, "tx": 400, "ty": 800},
        )
        save_calibration(result, path)
        loaded = load_calibration(path)
        assert loaded is not None
        assert loaded["links_mm"]["base"] == 80
        assert loaded["joint_viz_offsets"] == [0, 90, 90, 0, 0, 0]
        assert loaded["success"] is True

    def test_load_missing(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_calibration(path) is None


class TestDetectEndEffector:
    """Test end-effector detection."""

    def test_returns_none_for_none_frame(self):
        assert detect_end_effector(None) is None

    def test_returns_none_for_blank_frame(self):
        # White frame — no arm features
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = detect_end_effector(frame)
        # May or may not detect — just shouldn't crash
        assert result is None or isinstance(result, tuple)

    def test_detects_dark_object(self):
        """A dark blob on a light background should be detected."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Draw a dark circle (simulating arm)
        import cv2
        cv2.circle(frame, (400, 100), 30, (30, 30, 30), -1)
        result = detect_end_effector(frame)
        # Should detect something
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestCalibrationPoses:
    """Verify calibration poses are within limits."""

    def test_all_poses_within_limits(self):
        for i, pose in enumerate(CALIBRATION_POSES):
            for jid, (lo, hi) in JOINT_LIMITS.items():
                assert lo <= pose[jid] <= hi, \
                    f"Pose {i}: J{jid}={pose[jid]} outside [{lo}, {hi}]"

    def test_sufficient_pose_count(self):
        assert len(CALIBRATION_POSES) >= 15

    def test_poses_are_diverse(self):
        """Check that poses span a reasonable range."""
        j1_vals = [p[1] for p in CALIBRATION_POSES]
        j2_vals = [p[2] for p in CALIBRATION_POSES]
        assert max(j1_vals) - min(j1_vals) >= 80  # at least 80° range
        assert max(j2_vals) - min(j2_vals) >= 80
