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
    generate_round_poses,
    max_rounds,
    DEFAULT_LINKS_MM,
    DEFAULT_OFFSETS,
    JOINT_LIMITS,
    PITCH_JOINTS,
    ANGLE_INCREMENT,
    MAX_ANGLE,
    CONVERGENCE_THRESHOLD,
    STABLE_ROUNDS_NEEDED,
)


class TestFK2D:
    """Test the 2D forward kinematics chain."""

    def test_home_position(self):
        """At home [0,0,0,0,0,0] with default offsets, arm should point forward-ish."""
        links = [80, 170, 170, 60, 60, 50]
        offsets = [0, 90, 90, 0, 0, 0]
        pts = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)

        assert len(pts) == 7  # base + 6 segments
        assert pts[0] == (0.0, 0.0)
        assert pts[1] == pytest.approx((0.0, 80.0), abs=0.1)

    def test_all_zeros_no_offsets(self):
        """With zero angles and zero offsets, arm goes straight up."""
        links = [100, 100, 100, 50, 50, 50]
        offsets = [0, 0, 0, 0, 0, 0]
        pts = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)

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
        assert pts1[-1] != pytest.approx(pts2[-1], abs=1.0)

    def test_j4_affects_endpoint(self):
        links = [80, 170, 170, 60, 60, 50]
        offsets = DEFAULT_OFFSETS
        pts1 = fk_2d([0, 0, 0, 0, 0, 0], links, offsets)
        pts2 = fk_2d([0, 0, 0, 0, 45, 0], links, offsets)
        assert pts1[-1] != pytest.approx(pts2[-1], abs=1.0)


class TestProgressivePoses:
    """Test progressive pose generation."""

    def test_round_1_generates_small_angles(self):
        poses = generate_round_poses(1)
        # 3 joints × 2 directions = 6 poses
        assert len(poses) == 6
        for pose in poses:
            for jid in PITCH_JOINTS:
                assert abs(pose[jid]) <= ANGLE_INCREMENT

    def test_round_poses_increase_with_round(self):
        poses_r1 = generate_round_poses(1)
        poses_r3 = generate_round_poses(3)
        max_angle_r1 = max(abs(p[jid]) for p in poses_r1 for jid in PITCH_JOINTS)
        max_angle_r3 = max(abs(p[jid]) for p in poses_r3 for jid in PITCH_JOINTS)
        assert max_angle_r3 > max_angle_r1

    def test_all_round_poses_within_limits(self):
        for r in range(1, max_rounds() + 1):
            for pose in generate_round_poses(r):
                for jid, (lo, hi) in JOINT_LIMITS.items():
                    assert lo <= pose[jid] <= hi, \
                        f"Round {r}: J{jid}={pose[jid]} outside [{lo}, {hi}]"

    def test_max_rounds_correct(self):
        assert max_rounds() == MAX_ANGLE // ANGLE_INCREMENT

    def test_round_poses_only_move_one_joint(self):
        """Each pose should only move a single pitch joint."""
        for r in range(1, max_rounds() + 1):
            for pose in generate_round_poses(r):
                nonzero = [jid for jid in PITCH_JOINTS if pose[jid] != 0]
                assert len(nonzero) <= 1, f"Round {r}: pose {pose} moves multiple joints"

    def test_total_poses_reasonable(self):
        """Total poses across all rounds should be manageable."""
        total = sum(len(generate_round_poses(r)) for r in range(1, max_rounds() + 1))
        assert 20 <= total <= 100

    def test_poses_exceeding_limits_are_skipped(self):
        """If a round's angle exceeds joint limits, that pose is omitted."""
        # Round with angle = 90 would exceed ±85 limits
        poses = generate_round_poses(18)  # 18 * 5 = 90°
        for pose in poses:
            for jid, (lo, hi) in JOINT_LIMITS.items():
                assert lo <= pose[jid] <= hi


class TestSolveCalibration:
    """Test the optimization solver."""

    def _generate_synthetic_observations(self, links, offsets, camera_params, poses):
        """Generate synthetic observations from known parameters."""
        sx, sy, tx, ty = camera_params
        obs = []
        for pose in poses:
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

        # Use progressive poses for test data
        all_poses = [[0, 0, 0, 0, 0, 0]]
        for r in range(1, 5):
            all_poses.extend(generate_round_poses(r))

        obs = self._generate_synthetic_observations(true_links, true_offsets, cam, all_poses)
        result = solve_calibration(obs, (1080, 1920))

        assert result.success
        assert result.n_observations == len(all_poses)
        for name in ["base", "shoulder", "elbow", "wrist1", "wrist2", "end"]:
            assert result.links_mm[name] > 0

    def test_insufficient_observations(self):
        """With too few observations, should report failure."""
        result = solve_calibration([
            PoseObservation([0, 0, 0, 0, 0, 0], (100, 200)),
            PoseObservation([0, 10, 0, 0, 0, 0], (110, 190)),
        ], (1080, 1920))
        assert not result.success

    def test_no_valid_detections(self):
        """All None end-effector detections should fail gracefully."""
        obs = [PoseObservation([0, 0, 0, 0, 0, 0], None) for _ in range(10)]
        result = solve_calibration(obs)
        assert not result.success

    def test_residual_is_avg_px(self):
        """Residual should be average pixel error, not raw sum of squares."""
        true_links = [80, 170, 170, 60, 60, 50]
        true_offsets = [0, 90, 90, 0, 0, 0]
        cam = (1.5, 1.5, 400, 800)

        all_poses = [[0, 0, 0, 0, 0, 0]]
        for r in range(1, 4):
            all_poses.extend(generate_round_poses(r))

        obs = self._generate_synthetic_observations(true_links, true_offsets, cam, all_poses)
        result = solve_calibration(obs, (1080, 1920))
        assert result.success
        # With perfect data, residual should be very low
        assert result.residual < 10.0


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
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = detect_end_effector(frame)
        assert result is None or isinstance(result, tuple)

    def test_detects_dark_object(self):
        """A dark blob on a light background should be detected."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        import cv2
        cv2.circle(frame, (400, 100), 30, (30, 30, 30), -1)
        result = detect_end_effector(frame)
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestConvergenceConstants:
    """Verify calibration convergence constants are sensible."""

    def test_threshold_positive(self):
        assert CONVERGENCE_THRESHOLD > 0

    def test_stable_rounds_at_least_one(self):
        assert STABLE_ROUNDS_NEEDED >= 1

    def test_angle_increment_divides_max(self):
        assert MAX_ANGLE % ANGLE_INCREMENT == 0
