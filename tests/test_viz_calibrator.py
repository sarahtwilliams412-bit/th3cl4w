"""Tests for src.vision.viz_calibrator — v2 (3D FK + camera projection)."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.viz_calibrator import (
    ArmDetector,
    CalibrationResult,
    PoseObservation,
    _dh_transform,
    fk_3d,
    project_3d_to_2d,
    solve_calibration,
    save_calibration,
    load_calibration,
    generate_round_poses,
    max_rounds,
    DH_D,
    DH_ALPHA,
    JOINT_LIMITS,
    CALIBRATION_JOINTS,
    ANGLE_INCREMENT,
    MAX_ANGLE,
    CONVERGENCE_THRESHOLD,
    STABLE_ROUNDS_NEEDED,
    LANDMARK_FRAMES,
)


# ---------------------------------------------------------------------------
# 3D FK Tests
# ---------------------------------------------------------------------------

class TestFK3D:
    """Test 3D forward kinematics using DH parameters."""

    def test_home_position_base_at_origin(self):
        pts = fk_3d([0, 0, 0, 0, 0, 0])
        assert len(pts) == 8
        np.testing.assert_allclose(pts[0], [0, 0, 0], atol=1e-10)

    def test_home_position_first_joint_elevated(self):
        pts = fk_3d([0, 0, 0, 0, 0, 0])
        assert pts[1][2] == pytest.approx(0.1215, abs=1e-4)

    def test_home_ee_position_reasonable(self):
        pts = fk_3d([0, 0, 0, 0, 0, 0])
        dist = np.linalg.norm(pts[-1])
        assert 0.2 < dist < 0.8

    def test_different_angles_different_ee(self):
        pts1 = fk_3d([0, 0, 0, 0, 0, 0])
        pts2 = fk_3d([0, 30, 0, 0, 0, 0])
        assert not np.allclose(pts1[-1], pts2[-1], atol=0.01)

    def test_j0_rotates_in_xy(self):
        # Use a non-straight pose so arm extends out from z-axis
        pts_0 = fk_3d([0, 45, 0, 0, 0, 0])
        pts_45 = fk_3d([45, 45, 0, 0, 0, 0])
        # Z should be similar (J0 rotates in XY plane)
        assert pts_0[-1][2] == pytest.approx(pts_45[-1][2], abs=0.01)
        # XY should differ
        assert not np.allclose(pts_0[-1][:2], pts_45[-1][:2], atol=0.01)

    def test_returns_8_positions(self):
        assert len(fk_3d([0] * 6)) == 8

    def test_pads_6_to_7_joints(self):
        assert len(fk_3d([0] * 6)) == 8

    def test_theta_offsets(self):
        pts1 = fk_3d([0]*6, [0]*7)
        pts2 = fk_3d([0]*6, [0, 0.1, 0, 0, 0, 0, 0])
        assert not np.allclose(pts1[-1], pts2[-1], atol=0.001)


class TestDHTransform:
    def test_identity_at_zero(self):
        T = _dh_transform(0, 0, 0, 0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_d_offset(self):
        T = _dh_transform(0.5, 0, 0, 0)
        assert T[2, 3] == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Camera Projection Tests
# ---------------------------------------------------------------------------

class TestProjection:
    def _make_cam(self, fx=1000, fy=1000, cx=960, cy=540, rvec=(0,0,0), tvec=(0,0,1)):
        R, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
        return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'R': R.tolist(), 't': list(tvec)}

    def test_simple_projection(self):
        cam = self._make_cam(tvec=(0, 0, 0))
        pts = project_3d_to_2d([np.array([0.0, 0.0, 1.0])], cam)
        assert pts[0] is not None
        assert pts[0][0] == pytest.approx(960, abs=1)
        assert pts[0][1] == pytest.approx(540, abs=1)

    def test_behind_camera_returns_none(self):
        cam = self._make_cam(tvec=(0, 0, 0))
        pts = project_3d_to_2d([np.array([0.0, 0.0, -1.0])], cam)
        assert pts[0] is None

    def test_offset_point(self):
        cam = self._make_cam(tvec=(0, 0, 0))
        pts = project_3d_to_2d([np.array([0.1, 0.0, 1.0])], cam)
        assert pts[0][0] == pytest.approx(1060, abs=1)
        assert pts[0][1] == pytest.approx(540, abs=1)


# ---------------------------------------------------------------------------
# Pose Generation Tests
# ---------------------------------------------------------------------------

class TestProgressivePoses:
    def test_round_1_generates_small_angles(self):
        poses = generate_round_poses(1)
        assert len(poses) == 6
        for pose in poses:
            for jid in CALIBRATION_JOINTS:
                assert abs(pose[jid]) <= ANGLE_INCREMENT

    def test_round_poses_increase_with_round(self):
        max_r1 = max(abs(p[jid]) for p in generate_round_poses(1) for jid in CALIBRATION_JOINTS)
        max_r3 = max(abs(p[jid]) for p in generate_round_poses(3) for jid in CALIBRATION_JOINTS)
        assert max_r3 > max_r1

    def test_all_within_limits(self):
        for r in range(1, max_rounds() + 1):
            for pose in generate_round_poses(r):
                for jid, (lo, hi) in JOINT_LIMITS.items():
                    assert lo <= pose[jid] <= hi

    def test_max_rounds_correct(self):
        assert max_rounds() == MAX_ANGLE // ANGLE_INCREMENT

    def test_single_joint_per_pose(self):
        for r in range(1, max_rounds() + 1):
            for pose in generate_round_poses(r):
                nonzero = [jid for jid in CALIBRATION_JOINTS if pose[jid] != 0]
                assert len(nonzero) <= 1

    def test_total_poses_reasonable(self):
        total = sum(len(generate_round_poses(r)) for r in range(1, max_rounds() + 1))
        assert 20 <= total <= 100


# ---------------------------------------------------------------------------
# Solver Tests
# ---------------------------------------------------------------------------

class TestSolveCalibration:
    def _make_synthetic_obs(self, cam_params, poses, theta_offsets=None):
        """Generate synthetic observations from known camera params."""
        if theta_offsets is None:
            theta_offsets = [0.0] * 7
        obs_list = []
        for pose in poses:
            fk_pts = fk_3d(pose, theta_offsets)
            cam1_lm = {}
            proj = project_3d_to_2d(fk_pts, cam_params)
            for name, idx in LANDMARK_FRAMES.items():
                if proj[idx] is not None:
                    cam1_lm[name] = (int(proj[idx][0]), int(proj[idx][1]))
            obs_list.append(PoseObservation(
                joint_angles=pose,
                cam1_landmarks=cam1_lm,
            ))
        return obs_list

    def _make_cam(self):
        R, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))
        return {'fx': 1200, 'fy': 1200, 'cx': 960, 'cy': 540,
                'R': R.tolist(), 't': [0.0, -0.1, 1.5]}

    def test_recovers_with_synthetic_data(self):
        cam = self._make_cam()
        poses = [[0, 0, 0, 0, 0, 0]]
        for r in range(1, 4):
            poses.extend(generate_round_poses(r))
        obs = self._make_synthetic_obs(cam, poses)
        result = solve_calibration(obs, {1: (1080, 1920)})
        assert result.success
        assert result.residual < 50.0 or result.residual == -1.0

    def test_insufficient_observations(self):
        result = solve_calibration(
            [PoseObservation([0]*6, cam1_landmarks={"end_effector": (100, 200)})],
            {1: (1080, 1920)},
        )
        assert not result.success

    def test_no_landmarks(self):
        obs = [PoseObservation([0]*6) for _ in range(10)]
        result = solve_calibration(obs, {1: (1080, 1920)})
        assert not result.success

    def test_result_has_theta_offsets(self):
        cam = self._make_cam()
        poses = [[0, 0, 0, 0, 0, 0]]
        for r in range(1, 3):
            poses.extend(generate_round_poses(r))
        obs = self._make_synthetic_obs(cam, poses)
        result = solve_calibration(obs, {1: (1080, 1920)})
        assert len(result.theta_offsets) == 7


# ---------------------------------------------------------------------------
# Save/Load Tests
# ---------------------------------------------------------------------------

class TestSaveLoadCalibration:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "calib.json"
        R, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.0]))
        result = CalibrationResult(
            cam1_params={'fx': 1000, 'fy': 1000, 'cx': 960, 'cy': 540,
                         'R': R.tolist(), 't': [0.0, -0.2, 1.0]},
            cam0_params=None,
            theta_offsets=[0, 0.02, 0, -0.03, 0, 0.01, 0],
            residual=5.2,
            n_observations=20,
            n_constraints=80,
            success=True,
        )
        save_calibration(result, path)
        loaded = load_calibration(path)
        assert loaded is not None
        assert loaded["version"] == 2
        assert "camera_params" in loaded
        assert "cam1" in loaded["camera_params"]
        cam1 = loaded["camera_params"]["cam1"]
        assert "rx" in cam1  # should be Rodrigues format
        assert "fx" in cam1
        assert loaded["dh_theta_offsets_deg"] is not None
        assert loaded["success"] is True
        # Legacy compat
        assert "links_mm" in loaded
        assert "joint_viz_offsets" in loaded

    def test_load_missing(self, tmp_path):
        assert load_calibration(tmp_path / "nope.json") is None


# ---------------------------------------------------------------------------
# Detection Tests
# ---------------------------------------------------------------------------

class TestArmDetector:
    def test_gold_segment_on_gold_image(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        gold_bgr = cv2.cvtColor(np.array([[[25, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.rectangle(frame, (200, 100), (220, 300), gold_bgr.tolist(), -1)
        detector = ArmDetector()
        result = detector.detect_gold_segment(frame)
        assert result is not None
        top, bot = result
        assert top[1] < bot[1]

    def test_gold_segment_on_blank(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        assert ArmDetector().detect_gold_segment(frame) is None

    def test_frame_differencing(self):
        home = np.ones((480, 640, 3), dtype=np.uint8) * 128
        pose = home.copy()
        cv2.circle(pose, (400, 100), 40, (255, 255, 255), -1)
        detector = ArmDetector()
        detector.set_home_frame(1, home)
        tip = detector.detect_via_differencing(pose, 1)
        assert tip is not None
        assert abs(tip[0] - 400) < 50
        assert abs(tip[1] - 100) < 50

    def test_no_home_frame_returns_none(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        assert ArmDetector().detect_via_differencing(frame, 1) is None


# ---------------------------------------------------------------------------
# Convergence Constants
# ---------------------------------------------------------------------------

class TestConvergenceConstants:
    def test_threshold_positive(self):
        assert CONVERGENCE_THRESHOLD > 0

    def test_stable_rounds_at_least_one(self):
        assert STABLE_ROUNDS_NEEDED >= 1

    def test_angle_increment_divides_max(self):
        assert MAX_ANGLE % ANGLE_INCREMENT == 0
