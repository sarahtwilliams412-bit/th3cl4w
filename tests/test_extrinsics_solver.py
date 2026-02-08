"""Tests for camera extrinsics solver."""

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.extrinsics_solver import (
    ExtrinsicsResult,
    compute_fk_ee_positions,
    compute_reprojection_error,
    get_default_camera_matrix,
    load_extrinsics,
    save_extrinsics,
    solve_camera_pnp,
    solve_from_bootstrap,
)


class TestSyntheticPnP:
    """Test PnP solve with synthetic data (known 3D→2D projection → solve → verify)."""

    def _make_synthetic_data(self, n_points=20, noise_px=0.0):
        """Generate synthetic 3D points, project them with known extrinsics, then return everything."""
        np.random.seed(42)

        # Known camera extrinsics
        true_rvec = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        true_tvec = np.array([0.2, -0.1, 1.5], dtype=np.float64)

        camera_matrix = get_default_camera_matrix()
        dist_coeffs = np.zeros(5, dtype=np.float64)

        # Generate 3D points in a volume roughly matching arm workspace
        obj_pts = np.random.uniform(-0.3, 0.3, (n_points, 3)).astype(np.float64)
        obj_pts[:, 2] += 0.5  # shift Z up so points are in front of camera

        # Project to 2D
        projected, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3), true_rvec, true_tvec,
            camera_matrix, dist_coeffs,
        )
        img_pts = projected.reshape(-1, 2).astype(np.float64)

        # Add noise
        if noise_px > 0:
            img_pts += np.random.normal(0, noise_px, img_pts.shape)

        return obj_pts, img_pts, true_rvec, true_tvec, camera_matrix, dist_coeffs

    def test_perfect_recovery(self):
        """PnP with perfect data should recover exact extrinsics."""
        obj_pts, img_pts, true_rvec, true_tvec, cam_mtx, dist = self._make_synthetic_data()

        rvec, tvec, inliers = solve_camera_pnp(obj_pts, img_pts, cam_mtx, dist)

        assert rvec is not None
        assert tvec is not None

        # Rotation should match within 0.1 degrees
        angle_err = np.linalg.norm(rvec.flatten() - true_rvec) * 180 / math.pi
        assert angle_err < 0.1, f"Rotation error: {angle_err:.4f}°"

        # Translation should match within 1mm
        trans_err = np.linalg.norm(tvec.flatten() - true_tvec) * 1000
        assert trans_err < 1.0, f"Translation error: {trans_err:.4f}mm"

    def test_noisy_recovery(self):
        """PnP with noisy data should still produce reasonable results."""
        obj_pts, img_pts, true_rvec, true_tvec, cam_mtx, dist = self._make_synthetic_data(
            n_points=20, noise_px=1.0
        )

        rvec, tvec, inliers = solve_camera_pnp(obj_pts, img_pts, cam_mtx, dist)

        assert rvec is not None
        angle_err = np.linalg.norm(rvec.flatten() - true_rvec) * 180 / math.pi
        assert angle_err < 2.0, f"Rotation error too large: {angle_err:.2f}°"

        trans_err = np.linalg.norm(tvec.flatten() - true_tvec) * 1000
        assert trans_err < 20.0, f"Translation error too large: {trans_err:.2f}mm"

    def test_minimum_points(self):
        """PnP should work with exactly 6 points (ITERATIVE minimum)."""
        obj_pts, img_pts, _, _, cam_mtx, dist = self._make_synthetic_data(n_points=6)
        rvec, tvec, inliers = solve_camera_pnp(obj_pts[:6], img_pts[:6], cam_mtx, dist)
        assert rvec is not None

    def test_four_points(self):
        """PnP should work with 4 points using SQPNP."""
        obj_pts, img_pts, _, _, cam_mtx, dist = self._make_synthetic_data(n_points=4)
        rvec, tvec, inliers = solve_camera_pnp(obj_pts, img_pts, cam_mtx, dist)
        assert rvec is not None

    def test_too_few_points(self):
        """PnP should fail gracefully with < 4 points."""
        obj_pts, img_pts, _, _, cam_mtx, dist = self._make_synthetic_data(n_points=3)
        rvec, tvec, inliers = solve_camera_pnp(
            obj_pts[:3], img_pts[:3], cam_mtx, dist
        )
        assert rvec is None


class TestReprojectionError:
    """Test reprojection error computation."""

    def test_zero_error_with_perfect_projection(self):
        """Reprojection error should be zero for perfectly projected points."""
        np.random.seed(42)
        rvec = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        tvec = np.array([0.2, -0.1, 1.5], dtype=np.float64)
        cam_mtx = get_default_camera_matrix()
        dist = np.zeros(5, dtype=np.float64)

        obj_pts = np.random.uniform(-0.3, 0.3, (10, 3)).astype(np.float64)
        obj_pts[:, 2] += 0.5

        projected, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3), rvec, tvec, cam_mtx, dist,
        )
        img_pts = projected.reshape(-1, 2)

        mean_err, max_err, per_point = compute_reprojection_error(
            obj_pts, img_pts, rvec, tvec, cam_mtx, dist,
        )

        assert mean_err < 0.01, f"Mean error should be ~0, got {mean_err}"
        assert max_err < 0.01, f"Max error should be ~0, got {max_err}"

    def test_nonzero_error_with_offset(self):
        """Adding pixel offset should produce measurable reprojection error."""
        rvec = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        tvec = np.array([0.2, -0.1, 1.5], dtype=np.float64)
        cam_mtx = get_default_camera_matrix()
        dist = np.zeros(5, dtype=np.float64)

        obj_pts = np.array([[0, 0, 0.5], [0.1, 0, 0.5], [0, 0.1, 0.5]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3), rvec, tvec, cam_mtx, dist,
        )
        img_pts = projected.reshape(-1, 2) + 5.0  # 5px offset

        mean_err, max_err, per_point = compute_reprojection_error(
            obj_pts, img_pts, rvec, tvec, cam_mtx, dist,
        )

        assert abs(mean_err - 5.0 * math.sqrt(2)) < 0.1  # diagonal offset


class TestSaveLoad:
    """Test save/load of extrinsics."""

    def test_roundtrip(self):
        """Save and load should produce identical data."""
        results = [
            ExtrinsicsResult(
                camera_id="cam0",
                rvec=[0.1, 0.2, 0.3],
                tvec=[1.0, 2.0, 3.0],
                reprojection_error_mean=2.5,
                reprojection_error_max=5.0,
                num_poses_used=10,
                num_inliers=9,
                camera_matrix=get_default_camera_matrix().tolist(),
                date="2026-02-08T22:00:00+00:00",
            ),
            ExtrinsicsResult(
                camera_id="cam1",
                rvec=[0.4, 0.5, 0.6],
                tvec=[4.0, 5.0, 6.0],
                reprojection_error_mean=1.5,
                reprojection_error_max=3.0,
                num_poses_used=15,
                num_inliers=14,
                camera_matrix=get_default_camera_matrix().tolist(),
                date="2026-02-08T22:00:00+00:00",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "extrinsics.json")
            save_extrinsics(results, path)

            loaded = load_extrinsics(path)
            assert loaded is not None
            assert "cameras" in loaded
            assert "cam0" in loaded["cameras"]
            assert "cam1" in loaded["cameras"]

            cam0 = loaded["cameras"]["cam0"]
            assert cam0["rvec"] == [0.1, 0.2, 0.3]
            assert cam0["tvec"] == [1.0, 2.0, 3.0]
            assert cam0["reprojection_error_mean"] == 2.5
            assert cam0["num_poses_used"] == 10

    def test_load_nonexistent(self):
        """Loading nonexistent file returns None."""
        assert load_extrinsics("/nonexistent/path.json") is None


class TestFKPositions:
    """Test FK end-effector position computation."""

    def test_home_position(self):
        """Home position should give known FK output.
        
        FK has a 90° elbow bend at home, so the arm extends forward (X)
        rather than being purely vertical.
        """
        positions = compute_fk_ee_positions([[0, 0, 0, 0, 0, 0]])
        assert len(positions) == 1
        ee = positions[0]
        # y should be 0 (no yaw)
        assert abs(ee[1]) < 0.01
        # Total link chain should be consistent
        # Just verify it's a reasonable position
        assert 0.0 < np.linalg.norm(ee) < 1.0, f"EE position out of range: {ee}"

    def test_multiple_poses(self):
        """Multiple poses should return multiple positions."""
        poses = [
            [0, 0, 0, 0, 0, 0],
            [30, 0, 0, 0, 0, 0],
            [0, -30, 0, 0, 0, 0],
        ]
        positions = compute_fk_ee_positions(poses)
        assert len(positions) == 3
        # Different poses should give different positions
        assert not np.allclose(positions[0], positions[1])
        assert not np.allclose(positions[0], positions[2])


class TestBootstrapSolve:
    """Test bootstrap solve using annotated poses."""

    def test_cam1_bootstrap(self):
        """cam1 has 9 correspondences — should solve."""
        result = solve_from_bootstrap("cam1")
        assert result is not None
        assert result.camera_id == "cam1"
        assert result.num_poses_used == 9
        # Reprojection error should be finite
        assert result.reprojection_error_mean < 100  # generous threshold for bootstrap

    def test_cam0_bootstrap(self):
        """cam0 has 5 correspondences — should solve (just enough)."""
        result = solve_from_bootstrap("cam0")
        assert result is not None
        assert result.camera_id == "cam0"
        assert result.num_poses_used == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
