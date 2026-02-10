"""Tests for the updated 3-camera calibration system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.vision.camera_model import CameraModel, load_intrinsics, CAMERA_ROLES


class TestLoadIntrinsics:
    def test_load_all_three_cameras(self):
        """All 3 cameras should have intrinsics in the shared file."""
        for cam_id in [0, 1, 2]:
            K, dist = load_intrinsics(cam_id)
            assert K is not None, f"cam{cam_id} intrinsics not found"
            assert dist is not None
            assert K.shape == (3, 3)
            assert K[0, 0] > 0  # fx > 0
            assert K[1, 1] > 0  # fy > 0

    def test_cam0_and_cam2_both_brio(self):
        """cam0 (overhead BRIO) and cam2 (side BRIO) are both BRIOs but may have different zoom/FOV."""
        K0, _ = load_intrinsics(0)
        K2, _ = load_intrinsics(2)
        assert K0 is not None and K2 is not None
        # Both are 1920x1080
        assert K0[0, 2] > 900  # cx near center
        assert K2[0, 2] > 900

    def test_cam1_different_from_brio(self):
        """cam1 (MX Brio) should have slightly different focal length."""
        K0, _ = load_intrinsics(0)
        K1, _ = load_intrinsics(1)
        # MX Brio has 91.5° FOV vs BRIO's 90° — different fx
        assert K0[0, 0] != K1[0, 0]


class TestCameraRoles:
    def test_roles_defined(self):
        assert CAMERA_ROLES[0] == "overhead"
        assert CAMERA_ROLES[1] == "arm-mounted"
        assert CAMERA_ROLES[2] == "side-view"


class TestCameraModel:
    def test_static_camera_load(self):
        """cam0 has extrinsics on disk — should load."""
        cm = CameraModel(0)
        loaded = cm.load()
        assert loaded
        assert cm.is_calibrated
        assert cm.role == "overhead"

    def test_missing_extrinsics_still_loads_intrinsics(self):
        """cam2 has no extrinsics yet but should still get intrinsics."""
        cm = CameraModel(2)
        cm.load()  # Returns False (no extrinsics) but intrinsics should load
        assert cm.has_intrinsics
        assert cm.K is not None

    def test_arm_camera_role(self):
        cm = CameraModel(1)
        assert cm.is_arm_mounted
        assert cm.role == "arm-mounted"

    def test_hand_eye_update(self):
        """Test that update_from_fk correctly computes extrinsics."""
        cm = CameraModel(1)
        K, dist = load_intrinsics(1)
        cm.K = K
        cm.dist = dist

        # Set identity hand-eye (camera at EE origin, same orientation)
        T_ee_cam = np.eye(4, dtype=np.float64)
        cm.set_hand_eye_transform(T_ee_cam, save=False)
        assert cm.has_hand_eye

        # EE at world origin looking down Z
        T_world_ee = np.eye(4, dtype=np.float64)
        T_world_ee[0, 3] = 0.3  # 300mm forward
        T_world_ee[2, 3] = 0.5  # 500mm up
        cm.update_from_fk(T_world_ee)

        assert cm.is_calibrated
        np.testing.assert_array_almost_equal(
            cm.camera_position, [0.3, 0.0, 0.5]
        )

    def test_world_to_pixel_roundtrip(self):
        """Project a point and back-project it — should be consistent."""
        cm = CameraModel(0)
        if not cm.load():
            pytest.skip("cam0 extrinsics not available")

        # Project a point on the table
        point = np.array([0.2, 0.1, 0.0])
        u, v = cm.world_to_pixel(point)

        # Back-project at Z=0
        recovered = cm.pixel_to_world_at_z(u, v, z=0.0)
        assert recovered is not None
        np.testing.assert_array_almost_equal(recovered, point, decimal=3)


class TestHandEyeCalibrator:
    def test_import(self):
        from src.vision.hand_eye_calibrator import HandEyeCalibrator
        hec = HandEyeCalibrator(camera_id=1)
        assert hec.num_observations == 0

    def test_needs_min_observations(self):
        from src.vision.hand_eye_calibrator import HandEyeCalibrator
        hec = HandEyeCalibrator(camera_id=1)
        result = hec.solve()
        assert result is None  # Need at least 3


class TestIntrinsicsFile:
    def test_file_structure(self):
        """Verify the intrinsics JSON has correct structure."""
        path = Path(__file__).parent.parent / "calibration_results" / "camera_intrinsics.json"
        data = json.loads(path.read_text())

        # Support "cameras" wrapper format
        cameras = data.get("cameras", {})
        assert len(cameras) >= 3, "Should have at least 3 cameras"

        for key in ["cam0", "cam1", "cam2"]:
            assert key in cameras, f"{key} missing"
            cam = cameras[key]
            assert "camera_matrix" in cam
            assert "image_size" in cam or "fov_diagonal_deg" in cam

    def test_naming_is_correct(self):
        """Verify cam0=overhead, cam1=arm, cam2=side."""
        path = Path(__file__).parent.parent / "calibration_results" / "camera_intrinsics.json"
        data = json.loads(path.read_text())
        cameras = data.get("cameras", {})
        assert "overhead" in cameras["cam0"].get("label", "")
        assert "arm" in cameras["cam1"].get("label", "")
        assert "side" in cameras["cam2"].get("label", "")
