"""Tests for point cloud generation."""

import numpy as np
import pytest


class TestBackproject:
    """Test depth map back-projection to 3D points."""

    def test_basic_backproject(self):
        from src.vision.pointcloud_generator import backproject_depth

        h, w = 48, 64
        # Flat depth at 1.0m
        depth = np.ones((h, w), dtype=np.float32)
        rgb = np.full((h, w, 3), 128, dtype=np.uint8)

        points = backproject_depth(depth, rgb, subsample=4)
        assert points.shape[1] == 6  # x,y,z,r,g,b
        assert len(points) > 0
        # All z should be ~1.0 (identity pose)
        assert np.allclose(points[:, 2], 1.0, atol=0.01)

    def test_backproject_with_pose(self):
        from src.vision.pointcloud_generator import backproject_depth

        h, w = 48, 64
        depth = np.ones((h, w), dtype=np.float32) * 0.5
        rgb = np.full((h, w, 3), 200, dtype=np.uint8)

        # Translate camera 1m up
        pose = np.eye(4)
        pose[1, 3] = 1.0  # y offset

        points = backproject_depth(depth, rgb, camera_pose=pose, subsample=4)
        assert len(points) > 0
        # y should be offset by ~1.0
        assert np.mean(points[:, 1]) > 0.5

    def test_empty_depth(self):
        from src.vision.pointcloud_generator import backproject_depth

        depth = np.zeros((48, 64), dtype=np.float32)  # all zero = filtered out
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        points = backproject_depth(depth, rgb)
        assert len(points) == 0

    def test_subsample(self):
        from src.vision.pointcloud_generator import backproject_depth

        depth = np.ones((100, 100), dtype=np.float32)
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128

        pts1 = backproject_depth(depth, rgb, subsample=1)
        pts4 = backproject_depth(depth, rgb, subsample=4)
        # subsample=4 should have ~16x fewer points
        assert len(pts1) > len(pts4) * 10


class TestMergeAndDownsample:
    def test_merge(self):
        from src.vision.pointcloud_generator import merge_point_clouds

        c1 = np.random.rand(100, 6).astype(np.float32)
        c2 = np.random.rand(50, 6).astype(np.float32)
        merged = merge_point_clouds([c1, c2])
        assert len(merged) == 150

    def test_merge_empty(self):
        from src.vision.pointcloud_generator import merge_point_clouds

        merged = merge_point_clouds([])
        assert len(merged) == 0

    def test_voxel_downsample(self):
        from src.vision.pointcloud_generator import voxel_downsample

        # 1000 points clustered in a small region
        pts = np.random.rand(1000, 6).astype(np.float32) * 0.01
        down = voxel_downsample(pts, voxel_size=0.005)
        assert len(down) < len(pts)
        assert down.shape[1] == 6


class TestCameraPose:
    def test_compute_from_joints(self):
        from src.vision.pointcloud_generator import compute_camera_pose_from_joints

        angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pose = compute_camera_pose_from_joints(angles)
        assert pose.shape == (4, 4)
        # Should be close to identity with hand-eye offset
        assert np.isfinite(pose).all()


class TestSavePly:
    def test_save_ply(self, tmp_path):
        from src.vision.pointcloud_generator import save_ply

        pts = np.random.rand(100, 6).astype(np.float32)
        pts[:, 3:] *= 255  # colors in 0-255 range

        path = str(tmp_path / "test.ply")
        ok = save_ply(pts, path)
        assert ok
        assert (tmp_path / "test.ply").exists()
        content = (tmp_path / "test.ply").read_text()
        assert "ply" in content
        assert "vertex 100" in content
