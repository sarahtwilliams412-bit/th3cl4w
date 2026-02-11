"""Tests for src/map/pointcloud — point cloud utilities."""
import numpy as np
import pytest


def test_backproject_depth_basic():
    from src.map.pointcloud import backproject_depth

    depth = np.ones((480, 640), dtype=np.float32) * 0.5  # 0.5m uniform
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)

    pts = backproject_depth(depth, rgb, subsample=8)
    assert pts.shape[1] == 6
    assert len(pts) > 0
    # All z should be ~0.5m (in camera frame, identity pose)
    assert np.abs(pts[:, 2].mean() - 0.5) < 0.1


def test_backproject_depth_empty():
    from src.map.pointcloud import backproject_depth

    depth = np.zeros((480, 640), dtype=np.float32)  # all zero = below min
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = backproject_depth(depth, rgb)
    assert len(pts) == 0


def test_merge_point_clouds():
    from src.map.pointcloud import merge_point_clouds

    a = np.random.randn(10, 6).astype(np.float32)
    b = np.random.randn(5, 6).astype(np.float32)
    merged = merge_point_clouds([a, b])
    assert len(merged) == 15


def test_merge_empty():
    from src.map.pointcloud import merge_point_clouds

    result = merge_point_clouds([])
    assert len(result) == 0


def test_voxel_downsample():
    from src.map.pointcloud import voxel_downsample

    pts = np.random.randn(1000, 6).astype(np.float32) * 0.01  # tight cluster
    down = voxel_downsample(pts, voxel_size=0.005)
    assert len(down) <= len(pts)
    assert len(down) > 0


def test_compute_camera_pose():
    from src.map.pointcloud import compute_camera_pose_from_joints

    pose = compute_camera_pose_from_joints([0, 0, 0, 0, 0, 0])
    assert pose.shape == (4, 4)
    # Should be a valid transform (bottom row = [0,0,0,1])
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-10)


def test_save_ply(tmp_path):
    from src.map.pointcloud import save_ply

    pts = np.random.randn(50, 6).astype(np.float32)
    pts[:, 3:] = np.abs(pts[:, 3:]) * 255
    path = str(tmp_path / "test.ply")
    ok = save_ply(pts, path)
    assert ok
    assert (tmp_path / "test.ply").exists()
