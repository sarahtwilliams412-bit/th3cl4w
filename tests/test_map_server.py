"""Tests for the map server modules."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map.scene import Scene, ArmSkeletonData, ObjectData
from src.map.arm_model import ArmModel
from src.map.collision_map import CollisionMap
from src.map.env_map import EnvMap, EnvMapConfig


# ---------------------------------------------------------------------------
# Scene tests
# ---------------------------------------------------------------------------

class TestScene:
    def test_empty_snapshot(self):
        s = Scene()
        snap = s.snapshot(full=True)
        assert snap["frame"] == 1
        assert "arm" in snap
        assert "env" in snap
        assert "objects" in snap

    def test_update_arm(self):
        s = Scene()
        skeleton = ArmSkeletonData(
            joints=[[0, 0, 0], [0.1, 0, 0.12]],
            links=[{"start": [0, 0, 0], "end": [0.1, 0, 0.12], "radius": 0.03}],
            gripper_mm=25.0,
            joint_angles_deg=[0, 10, 20, 0, 0, 0, 0],
        )
        s.update_arm(skeleton)
        snap = s.snapshot()
        assert snap["arm"]["gripper_mm"] == 25.0
        assert len(snap["arm"]["joints"]) == 2

    def test_update_objects(self):
        s = Scene()
        objs = [ObjectData(id="cup", label="red cup", position_mm=[200, 100, 50], reachable=True)]
        s.update_objects(objs)
        snap = s.snapshot()
        assert len(snap["objects"]) == 1
        assert snap["objects"][0]["id"] == "cup"

    def test_update_point_cloud(self):
        s = Scene()
        cloud = np.random.rand(100, 6).astype(np.float32)
        s.update_point_cloud(cloud)
        snap = s.snapshot(full=True)
        assert snap["env"]["stats"]["total_points"] == 100
        assert snap["env"]["update_mode"] == "full"

    def test_clear_point_cloud(self):
        s = Scene()
        s.update_point_cloud(np.random.rand(50, 6).astype(np.float32))
        s.clear_point_cloud()
        snap = s.snapshot(full=True)
        assert snap["env"]["stats"]["total_points"] == 0

    def test_frame_increments(self):
        s = Scene()
        s.snapshot()
        s.snapshot()
        snap = s.snapshot()
        assert snap["frame"] == 3

    def test_layer_filtering(self):
        s = Scene()
        snap = s.snapshot(layers={"arm"})
        assert "arm" in snap
        assert "env" not in snap
        assert "objects" not in snap

    def test_reach_envelope(self):
        s = Scene()
        assert s.get_reach_envelope() is None
        s.set_reach_envelope({"vertices": [[0, 0, 0]], "faces": []})
        assert s.get_reach_envelope() is not None

    def test_get_point_cloud_raw(self):
        s = Scene()
        cloud = np.random.rand(30, 6).astype(np.float32)
        s.update_point_cloud(cloud)
        raw = s.get_point_cloud_raw()
        assert raw.shape == (30, 6)
        # Ensure it's a copy
        raw[0, 0] = 999.0
        assert s.get_point_cloud_raw()[0, 0] != 999.0


# ---------------------------------------------------------------------------
# ArmModel tests
# ---------------------------------------------------------------------------

class TestArmModel:
    def test_zero_angles(self):
        model = ArmModel()
        skeleton = model.update([0, 0, 0, 0, 0, 0], gripper_mm=0)
        # Should have 8 joint positions (base + 7 joints)
        assert len(skeleton.joints) == 8
        assert len(skeleton.links) == 7
        # Base should be at origin
        assert np.allclose(skeleton.joints[0], [0, 0, 0], atol=1e-6)

    def test_nonzero_angles(self):
        model = ArmModel()
        skeleton = model.update([30, -20, 10, 0, 15, 0], gripper_mm=50)
        assert skeleton.gripper_mm == 50.0
        assert len(skeleton.joints) == 8
        # End-effector should not be at origin
        ee = skeleton.joints[-1]
        assert not np.allclose(ee, [0, 0, 0], atol=0.01)

    def test_ee_pose_is_4x4(self):
        model = ArmModel()
        skeleton = model.update([0, 0, 0, 0, 0, 0])
        assert skeleton.ee_pose is not None
        assert len(skeleton.ee_pose) == 4
        assert len(skeleton.ee_pose[0]) == 4

    def test_reach_envelope(self):
        model = ArmModel()
        env = model.compute_reach_envelope()
        assert "vertices" in env
        assert "faces" in env
        assert len(env["vertices"]) > 100
        assert len(env["faces"]) > 100
        assert env["radius_m"] == 0.55

    def test_link_radii(self):
        model = ArmModel()
        skeleton = model.update([0, 0, 0, 0, 0, 0])
        for link in skeleton.links:
            assert "start" in link
            assert "end" in link
            assert "radius" in link
            assert link["radius"] > 0


# ---------------------------------------------------------------------------
# CollisionMap tests
# ---------------------------------------------------------------------------

class TestCollisionMap:
    def test_empty_map(self):
        cm = CollisionMap(voxel_size_m=0.01)
        assert cm.check_point([0, 0, 0]) == "unknown"
        assert cm.get_occupied_count() == 0

    def test_update_from_cloud(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
        cm.update_from_cloud(pts)
        assert cm.get_occupied_count() > 0

    def test_check_point_occupied(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        cm.update_from_cloud(pts)
        # Point in same voxel should be occupied
        assert cm.check_point([0.1, 0.1, 0.1]) == "occupied"

    def test_check_point_free(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        cm.update_from_cloud(pts)
        # Distant point should be free
        assert cm.check_point([0.5, 0.5, 0.5]) == "free"

    def test_check_sphere(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        cm.update_from_cloud(pts)
        # Sphere centered on obstacle
        assert cm.check_sphere([0.1, 0.1, 0.1], 0.02) is True
        # Sphere far away
        assert cm.check_sphere([0.5, 0.5, 0.5], 0.02) is False

    def test_check_path(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        cm.update_from_cloud(pts)
        # Path through obstacle
        collisions = cm.check_path([[0.1, 0.1, 0.1]], radius_m=0.02)
        assert len(collisions) > 0
        assert collisions[0]["index"] == 0
        # Clear path
        collisions = cm.check_path([[0.5, 0.5, 0.5]], radius_m=0.02)
        assert len(collisions) == 0

    def test_update_from_objects(self):
        cm = CollisionMap(voxel_size_m=0.01)
        objects = [{"position_mm": [100, 100, 50], "bbox_mm": [50, 50, 100]}]
        cm.update_from_objects(objects)
        assert cm.get_occupied_count() > 0

    def test_occupied_centers(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
        cm.update_from_cloud(pts)
        centers = cm.get_occupied_centers()
        assert centers.shape[1] == 3
        assert len(centers) > 0


# ---------------------------------------------------------------------------
# EnvMap tests
# ---------------------------------------------------------------------------

class TestEnvMap:
    def test_initial_state(self):
        em = EnvMap()
        assert em.get_stats()["total_points"] == 0
        assert len(em.get_cloud()) == 0

    def test_clear(self):
        em = EnvMap()
        em.clear()
        assert em.get_stats()["total_points"] == 0

    def test_config(self):
        cfg = EnvMapConfig()
        cfg.update(voxel_size_m=0.02, max_points=100000)
        assert cfg.voxel_size_m == 0.02
        assert cfg.max_points == 100000

    def test_config_to_dict(self):
        cfg = EnvMapConfig()
        d = cfg.to_dict()
        assert "voxel_size_m" in d
        assert "max_points" in d
        assert "depth_min_m" in d


# ---------------------------------------------------------------------------
# Integration: ArmModel → Scene
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_arm_to_scene(self):
        s = Scene()
        model = ArmModel()
        skeleton = model.update([0, 0, 0, 0, 0, 0])
        s.update_arm(skeleton)
        snap = s.snapshot()
        assert len(snap["arm"]["joints"]) == 8

    def test_collision_with_objects(self):
        cm = CollisionMap(voxel_size_m=0.01)
        pts = np.array([[0.1, 0.1, 0.05]], dtype=np.float32)
        cm.update_from_cloud(pts)
        cm.update_from_objects([{"position_mm": [200, 200, 50], "bbox_mm": [40, 40, 100]}])
        # Both point cloud voxels and object voxels should be counted
        assert cm.get_occupied_count() > 1

    def test_full_scene_snapshot_serializable(self):
        """Ensure snapshot can be JSON serialized."""
        s = Scene()
        model = ArmModel()
        skeleton = model.update([10, -5, 20, 0, 15, 0])
        s.update_arm(skeleton)
        s.update_objects([ObjectData(id="test", label="test obj", position_mm=[100, 50, 25])])
        cloud = np.random.rand(50, 6).astype(np.float32)
        s.update_point_cloud(cloud)

        snap = s.snapshot(full=True)
        # Should not raise
        json_str = json.dumps(snap)
        assert len(json_str) > 100

        # Roundtrip
        parsed = json.loads(json_str)
        assert parsed["arm"]["joints"] is not None
        assert len(parsed["objects"]) == 1


# ---------------------------------------------------------------------------
# FastAPI endpoint tests (offline, no server needed)
# ---------------------------------------------------------------------------

class TestMapServerApp:
    """Test the FastAPI app can be imported and routes exist."""

    def test_import(self):
        from web.map_server import app
        assert app is not None
        assert app.title == "th3cl4w Map Server"

    def test_routes_exist(self):
        from web.map_server import app
        routes = {r.path for r in app.routes}
        assert "/api/map/status" in routes
        assert "/api/map/scene" in routes
        assert "/api/map/arm" in routes
        assert "/api/map/collision/check" in routes
        assert "/api/map/objects" in routes
        assert "/api/map/scan/start" in routes
        assert "/api/map/env/clear" in routes
        assert "/api/map/reach-envelope" in routes
        assert "/ws/map" in routes
