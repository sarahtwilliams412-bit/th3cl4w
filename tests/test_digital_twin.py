"""Tests for the digital twin module."""

import time

import numpy as np
import pytest

from src.vision.digital_twin import (
    DigitalTwin,
    DigitalWaypoint,
    DigitalTwinSnapshot,
    WaypointStatus,
    ArmState,
)
from src.vision.vla_model import DetectedObject3D, ObjectShape

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_object(
    obj_id: str = "obj_1",
    position=(200.0, 100.0, 50.0),
    dims=(40.0, 30.0, 40.0),
    reachable=True,
) -> DetectedObject3D:
    return DetectedObject3D(
        object_id=obj_id,
        label="test",
        position_mm=np.array(position),
        dimensions_mm=np.array(dims),
        shape=ObjectShape.RECTANGULAR,
        confidence=0.8,
        reachable=reachable,
        reach_distance_mm=float(np.linalg.norm(position[:2])),
        mesh_vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        mesh_faces=[[0, 1, 2], [0, 1, 3]],
    )


# ---------------------------------------------------------------------------
# DigitalTwin construction tests
# ---------------------------------------------------------------------------


class TestDigitalTwinInit:
    def test_default_construction(self):
        twin = DigitalTwin()
        assert len(twin.get_objects()) == 0
        assert len(twin.get_waypoints()) == 0

    def test_with_kinematics(self):
        # Just test that it accepts the parameter (no actual kinematics needed)
        twin = DigitalTwin(kinematics=None)
        assert twin._kinematics is None


# ---------------------------------------------------------------------------
# Object management tests
# ---------------------------------------------------------------------------


class TestObjectManagement:
    def test_update_objects(self):
        twin = DigitalTwin()
        obj1 = _make_object("obj_1")
        obj2 = _make_object("obj_2", position=(300.0, 0.0, 0.0))
        twin.update_objects([obj1, obj2])
        assert len(twin.get_objects()) == 2

    def test_add_object(self):
        twin = DigitalTwin()
        obj = _make_object()
        twin.add_object(obj)
        assert len(twin.get_objects()) == 1

    def test_remove_object(self):
        twin = DigitalTwin()
        obj = _make_object("obj_1")
        twin.add_object(obj)
        assert len(twin.get_objects()) == 1
        twin.remove_object("obj_1")
        assert len(twin.get_objects()) == 0

    def test_remove_nonexistent(self):
        twin = DigitalTwin()
        twin.remove_object("nonexistent")  # should not raise

    def test_get_reachable(self):
        twin = DigitalTwin()
        obj1 = _make_object("obj_1", reachable=True)
        obj2 = _make_object("obj_2", position=(1000.0, 1000.0, 0.0), reachable=False)
        twin.update_objects([obj1, obj2])

        reachable = twin.get_reachable_objects()
        assert len(reachable) == 1
        assert reachable[0].object_id == "obj_1"

    def test_update_replaces(self):
        twin = DigitalTwin()
        twin.update_objects([_make_object("obj_1"), _make_object("obj_2")])
        assert len(twin.get_objects()) == 2

        twin.update_objects([_make_object("obj_3")])
        assert len(twin.get_objects()) == 1


# ---------------------------------------------------------------------------
# Arm state tests
# ---------------------------------------------------------------------------


class TestArmState:
    def test_update_arm_state(self):
        twin = DigitalTwin()
        angles = np.array([0.0, -45.0, 0.0, 90.0, 0.0, -45.0])
        twin.update_arm_state(angles, gripper_mm=30.0)

        snap = twin.snapshot()
        assert snap.arm is not None
        assert snap.arm.gripper_mm == 30.0
        np.testing.assert_array_almost_equal(snap.arm.joint_angles_deg, angles)

    def test_arm_state_to_dict(self):
        state = ArmState(
            joint_angles_deg=np.zeros(6),
            joint_positions_3d=[np.zeros(3)] * 3,
            end_effector_pose=np.eye(4),
            gripper_mm=10.0,
            timestamp=1.0,
        )
        d = state.to_dict()
        assert len(d["joint_angles_deg"]) == 6
        assert d["gripper_mm"] == 10.0


# ---------------------------------------------------------------------------
# Waypoint management tests
# ---------------------------------------------------------------------------


class TestWaypointManagement:
    def test_add_waypoint_cartesian(self):
        twin = DigitalTwin()
        wp = twin.add_waypoint(
            position_mm=np.array([200.0, 100.0, 50.0]),
            label="Target A",
        )
        assert wp.waypoint_id == "wp_1"
        assert wp.status == WaypointStatus.PENDING
        assert len(twin.get_waypoints()) == 1

    def test_add_waypoint_joint(self):
        twin = DigitalTwin()
        wp = twin.add_waypoint(
            joint_angles_deg=np.array([0.0, -45.0, 0.0, 90.0, 0.0, -45.0]),
            label="Joint Target",
        )
        assert wp.joint_angles_deg is not None

    def test_add_multiple_waypoints(self):
        twin = DigitalTwin()
        twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        twin.add_waypoint(position_mm=np.array([200.0, 0.0, 0.0]))
        twin.add_waypoint(position_mm=np.array([300.0, 0.0, 0.0]))
        assert len(twin.get_waypoints()) == 3

    def test_remove_waypoint(self):
        twin = DigitalTwin()
        wp = twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        assert twin.remove_waypoint(wp.waypoint_id) is True
        assert len(twin.get_waypoints()) == 0

    def test_remove_nonexistent_waypoint(self):
        twin = DigitalTwin()
        assert twin.remove_waypoint("wp_999") is False

    def test_reorder_waypoints(self):
        twin = DigitalTwin()
        wp1 = twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]), label="A")
        wp2 = twin.add_waypoint(position_mm=np.array([200.0, 0.0, 0.0]), label="B")
        wp3 = twin.add_waypoint(position_mm=np.array([300.0, 0.0, 0.0]), label="C")

        # Reverse order
        twin.reorder_waypoints([wp3.waypoint_id, wp2.waypoint_id, wp1.waypoint_id])
        ordered = twin.get_waypoints()
        assert ordered[0].waypoint_id == wp3.waypoint_id
        assert ordered[1].waypoint_id == wp2.waypoint_id
        assert ordered[2].waypoint_id == wp1.waypoint_id

    def test_get_pending_waypoints(self):
        twin = DigitalTwin()
        wp1 = twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        wp2 = twin.add_waypoint(position_mm=np.array([200.0, 0.0, 0.0]))
        twin.mark_waypoint_reached(wp1.waypoint_id)

        pending = twin.get_pending_waypoints()
        assert len(pending) == 1
        assert pending[0].waypoint_id == wp2.waypoint_id

    def test_mark_waypoint_reached(self):
        twin = DigitalTwin()
        wp = twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        twin.mark_waypoint_reached(wp.waypoint_id)

        wps = twin.get_waypoints()
        assert wps[0].status == WaypointStatus.REACHED
        assert wps[0].reached_at > 0

    def test_mark_waypoint_active(self):
        twin = DigitalTwin()
        wp = twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        twin.mark_waypoint_active(wp.waypoint_id)

        wps = twin.get_waypoints()
        assert wps[0].status == WaypointStatus.ACTIVE

    def test_clear_waypoints(self):
        twin = DigitalTwin()
        twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))
        twin.add_waypoint(position_mm=np.array([200.0, 0.0, 0.0]))
        twin.clear_waypoints()
        assert len(twin.get_waypoints()) == 0


# ---------------------------------------------------------------------------
# Trajectory preview tests
# ---------------------------------------------------------------------------


class TestTrajectoryPreview:
    def test_set_preview(self):
        twin = DigitalTwin()
        points = [
            {"position_mm": [100, 0, 50], "time": 0.0},
            {"position_mm": [200, 0, 50], "time": 1.0},
        ]
        twin.set_trajectory_preview(points)

        snap = twin.snapshot()
        assert snap.trajectory_preview is not None
        assert len(snap.trajectory_preview) == 2

    def test_clear_preview(self):
        twin = DigitalTwin()
        twin.set_trajectory_preview([{"position_mm": [0, 0, 0], "time": 0.0}])
        twin.clear_trajectory_preview()

        snap = twin.snapshot()
        assert snap.trajectory_preview is None


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self):
        twin = DigitalTwin()
        snap = twin.snapshot()
        assert snap.arm is None
        assert len(snap.objects) == 0
        assert len(snap.waypoints) == 0

    def test_full_snapshot(self):
        twin = DigitalTwin()

        # Add objects
        twin.update_objects([_make_object("obj_1"), _make_object("obj_2")])

        # Add arm state
        twin.update_arm_state(np.zeros(6), gripper_mm=20.0)

        # Add waypoints
        twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))

        snap = twin.snapshot()
        assert snap.arm is not None
        assert len(snap.objects) == 2
        assert len(snap.waypoints) == 1

    def test_snapshot_serialization(self):
        twin = DigitalTwin()
        twin.update_objects([_make_object()])
        twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))

        snap = twin.snapshot()
        d = snap.to_dict()
        assert "objects" in d
        assert "waypoints" in d
        assert "workspace" in d

        # Should be JSON serializable
        import json

        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_factory3d_update(self):
        twin = DigitalTwin()
        twin.update_objects([_make_object()])
        update = twin.get_factory3d_update()
        assert update["type"] == "digital_twin_update"
        assert "data" in update

    def test_object_mesh_in_snapshot(self):
        twin = DigitalTwin()
        obj = _make_object()
        twin.update_objects([obj])

        snap = twin.snapshot()
        assert len(snap.objects) == 1
        assert "mesh" in snap.objects[0]
        assert len(snap.objects[0]["mesh"]["vertices"]) == 4


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats(self):
        twin = DigitalTwin()
        twin.update_objects([_make_object(reachable=True)])
        twin.add_waypoint(position_mm=np.array([100.0, 0.0, 0.0]))

        stats = twin.get_stats()
        assert stats["objects"] == 1
        assert stats["reachable_objects"] == 1
        assert stats["waypoints_total"] == 1
        assert stats["waypoints_pending"] == 1


# ---------------------------------------------------------------------------
# DigitalWaypoint tests
# ---------------------------------------------------------------------------


class TestDigitalWaypoint:
    def test_to_dict_cartesian(self):
        wp = DigitalWaypoint(
            waypoint_id="wp_1",
            label="Target",
            position_mm=np.array([100.0, 200.0, 50.0]),
            gripper_mm=40.0,
            speed_factor=0.5,
            status=WaypointStatus.PENDING,
            order=1,
            created_at=1.0,
        )
        d = wp.to_dict()
        assert d["waypoint_id"] == "wp_1"
        assert d["position_mm"] == [100.0, 200.0, 50.0]
        assert d["status"] == "pending"

    def test_to_dict_joint(self):
        wp = DigitalWaypoint(
            waypoint_id="wp_2",
            label="Joint Target",
            joint_angles_deg=np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0]),
            status=WaypointStatus.REACHED,
            order=2,
            created_at=1.0,
            reached_at=2.0,
        )
        d = wp.to_dict()
        assert d["waypoint_id"] == "wp_2"
        assert len(d["joint_angles_deg"]) == 6
        assert d["status"] == "reached"
        assert "reached_at" in d
