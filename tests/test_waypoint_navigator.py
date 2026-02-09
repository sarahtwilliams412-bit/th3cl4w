"""Tests for the waypoint navigator module."""

import numpy as np
import pytest

pytest.importorskip("scipy", reason="scipy not installed")

from src.planning.waypoint_navigator import (
    WaypointNavigator,
    WaypointPlanEntry,
    NavigationPlan,
    NavigationStatus,
    PathOrderStrategy,
)
from src.planning.motion_planner import MotionPlanner, NUM_ARM_JOINTS
from src.vision.digital_twin import DigitalTwin, WaypointStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _home_angles() -> np.ndarray:
    return np.zeros(NUM_ARM_JOINTS)


def _ready_angles() -> np.ndarray:
    return np.array([0.0, -45.0, 0.0, 90.0, 0.0, -45.0])


# ---------------------------------------------------------------------------
# WaypointPlanEntry tests
# ---------------------------------------------------------------------------


class TestWaypointPlanEntry:
    def test_to_dict(self):
        entry = WaypointPlanEntry(
            waypoint_id="wp_1",
            label="Test",
            joint_angles_deg=np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0]),
            gripper_mm=30.0,
            speed_factor=0.6,
            position_mm=np.array([200.0, 100.0, 50.0]),
            ik_success=True,
            ik_error_mm=5.0,
            collision_free=True,
        )
        d = entry.to_dict()
        assert d["waypoint_id"] == "wp_1"
        assert len(d["joint_angles_deg"]) == 6
        assert d["ik_success"] is True
        assert d["position_mm"] is not None

    def test_collision_objects_in_dict(self):
        entry = WaypointPlanEntry(
            waypoint_id="wp_2",
            label="Blocked",
            joint_angles_deg=np.zeros(6),
            collision_free=False,
            collision_objects=["obj_1", "obj_2"],
        )
        d = entry.to_dict()
        assert d["collision_free"] is False
        assert len(d["collision_objects"]) == 2


# ---------------------------------------------------------------------------
# NavigationPlan tests
# ---------------------------------------------------------------------------


class TestNavigationPlan:
    def test_empty_plan(self):
        plan = NavigationPlan()
        assert plan.num_waypoints == 0
        assert plan.success is True  # READY status

    def test_failed_plan(self):
        plan = NavigationPlan(status=NavigationStatus.FAILED)
        assert plan.success is False

    def test_to_dict(self):
        plan = NavigationPlan(
            status=NavigationStatus.READY,
            message="Test plan",
            total_distance_deg=100.0,
            estimated_duration_s=5.0,
            planning_time_ms=10.0,
        )
        d = plan.to_dict()
        assert d["status"] == "ready"
        assert d["message"] == "Test plan"


# ---------------------------------------------------------------------------
# WaypointNavigator construction tests
# ---------------------------------------------------------------------------


class TestNavigatorInit:
    def test_default_construction(self):
        nav = WaypointNavigator()
        assert nav.max_ik_error == 30.0
        assert nav.collision_clearance == 40.0

    def test_custom_params(self):
        nav = WaypointNavigator(
            max_ik_error_mm=50.0,
            collision_clearance_mm=60.0,
        )
        assert nav.max_ik_error == 50.0
        assert nav.collision_clearance == 60.0


# ---------------------------------------------------------------------------
# Navigation planning tests
# ---------------------------------------------------------------------------


class TestNavigationPlanning:
    def test_no_waypoints(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()
        plan = nav.plan_navigation(twin, _home_angles())
        assert plan.status == NavigationStatus.COMPLETED
        assert plan.num_waypoints == 0

    def test_joint_space_waypoints(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        # Add joint-space waypoints
        twin.add_waypoint(
            joint_angles_deg=np.array([10.0, -30.0, 0.0, 60.0, 0.0, -20.0]),
            label="WP1",
        )
        twin.add_waypoint(
            joint_angles_deg=np.array([20.0, -40.0, 10.0, 70.0, -10.0, -30.0]),
            label="WP2",
        )

        plan = nav.plan_navigation(twin, _home_angles())
        assert plan.success is True
        assert plan.num_waypoints == 2
        assert plan.trajectory is not None
        assert plan.trajectory.num_points > 0

    def test_return_home(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        twin.add_waypoint(
            joint_angles_deg=_ready_angles(),
            label="Ready",
        )

        plan = nav.plan_navigation(twin, _home_angles(), return_home=True)
        assert plan.success is True
        # Should have original waypoint + home
        assert plan.num_waypoints == 2
        assert plan.entries[-1].waypoint_id == "wp_home"


class TestPathOrdering:
    def test_nearest_neighbor_ordering(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        # Add waypoints: far one first, close one second
        twin.add_waypoint(
            joint_angles_deg=np.array([90.0, -45.0, 0.0, 90.0, 0.0, -45.0]),
            label="Far",
        )
        twin.add_waypoint(
            joint_angles_deg=np.array([5.0, -5.0, 0.0, 5.0, 0.0, -5.0]),
            label="Close",
        )

        plan = nav.plan_navigation(
            twin,
            _home_angles(),
            order_strategy=PathOrderStrategy.NEAREST_NEIGHBOR,
        )
        assert plan.success is True
        # Nearest neighbor should pick "Close" first
        assert plan.entries[0].label == "Close"

    def test_user_defined_ordering(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        twin.add_waypoint(
            joint_angles_deg=np.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            label="First",
        )
        twin.add_waypoint(
            joint_angles_deg=np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            label="Second",
        )

        plan = nav.plan_navigation(
            twin,
            _home_angles(),
            order_strategy=PathOrderStrategy.USER_DEFINED,
        )
        # User-defined order should be preserved
        assert plan.entries[0].label == "First"
        assert plan.entries[1].label == "Second"


# ---------------------------------------------------------------------------
# Plan from positions tests
# ---------------------------------------------------------------------------


class TestPlanFromPositions:
    def test_single_position(self):
        nav = WaypointNavigator()
        positions = [np.array([200.0, 0.0, 200.0])]
        plan = nav.plan_from_positions(positions, _home_angles())
        assert plan.num_waypoints == 1

    def test_multiple_positions(self):
        nav = WaypointNavigator()
        positions = [
            np.array([200.0, 0.0, 200.0]),
            np.array([200.0, 100.0, 200.0]),
            np.array([300.0, 0.0, 150.0]),
        ]
        plan = nav.plan_from_positions(positions, _home_angles())
        assert plan.num_waypoints == 3


# ---------------------------------------------------------------------------
# Progress tracking tests
# ---------------------------------------------------------------------------


class TestProgressTracking:
    def test_track_progress(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        target = np.array([10.0, -30.0, 0.0, 60.0, 0.0, -20.0])
        twin.add_waypoint(joint_angles_deg=target, label="WP1")

        plan = nav.plan_navigation(twin, _home_angles())

        # Simulate being at the target
        status = nav.track_progress(plan, target, digital_twin=twin)
        assert status["reached"] == 1
        assert status["remaining"] == 0
        assert status["completed"] is True

    def test_track_progress_not_reached(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        target = np.array([60.0, -30.0, 0.0, 60.0, 0.0, -20.0])
        twin.add_waypoint(joint_angles_deg=target, label="WP1")

        plan = nav.plan_navigation(twin, _home_angles())

        # Still at home
        status = nav.track_progress(plan, _home_angles(), digital_twin=twin)
        assert status["reached"] == 0
        assert status["remaining"] == 1
        assert status["completed"] is False

    def test_trajectory_distance(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        twin.add_waypoint(
            joint_angles_deg=np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            label="WP1",
        )

        plan = nav.plan_navigation(twin, _home_angles())
        assert plan.total_distance_deg > 0

    def test_segments_generated(self):
        nav = WaypointNavigator()
        twin = DigitalTwin()

        twin.add_waypoint(
            joint_angles_deg=np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        twin.add_waypoint(
            joint_angles_deg=np.array([60.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        plan = nav.plan_navigation(twin, _home_angles())
        assert len(plan.segments) == 2  # start->wp1, wp1->wp2
