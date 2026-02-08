"""
Tests for the PickExecutor module.

Uses mocked tracker and grasp planner to test the orchestration logic.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

cv2 = pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

from src.planning.pick_executor import PickExecutor, PickResult, PickPhase
from src.planning.motion_planner import MotionPlanner, Trajectory, TrajectoryPoint
from src.planning.task_planner import TaskPlanner

# ═══════════════════════════════════════════════════════════════════════════
# Mock helpers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MockTrackedObject:
    label: str = "redbull"
    position_mm: np.ndarray = None
    position_cam_mm: np.ndarray = None
    size_mm: tuple = (53.0, 135.0, 53.0)
    confidence: float = 0.85
    bbox_left: tuple = (200, 150, 80, 120)
    bbox_right: tuple = (180, 150, 80, 120)
    centroid_left: tuple = (240, 210)
    centroid_right: tuple = (220, 210)
    depth_mm: float = 400.0

    def __post_init__(self):
        if self.position_mm is None:
            self.position_mm = np.array([250.0, 0.0, 50.0])
        if self.position_cam_mm is None:
            self.position_cam_mm = np.array([0.0, -50.0, 250.0])


@dataclass
class MockTrackingResult:
    objects: list = None
    depth_map: np.ndarray = None
    left_frame: np.ndarray = None
    right_frame: np.ndarray = None
    elapsed_ms: float = 50.0
    status: str = "ok"
    message: str = "Tracked 1 object(s)"

    def __post_init__(self):
        if self.objects is None:
            self.objects = [MockTrackedObject()]


@dataclass
class MockGraspPlan:
    approach_pose: np.ndarray = None
    grasp_pose: np.ndarray = None
    retreat_pose: np.ndarray = None
    approach_angles_deg: np.ndarray = None
    grasp_angles_deg: np.ndarray = None
    retreat_angles_deg: np.ndarray = None
    gripper_open_mm: float = 60.0
    gripper_close_mm: float = 10.0
    object_position_mm: np.ndarray = None
    object_label: str = "redbull"
    confidence: float = 0.9
    feasible: bool = True
    message: str = "Grasp plan ready"

    def __post_init__(self):
        if self.approach_pose is None:
            self.approach_pose = np.eye(4)
        if self.grasp_pose is None:
            self.grasp_pose = np.eye(4)
        if self.retreat_pose is None:
            self.retreat_pose = np.eye(4)
        if self.approach_angles_deg is None:
            self.approach_angles_deg = np.array([0.0, -30.0, 0.0, 60.0, 0.0, 0.0])
        if self.grasp_angles_deg is None:
            self.grasp_angles_deg = np.array([0.0, -45.0, 0.0, 75.0, 0.0, 0.0])
        if self.retreat_angles_deg is None:
            self.retreat_angles_deg = np.array([0.0, -20.0, 0.0, 50.0, 0.0, 0.0])
        if self.object_position_mm is None:
            self.object_position_mm = np.array([250.0, 0.0, 50.0])


def make_mock_tracker(result=None):
    tracker = MagicMock()
    if result is None:
        result = MockTrackingResult()
    tracker.track.return_value = result
    return tracker


def make_mock_grasp_planner(plan=None):
    planner = MagicMock()
    if plan is None:
        plan = MockGraspPlan()
    planner.plan_grasp.return_value = plan
    return planner


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPickExecutor:
    def setup_method(self):
        self.executor = PickExecutor()

    def test_init(self):
        assert self.executor.phase == PickPhase.IDLE
        assert self.executor.planner is not None
        assert self.executor.task_planner is not None

    def test_plan_pick_success(self):
        """Successful pick: detect -> plan -> trajectory."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        current = np.zeros(6)

        tracker = make_mock_tracker()
        grasp_planner = make_mock_grasp_planner()

        result = self.executor.plan_pick(
            left,
            right,
            tracker,
            grasp_planner,
            current,
            target_label="redbull",
        )

        assert result.success
        assert result.phase == PickPhase.COMPLETE
        assert result.trajectory is not None
        assert len(result.trajectory.points) > 0
        assert result.elapsed_ms > 0
        assert result.grasp_plan is not None

    def test_plan_pick_no_detection(self):
        """No objects detected should fail gracefully."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        current = np.zeros(6)

        empty_result = MockTrackingResult(objects=[], message="No objects found")
        tracker = make_mock_tracker(empty_result)
        grasp_planner = make_mock_grasp_planner()

        result = self.executor.plan_pick(
            left,
            right,
            tracker,
            grasp_planner,
            current,
        )

        assert not result.success
        assert result.phase == PickPhase.FAILED
        assert "No" in result.message

    def test_plan_pick_infeasible_grasp(self):
        """Infeasible grasp plan should fail."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        current = np.zeros(6)

        tracker = make_mock_tracker()
        bad_plan = MockGraspPlan(feasible=False, message="Out of reach")
        grasp_planner = make_mock_grasp_planner(bad_plan)

        result = self.executor.plan_pick(
            left,
            right,
            tracker,
            grasp_planner,
            current,
        )

        assert not result.success
        assert "Grasp planning failed" in result.message

    def test_plan_pick_from_position(self):
        """Plan pick from a known position (skip detection)."""
        current = np.zeros(6)
        pos = np.array([250.0, 0.0, 50.0])
        grasp_planner = make_mock_grasp_planner()

        result = self.executor.plan_pick_from_position(
            pos,
            grasp_planner,
            current,
        )

        assert result.success
        assert result.trajectory is not None
        assert len(result.trajectory.points) > 0

    def test_plan_pick_from_position_infeasible(self):
        """Infeasible position should fail."""
        current = np.zeros(6)
        pos = np.array([250.0, 0.0, 50.0])
        bad_plan = MockGraspPlan(feasible=False, message="Out of reach")
        grasp_planner = make_mock_grasp_planner(bad_plan)

        result = self.executor.plan_pick_from_position(
            pos,
            grasp_planner,
            current,
        )

        assert not result.success

    def test_select_best_target_single(self):
        """Single object should be returned directly."""
        obj = MockTrackedObject(confidence=0.8)
        result = self.executor._select_best_target([obj])
        assert result is obj

    def test_select_best_target_prefer_confidence(self):
        """Higher confidence should be preferred."""
        obj_low = MockTrackedObject(confidence=0.5, position_mm=np.array([200.0, 0.0, 50.0]))
        obj_high = MockTrackedObject(confidence=0.9, position_mm=np.array([300.0, 0.0, 50.0]))
        result = self.executor._select_best_target([obj_low, obj_high])
        assert result is obj_high

    def test_get_status_idle(self):
        status = self.executor.get_status()
        assert status["phase"] == "idle"

    def test_get_status_after_pick(self):
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        current = np.zeros(6)

        tracker = make_mock_tracker()
        grasp_planner = make_mock_grasp_planner()

        self.executor.plan_pick(left, right, tracker, grasp_planner, current)

        status = self.executor.get_status()
        assert "last_result" in status
        assert status["last_result"]["success"]
        assert status["last_result"]["has_trajectory"]

    def test_build_manual_trajectory(self):
        """Manual trajectory builder should produce a valid trajectory."""
        current = np.zeros(6)
        plan = MockGraspPlan()

        traj = self.executor._build_manual_trajectory(current, 0.0, plan)
        assert traj is not None
        assert isinstance(traj, Trajectory)
        assert len(traj.points) > 0
        assert traj.duration > 0


class TestPickPhase:
    def test_enum_values(self):
        assert PickPhase.IDLE.value == "idle"
        assert PickPhase.DETECTING.value == "detecting"
        assert PickPhase.PLANNING.value == "planning"
        assert PickPhase.COMPLETE.value == "complete"
        assert PickPhase.FAILED.value == "failed"


class TestPickResult:
    def test_creation(self):
        result = PickResult(
            phase=PickPhase.COMPLETE,
            success=True,
            message="Pick planned",
            elapsed_ms=100.0,
        )
        assert result.success
        assert result.phase == PickPhase.COMPLETE
        assert result.trajectory is None
        assert result.detected_objects == []

    def test_with_trajectory(self):
        traj = Trajectory(
            points=[
                TrajectoryPoint(
                    time=0.0,
                    positions=np.zeros(6),
                    velocities=np.zeros(6),
                    accelerations=np.zeros(6),
                ),
            ]
        )
        result = PickResult(
            phase=PickPhase.COMPLETE,
            success=True,
            trajectory=traj,
        )
        assert result.trajectory is not None
        assert len(result.trajectory.points) == 1
