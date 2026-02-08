"""
Tests for the VisualGraspPlanner module.

Tests grasp pose computation, IK feasibility, and difficulty estimation.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

from src.vision.grasp_planner import (
    VisualGraspPlanner,
    GraspPlan,
    REDBULL_CAN_HEIGHT_MM,
    REDBULL_CAN_DIAMETER_MM,
    GRIPPER_MAX_OPEN_MM,
)
from src.kinematics.kinematics import D1Kinematics
from src.planning.motion_planner import MotionPlanner


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualGraspPlanner:
    def setup_method(self):
        self.kin = D1Kinematics()
        self.planner = VisualGraspPlanner(kinematics=self.kin)

    def test_init_defaults(self):
        planner = VisualGraspPlanner()
        assert planner.approach_height == 100.0
        assert planner.retreat_height == 120.0
        assert planner.max_reach == 500.0
        assert planner.min_reach == 80.0

    def test_plan_grasp_redbull(self):
        """Plan a grasp for a Red Bull can at a reachable position."""
        pos = np.array([250.0, 0.0, 0.0])
        plan = self.planner.plan_grasp(pos, object_label="redbull")

        assert isinstance(plan, GraspPlan)
        assert plan.object_label == "redbull"
        np.testing.assert_array_equal(plan.object_position_mm, pos)
        assert plan.approach_pose.shape == (4, 4)
        assert plan.grasp_pose.shape == (4, 4)
        assert plan.retreat_pose.shape == (4, 4)
        assert plan.approach_angles_deg.shape == (6,)
        assert plan.grasp_angles_deg.shape == (6,)
        assert plan.retreat_angles_deg.shape == (6,)
        assert plan.gripper_open_mm > plan.gripper_close_mm

    def test_plan_grasp_out_of_reach(self):
        """Object too far should return infeasible plan."""
        pos = np.array([600.0, 0.0, 0.0])  # beyond 500mm max reach
        plan = self.planner.plan_grasp(pos)
        assert not plan.feasible
        assert "max reach" in plan.message

    def test_plan_grasp_too_close(self):
        """Object too close should return infeasible plan."""
        pos = np.array([30.0, 0.0, 0.0])  # within 80mm min reach
        plan = self.planner.plan_grasp(pos)
        assert not plan.feasible
        assert "min reach" in plan.message

    def test_plan_grasp_too_wide(self):
        """Object wider than gripper should return infeasible plan."""
        pos = np.array([250.0, 0.0, 0.0])
        plan = self.planner.plan_grasp(
            pos, object_size_mm=(100.0, 100.0, 100.0)  # 100mm > 65mm gripper
        )
        assert not plan.feasible
        assert "gripper" in plan.message.lower()

    def test_plan_grasp_custom_size(self):
        """Custom object size should override defaults."""
        pos = np.array([250.0, 0.0, 0.0])
        plan = self.planner.plan_grasp(
            pos, object_label="custom", object_size_mm=(40.0, 60.0, 40.0)
        )
        assert plan.gripper_open_mm == pytest.approx(60.0)  # 40 + 20
        assert plan.gripper_close_mm == pytest.approx(35.0)  # 40 - 5

    def test_plan_grasp_gripper_settings(self):
        """Gripper open should be object width + clearance."""
        pos = np.array([250.0, 0.0, 0.0])
        plan = self.planner.plan_grasp(pos, object_label="redbull")
        assert plan.gripper_open_mm >= REDBULL_CAN_DIAMETER_MM
        assert plan.gripper_close_mm < REDBULL_CAN_DIAMETER_MM

    def test_plan_grasp_approach_above_grasp(self):
        """Approach pose should be higher than grasp pose (top-down)."""
        pos = np.array([250.0, 0.0, 100.0])
        plan = self.planner.plan_grasp(pos, grasp_from_top=True)
        # Z component of approach should be higher than grasp
        assert plan.approach_pose[2, 3] > plan.grasp_pose[2, 3]

    def test_plan_grasp_retreat_above_grasp(self):
        """Retreat pose should be higher than grasp pose."""
        pos = np.array([250.0, 0.0, 100.0])
        plan = self.planner.plan_grasp(pos, grasp_from_top=True)
        assert plan.retreat_pose[2, 3] > plan.grasp_pose[2, 3]

    def test_plan_grasp_with_current_angles(self):
        """Providing current angles should seed IK for better convergence."""
        pos = np.array([250.0, 0.0, 100.0])
        current = np.array([0.0, -30.0, 0.0, 45.0, 0.0, 0.0])
        plan = self.planner.plan_grasp(pos, current_angles_deg=current)
        # Should still produce a plan (may or may not be feasible)
        assert isinstance(plan, GraspPlan)

    def test_plan_grasp_side(self):
        """Side grasp should produce different poses than top-down."""
        pos = np.array([250.0, 0.0, 100.0])
        plan_top = self.planner.plan_grasp(pos, grasp_from_top=True)
        plan_side = self.planner.plan_grasp(pos, grasp_from_top=False)
        # The grasp poses should be different
        assert not np.allclose(plan_top.grasp_pose, plan_side.grasp_pose)

    def test_failed_plan_structure(self):
        """Failed plan should have proper structure."""
        plan = self.planner._failed_plan(
            np.array([0.0, 0.0, 0.0]), "test", "test failure"
        )
        assert not plan.feasible
        assert plan.confidence == 0.0
        assert plan.message == "test failure"
        assert plan.approach_angles_deg.shape == (6,)

    def test_get_object_dimensions_redbull(self):
        w, h, d = self.planner._get_object_dimensions("redbull")
        assert w == pytest.approx(REDBULL_CAN_DIAMETER_MM)
        assert h == pytest.approx(REDBULL_CAN_HEIGHT_MM)

    def test_get_object_dimensions_override(self):
        w, h, d = self.planner._get_object_dimensions("anything", (10.0, 20.0, 30.0))
        assert w == 10.0
        assert h == 20.0
        assert d == 30.0

    def test_get_object_dimensions_unknown(self):
        """Unknown objects should get default dimensions."""
        w, h, d = self.planner._get_object_dimensions("mystery_object")
        assert w > 0
        assert h > 0

    def test_top_down_grasp_pose_structure(self):
        pos = np.array([300.0, 0.0, 50.0])
        T = self.planner._top_down_grasp_pose(pos, 135.0)
        assert T.shape == (4, 4)
        # Should be a valid homogeneous transform
        assert T[3, 3] == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(T[3, :3], [0, 0, 0])

    def test_offset_pose(self):
        pose = np.eye(4)
        pose[2, 3] = 0.1  # 100mm in meters
        offset = self.planner._offset_pose(pose, dz=50.0)
        assert offset[2, 3] == pytest.approx(0.15)  # 100 + 50 = 150mm

    def test_offset_pose_along_approach(self):
        pose = np.eye(4)
        # Z axis of pose is [0, 0, 1] (identity)
        offset = self.planner._offset_pose_along_approach(pose, 100.0)
        assert offset[2, 3] == pytest.approx(0.1)  # 100mm along Z


class TestGraspDifficulty:
    def setup_method(self):
        self.planner = VisualGraspPlanner()

    def test_easy_grasp(self):
        """Object directly in front, mid-height."""
        result = self.planner.estimate_grasp_difficulty(np.array([200.0, 0.0, 100.0]))
        assert result["reachable"]
        assert result["difficulty"] < 0.5

    def test_edge_of_workspace(self):
        """Object at edge of workspace should be harder."""
        result = self.planner.estimate_grasp_difficulty(np.array([450.0, 0.0, 0.0]))
        assert result["difficulty"] > 0.2

    def test_unreachable(self):
        """Object beyond reach should report not reachable."""
        result = self.planner.estimate_grasp_difficulty(np.array([600.0, 0.0, 0.0]))
        assert not result["reachable"]

    def test_difficulty_fields(self):
        """Check all expected fields are present."""
        result = self.planner.estimate_grasp_difficulty(np.array([200.0, 100.0, 50.0]))
        assert "reach_xy_mm" in result
        assert "height_mm" in result
        assert "angle_deg" in result
        assert "difficulty" in result
        assert "reachable" in result
        assert 0.0 <= result["difficulty"] <= 1.0


class TestGraspPlanDataclass:
    def test_creation(self):
        plan = GraspPlan(
            approach_pose=np.eye(4),
            grasp_pose=np.eye(4),
            retreat_pose=np.eye(4),
            approach_angles_deg=np.zeros(6),
            grasp_angles_deg=np.zeros(6),
            retreat_angles_deg=np.zeros(6),
            gripper_open_mm=60.0,
            gripper_close_mm=10.0,
            object_position_mm=np.array([200.0, 0.0, 50.0]),
            object_label="redbull",
            confidence=0.9,
            feasible=True,
            message="ok",
        )
        assert plan.feasible
        assert plan.confidence == 0.9
        assert plan.object_label == "redbull"
