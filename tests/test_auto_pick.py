"""Tests for the autonomous pick pipeline."""

import math
import pytest
from src.planning.auto_pick import AutoPick, AutoPickPhase, AutoPickState, PickResult


class TestPlanJoints:
    """Test the geometric joint planner."""

    def test_reference_pose(self):
        """Planner should reproduce the known reference pose."""
        x = 100 * math.cos(math.radians(1.0))
        y = 100 * math.sin(math.radians(1.0))
        joints = AutoPick.plan_joints(x, y, 0.0)
        assert joints == [1.0, 25.9, 6.7, 0.0, 90.0, 0.0]

    def test_j0_base_yaw(self):
        """J0 should be atan2(y, x) in degrees."""
        joints = AutoPick.plan_joints(100, 100, 0)
        assert abs(joints[0] - 45.0) < 0.1

        joints = AutoPick.plan_joints(100, 0, 0)
        assert abs(joints[0]) < 0.1

        joints = AutoPick.plan_joints(0, 100, 0)
        assert abs(joints[0] - 90.0) < 0.1

    def test_j4_always_90(self):
        """J4 should always be 90° for top-down approach."""
        for x, y in [(50, 0), (100, 50), (200, -100)]:
            joints = AutoPick.plan_joints(x, y, 0)
            assert joints[4] == 90.0

    def test_j1_scales_with_distance(self):
        """J1 should increase with horizontal distance."""
        j_near = AutoPick.plan_joints(50, 0, 0)
        j_mid = AutoPick.plan_joints(100, 0, 0)
        j_far = AutoPick.plan_joints(200, 0, 0)
        assert j_near[1] < j_mid[1] < j_far[1]

    def test_origin_gives_zero_reach(self):
        """Target at origin should give J1=J2≈0."""
        joints = AutoPick.plan_joints(0, 0, 0)
        assert joints[1] == 0.0
        assert joints[2] == 0.0

    def test_j3_j5_always_zero(self):
        """J3 and J5 should always be 0."""
        joints = AutoPick.plan_joints(150, 75, 0)
        assert joints[3] == 0.0
        assert joints[5] == 0.0

    def test_height_correction(self):
        """Higher objects should need less J1."""
        j_table = AutoPick.plan_joints(100, 0, 0)
        j_raised = AutoPick.plan_joints(100, 0, 50)
        assert j_raised[1] < j_table[1]

    def test_j1_clamped(self):
        """J1 should be clamped to [0, 80]."""
        joints = AutoPick.plan_joints(500, 0, 0)  # very far
        assert joints[1] <= 80.0
        assert joints[1] >= 0.0


class TestAutoPickState:
    """Test state management."""

    def test_initial_state(self):
        ap = AutoPick()
        assert ap.state.phase == AutoPickPhase.IDLE
        assert not ap.running

    def test_get_status(self):
        ap = AutoPick()
        status = ap.get_status()
        assert status["phase"] == "idle"
        assert status["running"] is False
        assert "log" in status

    def test_stop_sets_flag(self):
        ap = AutoPick()
        ap.stop()
        assert ap._stop_requested is True
