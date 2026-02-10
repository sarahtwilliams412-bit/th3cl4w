"""Tests for VirtualGripDetector."""

import math
import pytest

from src.planning.virtual_grip import VirtualGripDetector, GripCheckResult


@pytest.fixture
def detector():
    return VirtualGripDetector()


class TestGripperPosition:
    """Test FK gripper position computation."""

    def test_home_position_gripper_high(self, detector):
        """Home [0,0,0,0,0,0] → gripper at z = d0+L1+L2+L3 = 651.5mm."""
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        expected_z = 121.5 + 208.5 + 208.5 + 113.0  # 651.5
        assert abs(pos["z"] - expected_z) < 0.1
        assert abs(pos["x"]) < 0.1
        assert abs(pos["y"]) < 0.1

    def test_reference_pose_gripper_low(self, detector):
        """Reference pose [1.0, 25.9, 6.7, 0.5, 88.7, 3.3] → gripper well below home.

        Geometric FK gives z≈426mm — significantly below home (651.5mm).
        The wrist pitch (J4=88.7°) tilts the last link nearly horizontal,
        bringing the tip down and forward. This is a reasonable reaching pose.
        """
        pos = detector.compute_gripper_position([1.0, 25.9, 6.7, 0.5, 88.7, 3.3])
        # Well below home position — arm reaching forward and down
        assert pos["z"] < 450, f"Expected z < 450mm (reaching pose), got {pos['z']}"
        # Significant forward reach
        assert pos["x"] > 200, f"Expected x > 200mm (reaching forward), got {pos['x']}"

    def test_arm_straight_forward(self, detector):
        """J1=90° → upper arm horizontal, rest follows."""
        pos = detector.compute_gripper_position([0, 90, 0, 0, 0, 0])
        # Upper arm horizontal: z = d0 + 0 + L2*cos(90) + L3*cos(90)
        # = 121.5 + 0 + 0 + 0 = 121.5
        assert abs(pos["z"] - 121.5) < 0.5
        # All links horizontal forward
        expected_r = 208.5 + 208.5 + 113.0
        assert abs(pos["x"] - expected_r) < 0.5


class TestGripCheck:
    """Test grip detection logic."""

    def test_grip_true_when_close_and_closed(self, detector):
        """Gripper near object + closed → gripped."""
        joints = [0, 90, 0, 0, 0, 0]  # arm horizontal
        pos = detector.compute_gripper_position(joints)
        obj = {
            "label": "red_cup",
            "position": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
            "width_mm": 30.0,
        }
        result = detector.check_grip(joints, gripper_width_mm=5.0, detected_objects=[obj])
        assert result.gripped is True
        assert result.object_label == "red_cup"
        assert result.distance_mm < 1.0

    def test_grip_false_when_far(self, detector):
        """Gripper far from object → not gripped even if closed."""
        joints = [0, 0, 0, 0, 0, 0]  # home position, high up
        obj = {
            "label": "ball",
            "position": {"x": 300, "y": 0, "z": 0},
            "width_mm": 25.0,
        }
        result = detector.check_grip(joints, gripper_width_mm=5.0, detected_objects=[obj])
        assert result.gripped is False
        assert "Too far" in result.message

    def test_grip_false_when_wide(self, detector):
        """Gripper near object but wide open → not gripped."""
        joints = [0, 90, 0, 0, 0, 0]
        pos = detector.compute_gripper_position(joints)
        obj = {
            "label": "block",
            "position": {"x": pos["x"] + 5, "y": pos["y"], "z": pos["z"]},
            "width_mm": 20.0,
        }
        result = detector.check_grip(joints, gripper_width_mm=80.0, detected_objects=[obj])
        assert result.gripped is False
        assert "too wide" in result.message.lower()

    def test_no_objects(self, detector):
        """No detected objects → not gripped."""
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 10.0, [])
        assert result.gripped is False
        assert "No objects" in result.message

    def test_closest_object_selected(self, detector):
        """Multiple objects → checks closest one."""
        joints = [0, 90, 0, 0, 0, 0]
        pos = detector.compute_gripper_position(joints)
        objects = [
            {"label": "far_obj", "position": {"x": 0, "y": 0, "z": 0}},
            {"label": "near_obj", "position": {"x": pos["x"] + 1, "y": pos["y"], "z": pos["z"]}},
        ]
        result = detector.check_grip(joints, 5.0, objects)
        assert result.object_label == "near_obj"
        assert result.gripped is True
