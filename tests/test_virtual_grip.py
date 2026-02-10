"""Tests for virtual grip detection (geometric FK + grip check)."""

import math
import numpy as np
import pytest

from src.planning.virtual_grip import GripCheckResult, VirtualGripDetector


@pytest.fixture
def detector():
    return VirtualGripDetector()


class TestComputeGripperPosition:
    """FK position tests."""

    def test_home_pose(self, detector):
        """Home [0,0,0,0,0,0]: arm up, then PI/2 elbow offset makes forearm horizontal.

        shoulder at z=330, forearm horizontal +x, wrist pitch 0 continues horizontal.
        Expected: x = L2 + L3 = 321.5, y = 0, z = d0 + L1 = 330.
        """
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        assert isinstance(pos, np.ndarray)
        np.testing.assert_allclose(pos, [321.5, 0.0, 330.0], atol=0.1)

    def test_reference_grab_pose(self, detector):
        """Reference grab at [1, 25.9, 6.7, 0.5, 88.7, 3.3].

        Should place gripper near table level — low z, near base in XY.
        The wrist pitch of ~88.7° after the elbow bends the gripper downward.
        """
        pos = detector.compute_gripper_position([1.0, 25.9, 6.7, 0.5, 88.7, 3.3])
        # z should be low (near table) — verified from JS FK: ~100mm
        assert pos[2] < 150, f"z={pos[2]:.1f} should be below 150mm (near table)"
        # Should be within reach distance from base axis
        xy_dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
        assert xy_dist < 250, f"XY distance {xy_dist:.1f}mm should be < 250mm"

    def test_link_lengths_match(self, detector):
        assert detector.D0 == 121.5
        assert detector.L1 == 208.5
        assert detector.L2 == 208.5
        assert detector.L3 == 113.0

    def test_base_yaw_rotates_xy(self, detector):
        """J0=90° should rotate the arm into the Y axis."""
        pos = detector.compute_gripper_position([90, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(pos[0], 0.0, atol=0.5)
        np.testing.assert_allclose(pos[1], 321.5, atol=0.5)
        np.testing.assert_allclose(pos[2], 330.0, atol=0.5)

    def test_returns_ndarray(self, detector):
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)


class TestCheckGrip:
    """Grip detection tests."""

    def test_gripped_close_and_tight(self, detector):
        """Gripper close to object and closed enough → gripped."""
        pos = detector.compute_gripper_position([1.0, 25.9, 6.7, 0.5, 88.7, 3.3])
        objects = [{"label": "can", "position_mm": list(pos), "width_mm": 66.0}]
        result = detector.check_grip([1.0, 25.9, 6.7, 0.5, 88.7, 3.3], 50.0, objects)
        assert result.gripped is True
        assert result.object_label == "can"
        assert result.distance_mm < 1.0
        assert "Gripped" in result.message

    def test_missed_too_far(self, detector):
        """Object too far → not gripped."""
        objects = [{"label": "can", "position_mm": [999, 999, 999], "width_mm": 66.0}]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, objects)
        assert result.gripped is False
        assert "too far" in result.message

    def test_missed_gripper_too_wide(self, detector):
        """Gripper wider than object + margin → not gripped."""
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        objects = [{"label": "can", "position_mm": list(pos), "width_mm": 66.0}]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 200.0, objects)
        assert result.gripped is False
        assert "gripper too wide" in result.message

    def test_empty_objects(self, detector):
        """No objects → not gripped."""
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, [])
        assert result.gripped is False
        assert result.message == "No objects detected"

    def test_2d_position_gets_z_zero(self, detector):
        """Object with only [x, y] gets z=0."""
        objects = [{"label": "cup", "position_mm": [100, 100], "width_mm": 80.0}]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, objects)
        assert isinstance(result, GripCheckResult)

    def test_closest_object_selected(self, detector):
        """When multiple objects, closest one is checked."""
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        objects = [
            {"label": "far", "position_mm": [999, 999, 999], "width_mm": 66.0},
            {"label": "close", "position_mm": list(pos), "width_mm": 66.0},
        ]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, objects)
        assert result.object_label == "close"
        assert result.gripped is True

    def test_distance_calculation(self, detector):
        """Known distance between two points."""
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        offset_pos = [pos[0] + 30, pos[1], pos[2]]
        objects = [{"label": "test", "position_mm": offset_pos, "width_mm": 66.0}]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, objects)
        np.testing.assert_allclose(result.distance_mm, 30.0, atol=0.1)

    def test_default_object_width(self, detector):
        """Object without width_mm defaults to 66mm (Red Bull can)."""
        pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
        objects = [{"label": "mystery", "position_mm": list(pos)}]
        result = detector.check_grip([0, 0, 0, 0, 0, 0], 50.0, objects)
        assert result.object_width_mm == 66.0
