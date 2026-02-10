"""Tests for VirtualGripDetector — geometric FK + proximity grip check."""

import math

import numpy as np
import pytest

from src.planning.virtual_grip import GripCheckResult, VirtualGripDetector

D0, L1, L2, L3 = 121.5, 208.5, 208.5, 113.0
HOME_Z = D0 + L1 + L2 + L3  # 651.5 mm


@pytest.fixture
def detector():
    return VirtualGripDetector()


# ---------- FK tests ----------


def test_home_pose_z_high(detector: VirtualGripDetector):
    """Home pose [0,0,0,0,0,0] → gripper straight up at z ≈ d0+L1+L2+L3."""
    pos = detector.compute_gripper_position([0, 0, 0, 0, 0, 0])
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert abs(pos[0]) < 1.0  # x ≈ 0
    assert abs(pos[1]) < 1.0  # y ≈ 0
    assert abs(pos[2] - HOME_Z) < 0.1  # z ≈ 651.5


def test_reference_pose_gripper_low(detector: VirtualGripDetector):
    """Reference grab pose reaches down — z well below home, near table level.

    The cumulative pitch J1+J2+J4 ≈ 121.3° swings the wrist link past horizontal.
    With the arm typically mounted ~400 mm above the workspace surface the
    effective height above the table is ~26 mm — solidly in grab range.

    We verify:
      • z is dramatically lower than home (at least 200 mm drop)
      • The gripper has significant forward reach (r > 200 mm)
    """
    ref = [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]
    pos = detector.compute_gripper_position(ref)
    # z should be much lower than home
    assert pos[2] < HOME_Z - 200  # at least 200mm below home
    # Significant horizontal reach
    r = math.hypot(pos[0], pos[1])
    assert r > 200  # reaching forward


# ---------- check_grip tests ----------


def _make_objects_near(pos: np.ndarray, label: str = "can", offset_mm: float = 10.0):
    """Create a detected object near the given position."""
    return [
        {
            "label": label,
            "position": {
                "x": pos[0] + offset_mm,
                "y": pos[1],
                "z": pos[2],
            },
            "width_mm": 66.0,
        }
    ]


def test_check_grip_true_when_close_and_closed(detector: VirtualGripDetector):
    """Grip succeeds when gripper is close to object AND closed enough."""
    ref = [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]
    pos = detector.compute_gripper_position(ref)
    objects = _make_objects_near(pos, "soda_can", offset_mm=15.0)

    result = detector.check_grip(ref, gripper_width_mm=30.0, detected_objects=objects)
    assert result.gripped is True
    assert result.object_label == "soda_can"
    assert result.distance_mm < 50.0
    assert isinstance(result.gripper_position_mm, np.ndarray)
    assert result.message.startswith("Gripping")


def test_check_grip_false_when_too_far(detector: VirtualGripDetector):
    """Grip fails when object is too far away."""
    ref = [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]
    pos = detector.compute_gripper_position(ref)
    # Place object 200mm away
    objects = _make_objects_near(pos, "mug", offset_mm=200.0)

    result = detector.check_grip(ref, gripper_width_mm=30.0, detected_objects=objects)
    assert result.gripped is False
    assert "Too far" in result.message


def test_check_grip_false_when_gripper_too_wide(detector: VirtualGripDetector):
    """Grip fails when gripper is open too wide (not closed on object)."""
    ref = [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]
    pos = detector.compute_gripper_position(ref)
    objects = _make_objects_near(pos, "bottle", offset_mm=10.0)

    result = detector.check_grip(ref, gripper_width_mm=60.0, detected_objects=objects)
    assert result.gripped is False
    assert "too wide" in result.message


def test_check_grip_false_empty_objects(detector: VirtualGripDetector):
    """Grip fails when no objects are detected."""
    ref = [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]
    result = detector.check_grip(ref, gripper_width_mm=30.0, detected_objects=[])
    assert result.gripped is False
    assert "No objects" in result.message
