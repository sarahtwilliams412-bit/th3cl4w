"""Tests for VLA action decoder — the safety-critical component."""

import pytest
from src.vla.action_decoder import ActionDecoder, ActionType, ArmAction, JOINT_LIMITS


@pytest.fixture
def decoder():
    return ActionDecoder()


@pytest.fixture
def home_joints():
    """All joints at home (0°)."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestJointDecoding:
    def test_basic_joint_action(self, decoder, home_joints):
        actions = [{"type": "joint", "id": 0, "delta": 5.0, "reason": "test"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert len(decoded) == 1
        a = decoded[0]
        assert a.action_type == ActionType.JOINT
        assert a.joint_id == 0
        assert a.target_angle == 5.0
        assert a.delta == 5.0
        assert not a.clamped
        assert not a.rejected

    def test_clamps_to_max_delta(self, decoder, home_joints):
        """Delta >10° should be clamped to 10°."""
        actions = [{"type": "joint", "id": 1, "delta": 25.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        a = decoded[0]
        assert a.delta == 10.0
        assert a.target_angle == 10.0
        assert a.clamped

    def test_clamps_negative_delta(self, decoder, home_joints):
        actions = [{"type": "joint", "id": 1, "delta": -20.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        a = decoded[0]
        assert a.delta == -10.0
        assert a.target_angle == -10.0
        assert a.clamped

    def test_clamps_to_joint_limits(self, decoder):
        """Joint at 75°, requesting +10° should clamp to limit (80°)."""
        joints = [0.0, 75.0, 0.0, 0.0, 0.0, 0.0]
        actions = [{"type": "joint", "id": 1, "delta": 10.0}]
        decoded = decoder.decode(actions, joints, 0.0)
        a = decoded[0]
        assert a.target_angle == 80.0  # J1 limit with 5° margin
        assert a.delta == 5.0
        assert a.clamped

    def test_rejects_invalid_joint_id(self, decoder, home_joints):
        actions = [{"type": "joint", "id": 7, "delta": 5.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].rejected

    def test_rejects_tiny_delta(self, decoder, home_joints):
        """Delta <0.5° is not worth executing."""
        actions = [{"type": "joint", "id": 0, "delta": 0.3}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].rejected

    def test_sequential_projected_state(self, decoder, home_joints):
        """Multiple actions on same joint should project state forward."""
        actions = [
            {"type": "joint", "id": 0, "delta": 10.0},
            {"type": "joint", "id": 0, "delta": 10.0},
        ]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].target_angle == 10.0
        assert decoded[1].target_angle == 20.0  # Projected from first action

    def test_absolute_angle_conversion(self, decoder, home_joints):
        """Some models output absolute angles instead of deltas."""
        actions = [{"type": "joint", "id": 0, "angle": 15.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        a = decoded[0]
        assert a.target_angle == 10.0  # Clamped from delta of 15
        assert a.clamped


class TestGripperDecoding:
    def test_basic_gripper(self, decoder, home_joints):
        actions = [{"type": "gripper", "position_mm": 50.0, "reason": "open"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        a = decoded[0]
        assert a.action_type == ActionType.GRIPPER
        assert a.gripper_mm == 50.0

    def test_gripper_clamped_to_range(self, decoder, home_joints):
        actions = [{"type": "gripper", "position_mm": 100.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].gripper_mm == 65.0

    def test_gripper_negative_clamped(self, decoder, home_joints):
        actions = [{"type": "gripper", "position_mm": -5.0}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].gripper_mm == 0.0


class TestSpecialActions:
    def test_verify_action(self, decoder, home_joints):
        actions = [{"type": "verify", "reason": "check alignment"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].action_type == ActionType.VERIFY
        assert not decoded[0].is_executable

    def test_done_action(self, decoder, home_joints):
        actions = [{"type": "done", "reason": "task complete"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].action_type == ActionType.DONE

    def test_unknown_action_becomes_verify(self, decoder, home_joints):
        actions = [{"type": "unknown_action"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        assert decoded[0].action_type == ActionType.VERIFY


class TestSafetySequencing:
    def test_blocks_simultaneous_shoulder_lift_elbow_extend(self, decoder, home_joints):
        """CRITICAL: shoulder lift (J1+) and elbow extend (J2+) must not happen simultaneously."""
        actions = [
            {"type": "joint", "id": 1, "delta": 5.0},  # shoulder UP
            {"type": "joint", "id": 2, "delta": 5.0},  # elbow EXTEND
        ]
        decoded = decoder.decode(actions, home_joints, 0.0)
        # Should have a verify checkpoint inserted between them
        types = [a.action_type for a in decoded]
        j1_idx = next(i for i, a in enumerate(decoded) if a.joint_id == 1)
        j2_idx = next(i for i, a in enumerate(decoded) if a.joint_id == 2)
        # There should be a verify between J1 and J2
        verify_between = any(
            decoded[i].action_type == ActionType.VERIFY for i in range(j1_idx + 1, j2_idx)
        )
        assert (
            verify_between
        ), "Safety verify must be inserted between shoulder lift and elbow extend"

    def test_allows_shoulder_lower_with_elbow_extend(self, decoder, home_joints):
        """Shoulder LOWER (J1-) with elbow extend is OK."""
        actions = [
            {"type": "joint", "id": 1, "delta": -5.0},  # shoulder DOWN (safe)
            {"type": "joint", "id": 2, "delta": 5.0},  # elbow EXTEND
        ]
        decoded = decoder.decode(actions, home_joints, 0.0)
        # No extra verify should be inserted
        verify_count = sum(1 for a in decoded if a.action_type == ActionType.VERIFY)
        assert verify_count == 0


class TestDescribe:
    def test_joint_describe(self, decoder, home_joints):
        actions = [{"type": "joint", "id": 0, "delta": 5.0, "reason": "rotate base"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        desc = decoded[0].describe()
        assert "J0" in desc
        assert "5.0°" in desc

    def test_gripper_describe(self, decoder, home_joints):
        actions = [{"type": "gripper", "position_mm": 50.0, "reason": "open wide"}]
        decoded = decoder.decode(actions, home_joints, 0.0)
        desc = decoded[0].describe()
        assert "50.0mm" in desc


class TestMixedActionSequence:
    def test_complex_pick_sequence(self, decoder, home_joints):
        """Simulate a realistic pick sequence from the model."""
        actions = [
            {"type": "joint", "id": 0, "delta": 8.0, "reason": "aim at can"},
            {"type": "joint", "id": 1, "delta": -10.0, "reason": "lean forward"},
            {"type": "joint", "id": 2, "delta": 10.0, "reason": "extend elbow"},
            {"type": "gripper", "position_mm": 55.0, "reason": "open gripper"},
            {"type": "verify", "reason": "check alignment"},
        ]
        decoded = decoder.decode(actions, home_joints, 0.0)
        executable = [a for a in decoded if a.is_executable]
        assert len(executable) == 4  # 3 joints + 1 gripper
        assert decoded[-1].action_type == ActionType.VERIFY
