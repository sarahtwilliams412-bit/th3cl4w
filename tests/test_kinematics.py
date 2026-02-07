"""Tests for forward kinematics."""

import math

import numpy as np
import pytest

from src.kinematics.dh_params import D1_DH_PARAMS, DHParam, NUM_ARM_JOINTS
from src.kinematics.forward import (
    _dh_transform,
    end_effector_position,
    forward_kinematics,
    joint_positions_to_transforms,
)


class TestDHParams:
    def test_num_joints(self):
        assert NUM_ARM_JOINTS == 6
        assert len(D1_DH_PARAMS) == 6

    def test_params_are_frozen(self):
        param = D1_DH_PARAMS[0]
        with pytest.raises(AttributeError):
            param.a = 99.0


class TestDHTransform:
    def test_identity_params_zero_angle(self):
        """With all-zero DH params and zero angle, result should be identity."""
        param = DHParam(a=0, alpha=0, d=0, theta=0)
        T = _dh_transform(param, 0.0)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_pure_translation_d(self):
        """Pure offset along z: d != 0, everything else zero."""
        param = DHParam(a=0, alpha=0, d=0.5, theta=0)
        T = _dh_transform(param, 0.0)
        expected = np.eye(4)
        expected[2, 3] = 0.5  # z translation
        np.testing.assert_array_almost_equal(T, expected)

    def test_pure_translation_a(self):
        """Pure link length: a != 0, everything else zero."""
        param = DHParam(a=0.3, alpha=0, d=0, theta=0)
        T = _dh_transform(param, 0.0)
        expected = np.eye(4)
        expected[0, 3] = 0.3  # x translation
        np.testing.assert_array_almost_equal(T, expected)

    def test_rotation_90_degrees(self):
        """90-degree rotation about z."""
        param = DHParam(a=0, alpha=0, d=0, theta=0)
        T = _dh_transform(param, math.pi / 2)
        # Should rotate x->y, y->-x
        assert abs(T[0, 0]) < 1e-10  # cos(90) ~ 0
        assert abs(T[0, 1] - (-1)) < 1e-10  # -sin(90) ~ -1
        assert abs(T[1, 0] - 1) < 1e-10  # sin(90) ~ 1

    def test_result_is_4x4(self):
        param = D1_DH_PARAMS[0]
        T = _dh_transform(param, 0.5)
        assert T.shape == (4, 4)
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])


class TestForwardKinematics:
    def test_zero_angles(self):
        """FK at zero configuration should give a valid transform."""
        joints = np.zeros(7)  # 7 joints (6 arm + gripper)
        T = forward_kinematics(joints)
        assert T.shape == (4, 4)
        # Last row
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_rotation_matrix_is_orthonormal(self):
        """The rotation part of the FK result should be orthonormal."""
        joints = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
        T = forward_kinematics(joints)
        R = T[:3, :3]
        # R^T @ R should be identity
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=10)
        # det(R) should be 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_too_few_joints_raises(self):
        with pytest.raises(ValueError, match="at least"):
            forward_kinematics(np.zeros(3))

    def test_transforms_chain_length(self):
        joints = np.zeros(7)
        transforms = joint_positions_to_transforms(joints)
        assert len(transforms) == NUM_ARM_JOINTS

    def test_end_effector_position_returns_3d(self):
        joints = np.zeros(7)
        pos = end_effector_position(joints)
        assert pos.shape == (3,)

    def test_reach_is_reasonable(self):
        """At zero config, end-effector should be within the 550mm reach."""
        joints = np.zeros(7)
        pos = end_effector_position(joints)
        distance = np.linalg.norm(pos)
        # Should be between 0 and 0.7m (550mm reach + some base height)
        assert 0.0 < distance < 0.7

    def test_different_configs_give_different_positions(self):
        pos1 = end_effector_position(np.zeros(7))
        pos2 = end_effector_position(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]))
        assert not np.allclose(pos1, pos2)
