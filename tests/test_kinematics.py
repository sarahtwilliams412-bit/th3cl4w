"""Tests for the D1 kinematics module."""

import numpy as np
import pytest

from src.kinematics.kinematics import D1Kinematics

KIN = D1Kinematics()


class TestForwardKinematics:
    def test_zero_config_is_valid_transform(self):
        """FK at zero config returns a valid 4x4 homogeneous transform."""
        T = KIN.forward_kinematics(np.zeros(7))
        assert T.shape == (4, 4)
        # Last row must be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
        # Rotation part is orthonormal
        R = T[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)

    def test_zero_config_reach(self):
        """At zero config the EE should be at the expected reach along z."""
        T = KIN.forward_kinematics(np.zeros(7))
        total_d = sum(dh.d for dh in KIN.dh_params)
        # With all-zero angles and a = 0 for all joints, EE z should equal sum of d's
        np.testing.assert_almost_equal(T[2, 3], total_d, decimal=6)


class TestJacobian:
    def test_shape(self):
        J = KIN.jacobian(np.zeros(7))
        assert J.shape == (6, 7)

    def test_numerically_matches_finite_diff(self):
        """Jacobian should roughly match finite-difference approximation."""
        q = np.random.default_rng(42).uniform(-0.5, 0.5, 7)
        J_analytic = KIN.jacobian(q)

        eps = 1e-6
        J_fd = np.zeros((6, 7))
        T0 = KIN.forward_kinematics(q)
        for i in range(7):
            q_plus = q.copy()
            q_plus[i] += eps
            T_plus = KIN.forward_kinematics(q_plus)
            # Position part
            J_fd[:3, i] = (T_plus[:3, 3] - T0[:3, 3]) / eps
            # Orientation part (approx via rotation difference)
            from scipy.spatial.transform import Rotation

            dR = T_plus[:3, :3] @ T0[:3, :3].T
            J_fd[3:, i] = Rotation.from_matrix(dR).as_rotvec() / eps

        np.testing.assert_array_almost_equal(J_analytic, J_fd, decimal=4)


class TestInverseKinematics:
    def test_fk_ik_roundtrip(self):
        """IK should recover joint angles that reproduce the FK target."""
        q_original = np.array([0.1, -0.2, 0.3, -0.4, 0.15, -0.1, 0.05])
        T_target = KIN.forward_kinematics(q_original)

        q_solved = KIN.inverse_kinematics(T_target, q_init=np.zeros(7))
        T_solved = KIN.forward_kinematics(q_solved)

        # Position should match closely
        np.testing.assert_array_almost_equal(T_solved[:3, 3], T_target[:3, 3], decimal=4)
        # Orientation should match closely
        np.testing.assert_array_almost_equal(T_solved[:3, :3], T_target[:3, :3], decimal=3)


class TestJointPositions:
    def test_count(self):
        """Should return n_joints + 1 positions (base + each joint)."""
        positions = KIN.get_joint_positions_3d(np.zeros(7))
        assert len(positions) == 8

    def test_base_at_origin(self):
        positions = KIN.get_joint_positions_3d(np.zeros(7))
        np.testing.assert_array_almost_equal(positions[0], [0, 0, 0])
