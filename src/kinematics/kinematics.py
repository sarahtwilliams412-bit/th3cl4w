"""
Forward/Inverse Kinematics for the Unitree D1 7-DOF robotic arm.

Uses standard Denavit-Hartenberg (DH) convention and numerical IK
via damped least-squares (Levenberg-Marquardt style).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from src.control.joint_service import NUM_JOINTS, get_dh_params as _get_dh_params

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DH Parameter container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DHParameters:
    """Standard DH parameters for a single revolute joint.

    Convention (Craig):
        a     — link length (along x_{i})
        d     — link offset (along z_{i-1})
        alpha — link twist  (about x_{i})  [radians]
        theta_offset — fixed offset added to the joint variable [radians]
    """

    a: float
    d: float
    alpha: float
    theta_offset: float = 0.0


# ---------------------------------------------------------------------------
# D1 DH table — imported from joint_service (single source of truth)
# ---------------------------------------------------------------------------

_D1_DH: list[DHParameters] = [
    DHParameters(a=dh.a, d=dh.d, alpha=dh.alpha, theta_offset=dh.theta_offset)
    for dh in _get_dh_params()
]

assert len(_D1_DH) == NUM_JOINTS, f"DH table length {len(_D1_DH)} != NUM_JOINTS {NUM_JOINTS}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dh_transform(dh: DHParameters, theta: float, out: np.ndarray | None = None) -> np.ndarray:
    """Return the 4×4 homogeneous transform for a single DH frame.

    Pass a pre-allocated (4,4) array as `out` to avoid allocation in hot paths.
    """
    t = theta + dh.theta_offset
    ct, st = math.cos(t), math.sin(t)
    ca, sa = math.cos(dh.alpha), math.sin(dh.alpha)
    if out is None:
        out = np.empty((4, 4))
    out[0, 0] = ct
    out[0, 1] = -st * ca
    out[0, 2] = st * sa
    out[0, 3] = dh.a * ct
    out[1, 0] = st
    out[1, 1] = ct * ca
    out[1, 2] = -ct * sa
    out[1, 3] = dh.a * st
    out[2, 0] = 0.0
    out[2, 1] = sa
    out[2, 2] = ca
    out[2, 3] = dh.d
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0
    return out


def _pose_error(T_target: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """6-vector pose error [position_error; orientation_error]."""
    pos_err = T_target[:3, 3] - T_current[:3, 3]
    # Orientation error via rotation matrix difference
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    rotvec = Rotation.from_matrix(R_err).as_rotvec()
    return np.concatenate([pos_err, rotvec])


# ---------------------------------------------------------------------------
# Main kinematics class
# ---------------------------------------------------------------------------


class D1Kinematics:
    """Kinematics solver for the Unitree D1 7-DOF arm."""

    def __init__(self, dh_params: list[DHParameters] | None = None) -> None:
        self.dh_params = dh_params if dh_params is not None else list(_D1_DH)
        self.n_joints = len(self.dh_params)

    # ----- forward kinematics -----

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute the 4×4 end-effector pose from joint angles."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        assert joint_angles.shape == (
            self.n_joints,
        ), f"Expected {self.n_joints} angles, got {joint_angles.shape}"
        T = np.eye(4)
        for dh, q in zip(self.dh_params, joint_angles):
            T = T @ _dh_transform(dh, q)
        return T

    # ----- joint positions -----

    def get_joint_positions_3d(self, joint_angles: np.ndarray) -> list[np.ndarray]:
        """Return the 3-D position of each joint frame origin (including base and EE)."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        T = np.eye(4)
        positions: list[np.ndarray] = [T[:3, 3].copy()]
        for dh, q in zip(self.dh_params, joint_angles):
            T = T @ _dh_transform(dh, q)
            positions.append(T[:3, 3].copy())
        return positions  # length = n_joints + 1

    # ----- geometric Jacobian -----

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute the 6×N geometric Jacobian (world frame)."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        T = np.eye(4)
        transforms: list[np.ndarray] = [T.copy()]
        for dh, q in zip(self.dh_params, joint_angles):
            T = T @ _dh_transform(dh, q)
            transforms.append(T.copy())

        p_ee = transforms[-1][:3, 3]
        J = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            z_i = transforms[i][:3, 2]  # z-axis of frame i
            p_i = transforms[i][:3, 3]  # origin of frame i
            J[:3, i] = np.cross(z_i, p_ee - p_i)  # linear velocity
            J[3:, i] = z_i  # angular velocity
        return J

    # ----- inverse kinematics (damped least-squares) -----

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        damping: float = 0.01,
    ) -> np.ndarray:
        """Numerical IK using damped least-squares (Levenberg-Marquardt).

        Parameters
        ----------
        target_pose : (4, 4) homogeneous transform of desired end-effector pose.
        q_init      : initial joint configuration; zeros if *None*.
        max_iter    : maximum iterations.
        tol         : convergence tolerance on pose error norm.
        damping     : damping factor λ for DLS.

        Returns
        -------
        joint_angles : (N,) array of joint angles.
        """
        target_pose = np.asarray(target_pose, dtype=float)
        q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(self.n_joints)

        for i in range(max_iter):
            T_cur = self.forward_kinematics(q)
            err = _pose_error(target_pose, T_cur)
            if np.linalg.norm(err) < tol:
                logger.debug("IK converged in %d iterations", i)
                return q

            J = self.jacobian(q)
            # DLS: dq = J^T (J J^T + λ²I)^{-1} e
            JJt = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = q + dq

        logger.warning(
            "IK did not converge after %d iterations (err=%.4e)", max_iter, np.linalg.norm(err)
        )
        return q
