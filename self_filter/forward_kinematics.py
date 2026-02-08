"""
Forward Kinematics for the D1 Arm — Self-Filter Pipeline

Computes link frame transforms and link segment endpoints using
standard DH convention. This is a standalone implementation for the
self-filter pipeline that works with configurable DH parameters
(from self_filter/config.yaml), separate from the main FK in
src/kinematics/kinematics.py.
"""

from __future__ import annotations

import numpy as np


class ForwardKinematics:
    """DH-parameter forward kinematics for serial link arms.

    Parameters
    ----------
    dh_params : list
        List of [a_mm, alpha_rad, d_mm, theta_offset_rad] per joint.
    """

    def __init__(self, dh_params: list[list[float]]):
        self.dh = np.array(dh_params, dtype=np.float64)
        self.n_joints = len(dh_params)

    def dh_matrix(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Standard DH transformation matrix (4x4).

        Parameters
        ----------
        a : float
            Link length (mm).
        alpha : float
            Link twist (rad).
        d : float
            Link offset (mm).
        theta : float
            Joint angle + offset (rad).

        Returns
        -------
        np.ndarray
            4x4 homogeneous transform.
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,   -st * ca,  st * sa,  a * ct],
            [st,    ct * ca, -ct * sa,  a * st],
            [0.0,   sa,       ca,       d],
            [0.0,   0.0,      0.0,      1.0],
        ])

    def compute_link_frames(self, joint_angles: np.ndarray) -> list[np.ndarray]:
        """Compute the 4x4 transform for each link frame.

        Parameters
        ----------
        joint_angles : np.ndarray
            float64[N] joint angles in radians.

        Returns
        -------
        list[np.ndarray]
            List of N+1 homogeneous transforms (4x4 each).
            frames[0] = base frame (identity)
            frames[i] = frame of joint i (1-indexed)
            frames[N] = end-effector frame
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        n = min(len(joint_angles), self.n_joints)

        frames = [np.eye(4)]
        T_cumulative = np.eye(4)

        for i in range(n):
            a, alpha, d, theta_offset = self.dh[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T_cumulative = T_cumulative @ T_i
            frames.append(T_cumulative.copy())

        return frames

    def link_endpoints(
        self, joint_angles: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Compute line segment endpoints for each link.

        Parameters
        ----------
        joint_angles : np.ndarray
            float64[N] joint angles in radians.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of N (start_xyz_mm, end_xyz_mm) pairs, one per link.
        """
        frames = self.compute_link_frames(joint_angles)
        segments = []
        for i in range(len(frames) - 1):
            start = frames[i][:3, 3].copy()
            end = frames[i + 1][:3, 3].copy()
            segments.append((start, end))
        return segments

    def end_effector_pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute end-effector pose (4x4 transform).

        Parameters
        ----------
        joint_angles : np.ndarray
            Joint angles in radians.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transform of the end-effector.
        """
        frames = self.compute_link_frames(joint_angles)
        return frames[-1]
