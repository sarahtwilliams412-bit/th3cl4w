"""
Forward kinematics for the Unitree D1 arm.

Computes end-effector pose from joint angles using the DH parameter table.
"""

import math
from typing import List

import numpy as np

from src.kinematics.dh_params import D1_DH_PARAMS, DHParam, NUM_ARM_JOINTS


def _dh_transform(param: DHParam, theta_var: float) -> np.ndarray:
    """Compute the 4x4 homogeneous transform for one DH link (modified DH).

    Args:
        param: DH parameters for this link.
        theta_var: Variable joint angle (radians).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    theta = theta_var + param.theta
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(param.alpha), math.sin(param.alpha)
    a, d = param.a, param.d

    return np.array([
        [ct,     -st,     0.0,   a],
        [st*ca,  ct*ca,  -sa,   -sa*d],
        [st*sa,  ct*sa,   ca,    ca*d],
        [0.0,    0.0,     0.0,   1.0],
    ], dtype=np.float64)


def joint_positions_to_transforms(
    joint_angles: np.ndarray,
    dh_params: List[DHParam] = D1_DH_PARAMS,
) -> List[np.ndarray]:
    """Compute the cumulative transform for each link in the chain.

    Args:
        joint_angles: Array of joint angles (radians), length must be
            >= len(dh_params).
        dh_params: DH parameter table.  Defaults to D1_DH_PARAMS.

    Returns:
        List of 4x4 homogeneous transforms, one per joint.  transforms[i]
        is T_0_i (base frame to frame i).
    """
    if len(joint_angles) < len(dh_params):
        raise ValueError(
            f"Expected at least {len(dh_params)} joint angles, got {len(joint_angles)}"
        )

    transforms: List[np.ndarray] = []
    T = np.eye(4, dtype=np.float64)
    for i, param in enumerate(dh_params):
        T = T @ _dh_transform(param, joint_angles[i])
        transforms.append(T.copy())
    return transforms


def forward_kinematics(
    joint_angles: np.ndarray,
    dh_params: List[DHParam] = D1_DH_PARAMS,
) -> np.ndarray:
    """Compute the end-effector pose from joint angles.

    Args:
        joint_angles: Array of joint angles (radians).  Only the first
            len(dh_params) elements are used (so passing all 7 D1 joints
            is fine — the gripper joint is ignored).
        dh_params: DH parameter table.  Defaults to D1_DH_PARAMS.

    Returns:
        4x4 homogeneous transformation matrix of the end-effector in the
        base frame.
    """
    transforms = joint_positions_to_transforms(joint_angles, dh_params)
    return transforms[-1]


def end_effector_position(joint_angles: np.ndarray) -> np.ndarray:
    """Convenience: return the 3D position (x, y, z) of the end-effector."""
    T = forward_kinematics(joint_angles)
    return T[:3, 3].copy()
