"""
Forward Kinematics engine — direct port of the JS fkPositions() from the th3cl4w web UI.

Geometric FK using Rz/Ry rotation chain (NOT DH parameters).
Coordinate frame: Z=up, X=forward, Y=left.
"""

import math
import numpy as np
from typing import Optional
import requests

# Link lengths (meters)
D0 = 0.1215  # base to shoulder height
L1 = 0.2085  # shoulder to elbow
L2 = 0.2085  # elbow to wrist
L3 = 0.1130  # wrist to end-effector


def _rz(a: float) -> np.ndarray:
    """Rotation about Z axis."""
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def _ry(a: float) -> np.ndarray:
    """Rotation about Y axis."""
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def fk_positions(joints_deg: list[float]) -> list[list[float]]:
    """
    Compute forward kinematics for the D1 arm.

    Args:
        joints_deg: 6 joint angles in degrees [J0..J5].

    Returns:
        5 positions [base, shoulder, elbow, wrist, end_effector],
        each as [x, y, z] in meters.

    Note: This is a direct port of the JS fkPositions(). The JS viz flips
    signs on J1/J2 relative to the real arm, but that's baked into this
    function exactly as the JS does it.
    """
    # Pad to 6 joints, convert to radians
    j = [(joints_deg[i] if i < len(joints_deg) else 0.0) * math.pi / 180.0 for i in range(6)]

    base = [0.0, 0.0, 0.0]
    shoulder = [0.0, 0.0, D0]

    # J0 yaw + J1 pitch
    R = _rz(j[0]) @ _ry(j[1])
    elbow = [shoulder[k] + v for k, v in enumerate(R @ [0, 0, L1])]

    # 90° home bend + J2
    R = R @ _ry(math.pi / 2 + j[2])
    wrist = [elbow[k] + v for k, v in enumerate(R @ [0, 0, L2])]

    # J3 forearm roll + J4 wrist pitch
    R = R @ _rz(j[3])
    R = R @ _ry(j[4])
    ee = [wrist[k] + v for k, v in enumerate(R @ [0, 0, L3])]

    return [base, shoulder, elbow, wrist, ee]


def project_to_camera(
    positions_3d: list[list[float]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Project 3D positions into camera pixel space using OpenCV.

    Args:
        positions_3d: List of [x, y, z] positions in meters.
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Distortion coefficients (can be zeros).
        rvec: Rodrigues rotation vector (3,) or (3,1).
        tvec: Translation vector (3,) or (3,1).

    Returns:
        List of (u, v) pixel coordinates.
    """
    import cv2

    pts = np.array(positions_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)

    projected, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs)
    return [(float(p[0][0]), float(p[0][1])) for p in projected]


def project_to_camera_pinhole(
    positions_3d: list[list[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rvec: list[float],
    tvec: list[float],
) -> list[Optional[tuple[float, float]]]:
    """
    Project 3D positions using manual pinhole math (matches JS projectPoint3D).

    Args:
        positions_3d: List of [x, y, z] positions.
        fx, fy, cx, cy: Intrinsic parameters.
        rvec: Rodrigues rotation vector [rx, ry, rz].
        tvec: Translation vector [tx, ty, tz].

    Returns:
        List of (u, v) pixel coordinates, or None if behind camera.
    """
    # Rodrigues vector to rotation matrix
    rv = np.array(rvec, dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        R = np.eye(3)
    else:
        k = rv / angle
        c, s = math.cos(angle), math.sin(angle)
        v = 1 - c
        R = np.array(
            [
                [k[0] * k[0] * v + c, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
                [k[1] * k[0] * v + k[2] * s, k[1] * k[1] * v + c, k[1] * k[2] * v - k[0] * s],
                [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, k[2] * k[2] * v + c],
            ]
        )

    t = np.array(tvec, dtype=np.float64)
    results = []
    for p in positions_3d:
        pc = R @ np.array(p) + t
        if pc[2] <= 0:
            results.append(None)
        else:
            results.append((fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy))
    return results


def joints_from_api(host: str = "localhost", port: int = 8080) -> list[float]:
    """
    Fetch current joint angles from the robot API.

    Returns:
        List of joint angles in degrees.
    """
    resp = requests.get(f"http://{host}:{port}/api/state", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    # Extract joint positions - adapt to actual API response format
    if "joints" in data:
        return [float(j) for j in data["joints"]]
    elif "joint_positions" in data:
        return [float(j) for j in data["joint_positions"]]
    elif "position" in data:
        return [float(j) for j in data["position"]]
    else:
        raise ValueError(f"Cannot find joint data in API response: {list(data.keys())}")
