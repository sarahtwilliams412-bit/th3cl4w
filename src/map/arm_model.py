"""FK-based arm skeleton model.

Computes 3D joint positions and link geometries from joint angles
using D1Kinematics, ready for Three.js rendering.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.kinematics.kinematics import D1Kinematics
from src.map.scene import ArmSkeletonData

logger = logging.getLogger(__name__)

# Link visual radii (meters) — approximate cylinder radii for rendering
LINK_RADII = [0.04, 0.035, 0.03, 0.03, 0.025, 0.025, 0.02, 0.015]


class ArmModel:
    """Maintains the arm's FK-based skeleton from joint angles."""

    def __init__(self):
        self._kin = D1Kinematics()
        self._last_angles_rad: Optional[np.ndarray] = None

    def update(
        self,
        joint_angles_deg: List[float],
        gripper_mm: float = 0.0,
    ) -> ArmSkeletonData:
        """Compute arm skeleton from joint angles.

        Args:
            joint_angles_deg: 6 or 7 joint angles in degrees.
            gripper_mm: Gripper opening in mm.

        Returns:
            ArmSkeletonData with joints, links, ee_pose.
        """
        # Pad to 7 joints if needed
        angles = list(joint_angles_deg)
        while len(angles) < 7:
            angles.append(0.0)
        angles = angles[:7]

        angles_rad = np.radians(angles)
        self._last_angles_rad = angles_rad

        # Get joint positions (base + 7 joints = 8 positions)
        positions = self._kin.get_joint_positions_3d(angles_rad)

        # Build links (cylinders between consecutive joints)
        joints_list = [p.tolist() for p in positions]
        links = []
        for i in range(len(positions) - 1):
            radius = LINK_RADII[i] if i < len(LINK_RADII) else 0.015
            links.append(
                {
                    "start": positions[i].tolist(),
                    "end": positions[i + 1].tolist(),
                    "radius": radius,
                }
            )

        # End-effector pose
        ee_pose = self._kin.forward_kinematics(angles_rad)

        return ArmSkeletonData(
            joints=joints_list,
            links=links,
            gripper_mm=gripper_mm,
            ee_pose=ee_pose.tolist(),
            joint_angles_deg=angles,
        )

    def compute_reach_envelope(self, n_samples: int = 5000, radius_m: float = 0.55) -> dict:
        """Pre-compute a reach envelope mesh (hemisphere of reachable points).

        Returns dict with 'vertices' and 'faces' for Three.js.
        """
        # Generate hemisphere vertices
        vertices = []
        n_rings = 20
        n_segments = 40

        for i in range(n_rings + 1):
            phi = (np.pi / 2) * (i / n_rings)  # 0 to pi/2 (upper hemisphere)
            r = radius_m * np.cos(phi)
            z = radius_m * np.sin(phi)
            for j in range(n_segments):
                theta = 2 * np.pi * j / n_segments
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                vertices.append([x, y, z])

        # Generate faces (triangle strips between rings)
        faces = []
        for i in range(n_rings):
            for j in range(n_segments):
                a = i * n_segments + j
                b = i * n_segments + (j + 1) % n_segments
                c = (i + 1) * n_segments + j
                d = (i + 1) * n_segments + (j + 1) % n_segments
                faces.append([a, b, c])
                faces.append([b, d, c])

        return {"vertices": vertices, "faces": faces, "radius_m": radius_m}
