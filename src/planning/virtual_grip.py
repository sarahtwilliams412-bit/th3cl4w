"""Virtual Grip Detection — geometric grip verification for simulation mode.

Computes gripper position via FK and checks proximity + gripper width
against detected objects to determine if a "grip" would succeed.

FK ported from arm3d.js forwardKinematics() — uses rotation matrices
with the same link lengths and joint conventions as the 3D simulator.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("th3cl4w.planning.virtual_grip")


@dataclass
class GripCheckResult:
    gripped: bool
    object_label: str = ""
    distance_mm: float = float("inf")
    gripper_width_mm: float = 0.0
    object_width_mm: float = 0.0
    gripper_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    message: str = ""


def _ry(a: float) -> np.ndarray:
    """Y-axis rotation matrix (3x3)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    """Z-axis rotation matrix (3x3)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class VirtualGripDetector:
    """Determines if sim gripper has 'gripped' a detected object.

    Uses the geometric FK from arm3d.js (ported to Python) to compute
    the gripper's 3D position from joint angles, then checks proximity
    to detected objects.
    """

    # Link lengths in mm (matching arm3d.js D1_LINKS)
    D0 = 121.5   # base to shoulder
    L1 = 208.5   # shoulder to elbow
    L2 = 208.5   # elbow to wrist
    L3 = 113.0   # wrist to end-effector

    def __init__(
        self,
        grip_distance_threshold_mm: float = 60.0,
        grip_width_margin_mm: float = 10.0,
    ):
        self.distance_threshold = grip_distance_threshold_mm
        self.width_margin = grip_width_margin_mm

    def compute_gripper_position(self, joints_deg: list[float]) -> np.ndarray:
        """Compute gripper XYZ position from joint angles using geometric FK.

        Exact port of arm3d.js forwardKinematics(). Computes in Z-up frame.

        Joint conventions (matching JS):
          J0 = base yaw (Rz)
          J1 = shoulder pitch (Ry)
          J2 = elbow pitch (Ry with +PI/2 offset)
          J3 = forearm roll (Rz)
          J4 = wrist pitch (Ry)
          J5 = gripper roll (unused for position)

        Returns position in mm as np.array([x, y, z]) in Z-up frame.
        """
        j = [math.radians(a) for a in joints_deg[:6]]

        shoulder = np.array([0.0, 0.0, self.D0])

        R = _rz(j[0]) @ _ry(j[1])
        elbow = shoulder + R @ np.array([0.0, 0.0, self.L1])

        R = R @ _ry(math.pi / 2 + j[2])
        wrist = elbow + R @ np.array([0.0, 0.0, self.L2])

        R = R @ _rz(j[3])
        R = R @ _ry(j[4])
        ee = wrist + R @ np.array([0.0, 0.0, self.L3])

        return ee

    def check_grip(
        self,
        joints_deg: list[float],
        gripper_width_mm: float,
        detected_objects: list[dict],
    ) -> GripCheckResult:
        """Check if the gripper would successfully grip any detected object.

        Args:
            joints_deg: Current 6 joint angles in degrees
            gripper_width_mm: Current gripper opening in mm
            detected_objects: List of dicts with at least:
                - "label": str
                - "position_mm": [x, y, z] or just [x, y] (z assumed 0)
                - "width_mm": float (object width for grip check)

        Returns:
            GripCheckResult with grip status and details.
        """
        gripper_pos = self.compute_gripper_position(joints_deg)

        best_match = None
        best_dist = float("inf")

        for obj in detected_objects:
            label = obj.get("label", "unknown")
            pos = obj.get("position_mm", [0, 0, 0])
            obj_width = obj.get("width_mm", 66.0)  # default Red Bull can

            # Ensure 3D
            if len(pos) < 3:
                pos = list(pos) + [0.0]
            obj_pos = np.array(pos[:3], dtype=float)

            dist = float(np.linalg.norm(gripper_pos - obj_pos))

            if dist < best_dist:
                best_dist = dist
                best_match = (label, obj_width, dist)

        if best_match is None:
            return GripCheckResult(
                gripped=False,
                gripper_position_mm=tuple(gripper_pos),
                gripper_width_mm=gripper_width_mm,
                message="No objects detected",
            )

        label, obj_width, dist = best_match

        close_enough = dist < self.distance_threshold
        grip_tight = gripper_width_mm < (obj_width + self.width_margin)
        gripped = close_enough and grip_tight

        return GripCheckResult(
            gripped=gripped,
            object_label=label,
            distance_mm=dist,
            gripper_width_mm=gripper_width_mm,
            object_width_mm=obj_width,
            gripper_position_mm=tuple(gripper_pos),
            message=f"{'Gripped' if gripped else 'Missed'} {label} at {dist:.1f}mm"
            + (
                f" (gripper too wide: {gripper_width_mm:.1f}>{obj_width + self.width_margin:.1f}mm)"
                if not grip_tight
                else ""
            )
            + (
                f" (too far: {dist:.1f}>{self.distance_threshold:.1f}mm)"
                if not close_enough
                else ""
            ),
        )
