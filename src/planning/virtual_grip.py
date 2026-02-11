"""
Virtual Grip Detector — Geometric FK + proximity-based grip detection for sim mode.

Uses simplified geometric forward kinematics to compute the gripper tip position
from joint angles, then checks proximity to detected objects.

Coordinate frame: X=forward, Y=left, Z=up.  Origin at base centre on the table.

D1 arm link lengths (mm):
    d0  = 121.5   base-to-shoulder height
    L1  = 208.5   upper arm
    L2  = 208.5   forearm
    L3  = 113.0   wrist-to-gripper tip

Joint mapping (6-DOF, indices 0-5):
    J0 — base yaw        (rotation in XY plane)
    J1 — shoulder pitch   (+ = forward/down from vertical)
    J2 — elbow pitch      (extends from shoulder angle)
    J3 — forearm roll     (ignored for position)
    J4 — wrist pitch      (extends from elbow angle)
    J5 — gripper roll     (ignored for position)

At home (all zeros) the arm points straight up:
    gripper z = d0 + L1 + L2 + L3 = 651.5 mm

Reference grab pose [1.0, 25.9, 6.7, 0.5, 88.7, 3.3]:
    The total pitch (J1+J2+J4 ≈ 121.3°) swings the last link past horizontal,
    bringing the gripper low enough for a tabletop grab.  The geometric FK gives
    z ≈ 426 mm in the base frame; since the arm is typically mounted ~400 mm
    above the workspace plane the effective height above the table is ~26 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GripCheckResult:
    """Result of a virtual grip check."""

    gripped: bool
    object_label: Optional[str] = None
    distance_mm: float = float("inf")
    gripper_width_mm: float = 0.0
    object_width_mm: float = 0.0
    gripper_position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    message: str = ""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class VirtualGripDetector:
    """Compute gripper position via geometric FK and check proximity to objects."""

    # Arm geometry (mm)
    D0 = 121.5  # base-to-shoulder height
    L1 = 208.5  # upper arm
    L2 = 208.5  # forearm
    L3 = 113.0  # wrist-to-tip

    # Default thresholds
    DEFAULT_GRIP_DISTANCE_MM = 50.0  # max 3-D distance to count as "close"
    DEFAULT_GRIP_WIDTH_MM = 45.0  # gripper must be narrower than this

    def compute_gripper_position(self, joints_deg: list[float]) -> np.ndarray:
        """Compute gripper tip XYZ (mm) from joint angles in degrees.

        Returns a 3-element numpy array [x, y, z] in the base frame.
        """
        j0 = math.radians(joints_deg[0])  # base yaw
        j1 = math.radians(joints_deg[1])  # shoulder pitch
        j2 = math.radians(joints_deg[2])  # elbow pitch
        # joints_deg[3] = forearm roll → ignored for position
        j4 = math.radians(joints_deg[4])  # wrist pitch
        # joints_deg[5] = gripper roll  → ignored for position

        # Cumulative pitch angle from vertical (0 = straight up)
        pitch1 = j1
        pitch2 = pitch1 + j2
        pitch3 = pitch2 + j4

        # Compute in the vertical plane (r=radial, z=height)
        z = self.D0
        r = 0.0

        r += self.L1 * math.sin(pitch1)
        z += self.L1 * math.cos(pitch1)

        r += self.L2 * math.sin(pitch2)
        z += self.L2 * math.cos(pitch2)

        r += self.L3 * math.sin(pitch3)
        z += self.L3 * math.cos(pitch3)

        # Project onto XY using base yaw
        x = r * math.cos(j0)
        y = r * math.sin(j0)

        return np.array([x, y, z])

    def check_grip(
        self,
        joints_deg: list[float],
        gripper_width_mm: float,
        detected_objects: list[dict[str, Any]],
        grip_distance_mm: float | None = None,
        grip_width_mm: float | None = None,
    ) -> GripCheckResult:
        """Check whether the gripper is gripping any detected object.

        Parameters
        ----------
        joints_deg : list of 6 floats
            Current joint angles in degrees.
        gripper_width_mm : float
            Current gripper opening width in mm.
        detected_objects : list of dicts
            Each must have ``label`` and ``position`` with x, y, z in mm.
            Optional ``width_mm`` for the object diameter.
        grip_distance_mm : float, optional
            Max 3-D distance (mm) to consider gripping.
        grip_width_mm : float, optional
            Gripper must be narrower than this to count as closed.
        """
        if grip_distance_mm is None:
            grip_distance_mm = self.DEFAULT_GRIP_DISTANCE_MM
        if grip_width_mm is None:
            grip_width_mm = self.DEFAULT_GRIP_WIDTH_MM

        pos = self.compute_gripper_position(joints_deg)

        if not detected_objects:
            return GripCheckResult(
                gripped=False,
                gripper_position_mm=pos,
                gripper_width_mm=gripper_width_mm,
                message="No objects detected",
            )

        # Find closest object
        closest_label: str | None = None
        closest_dist = float("inf")
        closest_obj_width = 0.0

        for obj in detected_objects:
            obj_pos = obj.get("position", {})
            op = np.array(
                [
                    obj_pos.get("x", 0.0),
                    obj_pos.get("y", 0.0),
                    obj_pos.get("z", 0.0),
                ]
            )
            dist = float(np.linalg.norm(pos - op))
            if dist < closest_dist:
                closest_dist = dist
                closest_label = obj.get("label", "unknown")
                closest_obj_width = obj.get("width_mm", 30.0)

        gripper_closed = gripper_width_mm < grip_width_mm
        close_enough = closest_dist <= grip_distance_mm
        gripped = gripper_closed and close_enough

        if gripped:
            msg = f"Gripping '{closest_label}' at {closest_dist:.1f}mm"
        elif not close_enough:
            msg = f"Too far from '{closest_label}': {closest_dist:.1f}mm > {grip_distance_mm}mm"
        else:
            msg = f"Gripper too wide ({gripper_width_mm:.1f}mm) to grip '{closest_label}'"

        return GripCheckResult(
            gripped=gripped,
            object_label=closest_label,
            distance_mm=round(closest_dist, 2),
            gripper_width_mm=gripper_width_mm,
            object_width_mm=closest_obj_width,
            gripper_position_mm=pos,
            message=msg,
        )
