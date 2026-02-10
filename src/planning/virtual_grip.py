"""
Virtual Grip Detector — Geometric FK + proximity-based grip detection for sim mode.

Uses simplified geometric forward kinematics (not full DH) to compute the
gripper tip position from joint angles, then checks proximity to detected objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GripCheckResult:
    """Result of a virtual grip check."""
    gripped: bool
    object_label: Optional[str] = None
    distance_mm: float = float("inf")
    gripper_width_mm: float = 0.0
    object_width_mm: float = 0.0
    position: Dict[str, float] = field(default_factory=dict)
    message: str = ""


class VirtualGripDetector:
    """Compute gripper position via geometric FK and check proximity to objects.

    D1 arm link lengths (mm):
        d0 = 121.5   base height
        L1 = 208.5   upper arm
        L2 = 208.5   forearm
        L3 = 113.0   wrist-to-gripper tip

    Joint mapping (6-DOF):
        J0 = base yaw (rotation about Z)
        J1 = shoulder pitch (+=forward, 0=vertical up)
        J2 = elbow pitch
        J3 = wrist roll (ignored for position)
        J4 = wrist pitch
        J5 = gripper (ignored for position)
    """

    D0 = 121.5   # base height mm
    L1 = 208.5   # upper arm mm
    L2 = 208.5   # forearm mm
    L3 = 113.0   # wrist-to-tip mm

    # Default thresholds
    GRIP_DISTANCE_MM = 50.0   # max distance to consider gripping
    GRIP_WIDTH_MM = 30.0      # gripper must be narrower than this to grip

    def compute_gripper_position(self, joints_deg: List[float]) -> Dict[str, float]:
        """Compute gripper tip XYZ in mm from joint angles in degrees.

        Coordinate frame: X=forward, Y=left, Z=up. Origin at base.

        The arm hangs vertical at home (all zeros) with tip at Z = d0+L1+L2+L3.
        J1 pitches forward from vertical: angle 0 = straight up, 90 = horizontal forward.
        J2 is elbow relative to upper arm. J4 is wrist pitch relative to forearm.
        """
        j0 = math.radians(joints_deg[0])  # base yaw
        j1 = math.radians(joints_deg[1])  # shoulder pitch from vertical
        j2 = math.radians(joints_deg[2])  # elbow pitch
        j4 = math.radians(joints_deg[4])  # wrist pitch

        # Cumulative pitch angle from vertical
        # Each joint adds to the tilt from vertical
        pitch1 = j1                    # after shoulder
        pitch2 = pitch1 + j2          # after elbow
        pitch3 = pitch2 + j4          # after wrist

        # Compute in the vertical plane (r = radial from base axis, z = height)
        # Start at base height
        z = self.D0
        r = 0.0

        # Upper arm: L1 along pitch1 from vertical
        r += self.L1 * math.sin(pitch1)
        z += self.L1 * math.cos(pitch1)

        # Forearm: L2 along pitch2 from vertical
        r += self.L2 * math.sin(pitch2)
        z += self.L2 * math.cos(pitch2)

        # Wrist-to-tip: L3 along pitch3 from vertical
        r += self.L3 * math.sin(pitch3)
        z += self.L3 * math.cos(pitch3)

        # Project radial distance onto X/Y using base yaw
        x = r * math.cos(j0)
        y = r * math.sin(j0)

        return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

    def check_grip(
        self,
        joints_deg: List[float],
        gripper_width_mm: float,
        detected_objects: List[Dict[str, Any]],
        grip_distance_mm: float = GRIP_DISTANCE_MM,
        grip_width_mm: float = GRIP_WIDTH_MM,
    ) -> GripCheckResult:
        """Check if gripper is gripping any detected object.

        Parameters
        ----------
        joints_deg : list of 6 floats
            Current joint angles in degrees.
        gripper_width_mm : float
            Current gripper opening width in mm.
        detected_objects : list of dicts
            Each must have 'label' and 'position' with x,y,z in mm.
            Optionally 'width_mm' for object size.
        grip_distance_mm : float
            Max distance to consider gripping.
        grip_width_mm : float
            Gripper must be narrower than this to count as closed.
        """
        pos = self.compute_gripper_position(joints_deg)

        if not detected_objects:
            return GripCheckResult(
                gripped=False,
                position=pos,
                gripper_width_mm=gripper_width_mm,
                message="No objects detected",
            )

        # Find closest object
        closest_label = None
        closest_dist = float("inf")
        closest_obj_width = 0.0

        for obj in detected_objects:
            obj_pos = obj.get("position", {})
            dx = pos["x"] - obj_pos.get("x", 0)
            dy = pos["y"] - obj_pos.get("y", 0)
            dz = pos["z"] - obj_pos.get("z", 0)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
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
            position=pos,
            message=msg,
        )
