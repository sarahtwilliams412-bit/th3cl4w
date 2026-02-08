"""
Visual Grasp Planner — Plans grasp poses from dual-camera object detection.

Given a tracked object's 3D position and estimated size, computes:
1. A grasp pose (4x4 homogeneous transform) for the end-effector
2. An approach pose (above the grasp pose for safe descent)
3. A retreat pose (lift after grasping)
4. Joint angles via inverse kinematics for each pose

Designed for the Unitree D1 arm picking up cylindrical objects like cans.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.kinematics.kinematics import D1Kinematics
from src.planning.motion_planner import (
    MotionPlanner,
    JOINT_LIMITS_DEG,
    NUM_ARM_JOINTS,
    GRIPPER_MAX_MM,
)

logger = logging.getLogger("th3cl4w.vision.grasp_planner")

# Red Bull can dimensions (mm) - standard 250ml
REDBULL_CAN_HEIGHT_MM = 135.0
REDBULL_CAN_DIAMETER_MM = 53.0

# D1 gripper specs
GRIPPER_MAX_OPEN_MM = 65.0
GRIPPER_GRASP_CLEARANCE_MM = 5.0  # extra mm beyond object diameter


@dataclass
class GraspPlan:
    """A complete grasp plan with poses and joint angles."""

    # Poses as 4x4 homogeneous transforms in arm-base frame
    approach_pose: np.ndarray  # (4,4) pose above the object
    grasp_pose: np.ndarray  # (4,4) pose at the grasping position
    retreat_pose: np.ndarray  # (4,4) pose after lifting

    # Joint angles in degrees
    approach_angles_deg: np.ndarray  # (6,) joint angles for approach
    grasp_angles_deg: np.ndarray  # (6,) joint angles for grasp
    retreat_angles_deg: np.ndarray  # (6,) joint angles for retreat

    # Gripper settings in mm
    gripper_open_mm: float  # opening for approach
    gripper_close_mm: float  # closing for grasp

    # Metadata
    object_position_mm: np.ndarray  # (3,) target object center
    object_label: str
    confidence: float
    feasible: bool  # whether IK succeeded for all poses
    message: str = ""


class VisualGraspPlanner:
    """Plan grasp approaches from visual object detection results.

    Computes end-effector poses for a top-down or angled grasp on detected
    objects, then solves IK to get joint angles.
    """

    def __init__(
        self,
        kinematics: Optional[D1Kinematics] = None,
        motion_planner: Optional[MotionPlanner] = None,
        approach_height_mm: float = 100.0,
        retreat_height_mm: float = 120.0,
        grasp_depth_mm: float = 0.0,
        max_reach_mm: float = 500.0,
        min_reach_mm: float = 80.0,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.planner = motion_planner or MotionPlanner(kinematics=self.kinematics)
        self.approach_height = approach_height_mm
        self.retreat_height = retreat_height_mm
        self.grasp_depth = grasp_depth_mm
        self.max_reach = max_reach_mm
        self.min_reach = min_reach_mm

    def plan_grasp(
        self,
        object_position_mm: np.ndarray,
        object_label: str = "redbull",
        object_size_mm: Optional[tuple[float, float, float]] = None,
        current_angles_deg: Optional[np.ndarray] = None,
        grasp_from_top: bool = True,
    ) -> GraspPlan:
        """Plan a grasp for an object at a known 3D position.

        Args:
            object_position_mm: (3,) XYZ position in arm-base frame (mm).
            object_label: Object type for size estimation.
            object_size_mm: Optional (width, height, depth) override in mm.
            current_angles_deg: Current joint angles for IK seeding.
            grasp_from_top: If True, approach from above; if False, approach horizontally.

        Returns:
            GraspPlan with poses, joint angles, and feasibility info.
        """
        obj_pos = np.asarray(object_position_mm, dtype=np.float64)

        # Validate reachability
        reach = np.linalg.norm(obj_pos[:2])  # XY distance from base
        if reach > self.max_reach:
            return self._failed_plan(
                obj_pos, object_label,
                f"Object at {reach:.0f}mm exceeds max reach {self.max_reach:.0f}mm",
            )
        if reach < self.min_reach:
            return self._failed_plan(
                obj_pos, object_label,
                f"Object at {reach:.0f}mm is within min reach {self.min_reach:.0f}mm",
            )

        # Determine object dimensions
        width, height, depth = self._get_object_dimensions(object_label, object_size_mm)

        # Compute gripper settings
        gripper_open_mm = min(GRIPPER_MAX_OPEN_MM, width + 20.0)
        gripper_close_mm = max(0.0, width - GRIPPER_GRASP_CLEARANCE_MM)

        # Check if object fits in gripper
        if width > GRIPPER_MAX_OPEN_MM:
            return self._failed_plan(
                obj_pos, object_label,
                f"Object width {width:.0f}mm exceeds gripper max {GRIPPER_MAX_OPEN_MM:.0f}mm",
            )

        # Compute grasp pose
        if grasp_from_top:
            grasp_pose = self._top_down_grasp_pose(obj_pos, height)
            approach_pose = self._offset_pose(grasp_pose, dz=self.approach_height)
            retreat_pose = self._offset_pose(grasp_pose, dz=self.retreat_height)
        else:
            grasp_pose = self._side_grasp_pose(obj_pos, height)
            approach_pose = self._offset_pose_along_approach(grasp_pose, -self.approach_height)
            retreat_pose = self._offset_pose(grasp_pose, dz=self.retreat_height)

        # Solve IK for each pose
        q_init = np.zeros(7)
        if current_angles_deg is not None:
            q_init[:6] = np.deg2rad(current_angles_deg[:6])

        approach_q = self.kinematics.inverse_kinematics(approach_pose, q_init=q_init)
        grasp_q = self.kinematics.inverse_kinematics(grasp_pose, q_init=approach_q)
        retreat_q = self.kinematics.inverse_kinematics(retreat_pose, q_init=grasp_q)

        approach_deg = np.rad2deg(approach_q[:6])
        grasp_deg = np.rad2deg(grasp_q[:6])
        retreat_deg = np.rad2deg(retreat_q[:6])

        # Check joint limits
        feasible = True
        message = "Grasp plan ready"

        for name, angles in [
            ("approach", approach_deg),
            ("grasp", grasp_deg),
            ("retreat", retreat_deg),
        ]:
            if not self.planner.validate_joint_angles(angles):
                feasible = False
                message = f"Joint limits violated at {name} pose"
                # Clamp to limits (still return the plan but mark infeasible)
                break

        # Clamp all to joint limits for safety
        approach_deg = self.planner.clamp_joint_angles(approach_deg)
        grasp_deg = self.planner.clamp_joint_angles(grasp_deg)
        retreat_deg = self.planner.clamp_joint_angles(retreat_deg)

        # Verify IK accuracy
        for name, angles_deg, target_pose in [
            ("approach", approach_deg, approach_pose),
            ("grasp", grasp_deg, grasp_pose),
            ("retreat", retreat_deg, retreat_pose),
        ]:
            q7 = np.zeros(7)
            q7[:6] = np.deg2rad(angles_deg)
            actual_pose = self.kinematics.forward_kinematics(q7)
            pos_err = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3]) * 1000  # to mm
            if pos_err > 50.0:  # 50mm tolerance
                feasible = False
                message = f"IK position error {pos_err:.1f}mm at {name} pose"
                logger.warning("IK error at %s: %.1f mm", name, pos_err)

        return GraspPlan(
            approach_pose=approach_pose,
            grasp_pose=grasp_pose,
            retreat_pose=retreat_pose,
            approach_angles_deg=approach_deg,
            grasp_angles_deg=grasp_deg,
            retreat_angles_deg=retreat_deg,
            gripper_open_mm=gripper_open_mm,
            gripper_close_mm=gripper_close_mm,
            object_position_mm=obj_pos,
            object_label=object_label,
            confidence=1.0 if feasible else 0.3,
            feasible=feasible,
            message=message,
        )

    def plan_grasp_from_tracked_object(
        self,
        tracked_obj,  # TrackedObject from arm_tracker
        current_angles_deg: Optional[np.ndarray] = None,
        grasp_from_top: bool = True,
    ) -> GraspPlan:
        """Plan a grasp from an arm_tracker.TrackedObject."""
        return self.plan_grasp(
            object_position_mm=tracked_obj.position_mm,
            object_label=tracked_obj.label,
            object_size_mm=tracked_obj.size_mm,
            current_angles_deg=current_angles_deg,
            grasp_from_top=grasp_from_top,
        )

    def _get_object_dimensions(
        self,
        label: str,
        override: Optional[tuple[float, float, float]] = None,
    ) -> tuple[float, float, float]:
        """Get estimated object dimensions (width, height, depth) in mm."""
        if override is not None and all(d > 0 for d in override):
            return override

        # Known object dimensions
        known_objects = {
            "redbull": (REDBULL_CAN_DIAMETER_MM, REDBULL_CAN_HEIGHT_MM, REDBULL_CAN_DIAMETER_MM),
            "can": (55.0, 120.0, 55.0),
            "bottle": (70.0, 200.0, 70.0),
            "cup": (80.0, 100.0, 80.0),
        }

        return known_objects.get(label, (50.0, 80.0, 50.0))

    def _top_down_grasp_pose(
        self, obj_pos_mm: np.ndarray, obj_height_mm: float
    ) -> np.ndarray:
        """Compute a top-down grasp pose.

        End-effector points straight down (Z axis pointing down in arm frame),
        gripper aligned to approach the object from above.
        Grasp point is at mid-height of the object.
        """
        # Position: above object center, at mid-height for grasping
        grasp_x = obj_pos_mm[0]
        grasp_y = obj_pos_mm[1]
        grasp_z = obj_pos_mm[2] + obj_height_mm / 2.0 + self.grasp_depth

        # Orientation: end-effector pointing down
        # Rotation: EE z-axis points down (-Z in arm frame)
        # We want the gripper fingers to close along a horizontal axis
        # Compute yaw angle to point gripper toward object from base
        yaw = math.atan2(grasp_y, grasp_x)

        # Rotation matrix: Z down, X toward object (radial), Y perpendicular
        R = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, -1.0],
        ])

        # Flip to get EE z-axis pointing down
        # The D1 EE convention: z-axis is the approach direction
        R_grasp = np.array([
            [math.cos(yaw), math.sin(yaw), 0.0],
            [math.sin(yaw), -math.cos(yaw), 0.0],
            [0.0, 0.0, -1.0],
        ])

        T = np.eye(4)
        T[:3, :3] = R_grasp
        T[:3, 3] = np.array([grasp_x, grasp_y, grasp_z]) / 1000.0  # mm to meters for FK

        return T

    def _side_grasp_pose(
        self, obj_pos_mm: np.ndarray, obj_height_mm: float
    ) -> np.ndarray:
        """Compute a side grasp pose (approach horizontally).

        End-effector approaches from the side, gripper horizontal.
        """
        grasp_x = obj_pos_mm[0]
        grasp_y = obj_pos_mm[1]
        grasp_z = obj_pos_mm[2] + obj_height_mm / 2.0

        yaw = math.atan2(grasp_y, grasp_x)

        # Horizontal approach: EE z-axis points toward object (radial)
        R_grasp = np.array([
            [0.0, 0.0, math.cos(yaw)],
            [0.0, -1.0, math.sin(yaw)],
            [1.0, 0.0, 0.0],
        ])

        T = np.eye(4)
        T[:3, :3] = R_grasp
        T[:3, 3] = np.array([grasp_x, grasp_y, grasp_z]) / 1000.0

        return T

    def _offset_pose(self, pose: np.ndarray, dz: float = 0.0) -> np.ndarray:
        """Create a copy of a pose offset vertically (in arm Z direction).

        dz is in mm.
        """
        T = pose.copy()
        T[2, 3] += dz / 1000.0  # mm to meters
        return T

    def _offset_pose_along_approach(self, pose: np.ndarray, distance_mm: float) -> np.ndarray:
        """Offset along the end-effector approach direction (Z axis of the pose)."""
        T = pose.copy()
        z_axis = pose[:3, 2]  # approach direction
        T[:3, 3] += z_axis * (distance_mm / 1000.0)
        return T

    def _failed_plan(
        self, obj_pos: np.ndarray, label: str, message: str
    ) -> GraspPlan:
        """Create a failed/infeasible grasp plan."""
        logger.warning("Grasp planning failed: %s", message)
        zero_pose = np.eye(4)
        return GraspPlan(
            approach_pose=zero_pose,
            grasp_pose=zero_pose,
            retreat_pose=zero_pose,
            approach_angles_deg=np.zeros(NUM_ARM_JOINTS),
            grasp_angles_deg=np.zeros(NUM_ARM_JOINTS),
            retreat_angles_deg=np.zeros(NUM_ARM_JOINTS),
            gripper_open_mm=GRIPPER_MAX_OPEN_MM,
            gripper_close_mm=0.0,
            object_position_mm=obj_pos,
            object_label=label,
            confidence=0.0,
            feasible=False,
            message=message,
        )

    def estimate_grasp_difficulty(self, object_position_mm: np.ndarray) -> dict:
        """Estimate how difficult a grasp would be at the given position.

        Returns a dict with reach, height, angle info and a difficulty score 0-1.
        """
        pos = np.asarray(object_position_mm, dtype=np.float64)
        reach_xy = np.linalg.norm(pos[:2])
        height = pos[2]
        angle_deg = math.degrees(math.atan2(pos[1], pos[0]))

        # Difficulty factors
        reach_factor = reach_xy / self.max_reach  # 0-1, higher = harder
        height_factor = abs(height) / 400.0  # normalize to workspace
        angle_factor = abs(angle_deg) / 135.0  # edge of workspace is harder

        difficulty = min(1.0, 0.3 * reach_factor + 0.3 * height_factor + 0.4 * angle_factor)

        return {
            "reach_xy_mm": round(reach_xy, 1),
            "height_mm": round(height, 1),
            "angle_deg": round(angle_deg, 1),
            "reach_factor": round(reach_factor, 3),
            "height_factor": round(height_factor, 3),
            "angle_factor": round(angle_factor, 3),
            "difficulty": round(difficulty, 3),
            "reachable": reach_xy <= self.max_reach and reach_xy >= self.min_reach,
        }
