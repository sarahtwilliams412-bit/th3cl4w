"""
Pick Executor — Orchestrates autonomous visual pick operations.

End-to-end pipeline:
1. Capture frames from both cameras
2. Track and localize target object via DualCameraArmTracker
3. Plan grasp approach via VisualGraspPlanner
4. Collision-check via workspace mapper
5. Generate and return executable trajectory via TaskPlanner

This module ties vision and planning together into an autonomous pick system.
All angles in DEGREES, gripper in mm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from shared.kinematics.kinematics import D1Kinematics
from ..planning.motion_planner import (
    MotionPlanner,
    Trajectory,
    TrajectoryPoint,
    Waypoint,
    NUM_ARM_JOINTS,
)
from ..planning.task_planner import TaskPlanner, TaskResult, TaskStatus, HOME_POSE, READY_POSE

logger = logging.getLogger("th3cl4w.planning.pick_executor")


class PickPhase(Enum):
    """Phases of a visual pick operation."""

    IDLE = "idle"
    SCANNING = "scanning"
    DETECTING = "detecting"
    PLANNING = "planning"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    RETREATING = "retreating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PickResult:
    """Result of an autonomous pick operation."""

    phase: PickPhase
    success: bool
    trajectory: Optional[Trajectory] = None
    grasp_plan: Optional[object] = None  # GraspPlan from grasp_planner
    detected_objects: list = field(default_factory=list)
    target_object: Optional[object] = None  # TrackedObject
    message: str = ""
    elapsed_ms: float = 0.0

    # Planning details
    approach_angles_deg: Optional[list[float]] = None
    grasp_angles_deg: Optional[list[float]] = None
    retreat_angles_deg: Optional[list[float]] = None
    gripper_open_mm: float = 60.0
    gripper_close_mm: float = 5.0


class PickExecutor:
    """Autonomous pick execution using dual cameras and motion planning.

    Orchestrates the full pipeline from visual detection to trajectory generation.
    Does NOT directly send commands to the arm — returns trajectories for the
    caller (web server) to execute through the command smoother.
    """

    def __init__(
        self,
        kinematics: Optional[D1Kinematics] = None,
        motion_planner: Optional[MotionPlanner] = None,
        task_planner: Optional[TaskPlanner] = None,
        approach_speed: float = 0.5,
        grasp_speed: float = 0.3,
        retreat_speed: float = 0.4,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.planner = motion_planner or MotionPlanner(kinematics=self.kinematics)
        self.task_planner = task_planner or TaskPlanner(motion_planner=self.planner)
        self.approach_speed = approach_speed
        self.grasp_speed = grasp_speed
        self.retreat_speed = retreat_speed

        # State
        self._phase = PickPhase.IDLE
        self._last_result: Optional[PickResult] = None

    @property
    def phase(self) -> PickPhase:
        return self._phase

    def plan_pick(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        arm_tracker,  # DualCameraArmTracker
        grasp_planner,  # VisualGraspPlanner
        current_angles_deg: np.ndarray,
        current_gripper_mm: float = 0.0,
        target_label: str = "redbull",
        workspace_mapper=None,
        collision_preview=None,
    ) -> PickResult:
        """Plan a complete pick operation from camera frames.

        This is the main entry point. It:
        1. Detects the target object using dual cameras
        2. Plans a grasp approach
        3. Generates a full pick trajectory
        4. Optionally checks for collisions

        Args:
            left_frame: BGR image from left camera.
            right_frame: BGR image from right camera.
            arm_tracker: DualCameraArmTracker instance.
            grasp_planner: VisualGraspPlanner instance.
            current_angles_deg: (6,) current joint angles in degrees.
            current_gripper_mm: Current gripper position in mm.
            target_label: Object to pick ("redbull", "red", "blue", "all").
            workspace_mapper: Optional WorkspaceMapper for collision checking.
            collision_preview: Optional CollisionPreview for path checking.

        Returns:
            PickResult with trajectory if successful.
        """
        t0 = time.monotonic()
        current = np.asarray(current_angles_deg, dtype=np.float64)

        # Phase 1: Detect
        self._phase = PickPhase.DETECTING
        logger.info("Pick executor: detecting '%s' in stereo frames", target_label)

        tracking_result = arm_tracker.track(
            left_frame, right_frame, target_label=target_label, annotate=True
        )

        if tracking_result.status != "ok" or not tracking_result.objects:
            self._phase = PickPhase.FAILED
            result = PickResult(
                phase=PickPhase.FAILED,
                success=False,
                detected_objects=[],
                message=f"No '{target_label}' object detected: {tracking_result.message}",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            self._last_result = result
            return result

        # Select best target (highest confidence, or closest)
        target = self._select_best_target(tracking_result.objects)
        logger.info(
            "Pick executor: target '%s' at position [%.0f, %.0f, %.0f] mm, conf=%.2f",
            target.label,
            target.position_mm[0],
            target.position_mm[1],
            target.position_mm[2],
            target.confidence,
        )

        # Phase 2: Plan grasp
        self._phase = PickPhase.PLANNING
        grasp_plan = grasp_planner.plan_grasp(
            object_position_mm=target.position_mm,
            object_label=target.label,
            object_size_mm=target.size_mm,
            current_angles_deg=current,
        )

        if not grasp_plan.feasible:
            self._phase = PickPhase.FAILED
            result = PickResult(
                phase=PickPhase.FAILED,
                success=False,
                grasp_plan=grasp_plan,
                detected_objects=tracking_result.objects,
                target_object=target,
                message=f"Grasp planning failed: {grasp_plan.message}",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            self._last_result = result
            return result

        # Phase 3: Generate trajectory
        trajectory = self._build_pick_trajectory(
            current_angles_deg=current,
            current_gripper_mm=current_gripper_mm,
            grasp_plan=grasp_plan,
        )

        if trajectory is None or len(trajectory.points) < 2:
            self._phase = PickPhase.FAILED
            result = PickResult(
                phase=PickPhase.FAILED,
                success=False,
                grasp_plan=grasp_plan,
                detected_objects=tracking_result.objects,
                target_object=target,
                message="Failed to generate pick trajectory",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            self._last_result = result
            return result

        # Phase 4: Collision check (optional)
        if workspace_mapper is not None and collision_preview is not None:
            preview = collision_preview.preview_trajectory(trajectory, workspace_mapper, step=3)
            if not preview.clear:
                logger.warning("Collision detected in pick trajectory: %s", preview.summary)
                # Still return the plan but warn
                collision_msg = f" (WARNING: {preview.summary})"
            else:
                collision_msg = ""
        else:
            collision_msg = ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._phase = PickPhase.COMPLETE

        result = PickResult(
            phase=PickPhase.COMPLETE,
            success=True,
            trajectory=trajectory,
            grasp_plan=grasp_plan,
            detected_objects=tracking_result.objects,
            target_object=target,
            message=f"Pick planned: {len(trajectory.points)} pts, "
            f"{trajectory.duration:.1f}s{collision_msg}",
            elapsed_ms=round(elapsed_ms, 1),
            approach_angles_deg=grasp_plan.approach_angles_deg.tolist(),
            grasp_angles_deg=grasp_plan.grasp_angles_deg.tolist(),
            retreat_angles_deg=grasp_plan.retreat_angles_deg.tolist(),
            gripper_open_mm=grasp_plan.gripper_open_mm,
            gripper_close_mm=grasp_plan.gripper_close_mm,
        )
        self._last_result = result
        return result

    def plan_pick_from_position(
        self,
        object_position_mm: np.ndarray,
        grasp_planner,  # VisualGraspPlanner
        current_angles_deg: np.ndarray,
        current_gripper_mm: float = 0.0,
        object_label: str = "redbull",
        object_size_mm: Optional[tuple[float, float, float]] = None,
    ) -> PickResult:
        """Plan a pick from a known 3D position (skip detection).

        Useful when the object position is already known from a previous
        detection pass or manual input.
        """
        t0 = time.monotonic()
        current = np.asarray(current_angles_deg, dtype=np.float64)
        obj_pos = np.asarray(object_position_mm, dtype=np.float64)

        self._phase = PickPhase.PLANNING
        grasp_plan = grasp_planner.plan_grasp(
            object_position_mm=obj_pos,
            object_label=object_label,
            object_size_mm=object_size_mm,
            current_angles_deg=current,
        )

        if not grasp_plan.feasible:
            self._phase = PickPhase.FAILED
            result = PickResult(
                phase=PickPhase.FAILED,
                success=False,
                grasp_plan=grasp_plan,
                message=f"Grasp planning failed: {grasp_plan.message}",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            self._last_result = result
            return result

        trajectory = self._build_pick_trajectory(
            current_angles_deg=current,
            current_gripper_mm=current_gripper_mm,
            grasp_plan=grasp_plan,
        )

        if trajectory is None or len(trajectory.points) < 2:
            self._phase = PickPhase.FAILED
            result = PickResult(
                phase=PickPhase.FAILED,
                success=False,
                grasp_plan=grasp_plan,
                message="Failed to generate pick trajectory",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )
            self._last_result = result
            return result

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._phase = PickPhase.COMPLETE

        result = PickResult(
            phase=PickPhase.COMPLETE,
            success=True,
            trajectory=trajectory,
            grasp_plan=grasp_plan,
            message=f"Pick planned from position: {len(trajectory.points)} pts, "
            f"{trajectory.duration:.1f}s",
            elapsed_ms=round(elapsed_ms, 1),
            approach_angles_deg=grasp_plan.approach_angles_deg.tolist(),
            grasp_angles_deg=grasp_plan.grasp_angles_deg.tolist(),
            retreat_angles_deg=grasp_plan.retreat_angles_deg.tolist(),
            gripper_open_mm=grasp_plan.gripper_open_mm,
            gripper_close_mm=grasp_plan.gripper_close_mm,
        )
        self._last_result = result
        return result

    def _build_pick_trajectory(
        self,
        current_angles_deg: np.ndarray,
        current_gripper_mm: float,
        grasp_plan,  # GraspPlan
    ) -> Optional[Trajectory]:
        """Build a multi-segment pick trajectory using the task planner.

        Sequence:
        1. Open gripper + move to approach pose
        2. Descend to grasp pose (slow)
        3. Close gripper (hold position)
        4. Retreat upward with object
        """
        try:
            result = self.task_planner.pick_and_place(
                current_pose=current_angles_deg,
                pick_pose=grasp_plan.grasp_angles_deg,
                place_pose=grasp_plan.retreat_angles_deg,  # "place" is retreat (we hold the object)
                approach_offset=grasp_plan.approach_angles_deg - grasp_plan.grasp_angles_deg,
                gripper_open_mm=grasp_plan.gripper_open_mm,
                gripper_close_mm=grasp_plan.gripper_close_mm,
                speed_factor=self.approach_speed,
                retreat_pose=grasp_plan.retreat_angles_deg,
            )

            if result.status == TaskStatus.SUCCESS:
                return result.trajectory

            # Fallback: build manually with linear segments
            logger.info("Task planner pick_and_place returned %s, building manually", result.status)
            return self._build_manual_trajectory(current_angles_deg, current_gripper_mm, grasp_plan)

        except Exception as e:
            logger.warning("pick_and_place failed: %s, building manual trajectory", e)
            return self._build_manual_trajectory(current_angles_deg, current_gripper_mm, grasp_plan)

    def _build_manual_trajectory(
        self,
        current_angles_deg: np.ndarray,
        current_gripper_mm: float,
        grasp_plan,
    ) -> Optional[Trajectory]:
        """Build pick trajectory from individual linear segments."""
        try:
            segments = []

            # 1. Open gripper + move to approach
            seg1 = self.planner.linear_joint_trajectory(
                current_angles_deg,
                grasp_plan.approach_angles_deg,
                self.approach_speed,
                current_gripper_mm,
                grasp_plan.gripper_open_mm,
            )
            segments.append(seg1)

            # 2. Descend to grasp (slow)
            seg2 = self.planner.linear_joint_trajectory(
                grasp_plan.approach_angles_deg,
                grasp_plan.grasp_angles_deg,
                self.grasp_speed,
                grasp_plan.gripper_open_mm,
                grasp_plan.gripper_open_mm,
            )
            segments.append(seg2)

            # 3. Close gripper (hold position)
            seg3 = self.planner.linear_joint_trajectory(
                grasp_plan.grasp_angles_deg,
                grasp_plan.grasp_angles_deg,
                0.3,  # duration factor for gripper close
                grasp_plan.gripper_open_mm,
                grasp_plan.gripper_close_mm,
            )
            segments.append(seg3)

            # 4. Retreat with object
            seg4 = self.planner.linear_joint_trajectory(
                grasp_plan.grasp_angles_deg,
                grasp_plan.retreat_angles_deg,
                self.retreat_speed,
                grasp_plan.gripper_close_mm,
                grasp_plan.gripper_close_mm,
            )
            segments.append(seg4)

            # Concatenate
            return self.task_planner._concat_trajectories(segments, label="visual_pick")

        except Exception as e:
            logger.error("Manual trajectory build failed: %s", e)
            return None

    def _select_best_target(self, objects: list) -> object:
        """Select the best target from detected objects.

        Prefers: highest confidence, then closest to arm base.
        """
        if len(objects) == 1:
            return objects[0]

        # Sort by confidence (descending), then by distance (ascending)
        def score(obj):
            distance = np.linalg.norm(obj.position_mm[:2])
            return (-obj.confidence, distance)

        return sorted(objects, key=score)[0]

    def get_status(self) -> dict:
        """Get current executor status."""
        status = {
            "phase": self._phase.value,
        }
        if self._last_result is not None:
            status["last_result"] = {
                "success": self._last_result.success,
                "message": self._last_result.message,
                "elapsed_ms": self._last_result.elapsed_ms,
                "detected_count": len(self._last_result.detected_objects),
                "has_trajectory": self._last_result.trajectory is not None,
            }
            if self._last_result.trajectory is not None:
                status["last_result"]["trajectory_points"] = len(
                    self._last_result.trajectory.points
                )
                status["last_result"]["trajectory_duration_s"] = round(
                    self._last_result.trajectory.duration, 1
                )
        return status
