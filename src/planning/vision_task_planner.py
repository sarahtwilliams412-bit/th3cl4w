"""
Vision Task Planner — Look at camera feeds, understand instructions, build plans.

Takes a SceneDescription from the SceneAnalyzer and a natural-language
instruction, then reasons about the scene and builds a concrete arm
trajectory plan using the existing TaskPlanner and MotionPlanner.

Integrates with the dual-camera independent vision system:
  cam0 (front/side): height estimation
  cam1 (overhead): workspace X/Y positioning

The planning pipeline:
  1. Parse the instruction to identify the intended action and target objects
  2. Match targets against detected scene objects
  3. Map workspace positions to arm joint-space poses
  4. Build a step-by-step reasoning chain
  5. Generate executable trajectories via TaskPlanner

All reasoning is rule-based (no external LLM required).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.planning.motion_planner import (
    MotionPlanner,
    Waypoint,
    Trajectory,
    NUM_ARM_JOINTS,
    JOINT_LIMITS_DEG,
)
from src.planning.task_planner import (
    TaskPlanner,
    TaskResult,
    TaskStatus,
    HOME_POSE,
    READY_POSE,
)
from src.vision.scene_analyzer import SceneDescription, SceneObject

logger = logging.getLogger("th3cl4w.planning.vision_task_planner")


class ActionType(Enum):
    """High-level action types the planner can recognize."""

    PICK_AND_PLACE = "pick_and_place"
    PICK_UP = "pick_up"
    PUSH = "push"
    POUR = "pour"
    WAVE = "wave"
    POINT_AT = "point_at"
    GO_HOME = "go_home"
    GO_READY = "go_ready"
    INSPECT = "inspect"
    UNKNOWN = "unknown"


@dataclass
class ReasoningStep:
    """A single step in the planner's reasoning chain."""

    step: int
    description: str
    detail: str = ""


@dataclass
class VisionTaskPlan:
    """Complete plan produced by the VisionTaskPlanner.

    Contains the reasoning chain, the resolved action, and the
    executable trajectory (if planning succeeded).
    """

    instruction: str
    action: ActionType
    reasoning: list[ReasoningStep] = field(default_factory=list)
    target_object: Optional[dict] = None
    destination_object: Optional[dict] = None
    task_result: Optional[TaskResult] = None
    success: bool = False
    error: str = ""

    @property
    def trajectory(self) -> Optional[Trajectory]:
        if self.task_result:
            return self.task_result.trajectory
        return None

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        result: dict = {
            "instruction": self.instruction,
            "action": self.action.value,
            "success": self.success,
            "error": self.error,
            "reasoning": [
                {
                    "step": r.step,
                    "description": r.description,
                    "detail": r.detail,
                }
                for r in self.reasoning
            ],
            "target_object": self.target_object,
            "destination_object": self.destination_object,
        }
        if self.task_result:
            result["trajectory"] = {
                "points": self.task_result.trajectory.num_points,
                "duration_s": round(self.task_result.trajectory.duration, 2),
                "status": self.task_result.status.value,
                "message": self.task_result.message,
            }
        return result


# -----------------------------------------------------------------------
# Instruction patterns — maps keywords to action types
# -----------------------------------------------------------------------

_ACTION_PATTERNS: list[tuple[str, ActionType]] = [
    (r"\b(pick\s*(up|it)?(\s+and\s+place)?|grab|grasp)\b", ActionType.PICK_AND_PLACE),
    (r"\b(place|put|move|set)\b.*\b(to|on|at|onto|near|next)\b", ActionType.PICK_AND_PLACE),
    (r"\b(push|slide|nudge|shove)\b", ActionType.PUSH),
    (r"\b(pour|tilt|dump|spill)\b", ActionType.POUR),
    (r"\b(wave|hello|hi|greet|bye|goodbye)\b", ActionType.WAVE),
    (r"\b(point|indicate|show|aim)\b.*\b(at|to|toward)\b", ActionType.POINT_AT),
    (r"\b(go\s*home|return\s*home|home\s*position)\b", ActionType.GO_HOME),
    (r"\b(ready|neutral|standby)\b", ActionType.GO_READY),
    (r"\b(look|inspect|examine|check|scan|see|observe)\b", ActionType.INSPECT),
]

# Color keywords for matching objects
_COLOR_KEYWORDS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "white",
    "black",
    "pink",
    "cyan",
]

# Spatial keywords for identifying destination
_POSITION_KEYWORDS = {
    "left": (-0.3, 0.0),
    "right": (0.3, 0.0),
    "center": (0.0, 0.0),
    "middle": (0.0, 0.0),
    "top": (0.0, -0.3),
    "bottom": (0.0, 0.3),
    "up": (0.0, -0.3),
    "down": (0.0, 0.3),
    "forward": (0.0, 0.0),
    "back": (0.0, 0.0),
}


class VisionTaskPlanner:
    """Plans arm tasks from camera scene + natural-language instructions.

    Uses workspace positions from the SceneAnalyzer (derived from independent
    camera calibration) when available, and falls back to image-based
    estimation otherwise.

    Pipeline:
      1. analyze_instruction() — parse what the user wants
      2. match_targets() — find referenced objects in the scene
      3. plan() — generate the full plan with reasoning and trajectory
    """

    def __init__(
        self,
        task_planner: Optional[TaskPlanner] = None,
        workspace_center_deg: Optional[np.ndarray] = None,
        workspace_range_deg: float = 60.0,
    ):
        self.task_planner = task_planner or TaskPlanner()
        # Center of the visual workspace in joint space (for image-based fallback)
        self.workspace_center = (
            workspace_center_deg if workspace_center_deg is not None else READY_POSE.copy()
        )
        # Degrees of joint travel that maps to the full camera field of view
        self.workspace_range = workspace_range_deg

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        instruction: str,
        scene: SceneDescription,
        current_pose: np.ndarray,
    ) -> VisionTaskPlan:
        """Build a complete plan from instruction + scene.

        Args:
            instruction: Natural-language instruction (e.g. "pick up the red object").
            scene: Structured scene from SceneAnalyzer.
            current_pose: Current arm joint angles in degrees (6,).

        Returns:
            VisionTaskPlan with reasoning, resolved targets, and trajectory.
        """
        plan = VisionTaskPlan(instruction=instruction, action=ActionType.UNKNOWN)
        reasoning = []
        step_num = 1

        # Step 1: Parse the instruction
        action, action_detail = self._parse_action(instruction)
        plan.action = action
        reasoning.append(
            ReasoningStep(
                step=step_num,
                description="Parse instruction",
                detail=f"Identified action: {action.value}. {action_detail}",
            )
        )
        step_num += 1

        # Step 2: Analyze the scene
        cameras_str = ", ".join(scene.cameras_used) if scene.cameras_used else "unknown"
        scene_detail = (
            f"Scene has {scene.object_count} objects (cameras: {cameras_str}). "
            f"{scene.summary.split(chr(10))[0]}"
            if scene.has_objects
            else f"No objects detected in scene (cameras: {cameras_str})."
        )
        reasoning.append(
            ReasoningStep(
                step=step_num,
                description="Analyze scene",
                detail=scene_detail,
            )
        )
        step_num += 1

        # Step 3: Match target objects
        target, target_detail = self._match_target(instruction, scene)
        if target:
            target_dict: dict = {
                "color": target.color,
                "region": target.region,
                "centroid_2d": list(target.centroid_2d),
                "area": round(target.area, 1),
                "source": target.source,
            }
            if target.centroid_3d is not None:
                target_dict["workspace_mm"] = [round(v, 1) for v in target.centroid_3d]
            plan.target_object = target_dict
        reasoning.append(
            ReasoningStep(
                step=step_num,
                description="Match target object",
                detail=target_detail,
            )
        )
        step_num += 1

        # Step 4: Determine destination (for pick-and-place)
        destination = None
        if action == ActionType.PICK_AND_PLACE:
            destination, dest_detail = self._match_destination(instruction, scene, target)
            if destination:
                dest_dict: dict = {
                    "color": destination.color,
                    "region": destination.region,
                    "centroid_2d": list(destination.centroid_2d),
                }
                if destination.centroid_3d is not None:
                    dest_dict["workspace_mm"] = [round(v, 1) for v in destination.centroid_3d]
                plan.destination_object = dest_dict
            reasoning.append(
                ReasoningStep(
                    step=step_num,
                    description="Determine destination",
                    detail=dest_detail,
                )
            )
            step_num += 1

        # Step 5: Map scene positions to joint space
        reasoning.append(
            ReasoningStep(
                step=step_num,
                description="Map to joint space",
                detail=self._describe_mapping(target, destination),
            )
        )
        step_num += 1

        # Step 6: Generate trajectory
        try:
            task_result = self._generate_trajectory(
                action, current_pose, target, destination, scene
            )
            plan.task_result = task_result
            plan.success = task_result.status == TaskStatus.SUCCESS

            reasoning.append(
                ReasoningStep(
                    step=step_num,
                    description="Generate trajectory",
                    detail=(
                        f"Planned {task_result.trajectory.num_points} points, "
                        f"{task_result.trajectory.duration:.1f}s duration. "
                        f"{task_result.message}"
                    ),
                )
            )
        except Exception as e:
            plan.success = False
            plan.error = str(e)
            reasoning.append(
                ReasoningStep(
                    step=step_num,
                    description="Generate trajectory",
                    detail=f"Failed: {e}",
                )
            )

        plan.reasoning = reasoning
        return plan

    # ------------------------------------------------------------------
    # Instruction parsing
    # ------------------------------------------------------------------

    def _parse_action(self, instruction: str) -> tuple[ActionType, str]:
        """Identify the intended action from an instruction string."""
        text = instruction.lower().strip()

        for pattern, action_type in _ACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return action_type, f"Matched pattern '{match.group()}' -> {action_type.value}"

        return ActionType.UNKNOWN, "No recognized action pattern found."

    # ------------------------------------------------------------------
    # Object matching
    # ------------------------------------------------------------------

    def _match_target(
        self, instruction: str, scene: SceneDescription
    ) -> tuple[Optional[SceneObject], str]:
        """Find the object in the scene that the instruction refers to."""
        if not scene.has_objects:
            return None, "No objects in scene to match."

        text = instruction.lower()

        # Try to match by color
        for color in _COLOR_KEYWORDS:
            if color in text:
                matches = scene.objects_by_color(color)
                if matches:
                    obj = matches[0]
                    pos_str = ""
                    if obj.centroid_3d is not None:
                        pos_str = (
                            f", workspace ({obj.centroid_3d[0]:.0f}, {obj.centroid_3d[1]:.0f})mm"
                        )
                    return obj, f"Matched '{color}' -> {obj.color} object in {obj.region}{pos_str}"

        # Try to match by position keywords
        for pos_word in ["left", "right", "top", "bottom", "center", "middle"]:
            if pos_word in text:
                for obj in scene.objects:
                    if pos_word in obj.region:
                        return (
                            obj,
                            f"Matched position '{pos_word}' -> {obj.color} object in {obj.region}",
                        )

        # Try to match by size
        if "large" in text or "big" in text or "biggest" in text:
            obj = scene.largest_object()
            if obj:
                return obj, f"Matched 'largest' -> {obj.color} object ({obj.size_category})"

        if "near" in text or "close" in text or "nearest" in text or "closest" in text:
            obj = scene.nearest_object()
            if obj:
                return obj, f"Matched 'nearest' -> {obj.color} object in {obj.region}"

        # Default: use the largest object
        obj = scene.largest_object()
        if obj:
            return (
                obj,
                f"No specific target match; defaulting to largest: {obj.color} object in {obj.region}",
            )

        return None, "Could not match any target object."

    def _match_destination(
        self,
        instruction: str,
        scene: SceneDescription,
        target: Optional[SceneObject],
    ) -> tuple[Optional[SceneObject], str]:
        """Find the destination for a pick-and-place action."""
        text = instruction.lower()

        # Check for explicit destination object by color (after "to/on/near/onto")
        dest_match = re.search(r"\b(?:to|on|onto|near|next\s*to|beside)\s+(?:the\s+)?(\w+)", text)
        if dest_match:
            dest_word = dest_match.group(1)
            if dest_word in _COLOR_KEYWORDS:
                matches = scene.objects_by_color(dest_word)
                if target:
                    matches = [m for m in matches if m.centroid_2d != target.centroid_2d]
                if matches:
                    return matches[0], f"Destination: {dest_word} object in {matches[0].region}"

            if dest_word in _POSITION_KEYWORDS:
                return None, f"Destination: {dest_word} side of workspace (no specific object)"

        # If two objects of different colors are mentioned, second is destination
        mentioned_colors = [c for c in _COLOR_KEYWORDS if c in text]
        if len(mentioned_colors) >= 2 and target:
            dest_color = (
                mentioned_colors[1] if mentioned_colors[0] == target.color else mentioned_colors[0]
            )
            matches = scene.objects_by_color(dest_color)
            if matches:
                return matches[0], f"Second color mentioned: {dest_color} -> destination"

        return None, "No specific destination object; will place at offset from pick location."

    # ------------------------------------------------------------------
    # Joint-space mapping
    # ------------------------------------------------------------------

    def _scene_to_joint_pose(
        self, obj: SceneObject, offset: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Map a scene object's position to a joint-space pose.

        Uses normalized 2D position from the overhead camera to interpolate
        across the arm's visual workspace range, centered on the workspace
        center pose.
        """
        # Map normalized image coords to joint offsets
        # In overhead view: x maps to J0 (base yaw), y maps to J1 (shoulder pitch)
        x_offset = (obj.normalized_x - 0.5) * self.workspace_range
        y_offset = (obj.normalized_y - 0.5) * self.workspace_range * 0.5

        pose = self.workspace_center.copy()
        pose[0] += x_offset  # J0 base yaw follows horizontal position
        pose[1] += y_offset * 0.5  # J1 shoulder pitch follows vertical
        pose[4] += y_offset * 0.3  # J4 wrist pitch fine-tunes vertical

        if offset is not None:
            pose += offset

        # Clamp to joint limits
        for i in range(NUM_ARM_JOINTS):
            pose[i] = float(np.clip(pose[i], JOINT_LIMITS_DEG[i, 0], JOINT_LIMITS_DEG[i, 1]))

        return pose

    def _describe_mapping(
        self,
        target: Optional[SceneObject],
        destination: Optional[SceneObject],
    ) -> str:
        """Describe how scene positions map to joint space."""
        parts = []
        if target:
            pose = self._scene_to_joint_pose(target)
            pos_info = f"image ({target.normalized_x:.2f}, {target.normalized_y:.2f})"
            if target.centroid_3d is not None:
                pos_info += (
                    f" / workspace ({target.centroid_3d[0]:.0f}, {target.centroid_3d[1]:.0f})mm"
                )
            parts.append(
                f"Target at {pos_info} " f"-> joint pose [{', '.join(f'{a:.1f}' for a in pose)}]"
            )
        if destination:
            pose = self._scene_to_joint_pose(destination)
            pos_info = f"image ({destination.normalized_x:.2f}, {destination.normalized_y:.2f})"
            if destination.centroid_3d is not None:
                pos_info += f" / workspace ({destination.centroid_3d[0]:.0f}, {destination.centroid_3d[1]:.0f})mm"
            parts.append(
                f"Destination at {pos_info} "
                f"-> joint pose [{', '.join(f'{a:.1f}' for a in pose)}]"
            )
        if not parts:
            parts.append("Using default workspace positions (no target matched).")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_trajectory(
        self,
        action: ActionType,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
        destination: Optional[SceneObject],
        scene: SceneDescription,
    ) -> TaskResult:
        """Generate an executable trajectory based on the parsed plan."""

        if action == ActionType.GO_HOME:
            return self.task_planner.go_home(current_pose)

        if action == ActionType.GO_READY:
            return self.task_planner.go_ready(current_pose)

        if action == ActionType.WAVE:
            return self.task_planner.wave(current_pose)

        if action == ActionType.PICK_AND_PLACE or action == ActionType.PICK_UP:
            return self._plan_pick_and_place(current_pose, target, destination)

        if action == ActionType.POUR:
            return self._plan_pour(current_pose, target)

        if action == ActionType.POINT_AT:
            return self._plan_point_at(current_pose, target)

        if action == ActionType.PUSH:
            return self._plan_push(current_pose, target)

        if action == ActionType.INSPECT:
            return self._plan_inspect(current_pose, target)

        # Unknown action — move to ready position as safe default
        return self.task_planner.go_ready(current_pose)

    def _plan_pick_and_place(
        self,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
        destination: Optional[SceneObject],
    ) -> TaskResult:
        """Plan a pick-and-place using scene object positions."""
        if target is None:
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message="No target object found in scene for pick-and-place.",
            )

        pick_pose = self._scene_to_joint_pose(target)

        if destination is not None:
            place_pose = self._scene_to_joint_pose(destination)
        else:
            # Default: place 30 degrees to the right of pick
            place_pose = pick_pose.copy()
            place_pose[0] += 30.0
            place_pose[0] = float(
                np.clip(place_pose[0], JOINT_LIMITS_DEG[0, 0], JOINT_LIMITS_DEG[0, 1])
            )

        cfg = _get_pick_config()
        return self.task_planner.pick_and_place(
            current_pose,
            pick_pose,
            place_pose,
            gripper_open_mm=cfg.get("gripper", "pick_open_mm"),
            gripper_close_mm=cfg.get("gripper", "pick_close_mm"),
            speed_factor=0.6,
        )

    def _plan_pour(
        self,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
    ) -> TaskResult:
        """Plan a pouring motion toward the target object."""
        if target is not None:
            pour_pose = self._scene_to_joint_pose(target)
        else:
            pour_pose = READY_POSE.copy()

        return self.task_planner.pour(
            current_pose,
            pour_pose,
            pour_angle=80.0,
            speed_factor=0.4,
            gripper_mm=20.0,
        )

    def _plan_point_at(
        self,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
    ) -> TaskResult:
        """Plan a pointing gesture toward the target."""
        if target is not None:
            point_pose = self._scene_to_joint_pose(target)
        else:
            point_pose = READY_POSE.copy()

        waypoints = [
            Waypoint(current_pose, gripper_mm=0.0, max_speed_factor=0.6),
            Waypoint(point_pose, gripper_mm=0.0, max_speed_factor=0.6),
        ]

        return self.task_planner.custom_sequence(waypoints, label="point_at")

    def _plan_push(
        self,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
    ) -> TaskResult:
        """Plan a pushing motion toward the target."""
        if target is None:
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message="No target object found in scene for push.",
            )

        approach_pose = self._scene_to_joint_pose(target)
        push_pose = approach_pose.copy()
        push_pose[1] -= 10.0
        push_pose[1] = float(np.clip(push_pose[1], JOINT_LIMITS_DEG[1, 0], JOINT_LIMITS_DEG[1, 1]))

        waypoints = [
            Waypoint(current_pose, gripper_mm=0.0, max_speed_factor=0.6),
            Waypoint(approach_pose, gripper_mm=0.0, max_speed_factor=0.5),
            Waypoint(push_pose, gripper_mm=0.0, max_speed_factor=0.3),
            Waypoint(approach_pose, gripper_mm=0.0, max_speed_factor=0.5),
            Waypoint(current_pose, gripper_mm=0.0, max_speed_factor=0.6),
        ]

        return self.task_planner.custom_sequence(waypoints, label="push")

    def _plan_inspect(
        self,
        current_pose: np.ndarray,
        target: Optional[SceneObject],
    ) -> TaskResult:
        """Plan an inspection motion — move near the target to look at it."""
        if target is not None:
            inspect_pose = self._scene_to_joint_pose(target)
            inspect_pose[1] += 10.0
        else:
            inspect_pose = READY_POSE.copy()

        for i in range(NUM_ARM_JOINTS):
            inspect_pose[i] = float(
                np.clip(inspect_pose[i], JOINT_LIMITS_DEG[i, 0], JOINT_LIMITS_DEG[i, 1])
            )

        waypoints = [
            Waypoint(current_pose, gripper_mm=30.0, max_speed_factor=0.5),
            Waypoint(inspect_pose, gripper_mm=30.0, max_speed_factor=0.4),
        ]

        return self.task_planner.custom_sequence(waypoints, label="inspect")
