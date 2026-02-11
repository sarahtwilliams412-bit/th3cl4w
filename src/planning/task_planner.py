"""
Task Planner for Unitree D1 Robotic Arm

High-level task sequences that compose motion primitives:
pick-and-place, pour, wave, and custom task sequences.

All angles in DEGREES, gripper in mm (0-65).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.planning.motion_planner import (
    MotionPlanner,
    Waypoint,
    Trajectory,
    TrajectoryPoint,
)
from src.control.joint_service import (
    NUM_ARM_JOINTS,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
    HOME_POSITION,
    READY_POSITION,
)

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class TaskResult:
    """Result of a task plan."""

    status: TaskStatus
    trajectory: Trajectory
    message: str = ""
    sub_trajectories: list[Trajectory] = field(default_factory=list)


# Common poses in degrees — imported from joint_service
HOME_POSE = HOME_POSITION.copy()
READY_POSE = READY_POSITION.copy()


class TaskPlanner:
    """Composes motion primitives into high-level tasks."""

    def __init__(self, motion_planner: MotionPlanner | None = None):
        self.planner = motion_planner or MotionPlanner()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _concat_trajectories(self, trajectories: list[Trajectory], label: str = "") -> Trajectory:
        """Concatenate multiple trajectories into one."""
        combined = Trajectory(label=label)
        time_offset = 0.0

        for idx, traj in enumerate(trajectories):
            for j, pt in enumerate(traj.points):
                if idx > 0 and j == 0:
                    continue  # skip duplicate junction point
                combined.points.append(
                    TrajectoryPoint(
                        time=pt.time + time_offset,
                        positions=pt.positions.copy(),
                        velocities=pt.velocities.copy(),
                        accelerations=pt.accelerations.copy(),
                        gripper_mm=pt.gripper_mm,
                    )
                )
            time_offset += traj.duration

        return combined

    def _move_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed: float = 1.0,
        gripper_start: float = 0.0,
        gripper_end: float = 0.0,
    ) -> Trajectory:
        return self.planner.linear_joint_trajectory(
            start,
            end,
            speed,
            gripper_start,
            gripper_end,
        )

    def _gripper_segment(
        self,
        pose: np.ndarray,
        gripper_start: float,
        gripper_end: float,
        duration_factor: float = 0.3,
    ) -> Trajectory:
        """Hold position while changing gripper."""
        return self.planner.linear_joint_trajectory(
            pose,
            pose,
            duration_factor,
            gripper_start,
            gripper_end,
        )

    # ------------------------------------------------------------------
    # Pick and Place
    # ------------------------------------------------------------------

    def pick_and_place(
        self,
        current_pose: np.ndarray,
        pick_pose: np.ndarray,
        place_pose: np.ndarray,
        approach_offset: np.ndarray | None = None,
        gripper_open_mm: float = 60.0,
        gripper_close_mm: float = 5.0,
        speed_factor: float = 0.8,
        retreat_pose: np.ndarray | None = None,
    ) -> TaskResult:
        """Plan a pick-and-place task.

        Sequence:
        1. Open gripper, move to approach above pick
        2. Move down to pick pose
        3. Close gripper
        4. Retreat up
        5. Move to approach above place
        6. Move down to place pose
        7. Open gripper
        8. Retreat
        """
        if approach_offset is None:
            approach_offset = np.array([0.0, -15.0, 0.0, 0.0, 0.0, 0.0])

        pick_approach = pick_pose + approach_offset
        place_approach = place_pose + approach_offset
        final = retreat_pose if retreat_pose is not None else current_pose

        try:
            segments = [
                # 1. Open gripper & go to pick approach
                self._move_segment(
                    current_pose, pick_approach, speed_factor, gripper_close_mm, gripper_open_mm
                ),
                # 2. Descend to pick
                self._move_segment(
                    pick_approach, pick_pose, speed_factor * 0.5, gripper_open_mm, gripper_open_mm
                ),
                # 3. Close gripper (grasp)
                self._gripper_segment(pick_pose, gripper_open_mm, gripper_close_mm),
                # 4. Retreat up
                self._move_segment(
                    pick_pose, pick_approach, speed_factor * 0.5, gripper_close_mm, gripper_close_mm
                ),
                # 5. Move to place approach
                self._move_segment(
                    pick_approach, place_approach, speed_factor, gripper_close_mm, gripper_close_mm
                ),
                # 6. Descend to place
                self._move_segment(
                    place_approach,
                    place_pose,
                    speed_factor * 0.5,
                    gripper_close_mm,
                    gripper_close_mm,
                ),
                # 7. Open gripper (release)
                self._gripper_segment(place_pose, gripper_close_mm, gripper_open_mm),
                # 8. Retreat
                self._move_segment(
                    place_pose, final, speed_factor, gripper_open_mm, gripper_open_mm
                ),
            ]

            combined = self._concat_trajectories(segments, label="pick_and_place")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message="Pick and place planned successfully",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Pick and place failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    # ------------------------------------------------------------------
    # Pour
    # ------------------------------------------------------------------

    def pour(
        self,
        current_pose: np.ndarray,
        pour_position: np.ndarray,
        pour_angle: float = 90.0,
        pour_joint: int = 5,
        speed_factor: float = 0.5,
        gripper_mm: float = 20.0,
    ) -> TaskResult:
        """Plan a pouring motion.

        Sequence:
        1. Move to pour position
        2. Tilt (rotate pour_joint by pour_angle)
        3. Hold briefly (via slow return)
        4. Un-tilt
        5. Return
        """
        try:
            tilted = pour_position.copy()
            tilted[pour_joint] = pour_position[pour_joint] + pour_angle
            # Clamp
            tilted = self.planner.clamp_joint_angles(tilted)

            segments = [
                self._move_segment(
                    current_pose, pour_position, speed_factor, gripper_mm, gripper_mm
                ),
                self._move_segment(
                    pour_position, tilted, speed_factor * 0.3, gripper_mm, gripper_mm
                ),
                self._move_segment(
                    tilted, pour_position, speed_factor * 0.3, gripper_mm, gripper_mm
                ),
                self._move_segment(
                    pour_position, current_pose, speed_factor, gripper_mm, gripper_mm
                ),
            ]

            combined = self._concat_trajectories(segments, label="pour")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message="Pour planned successfully",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Pour failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    # ------------------------------------------------------------------
    # Wave
    # ------------------------------------------------------------------

    def wave(
        self,
        current_pose: np.ndarray,
        n_waves: int = 3,
        wave_amplitude: float = 30.0,
        wave_joint: int = 5,
        speed_factor: float = 0.8,
        gripper_mm: float = 65.0,
    ) -> TaskResult:
        """Plan a waving motion.

        Moves to a raised position, then oscillates a joint back and forth.
        """
        try:
            # Wave start position
            wave_base = np.array([0.0, -60.0, 0.0, 90.0, 0.0, 0.0])

            waypoints = [Waypoint(current_pose, gripper_mm, speed_factor)]
            waypoints.append(Waypoint(wave_base, gripper_mm, speed_factor))

            for i in range(n_waves):
                wp_left = wave_base.copy()
                wp_left[wave_joint] = wave_base[wave_joint] + wave_amplitude
                wp_left = self.planner.clamp_joint_angles(wp_left)

                wp_right = wave_base.copy()
                wp_right[wave_joint] = wave_base[wave_joint] - wave_amplitude
                wp_right = self.planner.clamp_joint_angles(wp_right)

                waypoints.append(Waypoint(wp_left, gripper_mm, speed_factor))
                waypoints.append(Waypoint(wp_right, gripper_mm, speed_factor))

            waypoints.append(Waypoint(wave_base, gripper_mm, speed_factor))
            waypoints.append(Waypoint(current_pose, gripper_mm, speed_factor))

            traj = self.planner.plan_waypoints(waypoints)
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=traj,
                message=f"Wave with {n_waves} cycles planned successfully",
            )
        except Exception as e:
            logger.error("Wave failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    # ------------------------------------------------------------------
    # Go Home
    # ------------------------------------------------------------------

    def go_home(
        self,
        current_pose: np.ndarray,
        speed_factor: float = 0.5,
        gripper_mm: float = 0.0,
    ) -> TaskResult:
        """Plan a safe return to home position."""
        try:
            traj = self.planner.linear_joint_trajectory(
                current_pose,
                HOME_POSE,
                speed_factor,
                gripper_mm,
                0.0,
            )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=traj,
                message="Home trajectory planned",
            )
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Failed: {e}",
            )

    # ------------------------------------------------------------------
    # Go to Ready
    # ------------------------------------------------------------------

    def go_ready(
        self,
        current_pose: np.ndarray,
        speed_factor: float = 0.6,
        gripper_mm: float = 30.0,
    ) -> TaskResult:
        """Plan move to ready/neutral position."""
        try:
            traj = self.planner.linear_joint_trajectory(
                current_pose,
                READY_POSE,
                speed_factor,
                gripper_mm,
                gripper_mm,
            )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=traj,
                message="Ready position trajectory planned",
            )
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Failed: {e}",
            )

    # ------------------------------------------------------------------
    # Custom sequence
    # ------------------------------------------------------------------

    def custom_sequence(
        self,
        waypoints: list[Waypoint],
        label: str = "custom",
    ) -> TaskResult:
        """Plan a custom sequence of waypoints."""
        try:
            if len(waypoints) < 2:
                return TaskResult(
                    status=TaskStatus.FAILED,
                    trajectory=Trajectory(),
                    message="Need at least 2 waypoints",
                )
            traj = self.planner.plan_waypoints(waypoints)
            traj.label = label
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=traj,
                message=f"Custom sequence '{label}' planned with {len(waypoints)} waypoints",
            )
        except Exception as e:
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Failed: {e}",
            )

    # ------------------------------------------------------------------
    # Open / Close articulated fixtures (cabinets, drawers, etc.)
    # ------------------------------------------------------------------

    def open_fixture(
        self,
        current_pose: np.ndarray,
        handle_pose: np.ndarray,
        open_direction: np.ndarray,
        pull_distance_deg: float = 20.0,
        speed_factor: float = 0.4,
        gripper_grip_mm: float = 15.0,
    ) -> TaskResult:
        """Plan opening an articulated fixture (cabinet, drawer, etc.).

        Sequence:
        1. Approach handle
        2. Grip handle
        3. Pull along open_direction
        4. Release handle
        5. Retreat
        """
        approach_offset = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        handle_approach = handle_pose + approach_offset
        opened_pose = handle_pose + open_direction * pull_distance_deg

        try:
            segments = [
                # Approach handle with open gripper
                self._move_segment(
                    current_pose, handle_approach, speed_factor, 0.0, 35.0
                ),
                # Move to handle
                self._move_segment(
                    handle_approach, handle_pose, speed_factor * 0.5, 35.0, 35.0
                ),
                # Grip handle
                self._gripper_segment(handle_pose, 35.0, gripper_grip_mm),
                # Pull open
                self._move_segment(
                    handle_pose, opened_pose, speed_factor * 0.3, gripper_grip_mm, gripper_grip_mm
                ),
                # Release handle
                self._gripper_segment(opened_pose, gripper_grip_mm, 35.0),
                # Retreat
                self._move_segment(
                    opened_pose, current_pose, speed_factor, 35.0, 0.0
                ),
            ]

            combined = self._concat_trajectories(segments, label="open_fixture")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message="Open fixture planned successfully",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Open fixture failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    def close_fixture(
        self,
        current_pose: np.ndarray,
        handle_pose: np.ndarray,
        close_direction: np.ndarray,
        push_distance_deg: float = 20.0,
        speed_factor: float = 0.4,
    ) -> TaskResult:
        """Plan closing a fixture by pushing it shut."""
        approach_offset = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        handle_approach = handle_pose + approach_offset
        closed_pose = handle_pose + close_direction * push_distance_deg

        try:
            segments = [
                self._move_segment(
                    current_pose, handle_approach, speed_factor, 0.0, 0.0
                ),
                self._move_segment(
                    handle_approach, handle_pose, speed_factor * 0.5, 0.0, 0.0
                ),
                # Push closed (gripper stays closed, use palm)
                self._move_segment(
                    handle_pose, closed_pose, speed_factor * 0.3, 0.0, 0.0
                ),
                self._move_segment(
                    closed_pose, current_pose, speed_factor, 0.0, 0.0
                ),
            ]

            combined = self._concat_trajectories(segments, label="close_fixture")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message="Close fixture planned successfully",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Close fixture failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    # ------------------------------------------------------------------
    # Knob rotation (stove knobs, faucets)
    # ------------------------------------------------------------------

    def turn_knob(
        self,
        current_pose: np.ndarray,
        knob_pose: np.ndarray,
        rotation_deg: float = 90.0,
        speed_factor: float = 0.3,
        gripper_grip_mm: float = 10.0,
    ) -> TaskResult:
        """Plan turning a knob by gripping and rotating wrist (J5).

        Sequence:
        1. Approach knob
        2. Grip knob
        3. Rotate J5 by rotation_deg
        4. Release
        5. Retreat
        """
        approach_offset = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        knob_approach = knob_pose + approach_offset

        rotated_pose = knob_pose.copy()
        rotated_pose[5] = knob_pose[5] + rotation_deg
        rotated_pose = self.planner.clamp_joint_angles(rotated_pose)

        try:
            segments = [
                self._move_segment(
                    current_pose, knob_approach, speed_factor, 0.0, 25.0
                ),
                self._move_segment(
                    knob_approach, knob_pose, speed_factor * 0.5, 25.0, 25.0
                ),
                self._gripper_segment(knob_pose, 25.0, gripper_grip_mm),
                self._move_segment(
                    knob_pose, rotated_pose, speed_factor * 0.2, gripper_grip_mm, gripper_grip_mm
                ),
                self._gripper_segment(rotated_pose, gripper_grip_mm, 25.0),
                self._move_segment(
                    rotated_pose, current_pose, speed_factor, 25.0, 0.0
                ),
            ]

            combined = self._concat_trajectories(segments, label="turn_knob")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message=f"Knob rotation {rotation_deg:.0f}° planned",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Turn knob failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )

    # ------------------------------------------------------------------
    # Button press
    # ------------------------------------------------------------------

    def press_button(
        self,
        current_pose: np.ndarray,
        button_pose: np.ndarray,
        press_depth_deg: float = 5.0,
        speed_factor: float = 0.3,
    ) -> TaskResult:
        """Plan pressing a button with closed gripper tip."""
        approach_offset = np.array([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        button_approach = button_pose + approach_offset
        pressed_pose = button_pose.copy()
        pressed_pose[1] = button_pose[1] + press_depth_deg  # Lean forward into button

        try:
            segments = [
                self._move_segment(
                    current_pose, button_approach, speed_factor, 0.0, 0.0
                ),
                self._move_segment(
                    button_approach, button_pose, speed_factor * 0.5, 0.0, 0.0
                ),
                # Press
                self._move_segment(
                    button_pose, pressed_pose, speed_factor * 0.2, 0.0, 0.0
                ),
                # Release
                self._move_segment(
                    pressed_pose, button_pose, speed_factor * 0.3, 0.0, 0.0
                ),
                self._move_segment(
                    button_pose, current_pose, speed_factor, 0.0, 0.0
                ),
            ]

            combined = self._concat_trajectories(segments, label="press_button")
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=combined,
                message="Button press planned",
                sub_trajectories=segments,
            )
        except Exception as e:
            logger.error("Press button failed: %s", e)
            return TaskResult(
                status=TaskStatus.FAILED,
                trajectory=Trajectory(),
                message=f"Planning failed: {e}",
            )


# ======================================================================
# Composite Task Planner
# ======================================================================

# Path to task decomposition graph
_TASK_GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "task_graph.json"
_GRASP_STRATEGIES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "grasp_strategies.json"


@dataclass
class CompositeTaskResult:
    """Result of a composite (multi-step) task plan."""

    status: TaskStatus
    task_name: str
    steps_planned: int
    steps_total: int
    sub_results: list[TaskResult] = field(default_factory=list)
    message: str = ""
    failed_step: Optional[str] = None


class CompositeTaskPlanner:
    """Plans multi-step kitchen tasks using the task decomposition graph.

    Decomposes composite task names (e.g., "PrepareCoffee") into sequences
    of atomic skills (open_cabinet, pick_from_cabinet, etc.) using the
    task graph derived from the NVIDIA Kitchen-Sim-Demos dataset.

    Usage:
        planner = CompositeTaskPlanner()
        result = planner.plan_composite("PrepareCoffee", current_pose, poses)
    """

    def __init__(
        self,
        task_planner: Optional[TaskPlanner] = None,
        task_graph_path: Optional[Path] = None,
        grasp_strategies_path: Optional[Path] = None,
    ):
        self._planner = task_planner or TaskPlanner()
        self._task_graph = self._load_task_graph(task_graph_path or _TASK_GRAPH_PATH)
        self._grasp_strategies = self._load_grasp_strategies(
            grasp_strategies_path or _GRASP_STRATEGIES_PATH
        )

    def _load_task_graph(self, path: Path) -> Dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        logger.warning("Task graph not found at %s, using empty graph", path)
        return {"atomic_skills": {}, "composite_tasks": {}}

    def _load_grasp_strategies(self, path: Path) -> Dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        logger.warning("Grasp strategies not found at %s", path)
        return {"strategies": {}, "object_strategy_mapping": {}}

    @property
    def available_tasks(self) -> List[str]:
        """List all available composite task names."""
        return list(self._task_graph.get("composite_tasks", {}).keys())

    @property
    def available_skills(self) -> List[str]:
        """List all available atomic skills."""
        return list(self._task_graph.get("atomic_skills", {}).keys())

    def get_task_steps(self, task_name: str) -> Optional[List[str]]:
        """Get the atomic skill sequence for a composite task."""
        task = self._task_graph.get("composite_tasks", {}).get(task_name)
        if task is None:
            return None
        return task.get("steps", [])

    def get_grasp_strategy(self, object_name: str) -> Optional[Dict]:
        """Look up the optimal grasp strategy for an object."""
        mapping = self._grasp_strategies.get("object_strategy_mapping", {})
        strategy_name = mapping.get(object_name)
        if strategy_name is None:
            return None
        return self._grasp_strategies.get("strategies", {}).get(strategy_name)

    def plan_composite(
        self,
        task_name: str,
        current_pose: np.ndarray,
        target_poses: Dict[str, np.ndarray],
        speed_factor: float = 0.5,
    ) -> CompositeTaskResult:
        """Plan a full composite task as a sequence of atomic skills.

        Args:
            task_name: Name of the composite task (e.g., "PrepareCoffee")
            current_pose: Current arm joint angles (degrees)
            target_poses: Dict mapping skill names / locations to joint poses
                Example: {"cabinet_handle": array, "counter": array, ...}
            speed_factor: Global speed scaling factor

        Returns:
            CompositeTaskResult with sub-results for each step.
        """
        steps = self.get_task_steps(task_name)
        if steps is None:
            return CompositeTaskResult(
                status=TaskStatus.FAILED,
                task_name=task_name,
                steps_planned=0,
                steps_total=0,
                message=f"Unknown composite task: '{task_name}'. "
                        f"Available: {', '.join(self.available_tasks)}",
            )

        sub_results: list[TaskResult] = []
        pose = current_pose.copy()

        for i, skill_name in enumerate(steps):
            skill_def = self._task_graph.get("atomic_skills", {}).get(skill_name, {})
            skill_type = skill_def.get("skill_type", "unknown")

            logger.info(
                "Composite task '%s' step %d/%d: %s (type=%s)",
                task_name, i + 1, len(steps), skill_name, skill_type,
            )

            result = self._plan_atomic_skill(
                skill_name, skill_type, pose, target_poses, speed_factor
            )
            sub_results.append(result)

            if result.status == TaskStatus.FAILED:
                return CompositeTaskResult(
                    status=TaskStatus.PARTIAL,
                    task_name=task_name,
                    steps_planned=i,
                    steps_total=len(steps),
                    sub_results=sub_results,
                    message=f"Failed at step {i + 1}/{len(steps)}: {skill_name}",
                    failed_step=skill_name,
                )

            # Update pose to end of this trajectory
            if result.trajectory.points:
                pose = np.array(result.trajectory.points[-1].positions)

        return CompositeTaskResult(
            status=TaskStatus.SUCCESS,
            task_name=task_name,
            steps_planned=len(steps),
            steps_total=len(steps),
            sub_results=sub_results,
            message=f"Composite task '{task_name}' planned: {len(steps)} steps",
        )

    def _plan_atomic_skill(
        self,
        skill_name: str,
        skill_type: str,
        current_pose: np.ndarray,
        target_poses: Dict[str, np.ndarray],
        speed_factor: float,
    ) -> TaskResult:
        """Plan a single atomic skill within a composite task."""

        # Get a target pose for this skill (fall back to ready pose if unknown)
        target = target_poses.get(skill_name, READY_POSE)

        if skill_type == "pick":
            pick_pose = target_poses.get(skill_name, target)
            place_pose = target_poses.get("counter", READY_POSE)
            return self._planner.pick_and_place(
                current_pose, pick_pose, place_pose, speed_factor=speed_factor,
            )

        elif skill_type == "place":
            place_pose = target_poses.get(skill_name, target)
            return self._planner.pick_and_place(
                current_pose, current_pose, place_pose, speed_factor=speed_factor,
            )

        elif skill_type == "articulation":
            direction = np.zeros(NUM_ARM_JOINTS)
            direction[0] = 1.0  # Default pull direction: base rotation
            return self._planner.open_fixture(
                current_pose, target, direction, speed_factor=speed_factor,
            )

        elif skill_type == "knob":
            return self._planner.turn_knob(
                current_pose, target, rotation_deg=90.0, speed_factor=speed_factor,
            )

        elif skill_type == "button":
            return self._planner.press_button(
                current_pose, target, speed_factor=speed_factor,
            )

        elif skill_type == "navigation":
            # Navigation is a no-op for fixed-base D1 arm
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=Trajectory(label="navigate_noop"),
                message="Navigation skipped (fixed-base arm)",
            )

        else:
            # Generic move to target
            return TaskResult(
                status=TaskStatus.SUCCESS,
                trajectory=self._planner.planner.linear_joint_trajectory(
                    current_pose, target, speed_factor, 0.0, 0.0,
                ),
                message=f"Generic move for skill '{skill_name}'",
            )
