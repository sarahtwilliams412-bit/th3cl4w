"""
Task Planner for Unitree D1 Robotic Arm

High-level task sequences that compose motion primitives:
pick-and-place, pour, wave, and custom task sequences.

All angles in DEGREES, gripper in mm (0-65).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.planning.motion_planner import (
    MotionPlanner, Waypoint, Trajectory, TrajectoryPoint, NUM_ARM_JOINTS,
    GRIPPER_MIN_MM, GRIPPER_MAX_MM,
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


# Common poses in degrees
HOME_POSE = np.zeros(NUM_ARM_JOINTS)
READY_POSE = np.array([0.0, -45.0, 0.0, 90.0, 0.0, -45.0])


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
                combined.points.append(TrajectoryPoint(
                    time=pt.time + time_offset,
                    positions=pt.positions.copy(),
                    velocities=pt.velocities.copy(),
                    accelerations=pt.accelerations.copy(),
                    gripper_mm=pt.gripper_mm,
                ))
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
            start, end, speed, gripper_start, gripper_end,
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
            pose, pose, duration_factor,
            gripper_start, gripper_end,
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
                self._move_segment(current_pose, pick_approach, speed_factor,
                                   gripper_close_mm, gripper_open_mm),
                # 2. Descend to pick
                self._move_segment(pick_approach, pick_pose, speed_factor * 0.5,
                                   gripper_open_mm, gripper_open_mm),
                # 3. Close gripper (grasp)
                self._gripper_segment(pick_pose, gripper_open_mm, gripper_close_mm),
                # 4. Retreat up
                self._move_segment(pick_pose, pick_approach, speed_factor * 0.5,
                                   gripper_close_mm, gripper_close_mm),
                # 5. Move to place approach
                self._move_segment(pick_approach, place_approach, speed_factor,
                                   gripper_close_mm, gripper_close_mm),
                # 6. Descend to place
                self._move_segment(place_approach, place_pose, speed_factor * 0.5,
                                   gripper_close_mm, gripper_close_mm),
                # 7. Open gripper (release)
                self._gripper_segment(place_pose, gripper_close_mm, gripper_open_mm),
                # 8. Retreat
                self._move_segment(place_pose, final, speed_factor,
                                   gripper_open_mm, gripper_open_mm),
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
                self._move_segment(current_pose, pour_position, speed_factor,
                                   gripper_mm, gripper_mm),
                self._move_segment(pour_position, tilted, speed_factor * 0.3,
                                   gripper_mm, gripper_mm),
                self._move_segment(tilted, pour_position, speed_factor * 0.3,
                                   gripper_mm, gripper_mm),
                self._move_segment(pour_position, current_pose, speed_factor,
                                   gripper_mm, gripper_mm),
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
                current_pose, HOME_POSE, speed_factor,
                gripper_mm, 0.0,
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
                current_pose, READY_POSE, speed_factor,
                gripper_mm, gripper_mm,
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
