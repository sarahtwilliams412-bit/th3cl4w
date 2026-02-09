"""
Waypoint Navigator — Plans and executes multi-waypoint navigation sequences.

Takes waypoints set in the digital twin (Factory 3D / real-world 3D model)
and generates collision-aware trajectories to navigate the arm through them.

Planning pipeline:
  1. Retrieve pending waypoints from the digital twin
  2. For Cartesian waypoints: solve IK to get joint angles
  3. Validate all waypoints against joint limits and workspace bounds
  4. Check for collisions with known objects in the world model
  5. Optimize the path order if requested (nearest-neighbor or user-defined)
  6. Generate smooth trajectories between consecutive waypoints
  7. Return the complete plan for execution

All angles in DEGREES, positions in mm.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.kinematics.kinematics import D1Kinematics
from src.planning.motion_planner import (
    MotionPlanner,
    Waypoint,
    Trajectory,
    TrajectoryPoint,
    NUM_ARM_JOINTS,
    JOINT_LIMITS_DEG,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
)
from src.planning.task_planner import TaskPlanner, TaskResult, TaskStatus, HOME_POSE

logger = logging.getLogger("th3cl4w.planning.waypoint_navigator")


class NavigationStatus(Enum):
    """Status of a navigation plan."""

    READY = "ready"  # plan generated, ready to execute
    EXECUTING = "executing"  # currently navigating
    COMPLETED = "completed"  # all waypoints reached
    FAILED = "failed"  # planning or execution failed
    PARTIAL = "partial"  # some waypoints reached, some failed


class PathOrderStrategy(Enum):
    """Strategy for ordering waypoints."""

    USER_DEFINED = "user_defined"  # respect user-set order
    NEAREST_NEIGHBOR = "nearest"  # greedy nearest-neighbor
    SHORTEST_PATH = "shortest_path"  # attempt TSP optimization


@dataclass
class WaypointPlanEntry:
    """A single waypoint in the navigation plan with resolved joint angles."""

    waypoint_id: str
    label: str
    joint_angles_deg: np.ndarray  # (6,) resolved joint angles
    gripper_mm: float = 30.0
    speed_factor: float = 0.6
    position_mm: Optional[np.ndarray] = None  # (3,) if Cartesian
    ik_success: bool = True
    ik_error_mm: float = 0.0
    collision_free: bool = True
    collision_objects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result: dict = {
            "waypoint_id": self.waypoint_id,
            "label": self.label,
            "joint_angles_deg": [round(v, 2) for v in self.joint_angles_deg.tolist()],
            "gripper_mm": self.gripper_mm,
            "speed_factor": self.speed_factor,
            "ik_success": self.ik_success,
            "ik_error_mm": round(self.ik_error_mm, 1),
            "collision_free": self.collision_free,
        }
        if self.position_mm is not None:
            result["position_mm"] = [round(v, 1) for v in self.position_mm.tolist()]
        if self.collision_objects:
            result["collision_objects"] = self.collision_objects
        return result


@dataclass
class NavigationPlan:
    """Complete navigation plan through a sequence of waypoints."""

    entries: list[WaypointPlanEntry] = field(default_factory=list)
    trajectory: Optional[Trajectory] = None
    status: NavigationStatus = NavigationStatus.READY
    message: str = ""
    total_distance_deg: float = 0.0  # sum of joint-space distances
    estimated_duration_s: float = 0.0
    planning_time_ms: float = 0.0

    # Segment trajectories (one per waypoint-to-waypoint transition)
    segments: list[Trajectory] = field(default_factory=list)

    # Which waypoints had issues
    failed_waypoints: list[str] = field(default_factory=list)
    collision_waypoints: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in (NavigationStatus.READY, NavigationStatus.COMPLETED)

    @property
    def num_waypoints(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        result: dict = {
            "status": self.status.value,
            "message": self.message,
            "num_waypoints": self.num_waypoints,
            "total_distance_deg": round(self.total_distance_deg, 1),
            "estimated_duration_s": round(self.estimated_duration_s, 2),
            "planning_time_ms": round(self.planning_time_ms, 1),
            "entries": [e.to_dict() for e in self.entries],
            "failed_waypoints": self.failed_waypoints,
            "collision_waypoints": self.collision_waypoints,
        }
        if self.trajectory:
            result["trajectory"] = {
                "num_points": self.trajectory.num_points,
                "duration_s": round(self.trajectory.duration, 2),
            }
        return result


class WaypointNavigator:
    """Plans and manages multi-waypoint navigation for the D1 arm.

    Integrates with:
    - DigitalTwin: receives waypoints
    - WorldModel: checks for collisions
    - MotionPlanner: generates trajectories
    - D1Kinematics: solves IK for Cartesian waypoints

    Planning approach:
    1. Resolve all waypoints to joint space (IK for Cartesian targets)
    2. Validate joint limits
    3. Check collisions along paths between consecutive waypoints
    4. Generate smooth trajectories with trapezoidal velocity profiles
    5. Package into a NavigationPlan for the executor
    """

    def __init__(
        self,
        kinematics: Optional[D1Kinematics] = None,
        motion_planner: Optional[MotionPlanner] = None,
        task_planner: Optional[TaskPlanner] = None,
        world_model=None,
        max_ik_error_mm: float = 30.0,
        collision_clearance_mm: float = 40.0,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.planner = motion_planner or MotionPlanner(kinematics=self.kinematics)
        self.task_planner = task_planner or TaskPlanner(motion_planner=self.planner)
        self.world_model = world_model
        self.max_ik_error = max_ik_error_mm
        self.collision_clearance = collision_clearance_mm

    # ------------------------------------------------------------------
    # Main planning entry point
    # ------------------------------------------------------------------

    def plan_navigation(
        self,
        digital_twin,
        current_angles_deg: np.ndarray,
        current_gripper_mm: float = 0.0,
        order_strategy: PathOrderStrategy = PathOrderStrategy.USER_DEFINED,
        return_home: bool = False,
    ) -> NavigationPlan:
        """Plan a navigation sequence through the digital twin's waypoints.

        Args:
            digital_twin: DigitalTwin instance with waypoints.
            current_angles_deg: (6,) current arm joint angles in degrees.
            current_gripper_mm: Current gripper position in mm.
            order_strategy: How to order the waypoints.
            return_home: If True, add a return-to-home waypoint at the end.

        Returns:
            NavigationPlan with trajectory and per-waypoint details.
        """
        t0 = time.monotonic()
        current = np.asarray(current_angles_deg, dtype=float)

        # Get pending waypoints from digital twin
        waypoints = digital_twin.get_pending_waypoints()
        if not waypoints:
            return NavigationPlan(
                status=NavigationStatus.COMPLETED,
                message="No pending waypoints to navigate",
            )

        # Step 1: Resolve all waypoints to joint space
        entries = []
        for wp in waypoints:
            entry = self._resolve_waypoint(wp, current)
            entries.append(entry)

        # Step 2: Order waypoints
        if order_strategy == PathOrderStrategy.NEAREST_NEIGHBOR:
            entries = self._order_nearest_neighbor(entries, current)
        elif order_strategy == PathOrderStrategy.SHORTEST_PATH:
            entries = self._order_shortest_path(entries, current)
        # USER_DEFINED: keep original order

        # Step 3: Add return-home if requested
        if return_home:
            home_entry = WaypointPlanEntry(
                waypoint_id="wp_home",
                label="Return Home",
                joint_angles_deg=HOME_POSE.copy(),
                gripper_mm=0.0,
                speed_factor=0.5,
            )
            entries.append(home_entry)

        # Step 4: Check collisions (if world model available)
        if self.world_model is not None:
            self._check_collisions(entries, current)

        # Step 5: Generate trajectories
        failed_wps = []
        collision_wps = []
        for e in entries:
            if not e.ik_success:
                failed_wps.append(e.waypoint_id)
            if not e.collision_free:
                collision_wps.append(e.waypoint_id)

        # Filter to valid entries
        valid_entries = [e for e in entries if e.ik_success]

        if not valid_entries:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return NavigationPlan(
                entries=entries,
                status=NavigationStatus.FAILED,
                message="No valid waypoints after IK resolution",
                failed_waypoints=failed_wps,
                collision_waypoints=collision_wps,
                planning_time_ms=elapsed_ms,
            )

        # Build motion planner waypoints
        trajectory, segments, total_dist = self._generate_trajectory(
            valid_entries, current, current_gripper_mm
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        status = NavigationStatus.READY
        message = f"Navigation planned: {len(valid_entries)} waypoints"
        if failed_wps:
            status = NavigationStatus.PARTIAL
            message += f", {len(failed_wps)} failed IK"
        if collision_wps:
            message += f", {len(collision_wps)} with collisions (proceeding with caution)"

        return NavigationPlan(
            entries=entries,
            trajectory=trajectory,
            segments=segments,
            status=status,
            message=message,
            total_distance_deg=total_dist,
            estimated_duration_s=trajectory.duration if trajectory else 0.0,
            planning_time_ms=elapsed_ms,
            failed_waypoints=failed_wps,
            collision_waypoints=collision_wps,
        )

    # ------------------------------------------------------------------
    # Plan from explicit waypoint list (no digital twin)
    # ------------------------------------------------------------------

    def plan_from_positions(
        self,
        positions_mm: list[np.ndarray],
        current_angles_deg: np.ndarray,
        current_gripper_mm: float = 0.0,
        speed_factor: float = 0.6,
        gripper_mm: float = 30.0,
    ) -> NavigationPlan:
        """Plan navigation from a list of Cartesian positions.

        Convenience method that doesn't require a digital twin.
        """
        t0 = time.monotonic()
        current = np.asarray(current_angles_deg, dtype=float)

        entries = []
        for i, pos in enumerate(positions_mm):
            pos = np.asarray(pos, dtype=float)

            # Build a pose matrix (position only, default orientation)
            T = self._position_to_pose(pos)
            q_init = np.zeros(7)
            q_init[:6] = np.deg2rad(current if i == 0 else entries[-1].joint_angles_deg)

            q_sol = self.kinematics.inverse_kinematics(T, q_init=q_init)
            angles_deg = np.rad2deg(q_sol[:6])
            angles_deg = self.planner.clamp_joint_angles(angles_deg)

            # Check IK accuracy
            q7_check = np.zeros(7)
            q7_check[:6] = np.deg2rad(angles_deg)
            actual_pose = self.kinematics.forward_kinematics(q7_check)
            ik_error = np.linalg.norm(actual_pose[:3, 3] * 1000.0 - pos)

            entry = WaypointPlanEntry(
                waypoint_id=f"pos_{i}",
                label=f"Position {i}",
                joint_angles_deg=angles_deg,
                gripper_mm=gripper_mm,
                speed_factor=speed_factor,
                position_mm=pos.copy(),
                ik_success=ik_error < self.max_ik_error,
                ik_error_mm=ik_error,
            )
            entries.append(entry)

        valid = [e for e in entries if e.ik_success]
        if not valid:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return NavigationPlan(
                entries=entries,
                status=NavigationStatus.FAILED,
                message="No valid positions after IK",
                planning_time_ms=elapsed_ms,
            )

        trajectory, segments, total_dist = self._generate_trajectory(
            valid, current, current_gripper_mm
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        return NavigationPlan(
            entries=entries,
            trajectory=trajectory,
            segments=segments,
            status=NavigationStatus.READY,
            message=f"Path planned: {len(valid)} positions",
            total_distance_deg=total_dist,
            estimated_duration_s=trajectory.duration if trajectory else 0.0,
            planning_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Waypoint resolution
    # ------------------------------------------------------------------

    def _resolve_waypoint(self, digital_waypoint, current_angles: np.ndarray) -> WaypointPlanEntry:
        """Resolve a DigitalWaypoint to joint angles via IK if needed."""
        wp = digital_waypoint

        # If joint angles are already specified, use them
        if wp.joint_angles_deg is not None:
            angles = np.asarray(wp.joint_angles_deg, dtype=float)
            angles = self.planner.clamp_joint_angles(angles)
            valid = self.planner.validate_joint_angles(angles)

            return WaypointPlanEntry(
                waypoint_id=wp.waypoint_id,
                label=wp.label,
                joint_angles_deg=angles,
                gripper_mm=wp.gripper_mm,
                speed_factor=wp.speed_factor,
                position_mm=wp.position_mm,
                ik_success=valid,
                ik_error_mm=0.0,
            )

        # Cartesian waypoint: solve IK
        if wp.position_mm is not None:
            pos = np.asarray(wp.position_mm, dtype=float)
            T = self._position_to_pose(pos, approach=wp.approach_from)

            q_init = np.zeros(7)
            q_init[:6] = np.deg2rad(current_angles)

            try:
                q_sol = self.kinematics.inverse_kinematics(T, q_init=q_init)
                angles_deg = np.rad2deg(q_sol[:6])
                angles_deg = self.planner.clamp_joint_angles(angles_deg)

                # Verify IK accuracy
                q7_check = np.zeros(7)
                q7_check[:6] = np.deg2rad(angles_deg)
                actual_pose = self.kinematics.forward_kinematics(q7_check)
                ik_error = np.linalg.norm(actual_pose[:3, 3] * 1000.0 - pos)

                return WaypointPlanEntry(
                    waypoint_id=wp.waypoint_id,
                    label=wp.label,
                    joint_angles_deg=angles_deg,
                    gripper_mm=wp.gripper_mm,
                    speed_factor=wp.speed_factor,
                    position_mm=pos.copy(),
                    ik_success=ik_error < self.max_ik_error,
                    ik_error_mm=ik_error,
                )
            except Exception as e:
                logger.warning("IK failed for waypoint %s: %s", wp.waypoint_id, e)
                return WaypointPlanEntry(
                    waypoint_id=wp.waypoint_id,
                    label=wp.label,
                    joint_angles_deg=current_angles.copy(),
                    gripper_mm=wp.gripper_mm,
                    speed_factor=wp.speed_factor,
                    position_mm=pos.copy(),
                    ik_success=False,
                    ik_error_mm=999.0,
                )

        # No position or joint angles specified
        return WaypointPlanEntry(
            waypoint_id=wp.waypoint_id,
            label=wp.label,
            joint_angles_deg=current_angles.copy(),
            gripper_mm=wp.gripper_mm,
            speed_factor=wp.speed_factor,
            ik_success=False,
            ik_error_mm=999.0,
        )

    def _position_to_pose(self, position_mm: np.ndarray, approach: str = "auto") -> np.ndarray:
        """Convert a 3D position to a 4x4 homogeneous transform.

        The orientation is set based on the approach direction:
        - "top": end-effector points straight down
        - "side": end-effector approaches horizontally
        - "auto": choose based on position height
        """
        pos_m = position_mm / 1000.0  # mm to meters
        x, y, z = pos_m

        if approach == "auto":
            # If target is above 150mm, approach from side; otherwise from top
            approach = "side" if position_mm[2] > 150.0 else "top"

        T = np.eye(4)
        T[:3, 3] = pos_m

        yaw = math.atan2(y, x)

        if approach == "top":
            # EE z-axis pointing down
            T[:3, :3] = np.array(
                [
                    [math.cos(yaw), math.sin(yaw), 0.0],
                    [math.sin(yaw), -math.cos(yaw), 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
        else:
            # EE z-axis pointing toward object (horizontal approach)
            T[:3, :3] = np.array(
                [
                    [0.0, 0.0, math.cos(yaw)],
                    [0.0, -1.0, math.sin(yaw)],
                    [1.0, 0.0, 0.0],
                ]
            )

        return T

    # ------------------------------------------------------------------
    # Path ordering
    # ------------------------------------------------------------------

    def _order_nearest_neighbor(
        self,
        entries: list[WaypointPlanEntry],
        start_angles: np.ndarray,
    ) -> list[WaypointPlanEntry]:
        """Order waypoints using greedy nearest-neighbor in joint space."""
        if len(entries) <= 1:
            return entries

        remaining = list(entries)
        ordered = []
        current = start_angles.copy()

        while remaining:
            # Find nearest waypoint to current position
            best_idx = 0
            best_dist = float("inf")
            for i, entry in enumerate(remaining):
                if entry.ik_success:
                    dist = np.linalg.norm(entry.joint_angles_deg - current)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            current = chosen.joint_angles_deg.copy()

        return ordered

    def _order_shortest_path(
        self,
        entries: list[WaypointPlanEntry],
        start_angles: np.ndarray,
    ) -> list[WaypointPlanEntry]:
        """Attempt to find shortest total path using 2-opt local search.

        Falls back to nearest-neighbor then improves with 2-opt swaps.
        """
        if len(entries) <= 2:
            return self._order_nearest_neighbor(entries, start_angles)

        # Start with nearest-neighbor ordering
        ordered = self._order_nearest_neighbor(entries, start_angles)

        # 2-opt improvement
        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(ordered) - 1):
                for j in range(i + 2, len(ordered)):
                    # Try reversing the segment between i+1 and j
                    new_order = (
                        ordered[: i + 1] + list(reversed(ordered[i + 1 : j + 1])) + ordered[j + 1 :]
                    )

                    old_cost = self._path_cost(ordered, start_angles)
                    new_cost = self._path_cost(new_order, start_angles)

                    if new_cost < old_cost:
                        ordered = new_order
                        improved = True

        return ordered

    def _path_cost(self, entries: list[WaypointPlanEntry], start: np.ndarray) -> float:
        """Compute total joint-space distance for a path ordering."""
        cost = 0.0
        current = start.copy()
        for entry in entries:
            if entry.ik_success:
                cost += np.linalg.norm(entry.joint_angles_deg - current)
                current = entry.joint_angles_deg
        return cost

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _check_collisions(self, entries: list[WaypointPlanEntry], start_angles: np.ndarray):
        """Check each waypoint for collisions against the world model."""
        if self.world_model is None:
            return

        for entry in entries:
            if not entry.ik_success:
                continue

            # Check the waypoint position itself
            q7 = np.zeros(7)
            q7[:6] = np.deg2rad(entry.joint_angles_deg)
            joint_positions = self.kinematics.get_joint_positions_3d(q7)

            # Check each joint position against world model obstacles
            for jp in joint_positions:
                pos_mm = jp * 1000.0  # m to mm
                collisions = self.world_model.check_collision(
                    pos_mm, radius_mm=self.collision_clearance
                )
                if collisions:
                    entry.collision_free = False
                    entry.collision_objects = [c.object_id for c in collisions]
                    break

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_trajectory(
        self,
        entries: list[WaypointPlanEntry],
        start_angles: np.ndarray,
        start_gripper: float,
    ) -> tuple[Optional[Trajectory], list[Trajectory], float]:
        """Generate a smooth trajectory through all valid waypoints."""
        if not entries:
            return None, [], 0.0

        # Build Waypoint objects for the motion planner
        mp_waypoints = [
            Waypoint(
                joint_angles=start_angles.copy(),
                gripper_mm=start_gripper,
                max_speed_factor=entries[0].speed_factor,
                label="start",
            )
        ]

        for entry in entries:
            mp_waypoints.append(
                Waypoint(
                    joint_angles=entry.joint_angles_deg.copy(),
                    gripper_mm=entry.gripper_mm,
                    max_speed_factor=entry.speed_factor,
                    label=entry.label,
                )
            )

        # Generate the full trajectory
        try:
            trajectory = self.planner.plan_waypoints(mp_waypoints)
            trajectory = self.planner.enforce_limits(trajectory)
        except Exception as e:
            logger.error("Trajectory generation failed: %s", e)
            return None, [], 0.0

        # Generate individual segments for per-waypoint tracking
        segments = []
        for i in range(len(mp_waypoints) - 1):
            try:
                seg = self.planner.linear_joint_trajectory(
                    mp_waypoints[i].joint_angles,
                    mp_waypoints[i + 1].joint_angles,
                    mp_waypoints[i + 1].max_speed_factor,
                    mp_waypoints[i].gripper_mm,
                    mp_waypoints[i + 1].gripper_mm,
                )
                segments.append(seg)
            except Exception as e:
                logger.warning("Segment %d generation failed: %s", i, e)

        # Compute total joint-space distance
        total_dist = 0.0
        current = start_angles.copy()
        for entry in entries:
            total_dist += float(np.linalg.norm(entry.joint_angles_deg - current))
            current = entry.joint_angles_deg

        return trajectory, segments, total_dist

    # ------------------------------------------------------------------
    # Execution tracking
    # ------------------------------------------------------------------

    def track_progress(
        self,
        plan: NavigationPlan,
        current_angles_deg: np.ndarray,
        digital_twin=None,
        threshold_deg: float = 5.0,
    ) -> dict:
        """Check which waypoints have been reached based on current arm position.

        Returns a status dict and updates the digital twin if provided.
        """
        current = np.asarray(current_angles_deg, dtype=float)
        status = {
            "total_waypoints": len(plan.entries),
            "reached": 0,
            "remaining": 0,
            "current_target": None,
        }

        for entry in plan.entries:
            if not entry.ik_success:
                continue

            dist = np.linalg.norm(entry.joint_angles_deg - current)
            if dist < threshold_deg:
                status["reached"] += 1
                if digital_twin is not None:
                    digital_twin.mark_waypoint_reached(entry.waypoint_id)
            else:
                status["remaining"] += 1
                if status["current_target"] is None:
                    status["current_target"] = entry.waypoint_id
                    if digital_twin is not None:
                        digital_twin.mark_waypoint_active(entry.waypoint_id)

        all_reached = status["remaining"] == 0
        if all_reached:
            plan.status = NavigationStatus.COMPLETED
        status["completed"] = all_reached

        return status
