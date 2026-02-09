"""
Digital Twin — Synchronized 3D environment for Factory 3D visualization.

Maintains a digital representation of the physical workspace that includes:
1. The arm itself (joint positions from FK)
2. Objects detected by the VLA model
3. Workspace boundaries and obstacles
4. User-set waypoints

This model is designed to sync with the Factory 3D WebGL visualization,
enabling the user to:
- See a live 3D view of the arm and its surroundings
- Place waypoints in the 3D environment
- Preview arm trajectories before execution
- Manipulate the arm by interacting with the digital model

All coordinates are in the arm-base frame (mm).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .vla_model import DetectedObject3D, ObjectShape

logger = logging.getLogger("th3cl4w.vision.digital_twin")

# Unitree D1 constants
ARM_MAX_REACH_MM = 550.0
ARM_LINK_LENGTHS_MM = [100.0, 220.0, 220.0, 60.0, 60.0, 40.0]  # approximate


class WaypointStatus(Enum):
    """Status of a user-set waypoint."""
    PENDING = "pending"          # waiting for arm to reach
    ACTIVE = "active"            # currently being approached
    REACHED = "reached"          # arm arrived
    SKIPPED = "skipped"          # skipped due to collision or unreachable
    CANCELLED = "cancelled"      # user cancelled


@dataclass
class DigitalWaypoint:
    """A waypoint in the digital twin that the arm should navigate to.

    Can be set in either Cartesian space (position_mm) or joint space
    (joint_angles_deg). If Cartesian, IK will be solved for joint angles.
    """

    waypoint_id: str
    label: str = ""

    # Cartesian position (in arm-base frame, mm)
    position_mm: Optional[np.ndarray] = None  # (3,) XYZ

    # Joint-space target (degrees)
    joint_angles_deg: Optional[np.ndarray] = None  # (6,)

    # Gripper target
    gripper_mm: float = 30.0

    # Navigation parameters
    speed_factor: float = 0.6
    approach_from: str = "auto"  # "top", "side", "auto"

    # Status
    status: WaypointStatus = WaypointStatus.PENDING
    order: int = 0  # execution order in the waypoint sequence

    # Metadata
    created_at: float = 0.0
    reached_at: float = 0.0

    def to_dict(self) -> dict:
        result: dict = {
            "waypoint_id": self.waypoint_id,
            "label": self.label,
            "gripper_mm": self.gripper_mm,
            "speed_factor": self.speed_factor,
            "approach_from": self.approach_from,
            "status": self.status.value,
            "order": self.order,
            "created_at": round(self.created_at, 3),
        }
        if self.position_mm is not None:
            result["position_mm"] = [round(v, 1) for v in self.position_mm.tolist()]
        if self.joint_angles_deg is not None:
            result["joint_angles_deg"] = [round(v, 2) for v in self.joint_angles_deg.tolist()]
        if self.reached_at > 0:
            result["reached_at"] = round(self.reached_at, 3)
        return result


@dataclass
class ArmState:
    """Current state of the arm in the digital twin."""

    joint_angles_deg: np.ndarray  # (6,) current joint angles
    joint_positions_3d: list[np.ndarray]  # list of (3,) XYZ for each joint
    end_effector_pose: np.ndarray  # (4,4) homogeneous transform
    gripper_mm: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "joint_angles_deg": [round(v, 2) for v in self.joint_angles_deg.tolist()],
            "joint_positions_3d": [
                [round(v, 1) for v in pos.tolist()] for pos in self.joint_positions_3d
            ],
            "end_effector_position_mm": [
                round(self.end_effector_pose[0, 3] * 1000, 1),
                round(self.end_effector_pose[1, 3] * 1000, 1),
                round(self.end_effector_pose[2, 3] * 1000, 1),
            ],
            "gripper_mm": round(self.gripper_mm, 1),
            "timestamp": round(self.timestamp, 4),
        }


@dataclass
class DigitalTwinSnapshot:
    """Complete snapshot of the digital twin state for Factory 3D rendering."""

    # Arm state
    arm: Optional[ArmState] = None

    # Detected objects with meshes
    objects: list[dict] = field(default_factory=list)

    # Workspace boundaries
    workspace_radius_mm: float = ARM_MAX_REACH_MM
    workspace_bounds: dict = field(default_factory=lambda: {
        "x_min": -600.0, "x_max": 600.0,
        "y_min": -600.0, "y_max": 600.0,
        "z_min": -50.0, "z_max": 500.0,
    })

    # Waypoints
    waypoints: list[dict] = field(default_factory=list)

    # Preview trajectory (if any)
    trajectory_preview: Optional[list[dict]] = None

    # Metadata
    timestamp: float = 0.0
    frame_number: int = 0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm.to_dict() if self.arm else None,
            "objects": self.objects,
            "workspace": {
                "radius_mm": self.workspace_radius_mm,
                "bounds": self.workspace_bounds,
            },
            "waypoints": self.waypoints,
            "trajectory_preview": self.trajectory_preview,
            "timestamp": round(self.timestamp, 4),
            "frame_number": self.frame_number,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class DigitalTwin:
    """Maintains a synchronized digital representation of the physical workspace.

    Integrates data from:
    - VLA model (detected 3D objects)
    - Arm kinematics (joint positions)
    - User input (waypoints)

    Outputs snapshots for Factory 3D WebGL rendering.
    """

    def __init__(self, kinematics=None):
        """
        Args:
            kinematics: D1Kinematics instance for FK computation.
                        If None, arm visualization uses approximate positions.
        """
        self._kinematics = kinematics
        self._lock = threading.Lock()

        # State
        self._objects: dict[str, DetectedObject3D] = {}
        self._waypoints: list[DigitalWaypoint] = []
        self._waypoint_counter = 0
        self._arm_state: Optional[ArmState] = None
        self._trajectory_preview: Optional[list[dict]] = None
        self._frame_number = 0

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def update_objects(self, objects: list[DetectedObject3D]):
        """Update the set of detected objects from the VLA model."""
        with self._lock:
            # Replace all objects with new detections
            new_objects = {}
            for obj in objects:
                new_objects[obj.object_id] = obj
            self._objects = new_objects
            self._frame_number += 1

    def add_object(self, obj: DetectedObject3D):
        """Add or update a single object."""
        with self._lock:
            self._objects[obj.object_id] = obj

    def remove_object(self, object_id: str):
        """Remove an object by ID."""
        with self._lock:
            self._objects.pop(object_id, None)

    def get_objects(self) -> list[DetectedObject3D]:
        """Get all objects in the digital twin."""
        with self._lock:
            return list(self._objects.values())

    def get_reachable_objects(self) -> list[DetectedObject3D]:
        """Get objects within the arm's reach envelope."""
        with self._lock:
            return [o for o in self._objects.values() if o.reachable]

    # ------------------------------------------------------------------
    # Arm state
    # ------------------------------------------------------------------

    def update_arm_state(
        self,
        joint_angles_deg: np.ndarray,
        gripper_mm: float = 0.0,
    ):
        """Update the arm's current joint state.

        Computes forward kinematics to determine joint 3D positions
        and end-effector pose.
        """
        angles = np.asarray(joint_angles_deg, dtype=float)

        # Compute FK positions
        joint_positions = []
        ee_pose = np.eye(4)

        if self._kinematics is not None:
            q7 = np.zeros(7)
            q7[:6] = np.deg2rad(angles)
            ee_pose = self._kinematics.forward_kinematics(q7)
            positions_3d = self._kinematics.get_joint_positions_3d(q7)
            joint_positions = [pos * 1000.0 for pos in positions_3d]  # m to mm
        else:
            # Approximate positions when kinematics unavailable
            joint_positions = self._approximate_joint_positions(angles)
            ee_pose = np.eye(4)
            if joint_positions:
                ee_pose[:3, 3] = joint_positions[-1] / 1000.0  # mm to m

        with self._lock:
            self._arm_state = ArmState(
                joint_angles_deg=angles.copy(),
                joint_positions_3d=joint_positions,
                end_effector_pose=ee_pose,
                gripper_mm=gripper_mm,
                timestamp=time.monotonic(),
            )

    def _approximate_joint_positions(
        self, angles_deg: np.ndarray
    ) -> list[np.ndarray]:
        """Rough joint position approximation when FK isn't available."""
        positions = [np.array([0.0, 0.0, 0.0])]  # base
        z_offset = 0.0

        for i, length in enumerate(ARM_LINK_LENGTHS_MM):
            angle_rad = math.radians(angles_deg[i] if i < len(angles_deg) else 0.0)
            prev = positions[-1]

            if i == 0:  # base rotation (yaw)
                pos = np.array([
                    length * math.cos(angle_rad),
                    length * math.sin(angle_rad),
                    z_offset,
                ])
            elif i % 2 == 1:  # pitch joints
                z_offset += length * math.cos(angle_rad)
                r = math.sqrt(prev[0] ** 2 + prev[1] ** 2)
                r += length * math.sin(angle_rad)
                base_angle = math.atan2(prev[1], prev[0]) if r > 0 else 0
                pos = np.array([
                    r * math.cos(base_angle),
                    r * math.sin(base_angle),
                    z_offset,
                ])
            else:
                pos = prev + np.array([0, 0, length])
                z_offset += length

            positions.append(pos)

        return positions

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------

    def add_waypoint(
        self,
        position_mm: Optional[np.ndarray] = None,
        joint_angles_deg: Optional[np.ndarray] = None,
        label: str = "",
        gripper_mm: float = 30.0,
        speed_factor: float = 0.6,
        approach_from: str = "auto",
    ) -> DigitalWaypoint:
        """Add a new waypoint to the sequence.

        Can be specified in Cartesian or joint space.
        """
        self._waypoint_counter += 1
        wp_id = f"wp_{self._waypoint_counter}"

        wp = DigitalWaypoint(
            waypoint_id=wp_id,
            label=label or f"Waypoint {self._waypoint_counter}",
            position_mm=np.asarray(position_mm, dtype=float) if position_mm is not None else None,
            joint_angles_deg=np.asarray(joint_angles_deg, dtype=float) if joint_angles_deg is not None else None,
            gripper_mm=gripper_mm,
            speed_factor=speed_factor,
            approach_from=approach_from,
            status=WaypointStatus.PENDING,
            order=self._waypoint_counter,
            created_at=time.monotonic(),
        )

        with self._lock:
            self._waypoints.append(wp)

        logger.info("Waypoint added: %s at order %d", wp_id, wp.order)
        return wp

    def remove_waypoint(self, waypoint_id: str) -> bool:
        """Remove a waypoint by ID."""
        with self._lock:
            before = len(self._waypoints)
            self._waypoints = [
                wp for wp in self._waypoints if wp.waypoint_id != waypoint_id
            ]
            removed = len(self._waypoints) < before
        if removed:
            logger.info("Waypoint removed: %s", waypoint_id)
        return removed

    def reorder_waypoints(self, waypoint_ids: list[str]):
        """Reorder waypoints by providing IDs in desired execution order."""
        with self._lock:
            wp_map = {wp.waypoint_id: wp for wp in self._waypoints}
            reordered = []
            for i, wp_id in enumerate(waypoint_ids):
                if wp_id in wp_map:
                    wp = wp_map[wp_id]
                    wp.order = i + 1
                    reordered.append(wp)
            # Add any waypoints not in the reorder list at the end
            remaining = [
                wp for wp in self._waypoints if wp.waypoint_id not in set(waypoint_ids)
            ]
            for wp in remaining:
                wp.order = len(reordered) + 1
                reordered.append(wp)
            self._waypoints = reordered

    def get_waypoints(self) -> list[DigitalWaypoint]:
        """Get all waypoints in execution order."""
        with self._lock:
            return sorted(self._waypoints, key=lambda wp: wp.order)

    def get_pending_waypoints(self) -> list[DigitalWaypoint]:
        """Get waypoints that haven't been reached yet."""
        with self._lock:
            return sorted(
                [wp for wp in self._waypoints if wp.status == WaypointStatus.PENDING],
                key=lambda wp: wp.order,
            )

    def mark_waypoint_reached(self, waypoint_id: str):
        """Mark a waypoint as reached."""
        with self._lock:
            for wp in self._waypoints:
                if wp.waypoint_id == waypoint_id:
                    wp.status = WaypointStatus.REACHED
                    wp.reached_at = time.monotonic()
                    logger.info("Waypoint reached: %s", waypoint_id)
                    break

    def mark_waypoint_active(self, waypoint_id: str):
        """Mark a waypoint as currently being approached."""
        with self._lock:
            for wp in self._waypoints:
                if wp.waypoint_id == waypoint_id:
                    wp.status = WaypointStatus.ACTIVE

    def clear_waypoints(self):
        """Clear all waypoints."""
        with self._lock:
            self._waypoints.clear()
            self._waypoint_counter = 0
        logger.info("All waypoints cleared")

    # ------------------------------------------------------------------
    # Trajectory preview
    # ------------------------------------------------------------------

    def set_trajectory_preview(self, trajectory_points: list[dict]):
        """Set a trajectory preview for visualization in Factory 3D.

        Each point should have: position_mm (list[3]), time (float)
        """
        with self._lock:
            self._trajectory_preview = trajectory_points

    def clear_trajectory_preview(self):
        """Clear the trajectory preview."""
        with self._lock:
            self._trajectory_preview = None

    # ------------------------------------------------------------------
    # Snapshot for Factory 3D
    # ------------------------------------------------------------------

    def snapshot(self) -> DigitalTwinSnapshot:
        """Take a complete snapshot for Factory 3D rendering."""
        with self._lock:
            arm = self._arm_state

            # Serialize objects with mesh data
            objects = []
            for obj in self._objects.values():
                obj_dict = obj.to_dict()
                if obj.mesh_vertices is not None:
                    obj_dict["mesh"] = {
                        "vertices": obj.mesh_vertices,
                        "faces": obj.mesh_faces,
                    }
                objects.append(obj_dict)

            waypoints = [wp.to_dict() for wp in sorted(self._waypoints, key=lambda w: w.order)]
            trajectory = self._trajectory_preview

        return DigitalTwinSnapshot(
            arm=arm,
            objects=objects,
            waypoints=waypoints,
            trajectory_preview=trajectory,
            timestamp=time.monotonic(),
            frame_number=self._frame_number,
        )

    def get_factory3d_update(self) -> dict:
        """Get a Factory 3D-compatible update payload.

        Returns a dict suitable for sending over WebSocket to the
        Factory 3D WebGL frontend.
        """
        snap = self.snapshot()
        return {
            "type": "digital_twin_update",
            "data": snap.to_dict(),
        }

    def get_stats(self) -> dict:
        """Get digital twin statistics."""
        with self._lock:
            n_objects = len(self._objects)
            n_reachable = sum(1 for o in self._objects.values() if o.reachable)
            n_waypoints = len(self._waypoints)
            n_pending = sum(1 for wp in self._waypoints if wp.status == WaypointStatus.PENDING)

        return {
            "objects": n_objects,
            "reachable_objects": n_reachable,
            "waypoints_total": n_waypoints,
            "waypoints_pending": n_pending,
            "has_arm_state": self._arm_state is not None,
            "has_trajectory_preview": self._trajectory_preview is not None,
            "frame_number": self._frame_number,
        }
