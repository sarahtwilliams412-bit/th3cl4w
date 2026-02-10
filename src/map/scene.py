"""Unified 3D scene graph for the map server.

Thread-safe central data store holding arm skeleton, point cloud,
objects, collision voxels, waypoints, and reach envelope.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np



import logging

logger = logging.getLogger(__name__)


@dataclass
class ArmSkeletonData:
    """Snapshot of the arm's 3D skeleton."""
    joints: List[List[float]] = field(default_factory=list)  # [[x,y,z], ...] 8 positions
    links: List[Dict[str, Any]] = field(default_factory=list)  # [{start, end, radius}, ...]
    gripper_mm: float = 0.0
    ee_pose: Optional[List[List[float]]] = None  # 4x4 matrix as nested list
    joint_angles_deg: List[float] = field(default_factory=list)


@dataclass
class EnvStats:
    total_points: int = 0
    voxel_count: int = 0
    bounds_min: List[float] = field(default_factory=lambda: [0, 0, 0])
    bounds_max: List[float] = field(default_factory=lambda: [0, 0, 0])
    last_update: float = 0.0


@dataclass
class ObjectData:
    id: str = ""
    label: str = ""
    position_mm: List[float] = field(default_factory=lambda: [0, 0, 0])
    bbox_mm: List[float] = field(default_factory=lambda: [0, 0, 0])
    confidence: float = 0.0
    reachable: bool = False


class Scene:
    """Thread-safe 3D scene graph."""

    def __init__(self):
        self._lock = threading.RLock()
        self._frame: int = 0

        # Arm
        self.arm = ArmSkeletonData()

        # Environment point cloud: Nx6 [x,y,z,r,g,b]
        self._point_cloud: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self._new_points: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self._removed_voxels: List[List[float]] = []
        self.env_stats = EnvStats()

        # Voxel occupancy centers for collision viz
        self._voxel_centers: np.ndarray = np.zeros((0, 3), dtype=np.float32)

        # Objects from location server
        self.objects: List[ObjectData] = []

        # Waypoints
        self.waypoints: List[Dict[str, Any]] = []

        # Reach envelope (precomputed vertices + faces)
        self._reach_envelope: Optional[Dict[str, Any]] = None

        # Track what changed since last broadcast
        self._dirty_layers: set = set()

    @property
    def frame(self) -> int:
        return self._frame

    def update_arm(self, skeleton: ArmSkeletonData) -> None:
        with self._lock:
            self.arm = skeleton
            self._dirty_layers.add("arm")

    def update_point_cloud(
        self,
        full_cloud: np.ndarray,
        new_points: Optional[np.ndarray] = None,
        voxel_centers: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            self._point_cloud = full_cloud
            self._new_points = new_points if new_points is not None else full_cloud
            if voxel_centers is not None:
                self._voxel_centers = voxel_centers

            # Update stats
            if len(full_cloud) > 0:
                self.env_stats.total_points = len(full_cloud)
                self.env_stats.bounds_min = full_cloud[:, :3].min(axis=0).tolist()
                self.env_stats.bounds_max = full_cloud[:, :3].max(axis=0).tolist()
            else:
                self.env_stats.total_points = 0
                self.env_stats.bounds_min = [0, 0, 0]
                self.env_stats.bounds_max = [0, 0, 0]
            self.env_stats.voxel_count = len(self._voxel_centers)
            self.env_stats.last_update = time.time()
            self._dirty_layers.add("env")

    def clear_point_cloud(self) -> None:
        with self._lock:
            self._point_cloud = np.zeros((0, 6), dtype=np.float32)
            self._new_points = np.zeros((0, 6), dtype=np.float32)
            self._voxel_centers = np.zeros((0, 3), dtype=np.float32)
            self.env_stats = EnvStats()
            self._dirty_layers.add("env")

    def update_objects(self, objects: List[ObjectData]) -> None:
        with self._lock:
            self.objects = objects
            self._dirty_layers.add("objects")

    def set_reach_envelope(self, envelope: Dict[str, Any]) -> None:
        with self._lock:
            self._reach_envelope = envelope

    def set_waypoints(self, waypoints: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.waypoints = waypoints
            self._dirty_layers.add("waypoints")

    def snapshot(self, full: bool = False, layers: Optional[set] = None) -> Dict[str, Any]:
        """Produce a JSON-serializable snapshot of the scene.

        Args:
            full: If True, include full point cloud. Otherwise delta only.
            layers: Set of layer names to include. None = all.
        """
        with self._lock:
            self._frame += 1
            include = layers or {"arm", "env", "objects", "collision", "waypoints"}

            data: Dict[str, Any] = {
                "timestamp": time.time(),
                "frame": self._frame,
            }

            if "arm" in include:
                data["arm"] = {
                    "joints": self.arm.joints,
                    "links": self.arm.links,
                    "gripper_mm": self.arm.gripper_mm,
                    "ee_pose": self.arm.ee_pose,
                    "joint_angles_deg": self.arm.joint_angles_deg,
                }

            if "env" in include:
                if full:
                    pts = self._point_cloud
                    mode = "full"
                else:
                    pts = self._new_points
                    mode = "delta"

                # Limit serialized points for WebSocket (cap at 50k for perf)
                max_ws_pts = 50000
                if len(pts) > max_ws_pts:
                    idx = np.random.choice(len(pts), max_ws_pts, replace=False)
                    pts = pts[idx]

                data["env"] = {
                    "update_mode": mode,
                    "new_points": pts.tolist() if len(pts) > 0 else [],
                    "removed_voxels": self._removed_voxels,
                    "stats": {
                        "total_points": self.env_stats.total_points,
                        "voxel_count": self.env_stats.voxel_count,
                        "bounds": {
                            "min": self.env_stats.bounds_min,
                            "max": self.env_stats.bounds_max,
                        },
                    },
                }
                # Clear delta after snapshot
                self._new_points = np.zeros((0, 6), dtype=np.float32)
                self._removed_voxels = []

            if "objects" in include:
                data["objects"] = [
                    {
                        "id": o.id,
                        "label": o.label,
                        "position_mm": o.position_mm,
                        "bbox_mm": o.bbox_mm,
                        "confidence": o.confidence,
                        "reachable": o.reachable,
                    }
                    for o in self.objects
                ]

            if "collision" in include and len(self._voxel_centers) > 0:
                max_vox = 10000
                vc = self._voxel_centers
                if len(vc) > max_vox:
                    idx = np.random.choice(len(vc), max_vox, replace=False)
                    vc = vc[idx]
                data["collision_voxels"] = vc.tolist()

            if "waypoints" in include:
                data["waypoints"] = self.waypoints

            self._dirty_layers.clear()
            return data

    def get_point_cloud_raw(self) -> np.ndarray:
        """Get the raw point cloud array (for PLY export etc.)."""
        with self._lock:
            return self._point_cloud.copy()

    def get_reach_envelope(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._reach_envelope
