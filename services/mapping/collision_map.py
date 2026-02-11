"""3D collision checking via voxel occupancy grid.

Builds collision volumes from env_map point cloud + object bounding boxes.
Exposes check_point, check_sphere, check_path for motion planner queries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CollisionMap:
    """3D voxel-based collision checker."""

    def __init__(self, voxel_size_m: float = 0.01):
        self.voxel_size = voxel_size_m
        # Occupied voxels stored as set of (ix, iy, iz) tuples for O(1) lookup
        self._occupied: set = set()
        self._object_voxels: set = set()  # Voxels from object bounding boxes

    def update_from_cloud(self, points: np.ndarray) -> None:
        """Rebuild occupancy grid from point cloud (Nx3 or Nx6)."""
        if len(points) == 0:
            self._occupied = set()
            return

        coords = points[:, :3]
        voxels = (coords / self.voxel_size).astype(np.int32)
        self._occupied = set(map(tuple, voxels.tolist()))

    def update_from_objects(self, objects: List[Dict[str, Any]]) -> None:
        """Add object bounding boxes to occupancy grid.

        Each object: {position_mm: [x,y,z], bbox_mm: [w,d,h]}
        """
        self._object_voxels = set()
        vs_mm = self.voxel_size * 1000  # voxel size in mm

        for obj in objects:
            pos = np.array(obj.get("position_mm", [0, 0, 0]), dtype=float)
            bbox = np.array(obj.get("bbox_mm", [0, 0, 0]), dtype=float)

            # Convert to meters for voxel grid
            pos_m = pos / 1000.0
            bbox_m = bbox / 1000.0

            # Fill voxels in bounding box
            half = bbox_m / 2.0
            lo = ((pos_m - half) / self.voxel_size).astype(int)
            hi = ((pos_m + half) / self.voxel_size).astype(int) + 1

            for ix in range(lo[0], hi[0]):
                for iy in range(lo[1], hi[1]):
                    for iz in range(lo[2], hi[2]):
                        self._object_voxels.add((ix, iy, iz))

    def _all_occupied(self) -> set:
        return self._occupied | self._object_voxels

    def _point_to_voxel(self, xyz_m: np.ndarray) -> Tuple[int, int, int]:
        v = (np.asarray(xyz_m) / self.voxel_size).astype(int)
        return tuple(v.tolist())

    def check_point(self, xyz_m: List[float]) -> str:
        """Check if a point is free/occupied/unknown.

        Args:
            xyz_m: [x, y, z] in meters.

        Returns:
            "free", "occupied", or "unknown"
        """
        voxel = self._point_to_voxel(np.array(xyz_m))
        occ = self._all_occupied()
        if not occ:
            return "unknown"
        if voxel in occ:
            return "occupied"
        return "free"

    def check_sphere(self, xyz_m: List[float], radius_m: float) -> bool:
        """Check if a sphere collides with any occupied voxel.

        Returns True if collision detected.
        """
        center = np.array(xyz_m)
        r_voxels = int(np.ceil(radius_m / self.voxel_size))
        cv = self._point_to_voxel(center)
        occ = self._all_occupied()

        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    v = (cv[0] + dx, cv[1] + dy, cv[2] + dz)
                    if v in occ:
                        # Check actual distance
                        voxel_center = (np.array(v) + 0.5) * self.voxel_size
                        dist = np.linalg.norm(voxel_center - center)
                        if dist <= radius_m + self.voxel_size * 0.5:
                            return True
        return False

    def check_path(
        self, points_m: List[List[float]], radius_m: float = 0.02
    ) -> List[Dict[str, Any]]:
        """Check a path (sequence of points) for collisions.

        Args:
            points_m: List of [x,y,z] waypoints in meters.
            radius_m: Collision check radius around each point.

        Returns:
            List of collision dicts: [{index, point, distance_to_obstacle}]
        """
        collisions = []
        occ = self._all_occupied()
        if not occ:
            return collisions

        for i, pt in enumerate(points_m):
            if self.check_sphere(pt, radius_m):
                # Find nearest occupied voxel for distance
                center = np.array(pt)
                cv = self._point_to_voxel(center)
                min_dist = float("inf")
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        for dz in range(-3, 4):
                            v = (cv[0] + dx, cv[1] + dy, cv[2] + dz)
                            if v in occ:
                                vc = (np.array(v) + 0.5) * self.voxel_size
                                d = float(np.linalg.norm(vc - center))
                                min_dist = min(min_dist, d)
                collisions.append(
                    {
                        "index": i,
                        "point": pt,
                        "distance_to_obstacle": round(min_dist, 4),
                    }
                )

        return collisions

    def get_occupied_count(self) -> int:
        return len(self._all_occupied())

    def get_occupied_centers(self) -> np.ndarray:
        """Get all occupied voxel centers as Nx3 array."""
        occ = self._all_occupied()
        if not occ:
            return np.zeros((0, 3), dtype=np.float32)
        voxels = np.array(list(occ), dtype=np.float32)
        return (voxels + 0.5) * self.voxel_size
