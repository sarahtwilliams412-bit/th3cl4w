"""
Bifocal Workspace Mapper — Quick 3D occupancy grid from stereo cameras.

Uses the two cameras as bifocal vision to build a fast voxel occupancy map
of the arm's workspace. Supports checkerboard calibration with tape measure
for real-world scale reference.

This is planning/measurement only — does NOT affect arm movement.
"""

import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np

from .calibration import StereoCalibrator
from .stereo_depth import StereoDepthEstimator

logger = logging.getLogger("th3cl4w.vision.workspace_mapper")


class WorkspaceMapper:
    """Builds a 3D occupancy grid from stereo camera depth maps.

    Quick and dirty voxel grid — not a work of art, just needs to work.
    The grid is centered on the arm base and covers the reachable workspace.
    """

    def __init__(
        self,
        calibrator: StereoCalibrator,
        # Workspace bounds in mm relative to arm base
        workspace_min: tuple[float, float, float] = (-600.0, -600.0, -100.0),
        workspace_max: tuple[float, float, float] = (600.0, 600.0, 700.0),
        voxel_size_mm: float = 30.0,  # ~3cm voxels — fast and rough
        depth_estimator: Optional[StereoDepthEstimator] = None,
    ):
        self.calibrator = calibrator
        self.voxel_size = voxel_size_mm
        self.ws_min = np.array(workspace_min, dtype=np.float32)
        self.ws_max = np.array(workspace_max, dtype=np.float32)

        # Grid dimensions
        self.grid_shape = tuple(
            int(np.ceil((self.ws_max[i] - self.ws_min[i]) / self.voxel_size)) for i in range(3)
        )
        # Occupancy grid: 0=free, 1=occupied, -1=unknown
        self._grid = np.full(self.grid_shape, -1, dtype=np.int8)

        # Depth estimator (lazy init if not provided)
        self._depth_est = depth_estimator

        # Scale factor from checkerboard/tape measure calibration
        self._scale_factor: float = 1.0  # mm per unit
        self._scale_calibrated = False

        # Threading
        self._lock = threading.Lock()
        self._enabled = False
        self._last_update_time: float = 0.0
        self._update_count: int = 0
        self._last_depth_map: Optional[np.ndarray] = None
        self._last_point_cloud: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        logger.info("Workspace mapper enabled")

    def disable(self):
        self._enabled = False
        logger.info("Workspace mapper disabled")

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        logger.info("Workspace mapper %s", "enabled" if self._enabled else "disabled")
        return self._enabled

    @property
    def scale_calibrated(self) -> bool:
        return self._scale_calibrated

    def _ensure_depth_estimator(self):
        """Lazy-create depth estimator from calibrator."""
        if self._depth_est is None and self.calibrator.is_calibrated:
            self._depth_est = StereoDepthEstimator(
                self.calibrator,
                num_disparities=64,  # fewer = faster
                block_size=7,  # larger = smoother but slower
            )

    def _world_to_voxel(self, point_mm: np.ndarray) -> Optional[tuple[int, int, int]]:
        """Convert world point (mm) to voxel grid index."""
        idx = ((point_mm - self.ws_min) / self.voxel_size).astype(int)
        if np.all(idx >= 0) and np.all(idx < self.grid_shape):
            return tuple(idx)
        return None

    def _voxel_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Convert voxel index to world center point (mm)."""
        return (
            self.ws_min
            + np.array([ix, iy, iz], dtype=np.float32) * self.voxel_size
            + self.voxel_size / 2
        )

    def update_from_frames(self, left: np.ndarray, right: np.ndarray) -> dict:
        """Process a stereo frame pair and update the occupancy grid.

        Returns a status dict with timing and point count info.
        """
        if not self._enabled:
            return {"status": "disabled"}

        t0 = time.monotonic()

        self._ensure_depth_estimator()
        if self._depth_est is None:
            return {"status": "not_calibrated", "error": "Stereo calibration required"}

        # Compute depth
        disparity, depth = self._depth_est.compute_depth(left, right, rectify=True)

        # Generate point cloud (only valid points)
        points = self._depth_est.compute_point_cloud(disparity, left)
        if len(points) == 0:
            return {"status": "no_points", "elapsed_ms": (time.monotonic() - t0) * 1000}

        # Apply scale correction
        xyz = points[:, :3] * self._scale_factor

        # Update occupancy grid
        with self._lock:
            # Age out old data slightly (decay toward unknown)
            # This keeps the map fresh without full resets
            if self._update_count % 10 == 0:
                occupied = self._grid == 1
                self._grid[occupied] = 0  # demote to free, will re-occupy if still there

            # Mark observed free space and occupied voxels
            n_occupied = 0
            # Subsample for speed — take every Nth point
            step = max(1, len(xyz) // 2000)
            for pt in xyz[::step]:
                voxel = self._world_to_voxel(pt)
                if voxel is not None:
                    self._grid[voxel] = 1
                    n_occupied += 1

            self._last_depth_map = depth
            self._last_point_cloud = xyz
            self._last_update_time = time.monotonic()
            self._update_count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000

        return {
            "status": "ok",
            "points_total": len(xyz),
            "points_sampled": len(xyz[::step]),
            "voxels_occupied": n_occupied,
            "elapsed_ms": round(elapsed_ms, 1),
            "update_count": self._update_count,
        }

    def calibrate_scale_from_checkerboard(
        self, left: np.ndarray, right: np.ndarray, known_square_mm: float
    ) -> dict:
        """Use a checkerboard pattern to calibrate real-world scale.

        Measures the apparent square size in the depth map and computes
        a correction factor against the known physical size.
        """
        corners = self.calibrator.find_corners(left)
        if corners is None:
            return {"ok": False, "error": "Checkerboard not found in left image"}

        # Measure apparent pixel distance between adjacent corners
        cols = self.calibrator.board_size[0]
        pixel_dists = []
        for i in range(len(corners) - 1):
            if (i + 1) % cols != 0:  # don't measure across rows
                d = np.linalg.norm(corners[i + 1][0] - corners[i][0])
                pixel_dists.append(d)

        if not pixel_dists:
            return {"ok": False, "error": "Not enough corners to measure"}

        avg_pixel_dist = np.mean(pixel_dists)

        # Use depth to estimate real distance
        self._ensure_depth_estimator()
        if self._depth_est is None:
            return {"ok": False, "error": "Depth estimator not ready"}

        disparity, depth = self._depth_est.compute_depth(left, right)

        # Get depth at checkerboard center
        cx, cy = corners[len(corners) // 2][0].astype(int)
        center_depth = self._depth_est.get_depth_at(depth, cx, cy)
        if center_depth <= 0:
            return {"ok": False, "error": "Could not measure depth at checkerboard"}

        # Apparent size: use pinhole model
        # real_size = pixel_size * depth / focal_length
        if self.calibrator.camera_matrix_left is not None:
            fx = self.calibrator.camera_matrix_left[0, 0]
            apparent_mm = avg_pixel_dist * center_depth / fx
        else:
            apparent_mm = avg_pixel_dist  # fallback — just pixels

        if apparent_mm > 0:
            self._scale_factor = known_square_mm / apparent_mm
            self._scale_calibrated = True

        return {
            "ok": True,
            "scale_factor": round(self._scale_factor, 4),
            "apparent_square_mm": round(apparent_mm, 2),
            "known_square_mm": known_square_mm,
            "depth_mm": round(center_depth, 1),
            "avg_pixel_dist": round(avg_pixel_dist, 1),
        }

    def calibrate_scale_from_tape_measure(
        self,
        left: np.ndarray,
        right: np.ndarray,
        known_length_mm: float,
        point1: tuple[int, int],
        point2: tuple[int, int],
    ) -> dict:
        """Calibrate scale using two user-marked points on a tape measure.

        point1, point2: pixel coordinates of two known-distance points
        known_length_mm: real-world distance between the two points
        """
        self._ensure_depth_estimator()
        if self._depth_est is None:
            return {"ok": False, "error": "Depth estimator not ready"}

        disparity, depth = self._depth_est.compute_depth(left, right)

        d1 = self._depth_est.get_depth_at(depth, point1[0], point1[1])
        d2 = self._depth_est.get_depth_at(depth, point2[0], point2[1])

        if d1 <= 0 or d2 <= 0:
            return {"ok": False, "error": "Could not measure depth at marked points"}

        # Back-project both points to 3D
        if self.calibrator.camera_matrix_left is not None:
            fx = self.calibrator.camera_matrix_left[0, 0]
            fy = self.calibrator.camera_matrix_left[1, 1]
            cx = self.calibrator.camera_matrix_left[0, 2]
            cy = self.calibrator.camera_matrix_left[1, 2]
        else:
            return {"ok": False, "error": "Camera matrix not available"}

        p3d_1 = np.array(
            [
                (point1[0] - cx) * d1 / fx,
                (point1[1] - cy) * d1 / fy,
                d1,
            ]
        )
        p3d_2 = np.array(
            [
                (point2[0] - cx) * d2 / fx,
                (point2[1] - cy) * d2 / fy,
                d2,
            ]
        )

        measured_mm = np.linalg.norm(p3d_1 - p3d_2)
        if measured_mm > 0:
            self._scale_factor = known_length_mm / measured_mm
            self._scale_calibrated = True

        return {
            "ok": True,
            "scale_factor": round(self._scale_factor, 4),
            "measured_mm": round(measured_mm, 2),
            "known_mm": known_length_mm,
            "point1_depth": round(d1, 1),
            "point2_depth": round(d2, 1),
        }

    def check_point(self, point_mm: np.ndarray) -> str:
        """Check if a 3D point is free, occupied, or unknown."""
        with self._lock:
            voxel = self._world_to_voxel(point_mm)
            if voxel is None:
                return "out_of_bounds"
            val = self._grid[voxel]
            if val == 1:
                return "occupied"
            elif val == 0:
                return "free"
            else:
                return "unknown"

    def check_path(
        self,
        points_mm: list[np.ndarray],
        radius_mm: float = 30.0,
    ) -> list[dict]:
        """Check a sequence of 3D points for collisions with occupied voxels.

        Returns a list of collision info dicts for points that hit obstacles.
        """
        collisions = []
        radius_voxels = max(1, int(radius_mm / self.voxel_size))

        with self._lock:
            for i, pt in enumerate(points_mm):
                voxel = self._world_to_voxel(pt)
                if voxel is None:
                    continue

                # Check neighborhood
                for dx in range(-radius_voxels, radius_voxels + 1):
                    for dy in range(-radius_voxels, radius_voxels + 1):
                        for dz in range(-radius_voxels, radius_voxels + 1):
                            nx = voxel[0] + dx
                            ny = voxel[1] + dy
                            nz = voxel[2] + dz
                            if (
                                0 <= nx < self.grid_shape[0]
                                and 0 <= ny < self.grid_shape[1]
                                and 0 <= nz < self.grid_shape[2]
                            ):
                                if self._grid[nx, ny, nz] == 1:
                                    collision_pt = self._voxel_to_world(nx, ny, nz)
                                    dist = np.linalg.norm(pt - collision_pt)
                                    if dist <= radius_mm:
                                        collisions.append(
                                            {
                                                "path_index": i,
                                                "point_mm": pt.tolist(),
                                                "obstacle_mm": collision_pt.tolist(),
                                                "distance_mm": round(dist, 1),
                                            }
                                        )
                                        break
                            if collisions and collisions[-1]["path_index"] == i:
                                break
                        if collisions and collisions[-1]["path_index"] == i:
                            break

        return collisions

    def get_occupancy_summary(self) -> dict:
        """Get a summary of the current occupancy grid state."""
        with self._lock:
            total = self._grid.size
            occupied = int(np.sum(self._grid == 1))
            free = int(np.sum(self._grid == 0))
            unknown = int(np.sum(self._grid == -1))

            return {
                "grid_shape": list(self.grid_shape),
                "voxel_size_mm": self.voxel_size,
                "total_voxels": total,
                "occupied": occupied,
                "free": free,
                "unknown": unknown,
                "occupied_pct": round(100 * occupied / max(total, 1), 1),
                "mapped_pct": round(100 * (occupied + free) / max(total, 1), 1),
                "scale_factor": round(self._scale_factor, 4),
                "scale_calibrated": self._scale_calibrated,
                "update_count": self._update_count,
                "enabled": self._enabled,
                "last_update_age_ms": (
                    round((time.monotonic() - self._last_update_time) * 1000, 0)
                    if self._last_update_time > 0
                    else None
                ),
            }

    def get_occupied_points(self, max_points: int = 500) -> list[list[float]]:
        """Get occupied voxel centers as 3D points for visualization.

        Returns list of [x, y, z] in mm, subsampled if needed.
        """
        with self._lock:
            indices = np.argwhere(self._grid == 1)
            if len(indices) == 0:
                return []

            # Subsample if too many
            if len(indices) > max_points:
                step = max(1, len(indices) // max_points)
                indices = indices[::step]

            points = []
            for idx in indices:
                center = self._voxel_to_world(idx[0], idx[1], idx[2])
                points.append([round(float(c), 1) for c in center])

            return points

    def clear(self):
        """Clear the occupancy grid."""
        with self._lock:
            self._grid.fill(-1)
            self._update_count = 0
            self._last_depth_map = None
            self._last_point_cloud = None
        logger.info("Workspace map cleared")

    def get_status(self) -> dict:
        """Full status for the API."""
        summary = self.get_occupancy_summary()
        summary["calibrator_ready"] = self.calibrator.is_calibrated
        return summary
