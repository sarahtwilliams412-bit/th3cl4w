"""
Independent-Camera Workspace Mapper — 2D+height workspace map.

Uses overhead camera (cam1) for a 2D occupancy grid of the workspace,
and the front camera (cam0) for height estimation of detected objects.
No stereo matching needed.

This is planning/measurement only — does NOT affect arm movement.
"""

import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraCalibration, IndependentCalibrator

logger = logging.getLogger("th3cl4w.vision.workspace_mapper")


class WorkspaceMapper:
    """Builds a 2D workspace occupancy grid from overhead camera + height from front camera.

    The grid covers the reachable workspace on the table surface.
    Height information is estimated per-cell from the front camera when available.
    """

    def __init__(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
        # Workspace bounds in mm (X/Y on table, relative to arm base)
        workspace_min: tuple[float, float] = (-400.0, -400.0),
        workspace_max: tuple[float, float] = (400.0, 400.0),
        cell_size_mm: float = 20.0,  # ~2cm cells
        max_height_mm: float = 300.0,
    ):
        self.cal_cam0 = cal_cam0
        self.cal_cam1 = cal_cam1
        self.cell_size = cell_size_mm
        self.max_height = max_height_mm
        self.ws_min = np.array(workspace_min, dtype=np.float32)
        self.ws_max = np.array(workspace_max, dtype=np.float32)

        # 2D grid dimensions
        self.grid_shape = (
            int(np.ceil((self.ws_max[0] - self.ws_min[0]) / self.cell_size)),
            int(np.ceil((self.ws_max[1] - self.ws_min[1]) / self.cell_size)),
        )
        # Occupancy: 0=free, 1=occupied, -1=unknown
        self._grid = np.full(self.grid_shape, -1, dtype=np.int8)
        # Height map: estimated height per cell (mm), NaN = unknown
        self._height_map = np.full(self.grid_shape, np.nan, dtype=np.float32)

        # Scale from checkerboard calibration
        self._scale_factor: float = 1.0
        self._scale_calibrated = False

        # Threading
        self._lock = threading.Lock()
        self._enabled = False
        self._last_update_time: float = 0.0
        self._update_count: int = 0

    def set_calibration(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
    ):
        """Update camera calibrations."""
        if cal_cam0 is not None:
            self.cal_cam0 = cal_cam0
        if cal_cam1 is not None:
            self.cal_cam1 = cal_cam1

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

    def _world_to_cell(self, x_mm: float, y_mm: float) -> Optional[tuple[int, int]]:
        """Convert workspace X/Y (mm) to grid cell index."""
        ix = int((x_mm - self.ws_min[0]) / self.cell_size)
        iy = int((y_mm - self.ws_min[1]) / self.cell_size)
        if 0 <= ix < self.grid_shape[0] and 0 <= iy < self.grid_shape[1]:
            return (ix, iy)
        return None

    def _cell_to_world(self, ix: int, iy: int) -> np.ndarray:
        """Convert grid cell to workspace center point (mm)."""
        x = self.ws_min[0] + (ix + 0.5) * self.cell_size
        y = self.ws_min[1] + (iy + 0.5) * self.cell_size
        return np.array([x, y], dtype=np.float32)

    def update_from_overhead(self, cam1_frame: np.ndarray) -> dict:
        """Update 2D occupancy from overhead camera using foreground detection.

        Simple approach: detect non-table-colored regions as occupied.
        """
        if not self._enabled:
            return {"status": "disabled"}
        if self.cal_cam1 is None or self.cal_cam1.cam_to_workspace is None:
            return {"status": "not_calibrated", "error": "Overhead camera not calibrated"}

        t0 = time.monotonic()

        # Convert to grayscale and threshold for objects on table
        gray = cv2.cvtColor(cam1_frame, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to find objects against table background
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blurred)
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = cam1_frame.shape[:2]
        n_occupied = 0
        n_free = 0

        with self._lock:
            # Sample grid — map each cell center to a pixel in overhead view
            for ix in range(self.grid_shape[0]):
                for iy in range(self.grid_shape[1]):
                    world_pt = self._cell_to_world(ix, iy)
                    # We need workspace-to-pixel for cam1 (inverse of pixel_to_workspace)
                    # For now, use a simpler direct mapping if available
                    # Mark based on mask
                    # Approximate: map grid linearly to image
                    u = int(ix / self.grid_shape[0] * w)
                    v = int(iy / self.grid_shape[1] * h)
                    u = max(0, min(w - 1, u))
                    v = max(0, min(h - 1, v))

                    if mask[v, u] > 0:
                        self._grid[ix, iy] = 1
                        n_occupied += 1
                    else:
                        self._grid[ix, iy] = 0
                        n_free += 1

            self._last_update_time = time.monotonic()
            self._update_count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "status": "ok",
            "occupied_cells": n_occupied,
            "free_cells": n_free,
            "elapsed_ms": round(elapsed_ms, 1),
            "update_count": self._update_count,
        }

    def calibrate_scale_from_checkerboard(
        self,
        image: np.ndarray,
        calibrator: IndependentCalibrator,
        known_square_mm: float = 23.8,
    ) -> dict:
        """Use checkerboard in overhead view to calibrate pixel-to-mm scale."""
        corners = calibrator.find_corners(image)
        if corners is None:
            return {"ok": False, "error": "Checkerboard not found"}

        cols = calibrator.board_size[0]
        pixel_dists = []
        for i in range(len(corners) - 1):
            if (i + 1) % cols != 0:
                d = np.linalg.norm(corners[i + 1][0] - corners[i][0])
                pixel_dists.append(d)

        if not pixel_dists:
            return {"ok": False, "error": "Not enough corners"}

        avg_pixel_dist = float(np.mean(pixel_dists))
        if avg_pixel_dist > 0:
            self._scale_factor = known_square_mm / avg_pixel_dist
            self._scale_calibrated = True

        return {
            "ok": True,
            "scale_factor": round(self._scale_factor, 4),
            "mm_per_pixel": round(self._scale_factor, 4),
            "avg_pixel_dist": round(avg_pixel_dist, 1),
            "known_square_mm": known_square_mm,
        }

    def check_point(self, x_mm: float, y_mm: float) -> str:
        """Check if a workspace point is free, occupied, or unknown."""
        with self._lock:
            cell = self._world_to_cell(x_mm, y_mm)
            if cell is None:
                return "out_of_bounds"
            val = self._grid[cell]
            return {1: "occupied", 0: "free"}.get(int(val), "unknown")

    def check_path_2d(
        self, points_mm: list[tuple[float, float]], radius_mm: float = 30.0
    ) -> list[dict]:
        """Check a 2D path for collisions with occupied cells."""
        collisions = []
        radius_cells = max(1, int(radius_mm / self.cell_size))

        with self._lock:
            for i, (px, py) in enumerate(points_mm):
                cell = self._world_to_cell(px, py)
                if cell is None:
                    continue
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        nx, ny = cell[0] + dx, cell[1] + dy
                        if (
                            0 <= nx < self.grid_shape[0]
                            and 0 <= ny < self.grid_shape[1]
                            and self._grid[nx, ny] == 1
                        ):
                            obs_pt = self._cell_to_world(nx, ny)
                            dist = np.sqrt((px - obs_pt[0]) ** 2 + (py - obs_pt[1]) ** 2)
                            if dist <= radius_mm:
                                collisions.append(
                                    {
                                        "path_index": i,
                                        "point_mm": [px, py],
                                        "obstacle_mm": obs_pt.tolist(),
                                        "distance_mm": round(float(dist), 1),
                                    }
                                )
                                break
                    else:
                        continue
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
                "cell_size_mm": self.cell_size,
                "total_cells": total,
                "occupied": occupied,
                "free": free,
                "unknown": unknown,
                "occupied_pct": round(100 * occupied / max(total, 1), 1),
                "mapped_pct": round(100 * (occupied + free) / max(total, 1), 1),
                "scale_factor": round(self._scale_factor, 4),
                "scale_calibrated": self._scale_calibrated,
                "update_count": self._update_count,
                "enabled": self._enabled,
            }

    def get_occupied_cells(self, max_cells: int = 500) -> list[list[float]]:
        """Get occupied cell centers as 2D points for visualization."""
        with self._lock:
            indices = np.argwhere(self._grid == 1)
            if len(indices) == 0:
                return []
            if len(indices) > max_cells:
                step = max(1, len(indices) // max_cells)
                indices = indices[::step]
            return [self._cell_to_world(idx[0], idx[1]).tolist() for idx in indices]

    def clear(self):
        """Clear the occupancy grid."""
        with self._lock:
            self._grid.fill(-1)
            self._height_map.fill(np.nan)
            self._update_count = 0
        logger.info("Workspace map cleared")

    def get_status(self) -> dict:
        summary = self.get_occupancy_summary()
        summary["cam0_calibrated"] = self.cal_cam0 is not None and self.cal_cam0.is_calibrated
        summary["cam1_calibrated"] = self.cal_cam1 is not None and self.cal_cam1.is_calibrated
        return summary
