"""
Obstacle Extractor

Subtracts the arm's voxel mask from the occupancy grid and computes
the Euclidean distance transform (EDT) for obstacle proximity detection.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt



import logging

logger = logging.getLogger(__name__)


class ObstacleExtractor:
    """Extract obstacle grid by subtracting arm volume.

    Parameters
    ----------
    obstacle_threshold : float
        Occupancy above this value is considered an obstacle.
    cell_size_mm : float
        Physical size of each voxel in mm, for distance conversion.
    """

    def __init__(self, obstacle_threshold: float = 0.3, cell_size_mm: float = 7.8):
        self.threshold = obstacle_threshold
        self.cell_size_mm = cell_size_mm

    def extract(
        self,
        occupancy_grid: np.ndarray,
        arm_mask: np.ndarray,
    ) -> dict:
        """Subtract arm from occupancy grid and compute distance field.

        Parameters
        ----------
        occupancy_grid : np.ndarray
            float32[N, N, N] — full scene occupancy from visual hull.
        arm_mask : np.ndarray
            bool[N, N, N] — where the arm is.

        Returns
        -------
        dict
            'obstacle_grid': float32[N,N,N] — arm removed
            'obstacle_binary': bool[N,N,N] — thresholded
            'distance_field': float32[N,N,N] — EDT in voxel units
            'min_obstacle_distance_mm': float — closest obstacle to arm
            'obstacle_voxel_count': int
        """
        # Subtract arm volume
        obstacle_grid = occupancy_grid.copy()
        obstacle_grid[arm_mask] = 0.0

        # Threshold to get binary obstacle map
        obstacle_binary = obstacle_grid > self.threshold

        # Euclidean distance transform
        # EDT computes distance FROM background TO nearest foreground.
        # We want distance TO nearest obstacle, so:
        # - distance_transform_edt(~obstacle_binary) gives distance from
        #   each non-obstacle voxel to the nearest obstacle voxel.
        if obstacle_binary.any():
            dist_from_obstacle = distance_transform_edt(
                ~obstacle_binary
            ).astype(np.float32)
        else:
            dist_from_obstacle = np.full(
                occupancy_grid.shape, 999.0, dtype=np.float32
            )

        # Find minimum distance from any arm voxel to nearest obstacle
        arm_distances = dist_from_obstacle[arm_mask]
        if arm_distances.size > 0:
            min_dist_voxels = float(arm_distances.min())
        else:
            min_dist_voxels = 999.0

        min_dist_mm = min_dist_voxels * self.cell_size_mm

        return {
            "obstacle_grid": obstacle_grid,
            "obstacle_binary": obstacle_binary,
            "distance_field": dist_from_obstacle,
            "min_obstacle_distance_mm": float(min_dist_mm),
            "obstacle_voxel_count": int(obstacle_binary.sum()),
        }
