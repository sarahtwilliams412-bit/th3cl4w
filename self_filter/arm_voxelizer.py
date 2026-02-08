"""
Arm Voxelizer

Converts the arm's link segments into a boolean occupancy mask in the
128^3 voxel grid. Each link is modeled as a capsule (cylinder with
hemispherical endcaps) and voxelized by computing the distance from
each voxel center to the nearest point on the link's line segment.

Performance optimization: only computes within the bounding box of each
link + radius, avoiding the full 128^3 grid per link.
"""

from __future__ import annotations

import numpy as np


class ArmVoxelizer:
    """Convert arm link segments to voxel occupancy mask.

    Parameters
    ----------
    link_radii_mm : list[float]
        Radius of each link capsule in mm.
    safety_margin_mm : float
        Additional inflation around each link.
    grid_resolution : int
        Grid size (default 128).
    cell_size_mm : float
        Physical size of each voxel in mm.
    grid_origin_mm : list[float]
        [x, y, z] position of voxel [0, 0, 0] in workspace mm.
    """

    def __init__(
        self,
        link_radii_mm: list[float],
        safety_margin_mm: float,
        grid_resolution: int = 128,
        cell_size_mm: float = 7.8,
        grid_origin_mm: list[float] | None = None,
    ):
        self.link_radii = np.array(link_radii_mm, dtype=np.float32) + safety_margin_mm
        self.N = grid_resolution
        self.cell_size = cell_size_mm
        self.origin = np.array(grid_origin_mm or [0.0, 0.0, 0.0], dtype=np.float32)

        # Pre-compute coordinate grid (compute once, reuse for all links)
        # coords[i,j,k] = [x,y,z] position of voxel center in grid coords
        coords = np.mgrid[0 : self.N, 0 : self.N, 0 : self.N]
        self._grid_coords = coords.astype(np.float32).transpose(1, 2, 3, 0)
        # shape: [N, N, N, 3]

    def _mm_to_grid(self, point_mm: np.ndarray) -> np.ndarray:
        """Convert mm coordinates to grid coordinates."""
        return (np.asarray(point_mm, dtype=np.float32) - self.origin) / self.cell_size

    def voxelize_capsule(
        self,
        p0_mm: np.ndarray,
        p1_mm: np.ndarray,
        radius_mm: float,
    ) -> np.ndarray:
        """Compute boolean mask of voxels within radius of a line segment.

        Optimized: only computes distance within the bounding box of the
        capsule, skipping most of the 128^3 grid.

        Parameters
        ----------
        p0_mm : np.ndarray
            Start point in workspace mm.
        p1_mm : np.ndarray
            End point in workspace mm.
        radius_mm : float
            Capsule radius in mm.

        Returns
        -------
        np.ndarray
            bool[N, N, N] — True where voxels are inside the capsule.
        """
        # Convert to grid coordinates
        p0 = self._mm_to_grid(p0_mm)
        p1 = self._mm_to_grid(p1_mm)
        r = radius_mm / self.cell_size

        # Compute bounding box in grid coords (with radius margin)
        bbox_min = np.minimum(p0, p1) - r - 1
        bbox_max = np.maximum(p0, p1) + r + 1

        # Clamp to grid bounds
        i_min = max(0, int(np.floor(bbox_min[0])))
        j_min = max(0, int(np.floor(bbox_min[1])))
        k_min = max(0, int(np.floor(bbox_min[2])))
        i_max = min(self.N, int(np.ceil(bbox_max[0])) + 1)
        j_max = min(self.N, int(np.ceil(bbox_max[1])) + 1)
        k_max = min(self.N, int(np.ceil(bbox_max[2])) + 1)

        # Skip if bounding box is entirely outside the grid
        if i_min >= i_max or j_min >= j_max or k_min >= k_max:
            return np.zeros((self.N, self.N, self.N), dtype=bool)

        # Extract the sub-grid of coordinates within the bounding box
        sub_coords = self._grid_coords[i_min:i_max, j_min:j_max, k_min:k_max]
        # shape: [di, dj, dk, 3]

        # Distance from points to line segment (vectorized)
        v = p1 - p0  # direction vector
        w = sub_coords - p0  # vectors from p0 to each point

        c1 = np.sum(w * v, axis=-1)  # dot(w, v)
        c2 = np.dot(v, v)            # dot(v, v) = |v|^2

        # Clamp parameter t to [0, 1] for closest point on segment
        t = np.clip(c1 / (c2 + 1e-10), 0.0, 1.0)

        # Closest point on segment for each voxel
        closest = p0 + t[..., np.newaxis] * v  # [di, dj, dk, 3]

        # Distance from each voxel to closest point
        diff = sub_coords - closest
        dist_sq = np.sum(diff * diff, axis=-1)  # [di, dj, dk]

        # Mark voxels within radius
        sub_mask = dist_sq < (r * r)

        # Place sub-mask into full-size mask
        mask = np.zeros((self.N, self.N, self.N), dtype=bool)
        mask[i_min:i_max, j_min:j_max, k_min:k_max] = sub_mask

        return mask

    def voxelize_arm(
        self,
        link_segments: list[tuple[np.ndarray, np.ndarray]],
        link_radii: np.ndarray | None = None,
    ) -> np.ndarray:
        """Voxelize all arm links into a single boolean mask.

        Parameters
        ----------
        link_segments : list[tuple[np.ndarray, np.ndarray]]
            List of (start_mm, end_mm) pairs from ForwardKinematics.
        link_radii : np.ndarray, optional
            Override radii per link. Uses self.link_radii if None.

        Returns
        -------
        np.ndarray
            bool[N, N, N] — True where the arm occupies space.
        """
        mask = np.zeros((self.N, self.N, self.N), dtype=bool)
        radii = link_radii if link_radii is not None else self.link_radii

        n_links = min(len(link_segments), len(radii))
        for i in range(n_links):
            p0, p1 = link_segments[i]
            mask |= self.voxelize_capsule(p0, p1, float(radii[i]))

        return mask
