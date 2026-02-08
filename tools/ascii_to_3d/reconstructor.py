"""Reconstruct a 3D voxel volume from two orthographic ASCII silhouettes.

Uses the *visual hull* (shape-from-silhouette) method:

    Given a **front view** (projection onto the XY plane, viewed along +Z)
    and a **side view** (projection onto the ZY plane, viewed along −X),
    a voxel at (x, y, z) is considered *filled* if and only if:

        front_silhouette[y, x]  AND  side_silhouette[y, z]

    where ``y`` is the vertical axis (shared between both views).

The two silhouettes must share the same vertical extent (height).  If they
differ in height the shorter one is centred and padded.  The horizontal
extents define the X (front width) and Z (side width) dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .parser import AsciiImage


@dataclass
class VoxelGrid:
    """Axis-aligned 3D binary occupancy grid.

    Axes
    ----
    - **X** → columns of the front view (left-to-right)
    - **Y** → rows (top-to-bottom, but flipped so Y increases upward)
    - **Z** → columns of the side view (left-to-right, i.e. depth)

    Attributes
    ----------
    voxels : np.ndarray
        Boolean 3D array of shape ``(size_x, size_y, size_z)``.
    size_x, size_y, size_z : int
        Dimensions of the grid.
    """

    voxels: np.ndarray  # shape (X, Y, Z), dtype bool
    size_x: int
    size_y: int
    size_z: int

    @property
    def filled_count(self) -> int:
        return int(self.voxels.sum())

    def filled_coords(self):
        """Yield (x, y, z) for every filled voxel."""
        xs, ys, zs = np.nonzero(self.voxels)
        return list(zip(xs.tolist(), ys.tolist(), zs.tolist()))

    def get_front_projection(self) -> np.ndarray:
        """Project onto XY plane (collapse Z axis) — should match front input."""
        return np.any(self.voxels, axis=2)  # (X, Y) → transpose for display

    def get_side_projection(self) -> np.ndarray:
        """Project onto ZY plane (collapse X axis) — should match side input."""
        return np.any(self.voxels, axis=0)  # (Y, Z)


def _align_heights(
    front: np.ndarray, side: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the shorter silhouette so both have equal height (axis-0)."""
    fh, sh = front.shape[0], side.shape[0]
    if fh == sh:
        return front, side

    target = max(fh, sh)

    def _pad(arr: np.ndarray, h: int) -> np.ndarray:
        if arr.shape[0] >= h:
            return arr
        pad_top = (h - arr.shape[0]) // 2
        pad_bot = h - arr.shape[0] - pad_top
        return np.pad(arr, ((pad_top, pad_bot), (0, 0)), constant_values=False)

    return _pad(front, target), _pad(side, target)


def reconstruct(front: AsciiImage, side: AsciiImage) -> VoxelGrid:
    """Build a 3D voxel grid from front and side ASCII silhouettes.

    Parameters
    ----------
    front : AsciiImage
        Front-view silhouette (XY plane).  Columns → X, rows → Y.
    side : AsciiImage
        Side-view silhouette (ZY plane).  Columns → Z, rows → Y.

    Returns
    -------
    VoxelGrid
        The reconstructed 3D volume.
    """
    # Grids: rows = Y (top-to-bottom), cols = X or Z
    # Flip vertically so row-0 is the *bottom* (Y=0 at bottom).
    front_grid = front.grid[::-1].copy()  # (H, W_front)
    side_grid = side.grid[::-1].copy()   # (H, W_side)

    front_grid, side_grid = _align_heights(front_grid, side_grid)

    height = front_grid.shape[0]  # Y
    size_x = front_grid.shape[1]  # X  (front width)
    size_z = side_grid.shape[1]   # Z  (side width)

    # Visual hull intersection:
    #   voxel[x, y, z] = front[y, x] AND side[y, z]
    # Broadcast: front[y, x] → (X, Y, 1),  side[y, z] → (1, Y, Z)
    front_3d = front_grid.T[:, :, np.newaxis]   # (X, Y, 1)
    side_3d = side_grid[np.newaxis, :, :]       # (1, Y, Z)

    voxels = front_3d & side_3d  # (X, Y, Z)

    return VoxelGrid(
        voxels=voxels,
        size_x=size_x,
        size_y=height,
        size_z=size_z,
    )
