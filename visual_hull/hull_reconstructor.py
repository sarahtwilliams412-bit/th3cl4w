"""
Visual Hull Reconstructor

Implements the core visual hull algorithm: intersect silhouettes from
two orthogonal views to produce a 128^3 3D occupancy grid.

With a top-down view providing density(x, y) and a profile view
providing density(x, z), the 3D occupancy is:

    grid[x, y, z] = min(top_density[x, y], prof_density[x, z])

This is a single NumPy broadcast operation — no loops, no projection math.
"""

from __future__ import annotations

import numpy as np


class VisualHullReconstructor:
    """Reconstruct 3D occupancy from two orthogonal ASCII views.

    Parameters
    ----------
    density_lut : np.ndarray
        float32[128] lookup table mapping ASCII codepoints to density [0, 1].
    shared_axis : str
        The spatial axis shared between both views ('x', 'y', or 'z').
        Default is 'x' — top-down maps (x, y), profile maps (x, z).
    coarse_threshold : float
        Occupancy threshold for coarse pass refinement trigger.
    """

    def __init__(
        self,
        density_lut: np.ndarray,
        shared_axis: str = "x",
        coarse_threshold: float = 0.1,
    ):
        self.density_lut = density_lut.astype(np.float32)
        self.shared_axis = shared_axis
        self.coarse_threshold = coarse_threshold

    def ascii_to_density(self, ascii_grid: np.ndarray) -> np.ndarray:
        """Convert uint8 ASCII codes to float32 density [0, 1].

        Parameters
        ----------
        ascii_grid : np.ndarray
            uint8 array of ASCII codepoints (any shape).

        Returns
        -------
        np.ndarray
            float32 density values in [0, 1], same shape as input.
        """
        clipped = np.clip(ascii_grid, 0, 127).astype(np.intp)
        return self.density_lut[clipped]

    def reconstruct(
        self,
        top_frame: np.ndarray,
        prof_frame: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct 128^3 occupancy grid from two ASCII views.

        Parameters
        ----------
        top_frame : np.ndarray
            uint8[128, 128] ASCII codepoints from top-down camera.
        prof_frame : np.ndarray
            uint8[128, 128] ASCII codepoints from profile camera.

        Returns
        -------
        np.ndarray
            float32[128, 128, 128] occupancy grid.
            Grid axes: [shared_axis, top_other_axis, prof_other_axis]
            For shared_axis='x': grid[x, y, z]
        """
        top_d = self.ascii_to_density(top_frame)    # [128, 128]
        prof_d = self.ascii_to_density(prof_frame)   # [128, 128]

        # THE CORE OPERATION — one line of NumPy
        # top_d[:, :, np.newaxis]  → [128, 128, 1]
        # prof_d[:, np.newaxis, :] → [128, 1, 128]
        # np.minimum broadcasts   → [128, 128, 128]
        grid = np.minimum(
            top_d[:, :, np.newaxis],
            prof_d[:, np.newaxis, :],
        )

        return grid

    def reconstruct_coarse_to_fine(
        self,
        top_frame: np.ndarray,
        prof_frame: np.ndarray,
    ) -> np.ndarray:
        """Two-pass reconstruction: coarse 32^3 then fine 128^3 where needed.

        First does a coarse pass at 32^3 resolution to identify occupied
        regions, then refines only those regions at full 128^3 resolution.

        Parameters
        ----------
        top_frame : np.ndarray
            uint8[128, 128] ASCII codepoints from top-down camera.
        prof_frame : np.ndarray
            uint8[128, 128] ASCII codepoints from profile camera.

        Returns
        -------
        np.ndarray
            float32[128, 128, 128] occupancy grid.
        """
        # Full-resolution density maps (computed once, used in both passes)
        top_d = self.ascii_to_density(top_frame)    # [128, 128]
        prof_d = self.ascii_to_density(prof_frame)   # [128, 128]

        # --- Coarse pass: downsample to 32×32 ---
        top_coarse = top_d[::4, ::4]    # [32, 32]
        prof_coarse = prof_d[::4, ::4]  # [32, 32]

        coarse_grid = np.minimum(
            top_coarse[:, :, np.newaxis],
            prof_coarse[:, np.newaxis, :],
        )  # [32, 32, 32]

        # Identify coarse cells that need refinement
        occupied_mask = coarse_grid > self.coarse_threshold  # [32, 32, 32]

        # Early exit: if nothing is occupied, return zeros
        if not occupied_mask.any():
            return np.zeros((128, 128, 128), dtype=np.float32)

        # --- Fine pass: full resolution, masked ---
        # Upscale mask to 128^3 (each coarse cell → 4×4×4 fine cells)
        fine_mask = np.kron(occupied_mask, np.ones((4, 4, 4), dtype=bool))

        # Compute full grid via broadcast
        full_grid = np.minimum(
            top_d[:, :, np.newaxis],
            prof_d[:, np.newaxis, :],
        )  # [128, 128, 128]

        # Zero out unoccupied regions
        fine_grid = np.where(fine_mask, full_grid, 0.0).astype(np.float32)

        return fine_grid
