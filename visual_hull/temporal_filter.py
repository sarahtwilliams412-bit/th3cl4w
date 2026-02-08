"""
Temporal Filtering for Occupancy Grids

Smooths the occupancy grid over time to reduce noise and flickering.
Implements exponential moving average with space-carving decay to
prevent transient occlusions from causing sudden changes.
"""

from __future__ import annotations

import numpy as np


class TemporalFilter:
    """Temporal smoothing filter for 3D occupancy grids.

    Uses exponential moving average (EMA) with space-carving decay:
    - Normal update: out = alpha * new + (1 - alpha) * prev
    - Rapid disappearance: if a voxel drops from >0.3 to <0.1,
      decay by 0.2/frame instead of jumping, preventing flicker
      from transient occlusions.

    Parameters
    ----------
    alpha : float
        Blending weight for new frames (higher = more responsive).
    shape : tuple
        Grid shape, default (128, 128, 128).
    decay_rate : float
        Per-frame decay for rapid disappearances.
    high_threshold : float
        Previous occupancy threshold triggering gradual decay.
    low_threshold : float
        New occupancy threshold below which gradual decay kicks in.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        shape: tuple[int, ...] = (128, 128, 128),
        decay_rate: float = 0.2,
        high_threshold: float = 0.3,
        low_threshold: float = 0.1,
    ):
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.prev_grid = np.zeros(shape, dtype=np.float32)
        self.frame_count = 0

    def update(self, new_grid: np.ndarray) -> np.ndarray:
        """Apply temporal filtering to a new occupancy grid.

        Parameters
        ----------
        new_grid : np.ndarray
            float32 occupancy grid (same shape as init).

        Returns
        -------
        np.ndarray
            Temporally filtered occupancy grid.
        """
        if self.frame_count == 0:
            self.prev_grid = new_grid.copy()
            self.frame_count += 1
            return self.prev_grid

        # Standard EMA blending
        blended = self.alpha * new_grid + (1.0 - self.alpha) * self.prev_grid

        # Space-carving decay: prevent flickering from transient occlusions
        # If a voxel was solid (>0.3) but now appears empty (<0.1),
        # decay gradually instead of jumping to zero
        rapid_disappear = (self.prev_grid > self.high_threshold) & (
            new_grid < self.low_threshold
        )
        if rapid_disappear.any():
            decayed = self.prev_grid[rapid_disappear] - self.decay_rate
            blended[rapid_disappear] = np.maximum(decayed, 0.0)

        self.prev_grid = blended.copy()
        self.frame_count += 1
        return self.prev_grid

    def reset(self) -> None:
        """Reset the filter state."""
        self.prev_grid[:] = 0.0
        self.frame_count = 0
