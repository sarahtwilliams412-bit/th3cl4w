"""Render a VoxelGrid as a 3D ASCII art preview in the terminal.

Two rendering modes are provided:

1. **Isometric projection** — a pseudo-3D view using Unicode block/shade
   characters to convey depth.  Good for a quick visual check.

2. **Slice view** — prints horizontal Y-slices of the voxel grid so you
   can inspect the internal structure layer by layer.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .reconstructor import VoxelGrid

# Shade characters from darkest (closest) to lightest (farthest)
_SHADE_CHARS = ["@", "#", "%", "=", "+", "-", ":", "."]


def render_isometric(voxel_grid: VoxelGrid, width: int = 72, height: int = 36) -> str:
    """Render the voxel grid as an isometric ASCII projection.

    Uses a simple isometric camera:
      screen_x = (x - z) * cos(30°)
      screen_y = y + (x + z) * sin(30°)

    Voxels closer to the camera overwrite those farther away, with depth
    encoded via shading characters.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        The voxel data to render.
    width, height : int
        Character dimensions of the output canvas.

    Returns
    -------
    str
        Multiline string of the rendered image.
    """
    voxels = voxel_grid.voxels
    sx, sy, sz = voxel_grid.size_x, voxel_grid.size_y, voxel_grid.size_z

    if voxel_grid.filled_count == 0:
        return "(empty voxel grid — no filled cells)"

    # Isometric projection parameters
    cos30 = math.cos(math.radians(30))
    sin30 = math.sin(math.radians(30))

    # Compute projected bounds to determine scaling
    corners = [
        (0, 0, 0), (sx, 0, 0), (0, sy, 0), (0, 0, sz),
        (sx, sy, 0), (sx, 0, sz), (0, sy, sz), (sx, sy, sz),
    ]
    proj_xs = [(x - z) * cos30 for x, y, z in corners]
    proj_ys = [y + (x + z) * sin30 for x, y, z in corners]

    px_min, px_max = min(proj_xs), max(proj_xs)
    py_min, py_max = min(proj_ys), max(proj_ys)

    px_range = px_max - px_min or 1
    py_range = py_max - py_min or 1

    # Leave a 1-char border
    usable_w = width - 2
    usable_h = height - 2
    scale = min(usable_w / px_range, usable_h / py_range)

    # Canvas: char + depth buffer
    canvas = [[" "] * width for _ in range(height)]
    depth_buf = [[float("inf")] * width for _ in range(height)]

    # Depth direction: higher (x + z) means farther from camera
    filled = np.argwhere(voxels)
    for x, y, z in filled:
        px = (x - z) * cos30
        py = y + (x + z) * sin30

        # Normalise to canvas coordinates
        cx = int((px - px_min) * scale) + 1
        cy = height - 1 - (int((py - py_min) * scale) + 1)  # flip Y

        if 0 <= cx < width and 0 <= cy < height:
            depth = float(x + z)  # simple depth metric
            if depth <= depth_buf[cy][cx]:
                depth_buf[cy][cx] = depth
                # Map depth to shade character
                max_depth = float(sx + sz) or 1.0
                shade_idx = int((depth / max_depth) * (len(_SHADE_CHARS) - 1))
                shade_idx = max(0, min(shade_idx, len(_SHADE_CHARS) - 1))
                canvas[cy][cx] = _SHADE_CHARS[shade_idx]

    return "\n".join("".join(row) for row in canvas)


def render_slices(voxel_grid: VoxelGrid) -> str:
    """Render horizontal slices of the voxel grid (one per Y level).

    Each slice shows the XZ cross-section at that height.

    Returns
    -------
    str
        Multiline string with labelled slices.
    """
    voxels = voxel_grid.voxels
    sx, sy, sz = voxel_grid.size_x, voxel_grid.size_y, voxel_grid.size_z

    parts: List[str] = []

    for y in range(sy - 1, -1, -1):
        layer = voxels[:, y, :]  # (X, Z)
        has_content = layer.any()
        if not has_content:
            continue

        parts.append(f"--- Y={y} ---")
        # Print with Z as columns, X as rows
        for x in range(sx):
            row_chars = []
            for z in range(sz):
                row_chars.append("#" if layer[x, z] else ".")
            parts.append(" ".join(row_chars))
        parts.append("")

    if not parts:
        return "(empty voxel grid)"

    return "\n".join(parts)


def render_projections(voxel_grid: VoxelGrid) -> str:
    """Render the three canonical projections (front, side, top).

    Useful for verifying that the reconstructed volume matches the input
    silhouettes.

    Returns
    -------
    str
        Multiline string showing all three projections.
    """
    voxels = voxel_grid.voxels

    front = np.any(voxels, axis=2)  # project along Z → (X, Y)
    side = np.any(voxels, axis=0)   # project along X → (Y, Z)
    top = np.any(voxels, axis=1)    # project along Y → (X, Z)

    def _grid_to_str(arr: np.ndarray, label: str) -> str:
        lines = [f"[{label}]"]
        # arr might be (A, B); treat first axis as rows, second as cols
        # Flip vertically so Y=0 is at the bottom
        for r in range(arr.shape[0] - 1, -1, -1):
            row = "".join("#" if arr[r, c] else "." for c in range(arr.shape[1]))
            lines.append(f"  {row}")
        return "\n".join(lines)

    sections = [
        _grid_to_str(front.T, "Front (XY)"),   # Transpose: rows=Y, cols=X
        _grid_to_str(side, "Side (YZ)"),         # rows=Y, cols=Z
        _grid_to_str(top, "Top (XZ)"),           # rows=X, cols=Z
    ]

    return "\n\n".join(sections)
