"""Convert a VoxelGrid into a triangle mesh.

Each filled voxel that has at least one exposed face (adjacent to an empty
voxel or at the grid boundary) contributes quads for its visible faces.
Each quad is split into two triangles for a clean, watertight mesh.

The resulting mesh can be exported directly to Wavefront OBJ format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .reconstructor import VoxelGrid

# Six face directions: (dx, dy, dz) and the four vertex offsets for the quad.
# Vertices are in counter-clockwise winding order when viewed from outside.
_FACE_DEFS: List[Tuple[Tuple[int, int, int], List[Tuple[float, float, float]]]] = [
    # +X face
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    # -X face
    ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
    # +Y face
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    # -Y face
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    # +Z face
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    # -Z face
    ((0, 0, -1), [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)]),
]


@dataclass
class TriMesh:
    """Simple triangle mesh.

    Attributes
    ----------
    vertices : np.ndarray
        Float array of shape (N, 3) — vertex positions.
    faces : np.ndarray
        Int array of shape (M, 3) — triangle indices (0-based).
    normals : np.ndarray
        Float array of shape (M, 3) — per-face normals.
    """

    vertices: np.ndarray  # (N, 3) float
    faces: np.ndarray     # (M, 3) int
    normals: np.ndarray   # (M, 3) float

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def face_count(self) -> int:
        return self.faces.shape[0]


def voxels_to_mesh(voxel_grid: VoxelGrid, scale: float = 1.0) -> TriMesh:
    """Generate a triangle mesh from filled voxels.

    Only boundary faces (faces adjacent to empty space) are emitted, so the
    result is a hollow shell rather than a solid block of internal geometry.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        The 3D occupancy grid to convert.
    scale : float
        Size of each voxel cube in world units (default 1.0).

    Returns
    -------
    TriMesh
        The generated mesh ready for export.
    """
    voxels = voxel_grid.voxels
    sx, sy, sz = voxel_grid.size_x, voxel_grid.size_y, voxel_grid.size_z

    vert_list: List[Tuple[float, float, float]] = []
    face_list: List[Tuple[int, int, int]] = []
    norm_list: List[Tuple[float, float, float]] = []

    # Build a vertex-index map to share vertices between adjacent quads
    vert_map: dict[Tuple[float, float, float], int] = {}

    def _vert_idx(v: Tuple[float, float, float]) -> int:
        if v not in vert_map:
            vert_map[v] = len(vert_list)
            vert_list.append(v)
        return vert_map[v]

    filled_coords = np.argwhere(voxels)  # (K, 3) array of (x, y, z)

    for x, y, z in filled_coords:
        for (dx, dy, dz), quad_offsets in _FACE_DEFS:
            nx, ny, nz = x + dx, y + dy, z + dz
            # Face is exposed if neighbour is out-of-bounds or empty
            if (
                nx < 0 or ny < 0 or nz < 0
                or nx >= sx or ny >= sy or nz >= sz
                or not voxels[nx, ny, nz]
            ):
                # Emit the quad as two triangles
                verts = [
                    (
                        (x + vo[0]) * scale,
                        (y + vo[1]) * scale,
                        (z + vo[2]) * scale,
                    )
                    for vo in quad_offsets
                ]
                i0 = _vert_idx(verts[0])
                i1 = _vert_idx(verts[1])
                i2 = _vert_idx(verts[2])
                i3 = _vert_idx(verts[3])

                face_list.append((i0, i1, i2))
                face_list.append((i0, i2, i3))

                normal = (float(dx), float(dy), float(dz))
                norm_list.append(normal)
                norm_list.append(normal)

    vertices = np.array(vert_list, dtype=np.float64) if vert_list else np.zeros((0, 3))
    faces = np.array(face_list, dtype=np.int64) if face_list else np.zeros((0, 3), dtype=np.int64)
    normals = np.array(norm_list, dtype=np.float64) if norm_list else np.zeros((0, 3))

    return TriMesh(vertices=vertices, faces=faces, normals=normals)
