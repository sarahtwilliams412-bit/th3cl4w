"""Tests for the ascii_to_3d tool.

Covers parsing, reconstruction, mesh generation, export, and rendering.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.ascii_to_3d.parser import AsciiImage, parse_ascii, parse_file
from tools.ascii_to_3d.reconstructor import VoxelGrid, reconstruct
from tools.ascii_to_3d.mesh import TriMesh, voxels_to_mesh
from tools.ascii_to_3d.exporter import export_obj, export_stl
from tools.ascii_to_3d.renderer import render_isometric, render_slices, render_projections

# ── Parser ──────────────────────────────────────────────────────────


class TestParser:
    def test_simple_square(self):
        art = "###\n# #\n###"
        img = parse_ascii(art)
        assert img.width == 3
        assert img.height == 3
        assert img.grid[0, 0] is np.True_
        assert img.grid[1, 1] is np.False_  # space in centre

    def test_spaces_are_empty(self):
        art = "# #\n   \n# #"
        img = parse_ascii(art)
        assert img.grid[1, 0] is np.False_
        assert img.grid[1, 1] is np.False_
        assert img.grid[1, 2] is np.False_

    def test_dots_are_empty(self):
        art = "#.#\n.#.\n#.#"
        img = parse_ascii(art)
        assert img.grid[0, 1] is np.False_
        assert img.grid[1, 0] is np.False_

    def test_trim_removes_empty_borders(self):
        art = "     \n  #  \n     "
        img = parse_ascii(art, trim=True)
        assert img.width == 1
        assert img.height == 1
        assert img.grid[0, 0] is np.True_

    def test_no_trim_preserves_borders(self):
        art = "     \n  #  \n     "
        img = parse_ascii(art, trim=False)
        assert img.width == 5
        assert img.height == 3

    def test_ragged_lines_padded(self):
        art = "##\n#\n###"
        img = parse_ascii(art)
        assert img.width == 3  # padded to longest line
        assert img.height == 3

    def test_filled_coords(self):
        art = "# \n #"
        img = parse_ascii(art)
        coords = img.filled_coords()
        assert (0, 0) in coords
        assert (1, 1) in coords
        assert len(coords) == 2

    def test_bounding_box(self):
        art = "   \n # \n   "
        img = parse_ascii(art, trim=False)
        bb = img.bounding_box()
        assert bb == (1, 1, 1, 1)

    def test_empty_grid(self):
        art = "   \n   "
        img = parse_ascii(art, trim=True)
        assert img.filled_count if hasattr(img, "filled_count") else True  # just shouldn't crash

    def test_parse_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("##\n##")
        img = parse_file(f)
        assert img.width == 2
        assert img.height == 2
        assert img.grid.all()

    def test_various_fill_characters(self):
        art = "@#%X*"
        img = parse_ascii(art)
        assert img.grid.all()


# ── Reconstructor ───────────────────────────────────────────────────


class TestReconstructor:
    def test_solid_cube(self):
        """Front and side both fully filled → solid cube."""
        front = parse_ascii("##\n##")
        side = parse_ascii("##\n##")
        vg = reconstruct(front, side)
        assert vg.size_x == 2
        assert vg.size_y == 2
        assert vg.size_z == 2
        assert vg.filled_count == 8  # 2x2x2

    def test_l_shape(self):
        """Non-square front and side produce expected volume."""
        front = parse_ascii("##\n##\n##")  # 2 wide, 3 tall
        side = parse_ascii("#\n#\n#")  # 1 deep, 3 tall
        vg = reconstruct(front, side)
        assert vg.size_x == 2
        assert vg.size_y == 3
        assert vg.size_z == 1
        assert vg.filled_count == 6

    def test_cross_shape(self):
        """A cross in front + full side should carve the shape."""
        front = parse_ascii(" # \n###\n # ")
        side = parse_ascii("###\n###\n###")
        vg = reconstruct(front, side)
        # Front cross has 5 filled cells, side is all filled
        # Result: each front cell extruded through full Z depth (3)
        assert vg.filled_count == 5 * 3

    def test_height_alignment(self):
        """Different heights should be aligned by padding."""
        front = parse_ascii("##\n##")  # height 2
        side = parse_ascii("#\n#\n#\n#")  # height 4
        vg = reconstruct(front, side)
        assert vg.size_y == 4  # max of the two

    def test_projections_match_input(self):
        """The reconstructed volume should project back to the inputs."""
        front = parse_ascii("###\n# #\n###")
        side = parse_ascii("##\n##\n##")
        vg = reconstruct(front, side)

        # Front projection (collapse Z)
        front_proj = vg.get_front_projection()  # (X, Y)
        # The front projection should match the original front grid (flipped)
        expected_front = front.grid[::-1].T  # (X, Y)
        np.testing.assert_array_equal(front_proj, expected_front)

    def test_single_pixel(self):
        front = parse_ascii("#")
        side = parse_ascii("#")
        vg = reconstruct(front, side)
        assert vg.filled_count == 1
        assert vg.size_x == 1
        assert vg.size_y == 1
        assert vg.size_z == 1


# ── Mesh ────────────────────────────────────────────────────────────


class TestMesh:
    def test_single_voxel_mesh(self):
        """A single voxel should produce a cube (6 faces × 2 triangles = 12)."""
        front = parse_ascii("#")
        side = parse_ascii("#")
        vg = reconstruct(front, side)
        mesh = voxels_to_mesh(vg)
        assert mesh.face_count == 12  # 6 faces, 2 tris each
        assert mesh.vertex_count == 8  # 8 corners of a cube

    def test_two_adjacent_voxels_share_face(self):
        """Two adjacent voxels should share internal face → fewer total faces."""
        front = parse_ascii("##")
        side = parse_ascii("#")
        vg = reconstruct(front, side)
        mesh = voxels_to_mesh(vg)
        # 2 separate cubes = 24 faces, but shared internal face removes 4 tris
        # 2×12 - 2×2 = 20 triangles
        assert mesh.face_count == 20

    def test_solid_2x2x2_mesh(self):
        """2×2×2 cube: only exterior faces emitted."""
        front = parse_ascii("##\n##")
        side = parse_ascii("##\n##")
        vg = reconstruct(front, side)
        mesh = voxels_to_mesh(vg)
        # 6 faces of the cube, each 2x2 = 4 quads = 8 tris per face
        assert mesh.face_count == 48

    def test_scale_parameter(self):
        """Scale should multiply vertex coordinates."""
        front = parse_ascii("#")
        side = parse_ascii("#")
        vg = reconstruct(front, side)
        mesh = voxels_to_mesh(vg, scale=2.0)
        assert mesh.vertices.max() == 2.0

    def test_empty_grid_produces_empty_mesh(self):
        """An empty voxel grid should produce an empty mesh."""
        vg = VoxelGrid(
            voxels=np.zeros((2, 2, 2), dtype=bool),
            size_x=2,
            size_y=2,
            size_z=2,
        )
        mesh = voxels_to_mesh(vg)
        assert mesh.face_count == 0
        assert mesh.vertex_count == 0

    def test_normals_are_unit_vectors(self):
        """Each face normal should be a unit axis-aligned vector."""
        front = parse_ascii("##\n##")
        side = parse_ascii("##\n##")
        vg = reconstruct(front, side)
        mesh = voxels_to_mesh(vg)
        for n in mesh.normals:
            length = np.linalg.norm(n)
            assert abs(length - 1.0) < 1e-10


# ── Exporter ────────────────────────────────────────────────────────


class TestExporter:
    def _make_mesh(self) -> TriMesh:
        front = parse_ascii("##\n##")
        side = parse_ascii("##\n##")
        vg = reconstruct(front, side)
        return voxels_to_mesh(vg)

    def test_export_obj(self, tmp_path: Path):
        mesh = self._make_mesh()
        out = tmp_path / "test.obj"
        export_obj(mesh, out, comment="test")
        content = out.read_text()
        assert content.startswith("# Wavefront OBJ")
        assert "v " in content
        assert "vn " in content
        assert "f " in content
        # Check vertex count matches
        v_lines = [l for l in content.splitlines() if l.startswith("v ")]
        assert len(v_lines) == mesh.vertex_count

    def test_export_stl(self, tmp_path: Path):
        mesh = self._make_mesh()
        out = tmp_path / "test.stl"
        export_stl(mesh, out, name="testobj")
        content = out.read_text()
        assert content.startswith("solid testobj")
        assert "endsolid testobj" in content
        assert content.count("facet normal") == mesh.face_count

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        mesh = self._make_mesh()
        out = tmp_path / "sub" / "dir" / "test.obj"
        export_obj(mesh, out)
        assert out.exists()

    def test_obj_faces_are_1_indexed(self, tmp_path: Path):
        mesh = self._make_mesh()
        out = tmp_path / "test.obj"
        export_obj(mesh, out)
        content = out.read_text()
        for line in content.splitlines():
            if line.startswith("f "):
                parts = line.split()[1:]
                for p in parts:
                    vi = int(p.split("//")[0])
                    assert vi >= 1  # OBJ is 1-indexed


# ── Renderer ────────────────────────────────────────────────────────


class TestRenderer:
    def _make_voxel_grid(self) -> VoxelGrid:
        front = parse_ascii("###\n# #\n###")
        side = parse_ascii("##\n##\n##")
        return reconstruct(front, side)

    def test_isometric_returns_string(self):
        vg = self._make_voxel_grid()
        result = render_isometric(vg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_isometric_respects_dimensions(self):
        vg = self._make_voxel_grid()
        result = render_isometric(vg, width=40, height=20)
        lines = result.split("\n")
        assert len(lines) == 20
        assert all(len(l) == 40 for l in lines)

    def test_slices_returns_string(self):
        vg = self._make_voxel_grid()
        result = render_slices(vg)
        assert isinstance(result, str)
        assert "---" in result  # slice headers

    def test_projections_returns_three_views(self):
        vg = self._make_voxel_grid()
        result = render_projections(vg)
        assert "[Front (XY)]" in result
        assert "[Side (YZ)]" in result
        assert "[Top (XZ)]" in result

    def test_empty_grid_isometric(self):
        vg = VoxelGrid(
            voxels=np.zeros((2, 2, 2), dtype=bool),
            size_x=2,
            size_y=2,
            size_z=2,
        )
        result = render_isometric(vg)
        assert "empty" in result.lower()

    def test_empty_grid_slices(self):
        vg = VoxelGrid(
            voxels=np.zeros((2, 2, 2), dtype=bool),
            size_x=2,
            size_y=2,
            size_z=2,
        )
        result = render_slices(vg)
        assert "empty" in result.lower()


# ── CLI (integration) ──────────────────────────────────────────────


class TestCLI:
    def test_cli_with_files(self, tmp_path: Path):
        from tools.ascii_to_3d.cli import main

        front = tmp_path / "front.txt"
        side = tmp_path / "side.txt"
        front.write_text("##\n##")
        side.write_text("##\n##")

        out = tmp_path / "out.obj"
        rc = main([str(front), str(side), "-o", str(out), "--stats"])
        assert rc == 0
        assert out.exists()

    def test_cli_with_inline(self, tmp_path: Path):
        from tools.ascii_to_3d.cli import main

        out = tmp_path / "out.obj"
        rc = main(
            [
                "--front-inline",
                "##\\n##",
                "--side-inline",
                "##\\n##",
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()

    def test_cli_preview_mode(self, tmp_path: Path, capsys):
        from tools.ascii_to_3d.cli import main

        front = tmp_path / "front.txt"
        side = tmp_path / "side.txt"
        front.write_text("###\n###\n###")
        side.write_text("###\n###\n###")

        rc = main([str(front), str(side), "--preview"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Isometric Preview" in captured.out

    def test_cli_missing_input(self):
        from tools.ascii_to_3d.cli import main

        with pytest.raises(SystemExit):
            main(["--front-inline", "##"])  # missing side

    def test_cli_stl_output(self, tmp_path: Path):
        from tools.ascii_to_3d.cli import main

        front = tmp_path / "front.txt"
        side = tmp_path / "side.txt"
        front.write_text("##\n##")
        side.write_text("##\n##")

        obj_out = tmp_path / "out.obj"
        stl_out = tmp_path / "out.stl"
        rc = main([str(front), str(side), "-o", str(obj_out), "--stl", str(stl_out)])
        assert rc == 0
        assert obj_out.exists()
        assert stl_out.exists()


# ── End-to-end with example files ──────────────────────────────────


EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "tools" / "ascii_to_3d" / "examples"


class TestExamples:
    @pytest.mark.parametrize("name", ["cup", "bottle", "house"])
    def test_example_roundtrip(self, name: str, tmp_path: Path):
        """Each example pair should parse, reconstruct, mesh, and export."""
        front_path = EXAMPLE_DIR / f"{name}_front.txt"
        side_path = EXAMPLE_DIR / f"{name}_side.txt"
        if not front_path.exists():
            pytest.skip(f"Example {name} not found")

        from tools.ascii_to_3d.parser import parse_file
        from tools.ascii_to_3d.reconstructor import reconstruct
        from tools.ascii_to_3d.mesh import voxels_to_mesh
        from tools.ascii_to_3d.exporter import export_obj

        front = parse_file(front_path)
        side = parse_file(side_path)
        vg = reconstruct(front, side)
        assert vg.filled_count > 0

        mesh = voxels_to_mesh(vg)
        assert mesh.face_count > 0
        assert mesh.vertex_count > 0

        out = tmp_path / f"{name}.obj"
        export_obj(mesh, out)
        assert out.stat().st_size > 0
