#!/usr/bin/env python3
"""CLI entry point for the ASCII-to-3D converter.

Usage examples
--------------

    # Convert two ASCII art files into an OBJ mesh
    python -m tools.ascii_to_3d.cli front.txt side.txt -o model.obj

    # Also produce an STL file
    python -m tools.ascii_to_3d.cli front.txt side.txt -o model.obj --stl model.stl

    # Show a terminal preview without writing files
    python -m tools.ascii_to_3d.cli front.txt side.txt --preview

    # Inline ASCII (useful for quick tests)
    python -m tools.ascii_to_3d.cli --front-inline '###
    # #
    ###' --side-inline '##
    ##
    ##' -o box.obj
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from .exporter import export_obj, export_stl
from .mesh import voxels_to_mesh
from .parser import parse_ascii, parse_file
from .reconstructor import reconstruct
from .renderer import render_isometric, render_projections, render_slices


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ascii_to_3d",
        description=(
            "Convert two ASCII art silhouettes (front + side views) "
            "into a 3D virtual object."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            The tool uses visual-hull reconstruction: each voxel is
            considered filled if it appears in BOTH the front and side
            silhouettes.  The result is exported as a Wavefront OBJ
            mesh that can be opened in Blender, MeshLab, or any 3D viewer.

            Any non-space character in the ASCII art is treated as a
            filled (solid) cell.  Spaces and dots are background.
        """),
    )

    # Input sources — either file paths or inline strings
    input_grp = p.add_argument_group("input")
    input_grp.add_argument(
        "front_file",
        nargs="?",
        help="Path to the front-view ASCII art file.",
    )
    input_grp.add_argument(
        "side_file",
        nargs="?",
        help="Path to the side-view ASCII art file.",
    )
    input_grp.add_argument(
        "--front-inline",
        dest="front_inline",
        help="Inline front-view ASCII art (use \\n for newlines).",
    )
    input_grp.add_argument(
        "--side-inline",
        dest="side_inline",
        help="Inline side-view ASCII art (use \\n for newlines).",
    )

    # Output options
    out_grp = p.add_argument_group("output")
    out_grp.add_argument(
        "-o", "--output",
        dest="obj_path",
        help="Write Wavefront OBJ file to this path.",
    )
    out_grp.add_argument(
        "--stl",
        dest="stl_path",
        help="Also write an ASCII STL file to this path.",
    )
    out_grp.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Voxel size in output mesh units (default: 1.0).",
    )

    # Preview options
    view_grp = p.add_argument_group("preview")
    view_grp.add_argument(
        "--preview",
        action="store_true",
        help="Print an isometric ASCII preview to the terminal.",
    )
    view_grp.add_argument(
        "--slices",
        action="store_true",
        help="Print horizontal Y-slices of the voxel grid.",
    )
    view_grp.add_argument(
        "--projections",
        action="store_true",
        help="Print front/side/top projections for verification.",
    )
    view_grp.add_argument(
        "--preview-width",
        type=int,
        default=72,
        help="Width of the isometric preview in characters (default: 72).",
    )
    view_grp.add_argument(
        "--preview-height",
        type=int,
        default=36,
        help="Height of the isometric preview in characters (default: 36).",
    )

    # Stats
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print voxel and mesh statistics.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Resolve inputs ──────────────────────────────────────────────
    front_text: str | None = None
    side_text: str | None = None

    if args.front_file:
        front_text = Path(args.front_file).read_text()
    elif args.front_inline:
        front_text = args.front_inline.replace("\\n", "\n")

    if args.side_file:
        side_text = Path(args.side_file).read_text()
    elif args.side_inline:
        side_text = args.side_inline.replace("\\n", "\n")

    if front_text is None or side_text is None:
        parser.error(
            "Both front and side views are required.  Provide either "
            "positional file paths or --front-inline / --side-inline."
        )
        return 1  # unreachable, but keeps mypy happy

    # ── Parse ───────────────────────────────────────────────────────
    front_img = parse_ascii(front_text)
    side_img = parse_ascii(side_text)

    print(f"Front view: {front_img.width}x{front_img.height}")
    print(f"Side view:  {side_img.width}x{side_img.height}")

    # ── Reconstruct ─────────────────────────────────────────────────
    voxel_grid = reconstruct(front_img, side_img)
    print(f"Voxel grid: {voxel_grid.size_x} x {voxel_grid.size_y} x {voxel_grid.size_z}")
    print(f"Filled voxels: {voxel_grid.filled_count}")

    # ── Mesh ────────────────────────────────────────────────────────
    mesh = voxels_to_mesh(voxel_grid, scale=args.scale)

    if args.stats:
        print(f"Mesh vertices: {mesh.vertex_count}")
        print(f"Mesh faces:    {mesh.face_count}")

    # ── Export ──────────────────────────────────────────────────────
    if args.obj_path:
        export_obj(mesh, args.obj_path, comment="Generated by ascii_to_3d")
        print(f"Wrote OBJ → {args.obj_path}")

    if args.stl_path:
        export_stl(mesh, args.stl_path)
        print(f"Wrote STL → {args.stl_path}")

    # ── Preview ─────────────────────────────────────────────────────
    if args.preview:
        print("\n=== Isometric Preview ===\n")
        print(render_isometric(voxel_grid, args.preview_width, args.preview_height))

    if args.slices:
        print("\n=== Y-Slices ===\n")
        print(render_slices(voxel_grid))

    if args.projections:
        print("\n=== Projections ===\n")
        print(render_projections(voxel_grid))

    # Warn if no output was requested
    if not any([args.obj_path, args.stl_path, args.preview, args.slices, args.projections]):
        print(
            "\nHint: use -o model.obj to export, or --preview for a terminal view."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
