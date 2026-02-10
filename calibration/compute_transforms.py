"""
Compute Grid-to-World Affine Transforms

Loads captured checkerboard frames, detects corners in grayscale-converted
ASCII images, and computes independent affine transforms for each camera
mapping grid coordinates to workspace millimeters.

Outputs calibration.json and density_lut.npy.

Usage:
    python -m calibration.compute_transforms
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from calibration.ascii_to_grayscale import (
    ascii_grid_to_grayscale,
    save_density_lut,
)

logger = logging.getLogger(__name__)

CAPTURE_DIR = Path(__file__).parent / "captures"
OUTPUT_DIR = Path(__file__).parent


def load_config() -> dict:
    """Load calibration config."""
    config_path = Path(__file__).parent / "config.yaml"
    if yaml is None:
        raise RuntimeError("pyyaml required")
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_corners(
    grayscale: np.ndarray, inner_corners: tuple[int, int]
) -> np.ndarray | None:
    """Detect checkerboard corners in a grayscale image.

    Parameters
    ----------
    grayscale : np.ndarray
        float32 [H, W] image in [0, 255].
    inner_corners : tuple
        (cols, rows) of inner corners.

    Returns
    -------
    np.ndarray or None
        Corner positions as float32 [N, 1, 2] or None if not found.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python required")

    # Convert to uint8 for OpenCV
    img_u8 = np.clip(grayscale, 0, 255).astype(np.uint8)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_eq = clahe.apply(img_u8)

    found, corners = cv2.findChessboardCorners(
        img_eq,
        inner_corners,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if found and corners is not None:
        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_eq, corners, (5, 5), (-1, -1), criteria)
        return corners

    return None


def compute_affine_transform(
    grid_points: np.ndarray,
    world_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute affine transform from grid coords to world coords.

    Uses least-squares to fit:
        [world_x, world_y] = A @ [grid_col, grid_row, 1]

    Parameters
    ----------
    grid_points : np.ndarray
        [N, 2] grid coordinates (col, row).
    world_points : np.ndarray
        [N, 2] world coordinates in mm.

    Returns
    -------
    tuple
        (affine_matrix [2, 3], residual_error_mm)
    """
    N = grid_points.shape[0]
    # Build system: [col, row, 1] → [wx, wy]
    A_mat = np.column_stack([grid_points, np.ones(N)])  # [N, 3]

    # Solve for each world dimension independently
    affine = np.zeros((2, 3), dtype=np.float64)
    total_residual = 0.0

    for dim in range(2):
        result, residuals, _, _ = np.linalg.lstsq(A_mat, world_points[:, dim], rcond=None)
        affine[dim] = result
        if len(residuals) > 0:
            total_residual += residuals[0]

    # Compute RMS residual error
    predicted = (affine @ A_mat.T).T  # [N, 2]
    errors = np.linalg.norm(predicted - world_points, axis=1)
    rms_error = float(np.sqrt(np.mean(errors**2)))

    return affine, rms_error


def compute_cell_size(affine: np.ndarray) -> float:
    """Estimate cell size in mm from the affine transform.

    Parameters
    ----------
    affine : np.ndarray
        [2, 3] affine matrix.

    Returns
    -------
    float
        Average cell size in mm.
    """
    # Scale factors from the 2x2 linear part
    sx = np.linalg.norm(affine[:, 0])
    sy = np.linalg.norm(affine[:, 1])
    return float((sx + sy) / 2.0)


def find_robot_base_cell(
    affine: np.ndarray,
    robot_base_mm: np.ndarray,
) -> list[int]:
    """Find the grid cell corresponding to the robot base position.

    Parameters
    ----------
    affine : np.ndarray
        [2, 3] affine matrix (grid→world).
    robot_base_mm : np.ndarray
        [2] robot base position in world mm.

    Returns
    -------
    list[int]
        [col, row] grid coordinates of robot base.
    """
    # Invert: world = A @ [col, row, 1]
    # → [col, row] = A_linear^-1 @ (world - translation)
    A_linear = affine[:, :2]
    translation = affine[:, 2]
    grid_pos = np.linalg.solve(A_linear, robot_base_mm - translation)
    return [int(round(grid_pos[0])), int(round(grid_pos[1]))]


def main() -> None:
    """Run the calibration transform computation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if cv2 is None:
        logger.info("ERROR: opencv-python required. Install with: pip install opencv-python")
        return

    config = load_config()
    inner_corners = tuple(config["checkerboard"]["inner_corners"])
    square_size_mm = config["checkerboard"]["square_size_mm"]
    robot_base_mm = np.array(config["workspace"]["robot_base_position_mm"][:2], dtype=np.float64)
    workspace_bounds = config["workspace"]["workspace_bounds_mm"]

    # Load captures
    captures = sorted(CAPTURE_DIR.glob("frame_*.npz"))
    if len(captures) < 5:
        logger.info(f"ERROR: Need at least 5 captures, found {len(captures)}")
        logger.info(f"Run 'python -m calibration.capture_frames' first.")
        return

    logger.info("Processing %d captures...", len(captures))

    # Generate checkerboard world coordinates
    n_cols, n_rows = inner_corners
    objp = np.zeros((n_cols * n_rows, 2), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            objp[r * n_cols + c] = [c * square_size_mm, r * square_size_mm]

    # Process each capture
    top_grid_points = []
    top_world_points = []
    prof_grid_points = []
    prof_world_points = []

    for cap_path in captures:
        data = np.load(str(cap_path))
        top_grid = data["top_down"]
        prof_grid = data["profile"]

        # Convert to grayscale
        top_gray = ascii_grid_to_grayscale(top_grid)
        prof_gray = ascii_grid_to_grayscale(prof_grid)

        # Detect corners
        top_corners = detect_corners(top_gray, inner_corners)
        prof_corners = detect_corners(prof_gray, inner_corners)

        if top_corners is not None:
            corners_2d = top_corners.reshape(-1, 2)
            top_grid_points.append(corners_2d)
            top_world_points.append(objp.copy())
            logger.info("  %s: top-down corners found (%d)", cap_path.name, len(corners_2d))
        else:
            logger.info("  %s: top-down corners NOT found", cap_path.name)

        if prof_corners is not None:
            corners_2d = prof_corners.reshape(-1, 2)
            prof_grid_points.append(corners_2d)
            prof_world_points.append(objp.copy())
            logger.info("  %s: profile corners found (%d)", cap_path.name, len(corners_2d))
        else:
            logger.info("  %s: profile corners NOT found", cap_path.name)

    # Compute transforms
    if len(top_grid_points) < 3:
        logger.info(f"ERROR: Need at least 3 top-down captures with corners, got {len(top_grid_points)}")
        return
    if len(prof_grid_points) < 3:
        logger.info(f"ERROR: Need at least 3 profile captures with corners, got {len(prof_grid_points)}")
        return

    all_top_grid = np.vstack(top_grid_points)
    all_top_world = np.vstack(top_world_points)
    all_prof_grid = np.vstack(prof_grid_points)
    all_prof_world = np.vstack(prof_world_points)

    top_affine, top_error = compute_affine_transform(all_top_grid, all_top_world)
    prof_affine, prof_error = compute_affine_transform(all_prof_grid, all_prof_world)

    cell_size = (compute_cell_size(top_affine) + compute_cell_size(prof_affine)) / 2.0

    # Determine robot base cell in each view
    top_base_cell = find_robot_base_cell(top_affine, robot_base_mm)
    prof_base_cell = find_robot_base_cell(
        prof_affine,
        np.array([robot_base_mm[0], 0.0]),  # Profile maps to (x, z), z=0 at base
    )

    logger.info("Top-down affine residual: %.2f mm", top_error)
    logger.info("Profile affine residual: %.2f mm", prof_error)
    logger.info("Average cell size: %.2f mm", cell_size)

    # Build calibration output
    calibration = {
        "density_lut_file": "calibration/density_lut.npy",
        "shared_axis": "x",
        "top_down": {
            "maps_to": ["x", "y"],
            "affine_grid_to_mm": top_affine.tolist(),
            "residual_error_mm": round(top_error, 2),
            "robot_base_cell": top_base_cell,
        },
        "profile": {
            "maps_to": ["x", "z"],
            "affine_grid_to_mm": prof_affine.tolist(),
            "residual_error_mm": round(prof_error, 2),
            "robot_base_cell": prof_base_cell,
        },
        "workspace_bounds_mm": workspace_bounds,
        "cell_size_mm": round(cell_size, 2),
        "grid_resolution": config.get("grid_resolution", 128),
    }

    # Save outputs
    cal_path = OUTPUT_DIR / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info("Calibration written to %s", cal_path)

    lut_path = OUTPUT_DIR / "density_lut.npy"
    save_density_lut(lut_path)
    logger.info("Density LUT written to %s", lut_path)

    # Summary
    logger.info("\n=== Calibration Results ===")
    logger.info(f"  Top-down: maps to {calibration['top_down']['maps_to']}")
    logger.info(f"    Affine residual: {top_error:.2f} mm")
    logger.info(f"    Robot base cell: {top_base_cell}")
    logger.info(f"  Profile: maps to {calibration['profile']['maps_to']}")
    logger.info(f"    Affine residual: {prof_error:.2f} mm")
    logger.info(f"    Robot base cell: {prof_base_cell}")
    logger.info(f"  Cell size: {cell_size:.2f} mm")
    logger.info(f"  Shared axis: {calibration['shared_axis']}")
    logger.info(f"\nOutputs:")
    logger.info(f"  {cal_path}")
    logger.info(f"  {lut_path}")


if __name__ == "__main__":
    main()
