"""
Calibration Validation Tool

Places a known-size object at a measured position, captures a frame pair,
and verifies the calibration transforms produce correct grid positions.

Usage:
    python -m calibration.validate

Pass criteria:
    - Position error < 10mm
    - Size error < 15%
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

from calibration.ascii_to_grayscale import ascii_grid_to_grayscale, load_density_lut

logger = logging.getLogger(__name__)

GRID_SIZE = 128
CAL_DIR = Path(__file__).parent


def load_calibration() -> dict:
    """Load calibration.json."""
    cal_path = CAL_DIR / "calibration.json"
    with open(cal_path) as f:
        return json.load(f)


def world_to_grid(affine: np.ndarray, world_mm: np.ndarray) -> np.ndarray:
    """Convert world coordinates (mm) to grid coordinates.

    Parameters
    ----------
    affine : np.ndarray
        [2, 3] affine matrix (grid → world).
    world_mm : np.ndarray
        [2] world position in mm.

    Returns
    -------
    np.ndarray
        [2] grid coordinates (col, row).
    """
    A_linear = affine[:, :2]
    translation = affine[:, 2]
    return np.linalg.solve(A_linear, world_mm - translation)


def find_object_bounds(
    grayscale: np.ndarray, threshold: float = 80.0
) -> tuple[int, int, int, int] | None:
    """Find bounding box of the brightest region in a grayscale image.

    Returns (min_col, min_row, max_col, max_row) or None.
    """
    mask = grayscale > threshold
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def validate_object(
    cal: dict,
    top_grid: np.ndarray,
    prof_grid: np.ndarray,
    known_position_mm: list[float],
    known_size_mm: list[float],
) -> dict:
    """Validate calibration against a known object.

    Parameters
    ----------
    cal : dict
        Calibration data from calibration.json.
    top_grid : np.ndarray
        uint8 [128, 128] top-down ASCII grid.
    prof_grid : np.ndarray
        uint8 [128, 128] profile ASCII grid.
    known_position_mm : list
        [x, y, z] known center position in mm.
    known_size_mm : list
        [width_x, depth_y, height_z] in mm.

    Returns
    -------
    dict
        Validation results including errors.
    """
    cell_size = cal["cell_size_mm"]

    top_affine = np.array(cal["top_down"]["affine_grid_to_mm"])
    prof_affine = np.array(cal["profile"]["affine_grid_to_mm"])

    # Convert to grayscale for object detection
    top_gray = ascii_grid_to_grayscale(top_grid)
    prof_gray = ascii_grid_to_grayscale(prof_grid)

    # Find object in top-down view (x, y)
    top_bounds = find_object_bounds(top_gray)
    # Find object in profile view (x, z)
    prof_bounds = find_object_bounds(prof_gray)

    result = {"passed": False, "details": {}}

    if top_bounds is None:
        result["details"]["error"] = "Object not detected in top-down view"
        return result
    if prof_bounds is None:
        result["details"]["error"] = "Object not detected in profile view"
        return result

    # Compute detected center in grid coords
    top_center_grid = np.array([
        (top_bounds[0] + top_bounds[2]) / 2.0,
        (top_bounds[1] + top_bounds[3]) / 2.0,
    ])
    prof_center_grid = np.array([
        (prof_bounds[0] + prof_bounds[2]) / 2.0,
        (prof_bounds[1] + prof_bounds[3]) / 2.0,
    ])

    # Convert to world coordinates
    top_center_mm = (top_affine @ np.append(top_center_grid, 1.0))
    prof_center_mm = (prof_affine @ np.append(prof_center_grid, 1.0))

    # Detected 3D position
    detected_x = (top_center_mm[0] + prof_center_mm[0]) / 2.0
    detected_y = top_center_mm[1]
    detected_z = prof_center_mm[1]

    # Position error
    pos_error = np.linalg.norm(
        np.array([detected_x, detected_y, detected_z])
        - np.array(known_position_mm)
    )

    # Size estimation from bounding boxes
    top_size_grid = np.array([
        top_bounds[2] - top_bounds[0],
        top_bounds[3] - top_bounds[1],
    ])
    prof_size_grid = np.array([
        prof_bounds[2] - prof_bounds[0],
        prof_bounds[3] - prof_bounds[1],
    ])

    detected_size_mm = [
        float((top_size_grid[0] + prof_size_grid[0]) / 2.0 * cell_size),
        float(top_size_grid[1] * cell_size),
        float(prof_size_grid[1] * cell_size),
    ]

    # Size error (relative)
    size_errors = []
    for detected, known in zip(detected_size_mm, known_size_mm):
        if known > 0:
            size_errors.append(abs(detected - known) / known)
    max_size_error = max(size_errors) if size_errors else 1.0

    passed = pos_error < 10.0 and max_size_error < 0.15

    result = {
        "passed": passed,
        "details": {
            "position_error_mm": round(float(pos_error), 2),
            "max_size_error_pct": round(float(max_size_error * 100), 1),
            "detected_position_mm": [
                round(detected_x, 1),
                round(detected_y, 1),
                round(detected_z, 1),
            ],
            "detected_size_mm": [round(s, 1) for s in detected_size_mm],
            "known_position_mm": known_position_mm,
            "known_size_mm": known_size_mm,
        },
    }
    return result


def main() -> None:
    """Interactive validation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if zmq is None:
        logger.info("ERROR: pyzmq required")
        sys.exit(1)

    cal = load_calibration()

    logger.info("=== Calibration Validation ===")
    logger.info("Place a known-size object at a measured position.")
    print()

    # Get known object parameters from user
    logger.info("Enter known object position (x_mm y_mm z_mm):")
    pos_input = input("> ").strip().split()
    known_pos = [float(v) for v in pos_input]

    logger.info("Enter known object size (width_mm depth_mm height_mm):")
    size_input = input("> ").strip().split()
    known_size = [float(v) for v in size_input]

    zmq_source = cal.get("zmq_frame_source", "tcp://localhost:5555")
    logger.info(f"\nCapturing frame from {zmq_source}...")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(zmq_source)
    sock.subscribe(b"")
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    try:
        msg = sock.recv()
    except zmq.error.Again:
        logger.info("ERROR: No frames received (timeout)")
        sys.exit(1)
    finally:
        sock.close()
        ctx.term()

    timestamp_ms = struct.unpack("<Q", msg[:8])[0]
    top = np.frombuffer(msg[8 : 8 + GRID_SIZE**2], dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    prof = np.frombuffer(msg[8 + GRID_SIZE**2 : 8 + 2 * GRID_SIZE**2], dtype=np.uint8).reshape(
        GRID_SIZE, GRID_SIZE
    )

    result = validate_object(cal, top, prof, known_pos, known_size)

    logger.info(f"\n=== Validation Results ===")
    logger.info(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
    for key, val in result["details"].items():
        logger.info(f"  {key}: {val}")


if __name__ == "__main__":
    main()
