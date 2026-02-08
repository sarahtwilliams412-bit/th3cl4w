"""
Camera extrinsics solver — computes camera-to-arm-base transform via PnP.

Uses calibration session data (FK 3D positions + detected/annotated 2D pixel
positions) to solve for each camera's extrinsic parameters independently.

Supports two modes:
1. Bootstrap: use manually annotated pixel positions from calibration plan
2. Live: detect end-effector in captured calibration frames
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root so src.* imports work
import sys
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.vision.fk_engine import fk_positions

logger = logging.getLogger("th3cl4w.calibration.extrinsics")

# Default intrinsics for 1920×1080 cameras (no intrinsic calibration yet)
DEFAULT_FX = 1000.0
DEFAULT_FY = 1000.0
DEFAULT_CX = 960.0
DEFAULT_CY = 540.0
DEFAULT_IMAGE_SIZE = (1920, 1080)


@dataclass
class ExtrinsicsResult:
    """Result of camera extrinsics calibration."""
    camera_id: str
    rvec: list[float]        # Rodrigues rotation vector (3,)
    tvec: list[float]        # Translation vector (3,)
    reprojection_error_mean: float
    reprojection_error_max: float
    num_poses_used: int
    num_inliers: int
    camera_matrix: list[list[float]]  # 3x3 intrinsic matrix used
    date: str


# Bootstrap poses from calibration plan (manually annotated pixel positions)
# Format: (joint_angles, cam0_pixel_or_None, cam1_pixel_or_None)
BOOTSTRAP_POSES = [
    ((0, 0, 0, 0, 0, 0),       (1130, 220), (760, 135)),
    ((45, 0, 0, 0, 0, 0),      (1430, 105), (680, 520)),
    ((-45, 0, 0, 0, 0, 0),     (920, 175),  (660, 290)),
    ((0, 45, 0, 0, 0, 0),      (1050, 220), (760, 270)),
    ((0, -45, 0, 0, 0, 0),     (540, 230),  (760, 470)),
    ((0, 0, 45, 0, 0, 0),      None,        (660, 330)),
    ((0, 0, -45, 0, 0, 0),     None,        (830, 490)),
    ((0, 0, 0, 0, 45, 0),      None,        (760, 290)),
    ((0, 0, 0, 0, -45, 0),     None,        (760, 200)),
]


def get_default_camera_matrix() -> np.ndarray:
    """Return default 3x3 camera intrinsic matrix for 1920x1080."""
    return np.array([
        [DEFAULT_FX, 0, DEFAULT_CX],
        [0, DEFAULT_FY, DEFAULT_CY],
        [0, 0, 1],
    ], dtype=np.float64)


def compute_fk_ee_positions(
    joint_angles_list: list[list[float]],
) -> list[np.ndarray]:
    """Compute end-effector 3D positions for a list of joint angle sets."""
    positions = []
    for angles in joint_angles_list:
        chain = fk_positions(angles)
        ee = chain[-1]  # end-effector is last position
        positions.append(np.array(ee, dtype=np.float64))
    return positions


def solve_camera_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    use_ransac: bool = True,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve camera extrinsics via PnP.

    Args:
        object_points: Nx3 float64 array of 3D points in arm-base frame
        image_points: Nx2 float64 array of corresponding pixel positions
        camera_matrix: 3x3 intrinsic matrix (uses defaults if None)
        dist_coeffs: distortion coefficients (assumes zero if None)
        use_ransac: whether to use RANSAC for outlier rejection

    Returns:
        (rvec, tvec, inliers) or (None, None, None) on failure
    """
    if camera_matrix is None:
        camera_matrix = get_default_camera_matrix()
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    obj = object_points.reshape(-1, 1, 3).astype(np.float64)
    img = image_points.reshape(-1, 1, 2).astype(np.float64)

    n = len(object_points)
    if n < 4:
        logger.error("Need at least 4 point correspondences, got %d", n)
        return None, None, None

    if use_ransac and n >= 6:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, camera_matrix, dist_coeffs,
            iterationsCount=1000,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            logger.error("solvePnPRansac failed")
            return None, None, None
    else:
        success, rvec, tvec = cv2.solvePnP(
            obj, img, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            logger.error("solvePnP failed")
            return None, None, None
        inliers = np.arange(n).reshape(-1, 1)

    # Refine with Levenberg-Marquardt
    rvec, tvec = cv2.solvePnPRefineLM(
        obj, img, camera_matrix, dist_coeffs, rvec, tvec,
    )

    return rvec, tvec, inliers


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> tuple[float, float, np.ndarray]:
    """
    Compute reprojection error.

    Returns:
        (mean_error, max_error, per_point_errors)
    """
    if camera_matrix is None:
        camera_matrix = get_default_camera_matrix()
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3).astype(np.float64),
        rvec, tvec, camera_matrix, dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    actual = image_points.reshape(-1, 2)

    errors = np.linalg.norm(projected - actual, axis=1)
    return float(np.mean(errors)), float(np.max(errors)), errors


def solve_from_bootstrap(
    camera_id: str = "cam0",
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[ExtrinsicsResult]:
    """
    Solve extrinsics using bootstrap poses from calibration plan.

    Uses manually annotated pixel positions. cam0 has 5 correspondences,
    cam1 has 9 correspondences.
    """
    cam_idx = 0 if camera_id == "cam0" else 1

    # Collect correspondences
    obj_pts = []
    img_pts = []
    for angles, cam0_px, cam1_px in BOOTSTRAP_POSES:
        px = cam0_px if cam_idx == 0 else cam1_px
        if px is None:
            continue
        chain = fk_positions(list(angles))
        ee = chain[-1]
        obj_pts.append(ee)
        img_pts.append(list(px))

    if len(obj_pts) < 4:
        logger.error("Not enough bootstrap correspondences for %s: %d", camera_id, len(obj_pts))
        return None

    object_points = np.array(obj_pts, dtype=np.float64)
    image_points = np.array(img_pts, dtype=np.float64)

    if camera_matrix is None:
        camera_matrix = get_default_camera_matrix()
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    rvec, tvec, inliers = solve_camera_pnp(
        object_points, image_points, camera_matrix, dist_coeffs,
        use_ransac=(len(obj_pts) >= 6),
    )
    if rvec is None:
        return None

    mean_err, max_err, _ = compute_reprojection_error(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs,
    )

    logger.info(
        "%s bootstrap solve: mean_err=%.2fpx, max_err=%.2fpx, inliers=%d/%d",
        camera_id, mean_err, max_err,
        len(inliers) if inliers is not None else len(obj_pts), len(obj_pts),
    )

    return ExtrinsicsResult(
        camera_id=camera_id,
        rvec=rvec.flatten().tolist(),
        tvec=tvec.flatten().tolist(),
        reprojection_error_mean=mean_err,
        reprojection_error_max=max_err,
        num_poses_used=len(obj_pts),
        num_inliers=len(inliers) if inliers is not None else len(obj_pts),
        camera_matrix=camera_matrix.tolist(),
        date=datetime.now(timezone.utc).isoformat(),
    )


def solve_from_session_data(
    captures: list[dict],
    camera_id: str = "cam0",
    pixel_positions: Optional[list[Optional[tuple[float, float]]]] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[ExtrinsicsResult]:
    """
    Solve extrinsics from calibration session data.

    Args:
        captures: list of capture dicts with 'actual_angles' field
        camera_id: which camera to solve for
        pixel_positions: pre-detected pixel positions per capture (None entries skipped)
        camera_matrix: intrinsic matrix
        dist_coeffs: distortion coefficients
    """
    if pixel_positions is None:
        logger.error("pixel_positions required (end-effector detection not yet implemented)")
        return None

    obj_pts = []
    img_pts = []
    for i, cap in enumerate(captures):
        if i >= len(pixel_positions) or pixel_positions[i] is None:
            continue
        chain = fk_positions(cap['actual_angles'])
        ee = chain[-1]
        obj_pts.append(ee)
        img_pts.append(list(pixel_positions[i]))

    if len(obj_pts) < 4:
        logger.error("Not enough correspondences for %s: %d", camera_id, len(obj_pts))
        return None

    object_points = np.array(obj_pts, dtype=np.float64)
    image_points = np.array(img_pts, dtype=np.float64)

    if camera_matrix is None:
        camera_matrix = get_default_camera_matrix()
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    rvec, tvec, inliers = solve_camera_pnp(
        object_points, image_points, camera_matrix, dist_coeffs,
    )
    if rvec is None:
        return None

    mean_err, max_err, _ = compute_reprojection_error(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs,
    )

    return ExtrinsicsResult(
        camera_id=camera_id,
        rvec=rvec.flatten().tolist(),
        tvec=tvec.flatten().tolist(),
        reprojection_error_mean=mean_err,
        reprojection_error_max=max_err,
        num_poses_used=len(obj_pts),
        num_inliers=len(inliers) if inliers is not None else len(obj_pts),
        camera_matrix=camera_matrix.tolist(),
        date=datetime.now(timezone.utc).isoformat(),
    )


def detect_ee_in_frame(
    frame_jpeg: bytes,
    method: str = "gold_hsv",
) -> Optional[tuple[float, float]]:
    """
    Detect end-effector (gripper tip) in a camera frame.

    Uses HSV color detection for gold-colored gripper accents.

    Args:
        frame_jpeg: JPEG-encoded frame bytes
        method: detection method ("gold_hsv" or "contour_extremity")

    Returns:
        (u, v) pixel position or None if not detected
    """
    img_array = np.frombuffer(frame_jpeg, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    if method == "gold_hsv":
        return _detect_gold_tip(frame)
    else:
        return _detect_contour_extremity(frame)


def _detect_gold_tip(frame: np.ndarray) -> Optional[tuple[float, float]]:
    """Detect gold-colored gripper tip using HSV thresholding."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gold/yellow range in HSV
    lower_gold = np.array([15, 80, 80])
    upper_gold = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area and find the most likely gripper tip
    valid = [c for c in contours if 50 < cv2.contourArea(c) < 50000]
    if not valid:
        return None

    # Use the highest contour (smallest y = highest in image = most likely tip)
    best = min(valid, key=lambda c: cv2.boundingRect(c)[1])
    M = cv2.moments(best)
    if M["m00"] < 1:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def _detect_contour_extremity(frame: np.ndarray) -> Optional[tuple[float, float]]:
    """Detect arm tip as the topmost extremity of the largest dark contour."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold for dark arm (matte black)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour is likely the arm
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None

    # Find the topmost point (smallest y)
    topmost = tuple(largest[largest[:, :, 1].argmin()][0])
    return (float(topmost[0]), float(topmost[1]))


def save_extrinsics(
    results: list[ExtrinsicsResult],
    path: str,
) -> None:
    """Save extrinsics results to JSON file."""
    output = {
        "cameras": {},
        "date": datetime.now(timezone.utc).isoformat(),
    }
    for r in results:
        output["cameras"][r.camera_id] = {
            "rvec": r.rvec,
            "tvec": r.tvec,
            "reprojection_error_mean": r.reprojection_error_mean,
            "reprojection_error_max": r.reprojection_error_max,
            "num_poses_used": r.num_poses_used,
            "num_inliers": r.num_inliers,
            "camera_matrix": r.camera_matrix,
            "date": r.date,
        }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info("Saved extrinsics to %s", path)


def load_extrinsics(path: str) -> Optional[dict]:
    """Load extrinsics from JSON file."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def run_bootstrap_calibration(
    output_path: str = None,
) -> dict[str, ExtrinsicsResult]:
    """
    Run full bootstrap calibration using annotated poses.

    Returns dict of camera_id -> ExtrinsicsResult.
    """
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "calibration_results" / "camera_extrinsics.json"
        )

    results = {}
    for cam_id in ["cam0", "cam1"]:
        result = solve_from_bootstrap(cam_id)
        if result is not None:
            results[cam_id] = result
            logger.info(
                "%s: mean_reproj=%.2fpx, max_reproj=%.2fpx",
                cam_id, result.reprojection_error_mean, result.reprojection_error_max,
            )
        else:
            logger.warning("Failed to solve extrinsics for %s", cam_id)

    if results:
        save_extrinsics(list(results.values()), output_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_bootstrap_calibration()
    for cam_id, r in results.items():
        print(f"\n{cam_id}:")
        print(f"  rvec: {[f'{v:.4f}' for v in r.rvec]}")
        print(f"  tvec: {[f'{v:.4f}' for v in r.tvec]}")
        print(f"  reproj mean: {r.reprojection_error_mean:.2f}px")
        print(f"  reproj max:  {r.reprojection_error_max:.2f}px")
        print(f"  poses used:  {r.num_poses_used}")
        print(f"  inliers:     {r.num_inliers}")
