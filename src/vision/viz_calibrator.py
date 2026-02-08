"""
Visualization Calibrator — Camera-based calibration of the V1 UI arm visualization.

Moves the arm through diverse poses, captures camera frames, detects the
end-effector (and optionally intermediate joints) in cam1 (side view), then
solves a least-squares optimization to find link lengths and joint offsets that
best match the observed pixel positions.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("th3cl4w.viz_calibrator")

# Default link lengths and offsets (fallback / initial guess)
DEFAULT_LINKS_MM = {
    "base": 80, "shoulder": 170, "elbow": 170,
    "wrist1": 60, "wrist2": 60, "end": 50,
}
DEFAULT_OFFSETS = [0, 90, 90, 0, 0, 0]

# Safety: joint limits with 5° margin
JOINT_LIMITS = {
    1: (-85.0, 85.0),
    2: (-85.0, 85.0),
    4: (-85.0, 85.0),
}

# Calibration poses: vary J1, J2, J4 across the range
# Each entry is [J0, J1, J2, J3, J4, J5]
CALIBRATION_POSES = [
    [0,   0,   0, 0,   0, 0],  # home
    [0,  30,   0, 0,   0, 0],
    [0, -30,   0, 0,   0, 0],
    [0,  60,   0, 0,   0, 0],
    [0, -60,   0, 0,   0, 0],
    [0,   0,  40, 0,   0, 0],
    [0,   0, -40, 0,   0, 0],
    [0,   0,  70, 0,   0, 0],
    [0,   0, -70, 0,   0, 0],
    [0,   0,   0, 0,  40, 0],
    [0,   0,   0, 0, -40, 0],
    [0,  30,  30, 0,   0, 0],
    [0, -30, -30, 0,   0, 0],
    [0,  40,  40, 0,  30, 0],
    [0, -40, -40, 0, -30, 0],
    [0,  60, -30, 0,  20, 0],
    [0, -20,  60, 0, -20, 0],
    [0,  50,  50, 0,  40, 0],
    [0, -50, -50, 0, -40, 0],
    [0,  20, -60, 0,  30, 0],
]

CAMERA_URL = "http://localhost:8081/snapshot/1"
ARM_API = "http://localhost:8080"
SETTLE_TIME = 2.0  # seconds to wait after moving
MOVE_STEP_DEG = 10.0  # max degrees per increment

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "static" / "v1" / "viz_calibration.json"


@dataclass
class PoseObservation:
    """A single calibration observation."""
    joint_angles: List[float]
    end_effector_px: Optional[Tuple[int, int]] = None
    joint_positions_px: Optional[List[Optional[Tuple[int, int]]]] = None
    timestamp: float = 0.0


@dataclass
class CalibrationResult:
    """Result of the calibration optimization."""
    links_mm: Dict[str, float]
    joint_viz_offsets: List[float]
    residual: float
    n_observations: int
    camera_params: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    message: str = ""


def fk_2d(joint_angles: List[float], links: List[float], offsets: List[float]) -> List[Tuple[float, float]]:
    """
    Compute 2D side-view FK chain points (same logic as the JS drawArm).
    Returns list of (x, y) points in mm, origin at base.
    Y axis points UP (positive = up).
    """
    pts = [(0.0, 0.0)]

    # Base: straight up
    pts.append((0.0, links[0]))

    cum_angle = math.pi / 2  # pointing up

    # Shoulder (J1)
    cum_angle += math.radians(joint_angles[1] + offsets[1])
    x = pts[1][0] + math.cos(cum_angle) * links[1]
    y = pts[1][1] + math.sin(cum_angle) * links[1]
    pts.append((x, y))

    # Elbow (J2)
    cum_angle += math.radians(joint_angles[2] + offsets[2])
    x = pts[2][0] + math.cos(cum_angle) * links[2]
    y = pts[2][1] + math.sin(cum_angle) * links[2]
    pts.append((x, y))

    # Wrist1 (J3 roll — no pitch change)
    x = pts[3][0] + math.cos(cum_angle) * links[3]
    y = pts[3][1] + math.sin(cum_angle) * links[3]
    pts.append((x, y))

    # Wrist2 (J4 pitch)
    cum_angle += math.radians(joint_angles[4] + offsets[4])
    x = pts[4][0] + math.cos(cum_angle) * links[4]
    y = pts[4][1] + math.sin(cum_angle) * links[4]
    pts.append((x, y))

    # End (J5 roll — no pitch change)
    x = pts[5][0] + math.cos(cum_angle) * links[5]
    y = pts[5][1] + math.sin(cum_angle) * links[5]
    pts.append((x, y))

    return pts


def detect_end_effector(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Detect the end-effector (gripper) in a camera frame.
    Uses multiple strategies: color detection, edge detection, contour analysis.
    Returns (x, y) pixel coordinates or None.
    """
    if frame is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # Strategy 1: Look for the metallic/dark gripper
    # The Unitree D1 gripper is typically dark gray/black
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Strategy 2: Look for the arm's blue/silver color
    # Unitree arms often have blue accents
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Strategy 3: Dark metallic regions (arm body)
    lower_dark = np.array([0, 0, 20])
    upper_dark = np.array([180, 80, 120])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Combine masks
    combined = cv2.bitwise_or(blue_mask, dark_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filter by area and find the topmost point of significant contours
    # (end-effector tends to be at extremes of the arm)
    min_area = (h * w) * 0.001  # at least 0.1% of image
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return None

    # Find the extreme point that's furthest from the base
    # Base is typically at bottom-center of the side view
    base_x, base_y = w // 2, h

    best_point = None
    best_dist = 0

    for contour in valid_contours:
        # Get extreme points
        for point in contour.reshape(-1, 2):
            px, py = int(point[0]), int(point[1])
            dist = math.sqrt((px - base_x) ** 2 + (py - base_y) ** 2)
            if dist > best_dist:
                best_dist = dist
                best_point = (px, py)

    return best_point


def detect_arm_joints(frame: np.ndarray) -> List[Optional[Tuple[int, int]]]:
    """
    Try to detect intermediate joint positions in the camera frame.
    Returns a list of detected positions [base, shoulder, elbow, wrist1, wrist2, end]
    with None for undetected joints.
    """
    # This is harder and less reliable — return empty for now
    # The optimization can still work with just end-effector positions
    return [None] * 6


async def capture_snapshot(camera_url: str = CAMERA_URL) -> Optional[np.ndarray]:
    """Capture a single frame from the camera server."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(camera_url)
            if resp.status_code != 200:
                logger.warning("Camera snapshot failed: %d", resp.status_code)
                return None
            arr = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        logger.error("Camera capture error: %s", e)
        return None


async def get_arm_state(api_base: str = ARM_API) -> Optional[Dict]:
    """Get current arm state from the server."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{api_base}/api/state")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.error("Failed to get arm state: %s", e)
    return None


async def move_joint_slowly(joint_id: int, target_deg: float,
                            api_base: str = ARM_API,
                            step_deg: float = MOVE_STEP_DEG) -> bool:
    """Move a single joint slowly in increments."""
    state = await get_arm_state(api_base)
    if state is None:
        return False

    current = state["joints"][joint_id]
    diff = target_deg - current

    if abs(diff) < 0.5:
        return True

    n_steps = max(1, int(abs(diff) / step_deg))
    for i in range(1, n_steps + 1):
        angle = current + (diff * i / n_steps)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.post(
                    f"{api_base}/api/command/set-joint",
                    json={"id": joint_id, "angle": round(angle, 1)},
                )
                if resp.status_code != 200:
                    data = resp.json()
                    if not data.get("ok", False):
                        logger.warning("Joint move failed: %s", data)
                        return False
        except Exception as e:
            logger.error("Joint move error: %s", e)
            return False
        await asyncio.sleep(0.3)

    return True


async def move_to_pose(pose: List[float], api_base: str = ARM_API) -> bool:
    """Move the arm to a target pose, one joint at a time, slowly."""
    # Move pitch joints (J1, J2, J4) which affect the side view
    for jid in [1, 2, 4, 0, 3, 5]:
        if abs(pose[jid]) > 0.1 or True:  # always send to ensure position
            ok = await move_joint_slowly(jid, pose[jid], api_base)
            if not ok:
                logger.warning("Failed to move J%d to %.1f", jid, pose[jid])
                return False
    return True


async def return_home(api_base: str = ARM_API) -> bool:
    """Return arm to home position [0,0,0,0,0,0]."""
    return await move_to_pose([0, 0, 0, 0, 0, 0], api_base)


def solve_calibration(observations: List[PoseObservation],
                      image_shape: Tuple[int, int] = (1080, 1920)) -> CalibrationResult:
    """
    Given observed end-effector pixel positions at known joint angles,
    find the best LINKS_MM and JOINT_VIZ_OFFSETS.

    The camera maps the 2D FK (mm) space to pixel space via an affine transform:
      px_x = sx * fk_x + tx
      px_y = -sy * fk_y + ty   (y-flip: FK y-up, pixel y-down)

    We jointly optimize: 6 link lengths, 3 offsets (J1,J2,J4), 4 camera params (sx,sy,tx,ty).
    """
    valid_obs = [o for o in observations if o.end_effector_px is not None]
    if len(valid_obs) < 5:
        return CalibrationResult(
            links_mm=DEFAULT_LINKS_MM.copy(),
            joint_viz_offsets=list(DEFAULT_OFFSETS),
            residual=float('inf'),
            n_observations=len(valid_obs),
            success=False,
            message=f"Not enough valid observations ({len(valid_obs)} < 5)",
        )

    h, w = image_shape

    # Parameter vector:
    # [base, shoulder, elbow, wrist1, wrist2, end, off1, off2, off4, sx, sy, tx, ty]
    link_names = ["base", "shoulder", "elbow", "wrist1", "wrist2", "end"]
    x0 = np.array([
        DEFAULT_LINKS_MM["base"],
        DEFAULT_LINKS_MM["shoulder"],
        DEFAULT_LINKS_MM["elbow"],
        DEFAULT_LINKS_MM["wrist1"],
        DEFAULT_LINKS_MM["wrist2"],
        DEFAULT_LINKS_MM["end"],
        DEFAULT_OFFSETS[1],  # off1
        DEFAULT_OFFSETS[2],  # off2
        DEFAULT_OFFSETS[4],  # off4
        w / 900.0,    # sx (scale x, approx)
        h / 900.0,    # sy (scale y)
        w * 0.35,     # tx (base pixel x)
        h * 0.85,     # ty (base pixel y)
    ])

    def residuals(params):
        links = list(params[:6])
        offsets = [0, params[6], params[7], 0, params[8], 0]
        sx, sy, tx, ty = params[9], params[10], params[11], params[12]

        total = 0.0
        for obs in valid_obs:
            fk_pts = fk_2d(obs.joint_angles, links, offsets)
            # End effector is last point
            ee_fk = fk_pts[-1]
            # Map to pixels
            pred_px = sx * ee_fk[0] + tx
            pred_py = -sy * ee_fk[1] + ty

            obs_px, obs_py = obs.end_effector_px
            total += (pred_px - obs_px) ** 2 + (pred_py - obs_py) ** 2

        return total

    # Bounds: links > 10mm, offsets ±180, camera params reasonable
    bounds = [
        (10, 300), (50, 400), (50, 400), (10, 200), (10, 200), (10, 200),  # links
        (-180, 180), (-180, 180), (-180, 180),  # offsets
        (0.1, 10), (0.1, 10),  # scale
        (0, w), (0, h),  # translation
    ]

    result = minimize(residuals, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000, 'ftol': 1e-10})

    p = result.x
    links_mm = {name: round(float(p[i]), 1) for i, name in enumerate(link_names)}
    offsets = [0, round(float(p[6]), 1), round(float(p[7]), 1), 0, round(float(p[8]), 1), 0]
    camera_params = {
        "sx": round(float(p[9]), 4),
        "sy": round(float(p[10]), 4),
        "tx": round(float(p[11]), 1),
        "ty": round(float(p[12]), 1),
    }

    return CalibrationResult(
        links_mm=links_mm,
        joint_viz_offsets=offsets,
        residual=round(float(result.fun), 2),
        n_observations=len(valid_obs),
        camera_params=camera_params,
        success=result.success,
        message=result.message if hasattr(result, 'message') else "",
    )


def save_calibration(result: CalibrationResult, path: Path = OUTPUT_PATH) -> None:
    """Save calibration result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "links_mm": result.links_mm,
        "joint_viz_offsets": result.joint_viz_offsets,
        "residual": result.residual,
        "n_observations": result.n_observations,
        "camera_params": result.camera_params,
        "success": result.success,
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Calibration saved to %s", path)


def load_calibration(path: Path = OUTPUT_PATH) -> Optional[Dict]:
    """Load saved calibration from JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.error("Failed to load calibration: %s", e)
        return None


async def run_calibration(
    poses: Optional[List[List[float]]] = None,
    api_base: str = ARM_API,
    camera_url: str = CAMERA_URL,
    progress_callback=None,
) -> CalibrationResult:
    """
    Full calibration routine:
    1. Move arm through poses
    2. Capture frames and detect end-effector
    3. Solve optimization
    4. Save results
    """
    if poses is None:
        poses = CALIBRATION_POSES

    observations = []
    total = len(poses)
    image_shape = (1080, 1920)  # default, updated from first frame

    try:
        for i, pose in enumerate(poses):
            if progress_callback:
                progress_callback(i, total, f"Moving to pose {i+1}/{total}")

            logger.info("Pose %d/%d: %s", i + 1, total, pose)

            # Move to pose
            ok = await move_to_pose(pose, api_base)
            if not ok:
                logger.warning("Skipping pose %d — move failed", i + 1)
                continue

            # Wait for arm to settle
            await asyncio.sleep(SETTLE_TIME)

            # Get actual joint angles
            state = await get_arm_state(api_base)
            if state is None:
                continue
            actual_angles = state["joints"]

            # Capture frame
            frame = await capture_snapshot(camera_url)
            if frame is None:
                logger.warning("Skipping pose %d — capture failed", i + 1)
                continue

            image_shape = frame.shape[:2]

            # Detect end-effector
            ee_px = detect_end_effector(frame)
            joint_px = detect_arm_joints(frame)

            obs = PoseObservation(
                joint_angles=actual_angles,
                end_effector_px=ee_px,
                joint_positions_px=joint_px,
                timestamp=time.time(),
            )
            observations.append(obs)

            logger.info("  Detected EE: %s", ee_px)

    finally:
        # Always return home
        if progress_callback:
            progress_callback(total, total, "Returning home")
        await return_home(api_base)

    if progress_callback:
        progress_callback(total, total, "Solving optimization")

    # Solve
    result = solve_calibration(observations, image_shape)

    # Save
    if result.success:
        save_calibration(result)

    return result
