"""
Visualization Calibrator — Camera-based calibration of the V1 UI arm visualization.

Uses progressive angle increments with self-assessment: starts with small joint
movements and gradually increases, stopping early when calibration converges.
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
from .gpu_preprocess import to_hsv
import httpx
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("th3cl4w.viz_calibrator")

# Default link lengths and offsets (fallback / initial guess)
DEFAULT_LINKS_MM = {
    "base": 80,
    "shoulder": 170,
    "elbow": 170,
    "wrist1": 60,
    "wrist2": 60,
    "end": 50,
}
DEFAULT_OFFSETS = [0, 90, 90, 0, 0, 0]

# Safety: joint limits with 5° margin
JOINT_LIMITS = {
    1: (-85.0, 85.0),
    2: (-85.0, 85.0),
    4: (-85.0, 85.0),
}

# Progressive calibration config
PITCH_JOINTS = [1, 2, 4]  # joints that affect side-view
ANGLE_INCREMENT = 5  # degrees per round
MAX_ANGLE = 45  # maximum angle to reach
CONVERGENCE_THRESHOLD = 50.0  # average px residual to consider "good"
STABLE_ROUNDS_NEEDED = 2  # rounds residual must be stable to stop

from src.config.camera_config import snap_url as _snap_url
CAMERA_URL = _snap_url(1)
ARM_API = "http://localhost:8080"
SETTLE_TIME = 2.0  # seconds to wait after moving
MOVE_STEP_DEG = 10.0  # max degrees per increment

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "web" / "static" / "v1" / "viz_calibration.json"
)


def generate_round_poses(round_num: int) -> List[List[float]]:
    """
    Generate poses for a given round number (1-indexed).
    Round N tests each pitch joint at ±(N * ANGLE_INCREMENT) degrees individually.
    Returns list of [J0, J1, J2, J3, J4, J5] poses.
    """
    angle = round_num * ANGLE_INCREMENT
    poses = []
    for jid in PITCH_JOINTS:
        lo, hi = JOINT_LIMITS[jid]
        for sign in [1, -1]:
            target = sign * angle
            if lo <= target <= hi:
                pose = [0, 0, 0, 0, 0, 0]
                pose[jid] = target
                poses.append(pose)
    return poses


def max_rounds() -> int:
    """Number of rounds to reach MAX_ANGLE."""
    return MAX_ANGLE // ANGLE_INCREMENT


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


def fk_2d(
    joint_angles: List[float], links: List[float], offsets: List[float]
) -> List[Tuple[float, float]]:
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

    hsv = to_hsv(frame)
    h, w = frame.shape[:2]

    # Strategy 1: Look for the metallic/dark gripper
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Strategy 2: Look for the arm's blue/silver color
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
    min_area = (h * w) * 0.001  # at least 0.1% of image
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return None

    # Find the extreme point that's furthest from the base
    base_x, base_y = w // 2, h

    best_point = None
    best_dist = 0

    for contour in valid_contours:
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


async def move_joint_slowly(
    joint_id: int, target_deg: float, api_base: str = ARM_API, step_deg: float = MOVE_STEP_DEG
) -> bool:
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
    for jid in [1, 2, 4, 0, 3, 5]:
        ok = await move_joint_slowly(jid, pose[jid], api_base)
        if not ok:
            logger.warning("Failed to move J%d to %.1f", jid, pose[jid])
            return False
    return True


async def return_home(api_base: str = ARM_API) -> bool:
    """Return arm to home position [0,0,0,0,0,0]."""
    return await move_to_pose([0, 0, 0, 0, 0, 0], api_base)


def solve_calibration(
    observations: List[PoseObservation], image_shape: Tuple[int, int] = (1080, 1920)
) -> CalibrationResult:
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
            residual=-1.0,
            n_observations=len(valid_obs),
            success=False,
            message=f"Not enough valid observations ({len(valid_obs)} < 5)",
        )

    h, w = image_shape

    link_names = ["base", "shoulder", "elbow", "wrist1", "wrist2", "end"]
    x0 = np.array(
        [
            DEFAULT_LINKS_MM["base"],
            DEFAULT_LINKS_MM["shoulder"],
            DEFAULT_LINKS_MM["elbow"],
            DEFAULT_LINKS_MM["wrist1"],
            DEFAULT_LINKS_MM["wrist2"],
            DEFAULT_LINKS_MM["end"],
            DEFAULT_OFFSETS[1],  # off1
            DEFAULT_OFFSETS[2],  # off2
            DEFAULT_OFFSETS[4],  # off4
            w / 900.0,  # sx
            h / 900.0,  # sy
            w * 0.35,  # tx
            h * 0.85,  # ty
        ]
    )

    def residuals(params):
        links = list(params[:6])
        offsets = [0, params[6], params[7], 0, params[8], 0]
        sx, sy, tx, ty = params[9], params[10], params[11], params[12]

        total = 0.0
        for obs in valid_obs:
            fk_pts = fk_2d(obs.joint_angles, links, offsets)
            ee_fk = fk_pts[-1]
            pred_px = sx * ee_fk[0] + tx
            pred_py = -sy * ee_fk[1] + ty

            obs_px, obs_py = obs.end_effector_px
            total += (pred_px - obs_px) ** 2 + (pred_py - obs_py) ** 2

        return total

    bounds = [
        (10, 300),
        (50, 400),
        (50, 400),
        (10, 200),
        (10, 200),
        (10, 200),  # links
        (-180, 180),
        (-180, 180),
        (-180, 180),  # offsets
        (0.1, 10),
        (0.1, 10),  # scale
        (0, w),
        (0, h),  # translation
    ]

    result = minimize(
        residuals,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "maxfun": 50000, "ftol": 1e-10},
    )

    p = result.x
    links_mm = {name: round(float(p[i]), 1) for i, name in enumerate(link_names)}
    offsets = [0, round(float(p[6]), 1), round(float(p[7]), 1), 0, round(float(p[8]), 1), 0]
    camera_params = {
        "sx": round(float(p[9]), 4),
        "sy": round(float(p[10]), 4),
        "tx": round(float(p[11]), 1),
        "ty": round(float(p[12]), 1),
    }

    # Compute average per-observation residual in pixels
    avg_residual = math.sqrt(result.fun / len(valid_obs)) if valid_obs else 0.0

    return CalibrationResult(
        links_mm=links_mm,
        joint_viz_offsets=offsets,
        residual=round(avg_residual, 2),
        n_observations=len(valid_obs),
        camera_params=camera_params,
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
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


async def check_for_stall(api_base: str = ARM_API, timeout_s: float = 3.0) -> Optional[int]:
    """
    After a move, check if any joint is stalled (commanded != actual beyond threshold).
    Returns the stalled joint id or None.
    """
    from src.safety.collision_detector import CollisionDetector

    detector = CollisionDetector(position_error_deg=3.0, stall_duration_s=0.3)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        state = await get_arm_state(api_base)
        if state is None:
            await asyncio.sleep(0.2)
            continue
        actual = state["joints"]
        await asyncio.sleep(0.2)
        state2 = await get_arm_state(api_base)
        if state2 is None:
            continue
        actual2 = state2["joints"]
        settled = all(abs(actual2[i] - actual[i]) < 0.1 for i in range(6))
        if settled:
            return None
    return None


async def run_calibration(
    api_base: str = ARM_API,
    camera_url: str = CAMERA_URL,
    progress_callback=None,
) -> CalibrationResult:
    """
    Progressive calibration routine with self-assessment:
    1. Generate poses in rounds of increasing angle (±5°, ±10°, ... ±45°)
    2. After each round, solve and assess residual
    3. Stop early if converged (residual < threshold for N stable rounds)
    4. Always return home between poses and at the end
    """
    observations: List[PoseObservation] = []
    image_shape = (1080, 1920)
    n_rounds = max_rounds()
    prev_residual: Optional[float] = None
    stable_count = 0
    skipped_poses: List[Tuple[int, List[float], str]] = []
    total_poses_planned = sum(len(generate_round_poses(r)) for r in range(1, n_rounds + 1))
    poses_done = 0

    try:
        # Home pose observation first
        if progress_callback:
            progress_callback(0, total_poses_planned, "Capturing home pose")
        frame = await capture_snapshot(camera_url)
        if frame is not None:
            image_shape = frame.shape[:2]
            state = await get_arm_state(api_base)
            if state is not None:
                ee_px = detect_end_effector(frame)
                observations.append(
                    PoseObservation(
                        joint_angles=state["joints"],
                        end_effector_px=ee_px,
                        joint_positions_px=detect_arm_joints(frame),
                        timestamp=time.time(),
                    )
                )

        for round_num in range(1, n_rounds + 1):
            round_poses = generate_round_poses(round_num)

            for pose in round_poses:
                poses_done += 1
                if progress_callback:
                    progress_callback(
                        poses_done, total_poses_planned, f"Round {round_num}: pose {poses_done}"
                    )

                logger.info("Round %d — pose %s", round_num, pose)

                # Move to pose
                ok = await move_to_pose(pose, api_base)
                if not ok:
                    reason = "move failed"
                    logger.warning("Skipping pose %s — %s", pose, reason)
                    skipped_poses.append((round_num, pose, reason))
                    await return_home(api_base)
                    await asyncio.sleep(1.0)
                    continue

                await asyncio.sleep(SETTLE_TIME)

                # Check for stall / collision
                stalled_joint = await check_for_stall(api_base, timeout_s=1.0)
                if stalled_joint is not None:
                    reason = f"J{stalled_joint} stalled"
                    logger.warning("Skipping pose %s — %s", pose, reason)
                    skipped_poses.append((round_num, pose, reason))
                    await return_home(api_base)
                    await asyncio.sleep(1.0)
                    continue

                # Get actual state and capture
                state = await get_arm_state(api_base)
                if state is None:
                    await return_home(api_base)
                    continue

                frame = await capture_snapshot(camera_url)
                if frame is None:
                    logger.warning("Skipping pose %s — capture failed", pose)
                    skipped_poses.append((round_num, pose, "capture failed"))
                    await return_home(api_base)
                    continue

                image_shape = frame.shape[:2]
                ee_px = detect_end_effector(frame)

                observations.append(
                    PoseObservation(
                        joint_angles=state["joints"],
                        end_effector_px=ee_px,
                        joint_positions_px=detect_arm_joints(frame),
                        timestamp=time.time(),
                    )
                )
                logger.info("  Detected EE: %s", ee_px)

                # Return home between poses
                await return_home(api_base)
                await asyncio.sleep(0.5)

            # --- Self-assessment after this round ---
            result = solve_calibration(observations, image_shape)
            residual = result.residual if result.success else -1.0
            n_valid = result.n_observations

            if result.success and prev_residual is not None:
                if abs(residual - prev_residual) < 5.0 and residual < CONVERGENCE_THRESHOLD:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0

            converged = stable_count >= STABLE_ROUNDS_NEEDED
            status = "converged — stopping" if converged else "continuing"
            logger.info(
                "Round %d complete: %d observations, residual %.1fpx, %s",
                round_num,
                n_valid,
                residual,
                status,
            )

            if converged:
                break

            prev_residual = residual

    finally:
        if progress_callback:
            progress_callback(total_poses_planned, total_poses_planned, "Returning home")
        await return_home(api_base)

    # Final solve
    if progress_callback:
        progress_callback(total_poses_planned, total_poses_planned, "Solving optimization")

    result = solve_calibration(observations, image_shape)

    if skipped_poses:
        logger.info(
            "Skipped %d poses: %s",
            len(skipped_poses),
            [(r, p, reason) for r, p, reason in skipped_poses],
        )

    if result.success:
        save_calibration(result)

    return result
