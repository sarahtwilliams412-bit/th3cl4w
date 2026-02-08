"""
3D Visualization Calibrator — Camera-based calibration of the V1 UI arm visualization.

Uses full 3D FK from DH parameters + pinhole camera projection.
Detection via frame differencing and gold segment color detection.
Both cameras (cam0 overhead, cam1 side) used for constraints.
Progressive angle increments with self-assessment.
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
from scipy.optimize import least_squares

logger = logging.getLogger("th3cl4w.viz_calibrator")

# ---------------------------------------------------------------------------
# DH parameters (from src/kinematics/kinematics.py) — FIXED, not optimized
# ---------------------------------------------------------------------------
# d values in meters: J1=0.1215, J2=0, J3=0.2085, J4=0, J5=0.2085, J6=0, J7=0.113
DH_D = [0.1215, 0.0, 0.2085, 0.0, 0.2085, 0.0, 0.1130]  # meters
DH_A = [0.0] * 7
DH_ALPHA = [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0]
DH_THETA_OFFSET = [0.0] * 7

# Safety: joint limits with 5° margin
JOINT_LIMITS = {
    0: (-45.0, 45.0),   # base yaw — limited for side camera
    1: (-85.0, 85.0),
    2: (-85.0, 85.0),
    4: (-85.0, 85.0),
}

# Calibration joints — now includes J0
CALIBRATION_JOINTS = [1, 2, 4]  # pitch joints for initial calibration
ANGLE_INCREMENT = 5
MAX_ANGLE = 45
CONVERGENCE_THRESHOLD = 20.0  # px average residual
STABLE_ROUNDS_NEEDED = 2

CAMERA_URLS = {
    0: "http://localhost:8081/snap/0",  # overhead
    1: "http://localhost:8081/snap/1",  # side
}
ARM_API = "http://localhost:8080"
SETTLE_TIME = 1.5
MOVE_STEP_DEG = 10.0

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "static" / "v1" / "viz_calibration.json"

# Key joint frames to track (indices into get_joint_positions_3d output, 0=base, 1-7=joints, 8 would be EE but we have 0..7)
# base=0, after_J1=1, after_J2=2, after_J3=3 (elbow region), after_J4=4, after_J5=5, after_J6=6, after_J7=7(EE)
LANDMARK_FRAMES = {
    'base': 0,
    'shoulder': 2,     # after J2 (shoulder pitch)
    'elbow': 4,        # after J4 (elbow pitch)
    'wrist': 6,        # after J6 (wrist pitch)
    'end_effector': 7, # after J7 (end-effector)
}


# ---------------------------------------------------------------------------
# 3D Forward Kinematics
# ---------------------------------------------------------------------------

def _dh_transform(d, a, alpha, theta):
    """4x4 homogeneous transform for one DH frame."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,    0.0,  1.0],
    ])


def fk_3d(joint_angles_deg: List[float], theta_offsets: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    Compute 3D positions of all joint frames using DH parameters.
    
    Args:
        joint_angles_deg: 7 joint angles in degrees (or 6 — padded to 7)
        theta_offsets: optional 7 offset values in radians to add to each joint
        
    Returns:
        List of 8 xyz positions (base + 7 joints) in meters
    """
    angles = list(joint_angles_deg)
    # Pad to 7 if only 6 provided (viz uses 6, arm has 7)
    while len(angles) < 7:
        angles.append(0.0)
    
    if theta_offsets is None:
        theta_offsets = DH_THETA_OFFSET
    
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    
    for i in range(7):
        theta = np.radians(angles[i]) + theta_offsets[i]
        T = T @ _dh_transform(DH_D[i], DH_A[i], DH_ALPHA[i], theta)
        positions.append(T[:3, 3].copy())
    
    return positions


# ---------------------------------------------------------------------------
# Pinhole Camera Projection
# ---------------------------------------------------------------------------

def project_3d_to_2d(points_3d: List[np.ndarray], camera_params: Dict) -> List[Optional[Tuple[float, float]]]:
    """
    Project 3D world points to 2D pixel coordinates using pinhole model.
    
    camera_params: {fx, fy, cx, cy, R (3x3), t (3,)}
    """
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    R = np.array(camera_params['R']).reshape(3, 3)
    t = np.array(camera_params['t'])
    
    projected = []
    for p in points_3d:
        # Transform to camera frame
        p_cam = R @ p + t
        
        if p_cam[2] <= 0.001:  # behind camera
            projected.append(None)
            continue
        
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        projected.append((float(u), float(v)))
    
    return projected


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class ArmDetector:
    """Multi-landmark arm detector using frame differencing + gold segment color."""
    
    def __init__(self):
        self.home_frames = {}  # cam_id -> blurred home frame
        self.base_positions = {}  # cam_id -> (x, y) pixel
    
    def set_home_frame(self, cam_id: int, frame: np.ndarray):
        """Store a reference frame at home position."""
        self.home_frames[cam_id] = cv2.GaussianBlur(frame, (7, 7), 0)
    
    def detect_gold_segment(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
        """
        Find the golden/yellow arm segment.
        Returns (top_point, bottom_point) representing shoulder→elbow endpoints.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Gold/yellow range for the anodized arm segment
        gold_mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([40, 255, 255]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 300:
            return None
        
        pts = c.reshape(-1, 2)
        # Top = shoulder, bottom = elbow (in image coords, y increases downward)
        top_idx = pts[:, 1].argmin()
        bot_idx = pts[:, 1].argmax()
        return (tuple(pts[top_idx]), tuple(pts[bot_idx]))
    
    def detect_via_differencing(self, frame: np.ndarray, cam_id: int) -> Optional[Tuple[int, int]]:
        """
        Find the end-effector via frame differencing with home position.
        Returns the pixel location of the point that moved furthest from base.
        """
        if cam_id not in self.home_frames:
            return None
        
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        diff = cv2.absdiff(blurred, self.home_frames[cam_id])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find point furthest from base (or image bottom-center as fallback)
        h, w = frame.shape[:2]
        base = self.base_positions.get(cam_id, (w // 2, int(h * 0.85)))
        
        best_pt, best_dist = None, 0
        for c in contours:
            if cv2.contourArea(c) < 150:
                continue
            for pt in c.reshape(-1, 2):
                d = math.hypot(pt[0] - base[0], pt[1] - base[1])
                if d > best_dist:
                    best_dist = d
                    best_pt = (int(pt[0]), int(pt[1]))
        
        return best_pt
    
    def detect_landmarks(self, frame: np.ndarray, cam_id: int) -> Dict[str, Optional[Tuple[int, int]]]:
        """Detect all available arm landmarks in a camera frame."""
        result = {}
        
        # Gold segment → shoulder + elbow
        gold = self.detect_gold_segment(frame)
        if gold:
            result['shoulder'] = gold[0]
            result['elbow'] = gold[1]
        
        # Frame differencing → end-effector
        ee = self.detect_via_differencing(frame, cam_id)
        if ee:
            result['end_effector'] = ee
        
        # Base is fixed
        if cam_id in self.base_positions:
            result['base'] = self.base_positions[cam_id]
        
        return result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PoseObservation:
    joint_angles: List[float]  # 6 or 7 angles in degrees
    cam0_landmarks: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    cam1_landmarks: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class CalibrationResult:
    cam1_params: Dict[str, Any]  # {fx, fy, cx, cy, R, t}
    cam0_params: Optional[Dict[str, Any]]  # overhead camera
    theta_offsets: List[float]  # 7 DH theta offsets in radians
    residual: float
    n_observations: int
    n_constraints: int
    success: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _rodrigues(rvec):
    """Rodrigues vector (3,) -> rotation matrix (3,3). Pure numpy, no opencv."""
    rvec = np.asarray(rvec, dtype=float)
    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return np.eye(3)
    k = rvec / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _params_to_cameras(params, n_cams=2):
    """
    Unpack parameter vector into camera params and theta offsets.
    
    Layout: [cam1_rvec(3), cam1_t(3), cam1_fx, cam1_fy, cam1_cx, cam1_cy,
             cam0_rvec(3), cam0_t(3), cam0_fx, cam0_fy, cam0_cx, cam0_cy,
             theta_offsets(7)]
    """
    idx = 0
    cams = []
    for _ in range(n_cams):
        rvec = params[idx:idx+3]; idx += 3
        t = params[idx:idx+3]; idx += 3
        fx, fy, cx, cy = params[idx:idx+4]; idx += 4
        R = _rodrigues(rvec)
        cams.append({'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'R': R.tolist(), 't': t.tolist()})
    
    theta_offsets = params[idx:idx+7].tolist()
    return cams, theta_offsets


def solve_calibration(observations: List[PoseObservation],
                      image_shapes: Dict[int, Tuple[int, int]]) -> CalibrationResult:
    """
    Solve for camera extrinsics/intrinsics and DH theta offsets using least_squares.
    Link lengths are FIXED from DH parameters.
    """
    # Filter observations with at least one landmark
    valid_obs = [o for o in observations 
                 if len(o.cam1_landmarks) > 0 or len(o.cam0_landmarks) > 0]
    
    if len(valid_obs) < 3:
        return CalibrationResult(
            cam1_params={}, cam0_params=None,
            theta_offsets=[0.0]*7, residual=-1.0,
            n_observations=len(valid_obs), n_constraints=0,
            success=False, message=f"Not enough observations ({len(valid_obs)} < 3)")
    
    # Count constraints
    n_constraints = 0
    for obs in valid_obs:
        n_constraints += len(obs.cam1_landmarks) * 2
        n_constraints += len(obs.cam0_landmarks) * 2
    
    use_cam0 = any(len(o.cam0_landmarks) > 0 for o in valid_obs)
    n_cams = 2 if use_cam0 else 1
    
    # Initial guess
    h1, w1 = image_shapes.get(1, (1080, 1920))
    
    # cam1 (side view): camera looking at arm from the side
    cam1_init = [
        0.0, 0.0, 0.0,        # rvec (no rotation = camera aligned with world)
        0.0, 0.0, 1.5,        # t: camera 1.5m in front of arm
        1200.0, 1200.0,       # fx, fy
        w1/2, h1/2,           # cx, cy
    ]
    
    x0 = list(cam1_init)
    
    if use_cam0:
        h0, w0 = image_shapes.get(0, (1080, 1920))
        # cam0 (overhead): looking down
        cam0_init = [
            np.pi/2, 0.0, 0.0,  # rvec: rotated to look down
            0.0, 0.0, 1.0,      # t: 1m above
            1200.0, 1200.0,     # fx, fy
            w0/2, h0/2,         # cx, cy
        ]
        x0.extend(cam0_init)
    
    # theta offsets — start at zero
    x0.extend([0.0] * 7)
    x0 = np.array(x0, dtype=float)
    
    # Bounds
    cam_bounds_lo = [
        -np.pi, -np.pi, -np.pi,  # rvec
        -3.0, -3.0, 0.1,         # t
        200.0, 200.0,             # fx, fy
        0.0, 0.0,                 # cx, cy
    ]
    cam_bounds_hi = [
        np.pi, np.pi, np.pi,
        3.0, 3.0, 5.0,
        5000.0, 5000.0,
        w1, h1,
    ]
    
    lb = list(cam_bounds_lo)
    ub = list(cam_bounds_hi)
    
    if use_cam0:
        h0, w0 = image_shapes.get(0, (1080, 1920))
        lb.extend([-np.pi, -np.pi, -np.pi, -3.0, -3.0, 0.1, 200.0, 200.0, 0.0, 0.0])
        ub.extend([np.pi, np.pi, np.pi, 3.0, 3.0, 5.0, 5000.0, 5000.0, w0, h0])
    
    # Theta offset bounds: small corrections only (±30°)
    for _ in range(7):
        lb.append(-np.radians(30))
        ub.append(np.radians(30))
    
    def residual_fn(params):
        cams, theta_offsets = _params_to_cameras(params, n_cams)
        
        residuals = []
        for obs in valid_obs:
            positions_3d = fk_3d(obs.joint_angles, theta_offsets)
            
            # cam1 landmarks
            if obs.cam1_landmarks:
                projected = project_3d_to_2d(positions_3d, cams[0])
                for name, obs_px in obs.cam1_landmarks.items():
                    if obs_px is None or name not in LANDMARK_FRAMES:
                        continue
                    frame_idx = LANDMARK_FRAMES[name]
                    pred = projected[frame_idx]
                    if pred is None:
                        residuals.extend([100.0, 100.0])  # penalty for behind-camera
                        continue
                    residuals.append(pred[0] - obs_px[0])
                    residuals.append(pred[1] - obs_px[1])
            
            # cam0 landmarks
            if use_cam0 and obs.cam0_landmarks:
                projected = project_3d_to_2d(positions_3d, cams[1])
                for name, obs_px in obs.cam0_landmarks.items():
                    if obs_px is None or name not in LANDMARK_FRAMES:
                        continue
                    frame_idx = LANDMARK_FRAMES[name]
                    pred = projected[frame_idx]
                    if pred is None:
                        residuals.extend([100.0, 100.0])
                        continue
                    residuals.append(pred[0] - obs_px[0])
                    residuals.append(pred[1] - obs_px[1])
        
        # Regularization on theta offsets (prefer small corrections)
        offset_start = 10 * n_cams  # after camera params
        for i in range(7):
            residuals.append(5.0 * params[offset_start + i])  # penalty
        
        return np.array(residuals)
    
    result = least_squares(
        residual_fn, x0,
        bounds=(lb, ub),
        method='trf',
        loss='huber',
        f_scale=10.0,
        max_nfev=2000,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    
    cams, theta_offsets = _params_to_cameras(result.x, n_cams)
    
    # Compute average pixel residual (excluding regularization terms)
    raw_residuals = result.fun[:-(7)]  # remove regularization
    n_data = len(raw_residuals) // 2
    avg_residual = float(np.sqrt(np.mean(raw_residuals**2))) if n_data > 0 else -1.0
    
    return CalibrationResult(
        cam1_params=cams[0],
        cam0_params=cams[1] if use_cam0 else None,
        theta_offsets=theta_offsets,
        residual=round(avg_residual, 2),
        n_observations=len(valid_obs),
        n_constraints=n_constraints,
        success=result.success,
        message=str(result.message),
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def _rotation_to_rodrigues(R):
    """Convert 3x3 rotation matrix to Rodrigues vector (3,). Pure numpy."""
    R = np.asarray(R, dtype=float)
    # Use scipy for this (more robust than hand-rolling)
    from scipy.spatial.transform import Rotation as SciRotation
    return SciRotation.from_matrix(R).as_rotvec()


def _cam_params_to_js(cam_params: Dict) -> Dict:
    """Convert internal camera params (R matrix + t) to JS format (Rodrigues rvec + t)."""
    if not cam_params:
        return {}
    R = np.array(cam_params['R']).reshape(3, 3)
    rvec = _rotation_to_rodrigues(R)
    t = cam_params['t']
    return {
        "fx": cam_params['fx'], "fy": cam_params['fy'],
        "cx": cam_params['cx'], "cy": cam_params['cy'],
        "rx": float(rvec[0]), "ry": float(rvec[1]), "rz": float(rvec[2]),
        "tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
    }


def save_calibration(result: CalibrationResult, path: Path = OUTPUT_PATH):
    """Save calibration to JSON in format compatible with JS frontend."""
    # Convert theta offsets from radians to degrees for JS
    theta_offsets_deg = [float(np.degrees(o)) for o in result.theta_offsets]

    # Build camera_params dict keyed by cam name for JS
    camera_params = {}
    if result.cam1_params:
        camera_params["cam1"] = _cam_params_to_js(result.cam1_params)
    if result.cam0_params:
        camera_params["cam0"] = _cam_params_to_js(result.cam0_params)

    data = {
        "version": 2,
        "projection_type": "pinhole",
        "dh_theta_offsets_deg": theta_offsets_deg,
        "camera_params": camera_params,
        "dh_d_mm": [d * 1000 for d in DH_D],
        "dh_alpha": list(DH_ALPHA),
        "residual": result.residual,
        "n_observations": result.n_observations,
        "n_constraints": result.n_constraints,
        "success": result.success,
        "timestamp": time.time(),
        # Legacy compat fields for old drawArm fallback
        "links_mm": {
            "base": DH_D[0] * 1000,
            "shoulder": DH_D[2] * 1000,
            "elbow": DH_D[4] * 1000,
            "wrist1": 0.0,
            "wrist2": 0.0,
            "end": DH_D[6] * 1000,
        },
        "joint_viz_offsets": [0, 90, 90, 0, 0, 0],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("3D calibration saved to %s", path)


def load_calibration(path: Path = OUTPUT_PATH) -> Optional[Dict]:
    """Load calibration from JSON."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception as e:
        logger.error("Failed to load calibration: %s", e)
        return None


# ---------------------------------------------------------------------------
# Arm movement helpers
# ---------------------------------------------------------------------------

async def capture_snapshot(cam_id: int) -> Optional[np.ndarray]:
    """Capture a snapshot from camera server."""
    url = CAMERA_URLS.get(cam_id)
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("Camera %d snapshot failed: %d", cam_id, resp.status_code)
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Camera %d capture error: %s", cam_id, e)
        return None


async def get_arm_state() -> Optional[Dict]:
    """Get current arm state."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{ARM_API}/api/state")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.error("Failed to get arm state: %s", e)
    return None


async def move_joint_slowly(joint_id: int, target_angle: float, step_deg: float = MOVE_STEP_DEG):
    """Move a single joint incrementally to target angle."""
    state = await get_arm_state()
    if not state:
        return False
    
    current = state.get('joints', [0]*6)
    if joint_id >= len(current):
        return False
    
    cur = current[joint_id]
    steps = max(1, int(abs(target_angle - cur) / step_deg))
    
    for s in range(1, steps + 1):
        angle = cur + (target_angle - cur) * s / steps
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.post(
                    f"{ARM_API}/api/command/set-joint",
                    json={"id": joint_id, "angle": round(angle, 1)},
                )
                if resp.status_code != 200:
                    logger.warning("Joint move failed: %s", resp.text)
                    return False
        except Exception as e:
            logger.error("Joint move error: %s", e)
            return False
        await asyncio.sleep(0.15)
    
    return True


async def move_to_home():
    """Move all joints to home [0,0,0,0,0,0]."""
    for jid in range(6):
        await move_joint_slowly(jid, 0.0)
    await asyncio.sleep(SETTLE_TIME)


# ---------------------------------------------------------------------------
# Progressive pose generation
# ---------------------------------------------------------------------------

def generate_round_poses(round_num: int) -> List[List[float]]:
    """
    Generate poses for round N (1-indexed).
    Each round tests each calibration joint at ±(N * ANGLE_INCREMENT).
    """
    angle = round_num * ANGLE_INCREMENT
    poses = []
    for jid in CALIBRATION_JOINTS:
        lo, hi = JOINT_LIMITS.get(jid, (-85, 85))
        for sign in [1, -1]:
            target = sign * angle
            if lo <= target <= hi:
                pose = [0.0] * 6
                pose[jid] = target
                poses.append(pose)
    return poses


def max_rounds() -> int:
    return MAX_ANGLE // ANGLE_INCREMENT


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

async def run_calibration(
    camera_url_override: Optional[str] = None,
    arm_api_override: Optional[str] = None,
) -> CalibrationResult:
    """
    Progressive 3D calibration routine:
    1. Capture home frames from both cameras
    2. Move through progressive poses, detect landmarks
    3. After each round, solve and assess
    4. Stop early if converged
    """
    global ARM_API
    if arm_api_override:
        ARM_API = arm_api_override
    
    detector = ArmDetector()
    observations = []
    image_shapes = {}
    
    logger.info("=== 3D Viz Calibration Starting ===")
    
    # Move to home first
    await move_to_home()
    await asyncio.sleep(SETTLE_TIME)
    
    # Capture home frames from both cameras
    for cam_id in [0, 1]:
        frame = await capture_snapshot(cam_id)
        if frame is not None:
            detector.set_home_frame(cam_id, frame)
            image_shapes[cam_id] = (frame.shape[0], frame.shape[1])
            logger.info("Home frame captured for cam%d: %dx%d", cam_id, frame.shape[1], frame.shape[0])
            
            # Detect base position (center-bottom region for side, center for overhead)
            if cam_id == 1:  # side
                detector.base_positions[cam_id] = (frame.shape[1] // 3, int(frame.shape[0] * 0.85))
            else:  # overhead
                detector.base_positions[cam_id] = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    if not image_shapes:
        return CalibrationResult(
            cam1_params={}, cam0_params=None,
            theta_offsets=[0.0]*7, residual=-1.0,
            n_observations=0, n_constraints=0,
            success=False, message="No cameras available")
    
    # Progressive rounds
    total_rounds = max_rounds()
    prev_residual = float('inf')
    stable_count = 0
    best_result = None
    
    for round_num in range(1, total_rounds + 1):
        poses = generate_round_poses(round_num)
        round_obs = 0
        
        logger.info("--- Round %d/%d: ±%d° (%d poses) ---",
                     round_num, total_rounds, round_num * ANGLE_INCREMENT, len(poses))
        
        for i, pose in enumerate(poses):
            # Move to pose
            move_ok = True
            for jid in range(6):
                if pose[jid] != 0.0:
                    ok = await move_joint_slowly(jid, pose[jid])
                    if not ok:
                        move_ok = False
                        break
            
            if not move_ok:
                logger.warning("Skipping pose %s — move failed", pose)
                await move_to_home()
                await asyncio.sleep(SETTLE_TIME)
                continue
            
            # Wait for settle and verify feedback
            await asyncio.sleep(SETTLE_TIME)
            
            state = await get_arm_state()
            if state:
                actual = state.get('joints', [0]*6)
                # Check if arm actually moved (reject stale feedback)
                max_err = 0
                for jid in range(6):
                    if pose[jid] != 0.0:
                        max_err = max(max_err, abs(actual[jid] - pose[jid]))
                
                if max_err > 15.0:  # arm didn't reach target — skip
                    logger.warning("Pose %s: arm didn't reach (max err %.1f°), skipping", pose, max_err)
                    await move_to_home()
                    await asyncio.sleep(SETTLE_TIME)
                    continue
            
            # Capture from both cameras and detect landmarks
            obs = PoseObservation(joint_angles=pose, timestamp=time.time())
            
            for cam_id in [0, 1]:
                frame = await capture_snapshot(cam_id)
                if frame is not None:
                    landmarks = detector.detect_landmarks(frame, cam_id)
                    if cam_id == 0:
                        obs.cam0_landmarks = landmarks
                    else:
                        obs.cam1_landmarks = landmarks
            
            total_landmarks = len(obs.cam0_landmarks) + len(obs.cam1_landmarks)
            if total_landmarks > 0:
                observations.append(obs)
                round_obs += 1
                logger.info("  Pose %s: %d landmarks (cam0: %s, cam1: %s)",
                           pose, total_landmarks,
                           list(obs.cam0_landmarks.keys()),
                           list(obs.cam1_landmarks.keys()))
            else:
                logger.warning("  Pose %s: no landmarks detected, skipping", pose)
            
            # Return to home
            await move_to_home()
            await asyncio.sleep(0.5)
        
        # Self-assessment after round
        if len(observations) >= 3:
            result = solve_calibration(observations, image_shapes)
            logger.info("Round %d: %d obs, %d constraints, residual=%.1fpx, %s",
                        round_num, result.n_observations, result.n_constraints,
                        result.residual, "converging" if result.success else "not converged")
            
            if result.residual > 0 and result.residual < CONVERGENCE_THRESHOLD:
                if abs(result.residual - prev_residual) < 5.0:
                    stable_count += 1
                else:
                    stable_count = 0
                
                if stable_count >= STABLE_ROUNDS_NEEDED:
                    logger.info("Converged after round %d! Residual: %.1fpx", round_num, result.residual)
                    best_result = result
                    break
            else:
                stable_count = 0
            
            prev_residual = result.residual if result.residual > 0 else prev_residual
            best_result = result
        
        logger.info("Round %d complete: %d new obs, %d total", round_num, round_obs, len(observations))
    
    # Final solve if we haven't converged
    if best_result is None and len(observations) >= 3:
        best_result = solve_calibration(observations, image_shapes)
    
    if best_result is None:
        best_result = CalibrationResult(
            cam1_params={}, cam0_params=None,
            theta_offsets=[0.0]*7, residual=-1.0,
            n_observations=len(observations), n_constraints=0,
            success=False, message="Insufficient observations")
    
    # Save
    if best_result.success:
        save_calibration(best_result)
    
    # Ensure home
    await move_to_home()
    
    logger.info("=== Calibration complete: %.1fpx residual, %d obs, %d constraints ===",
                best_result.residual, best_result.n_observations, best_result.n_constraints)
    
    return best_result
