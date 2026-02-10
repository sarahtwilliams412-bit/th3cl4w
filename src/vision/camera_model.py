"""Camera model with calibrated extrinsics for pixel↔world transforms.

Supports three camera types:
  - Static cameras (cam0 overhead, cam2 side): fixed extrinsics loaded from file
  - Arm-mounted camera (cam1): hand-eye calibration — camera pose relative to
    end-effector is fixed, world pose = FK(joints) × T_ee_cam
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"

# Camera roles
ROLE_OVERHEAD = "overhead"
ROLE_ARM = "arm-mounted"
ROLE_SIDE = "side-view"

# Map camera IDs to roles
CAMERA_ROLES = {
    0: ROLE_OVERHEAD,
    1: ROLE_ARM,
    2: ROLE_SIDE,
}


def load_intrinsics(camera_id: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load intrinsics (K, dist) for a camera from camera_intrinsics.json.

    Returns (K, dist) or (None, None) if not found.
    """
    intrinsics_path = CALIBRATION_DIR / "camera_intrinsics.json"
    if not intrinsics_path.exists():
        return None, None
    try:
        data = json.loads(intrinsics_path.read_text())
        # Try multiple formats:
        # 1. "cameras" wrapper: {"cameras": {"cam0": {...}, ...}}
        # 2. camera_index map: {"camera_index": {"0": "cam0_overhead"}, "cam0_overhead": {...}}
        # 3. Direct top-level: {"cam0_overhead": {...}}
        cam_data = None
        cameras = data.get("cameras", {})
        if cameras:
            key = f"cam{camera_id}"
            if key in cameras:
                cam_data = cameras[key]
        if cam_data is None:
            index = data.get("camera_index", {})
            key = index.get(str(camera_id))
            if key and key in data:
                cam_data = data[key]
        if cam_data is None:
            for k in data:
                if k.startswith(f"cam{camera_id}_"):
                    cam_data = data[k]
                    break
        if cam_data is None:
            return None, None
        cm = cam_data["camera_matrix"]
        K = np.array([
            [cm["fx"], 0.0, cm["cx"]],
            [0.0, cm["fy"], cm["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dc = cam_data.get("distortion_coefficients", {})
        dist = np.array([
            dc.get("k1", 0.0), dc.get("k2", 0.0),
            dc.get("p1", 0.0), dc.get("p2", 0.0),
            dc.get("k3", 0.0),
        ], dtype=np.float64)
        return K, dist
    except Exception as e:
        logger.warning("Failed to load intrinsics for cam%d: %s", camera_id, e)
        return None, None


class CameraModel:
    """Calibrated camera with intrinsics + extrinsics for 3D↔2D projection.

    For static cameras (overhead, side), extrinsics are loaded from file.
    For arm-mounted camera, use set_hand_eye_transform() and update pose
    via update_from_fk() each time the arm moves.
    """

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.role = CAMERA_ROLES.get(camera_id, "unknown")
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.camera_position: Optional[np.ndarray] = None
        self._loaded = False

        # Hand-eye calibration (arm-mounted camera only)
        # T_ee_cam: 4x4 transform from end-effector frame to camera frame
        self._T_ee_cam: Optional[np.ndarray] = None
        self._hand_eye_loaded = False

    def load(self, path: Optional[str] = None) -> bool:
        """Load calibration from JSON file.

        For static cameras, loads full extrinsics.
        For arm-mounted camera, loads hand-eye transform if extrinsics file
        doesn't exist. Also loads intrinsics from camera_intrinsics.json.
        """
        # Always try to load intrinsics from the shared file
        K_intr, dist_intr = load_intrinsics(self.camera_id)

        if path is None:
            path = str(CALIBRATION_DIR / f"camera{self.camera_id}_extrinsics.json")

        try:
            with open(path) as f:
                data = json.load(f)

            self.K = np.array(data["K"], dtype=np.float64)
            self.dist = np.array(data["dist"], dtype=np.float64)
            self.rvec = np.array(data["rvec"], dtype=np.float64).reshape(3, 1)
            self.tvec = np.array(data["tvec"], dtype=np.float64).reshape(3, 1)
            self.R = np.array(data["R"], dtype=np.float64)
            self.camera_position = np.array(
                data.get("camera_position_world_m", [0, 0, 0]), dtype=np.float64
            )
            self._loaded = True
            logger.info(
                "Camera %d (%s) calibration loaded: "
                "reproj=%spx, pos=(%.2f, %.2f, %.2f)m",
                self.camera_id, self.role,
                data.get("reprojection_error_mean_px", "?"),
                self.camera_position[0], self.camera_position[1], self.camera_position[2],
            )
            return True
        except FileNotFoundError:
            # For arm camera, try loading hand-eye transform instead
            if self.role == ROLE_ARM:
                he_loaded = self._load_hand_eye()
                if he_loaded and K_intr is not None:
                    self.K = K_intr
                    self.dist = dist_intr
                    logger.info(
                        "Camera %d (%s): hand-eye loaded, intrinsics from shared file",
                        self.camera_id, self.role,
                    )
                    return True
            # For other cameras, just load intrinsics
            if K_intr is not None:
                self.K = K_intr
                self.dist = dist_intr
                logger.info(
                    "Camera %d (%s): intrinsics loaded (no extrinsics yet)",
                    self.camera_id, self.role,
                )
            return False
        except Exception as e:
            logger.warning("Failed to load camera %d calibration: %s", self.camera_id, e)
            if K_intr is not None:
                self.K = K_intr
                self.dist = dist_intr
            return False

    def _load_hand_eye(self) -> bool:
        """Load hand-eye transform from file."""
        path = CALIBRATION_DIR / f"camera{self.camera_id}_hand_eye.json"
        try:
            with open(path) as f:
                data = json.load(f)
            self._T_ee_cam = np.array(data["T_ee_cam"], dtype=np.float64)
            self._hand_eye_loaded = True
            logger.info("Camera %d hand-eye transform loaded", self.camera_id)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning("Failed to load hand-eye for cam%d: %s", self.camera_id, e)
            return False

    @property
    def is_calibrated(self) -> bool:
        """True if we can do world↔pixel transforms."""
        return self._loaded

    @property
    def is_arm_mounted(self) -> bool:
        return self.role == ROLE_ARM

    @property
    def has_hand_eye(self) -> bool:
        return self._hand_eye_loaded

    @property
    def has_intrinsics(self) -> bool:
        return self.K is not None

    def set_hand_eye_transform(self, T_ee_cam: np.ndarray, save: bool = True):
        """Set the hand-eye calibration transform.

        Args:
            T_ee_cam: 4x4 transform from end-effector frame to camera frame.
                      Maps points in EE frame to camera frame.
            save: If True, persist to calibration_results/
        """
        self._T_ee_cam = np.array(T_ee_cam, dtype=np.float64)
        self._hand_eye_loaded = True
        if save:
            self._save_hand_eye()

    def _save_hand_eye(self):
        """Persist hand-eye transform to JSON."""
        if self._T_ee_cam is None:
            return
        path = CALIBRATION_DIR / f"camera{self.camera_id}_hand_eye.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_id": self.camera_id,
            "role": self.role,
            "T_ee_cam": self._T_ee_cam.tolist(),
            "description": "Transform from end-effector frame to camera frame. "
                           "World pose = T_world_ee @ inv(T_ee_cam)",
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved hand-eye transform for cam%d", self.camera_id)

    def update_from_fk(self, T_world_ee: np.ndarray):
        """Update camera extrinsics from current arm FK pose.

        For arm-mounted camera only. Call this whenever the arm moves.

        Args:
            T_world_ee: 4x4 transform from world frame to end-effector frame
                        (i.e., the FK result — EE pose in world coordinates).
        """
        if not self._hand_eye_loaded or self._T_ee_cam is None:
            raise RuntimeError("Hand-eye transform not loaded for cam%d" % self.camera_id)
        if self.K is None:
            raise RuntimeError("Intrinsics not loaded for cam%d" % self.camera_id)

        # T_world_cam = T_world_ee @ inv(T_ee_cam)
        # T_ee_cam maps EE→cam, so inv maps cam→EE, then T_world_ee maps EE→world
        # Actually: T_world_cam = T_world_ee @ T_ee_cam
        # if T_ee_cam is defined as the camera pose IN the EE frame
        # Let's be precise:
        #   T_ee_cam: transforms points from camera frame to EE frame
        #   T_world_ee: transforms points from EE frame to world frame
        #   T_world_cam = T_world_ee @ T_ee_cam: transforms camera-frame points to world
        T_world_cam = T_world_ee @ self._T_ee_cam

        # Extract R, t for OpenCV convention:
        # OpenCV rvec/tvec: world point → camera point
        # T_world_cam maps camera→world, so T_cam_world = inv(T_world_cam)
        T_cam_world = np.linalg.inv(T_world_cam)
        R = T_cam_world[:3, :3]
        t = T_cam_world[:3, 3]

        self.R = R
        self.rvec, _ = cv2.Rodrigues(R)
        self.tvec = t.reshape(3, 1)
        self.camera_position = T_world_cam[:3, 3].copy()
        self._loaded = True

    def world_to_pixel(self, point_3d: np.ndarray) -> tuple[float, float]:
        """Project a 3D world point to 2D pixel coordinates.

        Args:
            point_3d: (3,) array in world frame (meters, Z=up)

        Returns:
            (u, v) pixel coordinates
        """
        if not self._loaded:
            raise RuntimeError("Camera not calibrated")

        pts = np.array([point_3d], dtype=np.float64)
        projected, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        u, v = projected[0, 0]
        return float(u), float(v)

    def pixel_to_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel to a 3D ray in world frame.

        Returns:
            (origin, direction) — ray origin (camera pos) and unit direction vector
        """
        if not self._loaded:
            raise RuntimeError("Camera not calibrated")

        # Undistort the pixel
        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        u_ud, v_ud = undistorted[0, 0]

        # Pixel to normalized camera coordinates
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_cam = (u_ud - cx) / fx
        y_cam = (v_ud - cy) / fy
        z_cam = 1.0

        # Camera frame direction to world frame
        dir_cam = np.array([x_cam, y_cam, z_cam])
        dir_world = self.R.T @ dir_cam
        dir_world = dir_world / np.linalg.norm(dir_world)

        return self.camera_position.copy(), dir_world

    def pixel_to_world_at_z(self, u: float, v: float, z: float = 0.0) -> Optional[np.ndarray]:
        """Back-project a pixel to a 3D world point at a given Z height.

        Args:
            u, v: pixel coordinates
            z: world Z coordinate (meters, Z=up)

        Returns:
            (3,) world point or None if ray is parallel to Z plane
        """
        origin, direction = self.pixel_to_ray(u, v)

        if abs(direction[2]) < 1e-8:
            return None
        t = (z - origin[2]) / direction[2]
        if t < 0:
            return None
        return origin + t * direction

    def pixel_to_world_at_distance(self, u: float, v: float, distance: float) -> np.ndarray:
        """Back-project a pixel to a 3D point at a given distance from camera.

        Args:
            u, v: pixel coordinates
            distance: distance from camera in meters

        Returns:
            (3,) world point
        """
        origin, direction = self.pixel_to_ray(u, v)
        return origin + distance * direction
