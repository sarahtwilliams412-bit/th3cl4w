"""Camera model with calibrated extrinsics for pixel↔world transforms."""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"


class CameraModel:
    """Calibrated camera with intrinsics + extrinsics for 3D↔2D projection."""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.camera_position: Optional[np.ndarray] = None
        self._loaded = False

    def load(self, path: Optional[str] = None) -> bool:
        """Load calibration from JSON file."""
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
                f"Camera {self.camera_id} calibration loaded: "
                f"reproj={data.get('reprojection_error_mean_px', '?')}px, "
                f"pos=({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f})m"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load camera {self.camera_id} calibration: {e}")
            return False

    @property
    def is_calibrated(self) -> bool:
        return self._loaded

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

    def pixel_to_world_at_z(
        self, u: float, v: float, z: float = 0.0
    ) -> Optional[np.ndarray]:
        """Back-project a pixel to a 3D world point at a given Z height.

        Useful for objects on a known surface (e.g., table at z=0.05m).

        Args:
            u, v: pixel coordinates
            z: world Z coordinate (meters, Z=up)

        Returns:
            (3,) world point or None if ray is parallel to Z plane
        """
        origin, direction = self.pixel_to_ray(u, v)

        # Intersect ray with Z=z plane
        if abs(direction[2]) < 1e-8:
            return None  # Ray parallel to plane

        t = (z - origin[2]) / direction[2]
        if t < 0:
            return None  # Behind camera

        point = origin + t * direction
        return point

    def pixel_to_world_at_distance(
        self, u: float, v: float, distance: float
    ) -> np.ndarray:
        """Back-project a pixel to a 3D point at a given distance from camera.

        Args:
            u, v: pixel coordinates
            distance: distance from camera in meters

        Returns:
            (3,) world point
        """
        origin, direction = self.pixel_to_ray(u, v)
        return origin + distance * direction
