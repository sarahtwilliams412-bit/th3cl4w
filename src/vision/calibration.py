"""
Independent Camera Calibrator — Calibrates each camera separately.

Each camera gets its own intrinsics, distortion coefficients, and
camera-to-workspace extrinsic transform. No stereo pair needed.

cam0 (front/side): Provides height (Z) information via vertical checkerboard.
cam1 (overhead/top-down): Provides X/Y workspace position via flat checkerboard.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.calibration")

# Default checkerboard: 15/16" = 23.8125mm squares
DEFAULT_SQUARE_SIZE_MM = 23.8
DEFAULT_BOARD_SIZE = (7, 5)  # inner corners (cols, rows)


@dataclass
class CameraCalibration:
    """Calibration data for a single camera."""

    camera_id: str
    image_size: tuple[int, int] = (640, 480)  # (width, height)
    camera_matrix: Optional[np.ndarray] = None  # 3x3 intrinsic matrix
    dist_coeffs: Optional[np.ndarray] = None  # distortion coefficients
    rvecs: list[np.ndarray] = field(default_factory=list)  # rotation vectors per image
    tvecs: list[np.ndarray] = field(default_factory=list)  # translation vectors per image
    reprojection_error: float = -1.0
    # Extrinsic: camera-to-workspace transform (4x4)
    cam_to_workspace: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0]) if self.camera_matrix is not None else 500.0

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1]) if self.camera_matrix is not None else 500.0

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2]) if self.camera_matrix is not None else 320.0

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2]) if self.camera_matrix is not None else 240.0

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image."""
        if not self.is_calibrated:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Convert pixel coordinates to a unit ray in camera frame.

        Returns (3,) normalized direction vector in camera coordinates.
        Camera frame: X right, Y down, Z forward.
        """
        if self.camera_matrix is not None:
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
        else:
            x = (u - 320.0) / 500.0
            y = (v - 240.0) / 500.0
        ray = np.array([x, y, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def pixel_to_workspace(self, u: float, v: float, known_z: float = 0.0) -> Optional[np.ndarray]:
        """Project a pixel onto the workspace plane at known Z height.

        For overhead camera: projects onto table surface (Z=0 by default).
        For front camera: projects onto a vertical plane at known depth.

        Returns (3,) workspace coordinates in mm, or None if no extrinsic.
        """
        if self.cam_to_workspace is None:
            return None

        ray_cam = self.pixel_to_ray(u, v)
        # Transform ray origin and direction to workspace frame
        R = self.cam_to_workspace[:3, :3]
        t = self.cam_to_workspace[:3, 3]

        ray_ws = R @ ray_cam
        origin_ws = t  # camera origin in workspace frame

        # Intersect with Z = known_z plane
        if abs(ray_ws[2]) < 1e-6:
            return None  # ray parallel to plane
        param = (known_z - origin_ws[2]) / ray_ws[2]
        if param < 0:
            return None  # behind camera
        point = origin_ws + param * ray_ws
        return point

    def save(self, path: str):
        """Save calibration to JSON file."""
        data = {
            "camera_id": self.camera_id,
            "image_size": list(self.image_size),
            "reprojection_error": self.reprojection_error,
        }
        if self.camera_matrix is not None:
            data["camera_matrix"] = self.camera_matrix.tolist()
        if self.dist_coeffs is not None:
            data["dist_coeffs"] = self.dist_coeffs.tolist()
        if self.cam_to_workspace is not None:
            data["cam_to_workspace"] = self.cam_to_workspace.tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration for %s to %s", self.camera_id, path)

    @classmethod
    def load(cls, path: str) -> "CameraCalibration":
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cal = cls(
            camera_id=data["camera_id"],
            image_size=tuple(data["image_size"]),
            reprojection_error=data.get("reprojection_error", -1.0),
        )
        if "camera_matrix" in data:
            cal.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        if "dist_coeffs" in data:
            cal.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        if "cam_to_workspace" in data:
            cal.cam_to_workspace = np.array(data["cam_to_workspace"], dtype=np.float64)
        return cal


class IndependentCalibrator:
    """Calibrates cameras independently using checkerboard patterns.

    Each camera is calibrated on its own — intrinsics + distortion from
    multiple checkerboard views, then an extrinsic transform from a
    known checkerboard placement in the workspace.
    """

    def __init__(
        self,
        board_size: tuple[int, int] = DEFAULT_BOARD_SIZE,
        square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    ):
        self.board_size = board_size
        self.square_size_mm = square_size_mm

        # Object points for the checkerboard (Z=0 plane)
        self.obj_points = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
        self.obj_points[:, :2] = (
            np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size_mm
        )

        # Collected calibration images per camera
        self._image_points: dict[str, list[np.ndarray]] = {}
        self._image_sizes: dict[str, tuple[int, int]] = {}

    def find_corners(self, image: np.ndarray, refine: bool = True) -> Optional[np.ndarray]:
        """Find checkerboard corners in an image.

        Returns Nx1x2 corner array or None if not found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found or corners is None:
            return None
        if refine:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def add_calibration_image(self, camera_id: str, image: np.ndarray) -> Optional[np.ndarray]:
        """Add a calibration image for a camera. Returns corners if found."""
        corners = self.find_corners(image)
        if corners is None:
            logger.warning("No checkerboard found for %s", camera_id)
            return None

        if camera_id not in self._image_points:
            self._image_points[camera_id] = []
        self._image_points[camera_id].append(corners)
        h, w = image.shape[:2]
        self._image_sizes[camera_id] = (w, h)
        logger.info(
            "Added calibration image for %s (%d total)",
            camera_id,
            len(self._image_points[camera_id]),
        )
        return corners

    def calibrate_camera(self, camera_id: str, min_images: int = 3) -> Optional[CameraCalibration]:
        """Compute intrinsics and distortion for a camera.

        Needs at least min_images checkerboard images.
        Returns CameraCalibration or None if insufficient data.
        """
        pts = self._image_points.get(camera_id, [])
        if len(pts) < min_images:
            logger.error("Need %d images for %s, have %d", min_images, camera_id, len(pts))
            return None

        image_size = self._image_sizes[camera_id]
        obj_pts = [self.obj_points] * len(pts)

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, pts, image_size, None, None)

        cal = CameraCalibration(
            camera_id=camera_id,
            image_size=image_size,
            camera_matrix=mtx,
            dist_coeffs=dist,
            rvecs=list(rvecs),
            tvecs=list(tvecs),
            reprojection_error=rms,
        )
        logger.info(
            "Calibrated %s: RMS=%.4f, fx=%.1f, fy=%.1f",
            camera_id,
            rms,
            cal.fx,
            cal.fy,
        )
        return cal

    def compute_extrinsic(
        self,
        cal: CameraCalibration,
        image: np.ndarray,
        workspace_origin_on_board: np.ndarray = None,
        board_to_workspace: np.ndarray = None,
    ) -> Optional[np.ndarray]:
        """Compute camera-to-workspace transform from a single checkerboard image.

        The checkerboard defines a local coordinate system. You can provide:
        - board_to_workspace: 4x4 transform from board frame to workspace frame
          (if the board is at a known position/orientation in the workspace)

        If neither is provided, the board frame IS the workspace frame.

        Returns 4x4 cam_to_workspace transform, also stored in cal.cam_to_workspace.
        """
        if not cal.is_calibrated:
            return None

        corners = self.find_corners(image)
        if corners is None:
            logger.warning("No checkerboard found for extrinsic computation")
            return None

        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            cal.camera_matrix,
            cal.dist_coeffs,
        )
        if not ok:
            return None

        # Camera-to-board transform
        R_cam_to_board, _ = cv2.Rodrigues(rvec)
        T_cam_to_board = np.eye(4, dtype=np.float64)
        T_cam_to_board[:3, :3] = R_cam_to_board
        T_cam_to_board[:3, 3] = tvec.flatten()

        # Board-to-workspace (identity if not specified)
        if board_to_workspace is None:
            board_to_workspace = np.eye(4, dtype=np.float64)

        # cam_to_workspace = board_to_workspace @ inv(T_cam_to_board)
        # Actually: T_cam_to_board maps board points to camera frame
        # So cam_to_workspace = board_to_workspace @ inv(T_cam_to_board) would be wrong
        # T_cam_to_board: P_camera = R * P_board + t
        # We want: P_workspace = board_to_workspace @ P_board
        # And: P_board = R^T * (P_camera - t)
        # So: P_workspace = board_to_workspace @ [R^T | -R^T*t] @ P_camera_hom
        T_board_from_cam = np.eye(4, dtype=np.float64)
        T_board_from_cam[:3, :3] = R_cam_to_board.T
        T_board_from_cam[:3, 3] = -R_cam_to_board.T @ tvec.flatten()

        cam_to_workspace = board_to_workspace @ T_board_from_cam

        # Actually we want the inverse direction for pixel_to_workspace:
        # We need to know where the camera is in workspace frame.
        # cam_to_workspace here transforms camera-frame points to workspace.
        # That's what we want.
        cal.cam_to_workspace = cam_to_workspace
        logger.info(
            "Computed extrinsic for %s, camera at workspace pos: %s",
            cal.camera_id,
            [round(float(x), 1) for x in cam_to_workspace[:3, 3]],
        )
        return cam_to_workspace

    def image_count(self, camera_id: str) -> int:
        return len(self._image_points.get(camera_id, []))
