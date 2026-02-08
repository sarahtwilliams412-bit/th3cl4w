"""
Stereo camera calibration using checkerboard patterns.

Captures image pairs, finds chessboard corners, calibrates each camera
individually, then performs stereo calibration to compute rectification maps.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.calibration")


class StereoCalibrator:
    """Stereo camera calibration and rectification."""

    def __init__(
        self,
        board_size: tuple[int, int] = (9, 6),
        square_size: float = 25.0,  # mm
        image_size: tuple[int, int] = (640, 480),
    ):
        self.board_size = board_size
        self.square_size = square_size
        self.image_size = image_size  # (width, height)

        # Calibration results
        self.camera_matrix_left: Optional[np.ndarray] = None
        self.dist_coeffs_left: Optional[np.ndarray] = None
        self.camera_matrix_right: Optional[np.ndarray] = None
        self.dist_coeffs_right: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None  # rotation between cameras
        self.T: Optional[np.ndarray] = None  # translation between cameras
        self.Q: Optional[np.ndarray] = None  # disparity-to-depth mapping matrix

        # Rectification maps
        self.map_left_x: Optional[np.ndarray] = None
        self.map_left_y: Optional[np.ndarray] = None
        self.map_right_x: Optional[np.ndarray] = None
        self.map_right_y: Optional[np.ndarray] = None

        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def _make_object_points(self) -> np.ndarray:
        """Generate 3D object points for the checkerboard."""
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.board_size[0], 0 : self.board_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def find_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find checkerboard corners in a grayscale or BGR image.

        Returns refined corner positions or None if not found.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def calibrate(
        self,
        image_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Run stereo calibration from a list of (left, right) image pairs.

        Returns the stereo reprojection error.
        Raises ValueError if fewer than 3 valid pairs are found.
        """
        objp = self._make_object_points()
        obj_points: list[np.ndarray] = []
        img_points_left: list[np.ndarray] = []
        img_points_right: list[np.ndarray] = []

        for i, (left, right) in enumerate(image_pairs):
            corners_l = self.find_corners(left)
            corners_r = self.find_corners(right)
            if corners_l is not None and corners_r is not None:
                obj_points.append(objp)
                img_points_left.append(corners_l)
                img_points_right.append(corners_r)
                logger.info("Pair %d: corners found", i)
            else:
                logger.warning(
                    "Pair %d: corners not found (L=%s, R=%s)",
                    i,
                    corners_l is not None,
                    corners_r is not None,
                )

        if len(obj_points) < 3:
            raise ValueError(f"Need at least 3 valid image pairs, got {len(obj_points)}")

        logger.info("Calibrating with %d valid pairs...", len(obj_points))
        h, w = self.image_size[1], self.image_size[0]

        # Individual camera calibration
        ret_l, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            obj_points, img_points_left, (w, h), None, None
        )
        ret_r, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            obj_points, img_points_right, (w, h), None, None
        )
        logger.info("Individual calibration RMS: left=%.4f, right=%.4f", ret_l, ret_r)

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        (
            rms,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.R,
            self.T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            (w, h),
            criteria=criteria,
            flags=flags,
        )

        logger.info("Stereo calibration RMS error: %.4f", rms)

        # Compute rectification maps
        self._compute_rectification_maps()
        self._calibrated = True
        return rms

    def _compute_rectification_maps(self):
        """Compute rectification transforms and projection matrices."""
        h, w = self.image_size[1], self.image_size[0]

        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            (w, h),
            self.R,
            self.T,
            alpha=0,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
        )

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply rectification maps to a stereo image pair.

        Raises RuntimeError if not calibrated.
        """
        if not self._calibrated:
            raise RuntimeError("Calibration required before rectification")

        rect_left = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def save(self, path: str | Path):
        """Save calibration data to a .npz file."""
        if not self._calibrated:
            raise RuntimeError("Nothing to save — not calibrated")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(path),
            camera_matrix_left=self.camera_matrix_left,
            dist_coeffs_left=self.dist_coeffs_left,
            camera_matrix_right=self.camera_matrix_right,
            dist_coeffs_right=self.dist_coeffs_right,
            R=self.R,
            T=self.T,
            Q=self.Q,
            map_left_x=self.map_left_x,
            map_left_y=self.map_left_y,
            map_right_x=self.map_right_x,
            map_right_y=self.map_right_y,
            image_size=np.array(self.image_size),
            board_size=np.array(self.board_size),
            square_size=np.array(self.square_size),
        )
        logger.info("Calibration saved to %s", path)

    def load(self, path: str | Path):
        """Load calibration data from a .npz file."""
        path = Path(path)
        data = np.load(str(path))

        self.camera_matrix_left = data["camera_matrix_left"]
        self.dist_coeffs_left = data["dist_coeffs_left"]
        self.camera_matrix_right = data["camera_matrix_right"]
        self.dist_coeffs_right = data["dist_coeffs_right"]
        self.R = data["R"]
        self.T = data["T"]
        self.Q = data["Q"]
        self.map_left_x = data["map_left_x"]
        self.map_left_y = data["map_left_y"]
        self.map_right_x = data["map_right_x"]
        self.map_right_y = data["map_right_y"]
        self.image_size = tuple(data["image_size"].tolist())
        self.board_size = tuple(data["board_size"].tolist())
        self.square_size = float(data["square_size"])
        self._calibrated = True
        logger.info("Calibration loaded from %s", path)
