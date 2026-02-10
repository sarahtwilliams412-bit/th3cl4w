"""ChArUco board detection for robust calibration target tracking.

ChArUco boards combine the accuracy of checkerboard corner refinement with
the robustness of ArUco markers. Benefits over plain checkerboards:
- Works with partial occlusion (arm blocking part of the board)
- Unique corner IDs enable unambiguous correspondence
- Handles steep viewing angles better
- No need to detect ALL corners — any subset works

Board specification (matching expert consensus):
  Dictionary: DICT_5X5_100
  Grid: 7 columns x 5 rows (of squares)
  Square size: 30mm
  Marker size: 22mm
  Print on rigid foam board to prevent warping

Usage:
    detector = ChArUcoDetector()
    corners, ids, charuco_corners, charuco_ids = detector.detect(frame)
    obj_pts, img_pts = detector.get_correspondences(frame)
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Board specification — consensus from GPT-5.2, Claude Opus 4.6, Gemini 3 Pro
CHARUCO_DICT = cv2.aruco.DICT_5X5_100
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.030  # 30mm
CHARUCO_MARKER_LENGTH_M = 0.022  # 22mm


class ChArUcoDetector:
    """Detects ChArUco board corners in camera frames.

    Replaces plain checkerboard detection with a more robust approach
    that handles partial occlusion and provides unique corner IDs.
    """

    def __init__(
        self,
        dictionary_id: int = CHARUCO_DICT,
        squares_x: int = CHARUCO_SQUARES_X,
        squares_y: int = CHARUCO_SQUARES_Y,
        square_length: float = CHARUCO_SQUARE_LENGTH_M,
        marker_length: float = CHARUCO_MARKER_LENGTH_M,
    ):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary,
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # Object points for all possible ChArUco corners
        self._all_obj_points = self.board.getChessboardCorners()

    @property
    def num_corners(self) -> int:
        """Total number of ChArUco corners on the board."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def detect(
        self,
        frame: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Detect ChArUco board in a frame.

        Args:
            frame: BGR or grayscale image.
            camera_matrix: 3x3 intrinsic matrix (improves corner refinement).
            dist_coeffs: Distortion coefficients.

        Returns:
            (aruco_corners, aruco_ids, charuco_corners, charuco_ids)
            Any may be None if detection fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Step 1: Detect ArUco markers
        aruco_corners, aruco_ids, rejected = self.aruco_detector.detectMarkers(gray)

        if aruco_ids is None or len(aruco_ids) < 2:
            logger.debug("ChArUco: fewer than 2 ArUco markers detected")
            return aruco_corners, aruco_ids, None, None

        # Step 2: Interpolate ChArUco corners from detected markers
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            aruco_corners,
            aruco_ids,
            gray,
            self.board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )

        if num_corners < 4:
            logger.debug(
                "ChArUco: only %d corners interpolated (need >= 4)",
                num_corners,
            )
            return aruco_corners, aruco_ids, None, None

        logger.debug(
            "ChArUco: detected %d ArUco markers, %d corners",
            len(aruco_ids),
            num_corners,
        )
        return aruco_corners, aruco_ids, charuco_corners, charuco_ids

    def get_correspondences(
        self,
        frame: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get 3D-2D correspondences from a ChArUco detection.

        Returns:
            (object_points, image_points) — Nx3 and Nx2 arrays,
            or (None, None) if detection fails.
        """
        _, _, charuco_corners, charuco_ids = self.detect(frame, camera_matrix, dist_coeffs)
        if charuco_corners is None or charuco_ids is None:
            return None, None

        obj_pts = self._all_obj_points[charuco_ids.flatten()]
        img_pts = charuco_corners.reshape(-1, 2)

        return obj_pts.astype(np.float64), img_pts.astype(np.float64)

    def estimate_pose(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate the board pose in the camera frame via PnP.

        Returns:
            (success, rvec, tvec) where rvec/tvec define the board-to-camera
            transform.
        """
        obj_pts, img_pts = self.get_correspondences(frame, camera_matrix, dist_coeffs)
        if obj_pts is None or len(obj_pts) < 4:
            return False, None, None

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
        if not ok:
            return False, None, None

        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts.reshape(-1, 1, 3),
            img_pts.reshape(-1, 1, 2),
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )

        return True, rvec, tvec

    def compute_reprojection_error(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[float]:
        """Compute mean reprojection error for a detected board.

        Returns mean pixel error, or None if detection fails.
        """
        obj_pts, img_pts = self.get_correspondences(frame, camera_matrix, dist_coeffs)
        if obj_pts is None:
            return None

        projected, _ = cv2.projectPoints(
            obj_pts.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - img_pts, axis=1)
        return float(np.mean(errors))

    def calibrate_camera_intrinsics(
        self,
        frames: list[np.ndarray],
        image_size: tuple[int, int],
        min_corners: int = 6,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        int,
    ]:
        """Calibrate camera intrinsics from multiple ChArUco board views.

        Args:
            frames: List of BGR images containing the ChArUco board.
            image_size: (width, height) of the images.
            min_corners: Minimum corners per frame to include it.

        Returns:
            (camera_matrix, dist_coeffs, rms_error, num_frames_used)
        """
        all_charuco_corners = []
        all_charuco_ids = []

        for i, frame in enumerate(frames):
            _, _, charuco_corners, charuco_ids = self.detect(frame)
            if charuco_corners is not None and len(charuco_corners) >= min_corners:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                logger.debug("Frame %d: insufficient corners, skipping", i)

        n_frames = len(all_charuco_corners)
        if n_frames < 3:
            logger.error(
                "Need at least 3 frames with ChArUco detections, got %d",
                n_frames,
            )
            return None, None, -1.0, 0

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            self.board,
            image_size,
            None,
            None,
        )

        logger.info(
            "ChArUco intrinsic calibration: RMS=%.4f, %d frames used, "
            "fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            rms,
            n_frames,
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
        )

        return camera_matrix, dist_coeffs, rms, n_frames

    def draw_detected(
        self,
        frame: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw detected ChArUco markers and corners on a frame copy."""
        vis = frame.copy()
        aruco_corners, aruco_ids, charuco_corners, charuco_ids = self.detect(
            frame, camera_matrix, dist_coeffs
        )

        if aruco_corners is not None and aruco_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, aruco_corners, aruco_ids)

        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

        return vis
