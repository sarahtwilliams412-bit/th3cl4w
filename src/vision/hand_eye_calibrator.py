"""Hand-eye calibration for the arm-mounted camera (cam1).

The arm camera moves with the end-effector. Its pose relative to the EE is fixed:
    T_world_cam = T_world_ee @ T_ee_cam

This module solves for T_ee_cam given multiple (T_world_ee, T_world_cam) pairs
collected by moving the arm to different poses while observing a fixed calibration
target (e.g., checkerboard on the table).

Two approaches:
1. OpenCV's calibrateHandEye (AX=XB solver) — requires 3+ pose pairs
2. Direct optimization: minimize reprojection error of known world points

Usage:
    calibrator = HandEyeCalibrator(camera_id=1)
    # Move arm to different poses, at each pose:
    calibrator.add_observation(T_world_ee, frame, ...)
    # Then solve:
    T_ee_cam = calibrator.solve()
    # Apply to CameraModel:
    camera_model.set_hand_eye_transform(T_ee_cam)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"


class HandEyeCalibrator:
    """Solves hand-eye calibration for arm-mounted camera."""

    def __init__(
        self,
        camera_id: int = 1,
        board_size: tuple[int, int] = (7, 5),
        square_size_mm: float = 23.8,
    ):
        self.camera_id = camera_id
        self.board_size = board_size
        self.square_size_mm = square_size_mm

        # Object points on checkerboard (Z=0 plane, in mm then converted to meters)
        self.obj_points = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
        self.obj_points[:, :2] = (
            np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
            * square_size_mm / 1000.0  # convert to meters
        )

        # Collected observations
        self._T_world_ee_list: list[np.ndarray] = []  # FK poses (4x4)
        self._R_cam_board_list: list[np.ndarray] = []  # camera-to-board rotations
        self._t_cam_board_list: list[np.ndarray] = []  # camera-to-board translations

        # Intrinsics (loaded from shared file)
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self._load_intrinsics()

        # Board pose in world frame (fixed — set once when board is placed)
        self._T_world_board: Optional[np.ndarray] = None

    def _load_intrinsics(self):
        """Load camera intrinsics from shared file."""
        from .camera_model import load_intrinsics
        self.K, self.dist = load_intrinsics(self.camera_id)
        if self.K is not None:
            logger.info("HandEye: loaded intrinsics for cam%d", self.camera_id)

    def set_board_pose(self, T_world_board: np.ndarray):
        """Set the checkerboard's pose in world frame (4x4).

        If the board is flat on the table at a known position, set this once.
        If not set, the solver will estimate it jointly.
        """
        self._T_world_board = np.array(T_world_board, dtype=np.float64)

    def add_observation(
        self,
        T_world_ee: np.ndarray,
        frame: np.ndarray,
    ) -> bool:
        """Add a calibration observation.

        Args:
            T_world_ee: 4x4 FK transform (end-effector pose in world frame)
            frame: BGR image from arm camera at this pose

        Returns:
            True if checkerboard was found and observation added.
        """
        if self.K is None:
            logger.error("No intrinsics loaded for cam%d", self.camera_id)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if not found or corners is None:
            logger.warning("HandEye: no checkerboard found in frame")
            return False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Solve PnP: board → camera
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points, corners, self.K, self.dist
        )
        if not ok:
            logger.warning("HandEye: solvePnP failed")
            return False

        R_cam_board, _ = cv2.Rodrigues(rvec)

        self._T_world_ee_list.append(np.array(T_world_ee, dtype=np.float64))
        self._R_cam_board_list.append(R_cam_board)
        self._t_cam_board_list.append(tvec)

        logger.info(
            "HandEye: observation %d added (board at t=%.3f, %.3f, %.3f in cam frame)",
            len(self._T_world_ee_list),
            tvec[0, 0], tvec[1, 0], tvec[2, 0],
        )
        return True

    @property
    def num_observations(self) -> int:
        return len(self._T_world_ee_list)

    def solve(self, method: int = cv2.CALIB_HAND_EYE_TSAI) -> Optional[np.ndarray]:
        """Solve for T_ee_cam using OpenCV's calibrateHandEye.

        This solves AX = XB where:
          A = relative gripper (EE) motion between poses
          B = relative camera motion between poses
          X = T_ee_cam (what we want)

        Args:
            method: OpenCV hand-eye method. Options:
                cv2.CALIB_HAND_EYE_TSAI (default)
                cv2.CALIB_HAND_EYE_PARK
                cv2.CALIB_HAND_EYE_HORAUD
                cv2.CALIB_HAND_EYE_DANIILIDIS

        Returns:
            4x4 T_ee_cam transform, or None if insufficient data.
        """
        n = self.num_observations
        if n < 3:
            logger.error("Need at least 3 observations, have %d", n)
            return None

        # OpenCV calibrateHandEye expects:
        #   R_gripper2base, t_gripper2base: list of EE rotations/translations in world frame
        #   R_target2cam, t_target2cam: list of board rotations/translations in camera frame
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for i in range(n):
            T = self._T_world_ee_list[i]
            R_gripper2base.append(T[:3, :3])
            t_gripper2base.append(T[:3, 3].reshape(3, 1))
            R_target2cam.append(self._R_cam_board_list[i])
            t_target2cam.append(self._t_cam_board_list[i])

        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method,
        )

        # Build T_ee_cam (camera pose in EE frame)
        # calibrateHandEye returns cam→EE transform
        T_ee_cam = np.eye(4, dtype=np.float64)
        T_ee_cam[:3, :3] = R_cam2ee
        T_ee_cam[:3, 3] = t_cam2ee.flatten()

        logger.info(
            "HandEye solved: T_ee_cam translation = (%.4f, %.4f, %.4f)m",
            T_ee_cam[0, 3], T_ee_cam[1, 3], T_ee_cam[2, 3],
        )

        return T_ee_cam

    def solve_and_save(self, method: int = cv2.CALIB_HAND_EYE_TSAI) -> Optional[np.ndarray]:
        """Solve and save the result."""
        T_ee_cam = self.solve(method)
        if T_ee_cam is None:
            return None

        path = CALIBRATION_DIR / f"camera{self.camera_id}_hand_eye.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_id": self.camera_id,
            "T_ee_cam": T_ee_cam.tolist(),
            "num_observations": self.num_observations,
            "method": "calibrateHandEye",
            "description": "Transform from end-effector frame to camera frame. "
                           "World pose: T_world_cam = T_world_ee @ T_ee_cam",
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved hand-eye calibration to %s", path)
        return T_ee_cam

    def clear(self):
        """Clear all collected observations."""
        self._T_world_ee_list.clear()
        self._R_cam_board_list.clear()
        self._t_cam_board_list.clear()
