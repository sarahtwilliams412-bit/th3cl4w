"""Fixed camera extrinsics calibration via calibrateRobotWorldHandEye().

Solves the AX=ZB problem for cameras that are static in the world frame
(overhead cam0, side cam2) while the robot arm holds a ChArUco board.

The solver finds both:
  - T_base_to_world (robot base in world/camera frame)
  - T_gripper_to_cam (only relevant for eye-in-hand, not used here)

For fixed cameras, what we actually need is T_cam_to_base (or equivalently
T_world_to_cam), which tells us where each fixed camera sits relative to
the robot's base frame.

Two approaches implemented:
1. OpenCV calibrateRobotWorldHandEye() — newest API, solves AX=ZB jointly
2. Global anchor fallback — simpler single-frame method per Gemini 3 Pro

Usage:
    calibrator = FixedCameraCalibrator(camera_id=0)
    # Mount ChArUco board on gripper. Move arm to diverse poses.
    # At each pose:
    calibrator.add_observation(T_base_ee, frame)
    # Then solve:
    T_cam_base = calibrator.solve()
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco_detector import ChArUcoDetector

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"


@dataclass
class FixedCameraResult:
    """Result of fixed camera extrinsics calibration."""

    camera_id: int
    T_cam_base: np.ndarray  # 4x4 camera-to-robot-base transform
    T_base_world: np.ndarray  # 4x4 robot-base-to-world transform
    method: str
    num_observations: int
    rotation_det: float
    reprojection_error: Optional[float] = None


def _validate_rotation(R: np.ndarray) -> tuple[float, float, bool]:
    """Validate a rotation matrix."""
    det = float(np.linalg.det(R))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3), "fro"))
    is_valid = abs(det - 1.0) < 0.01 and orth_err < 0.01
    return det, orth_err, is_valid


class FixedCameraCalibrator:
    """Calibrates fixed cameras using robot arm holding a ChArUco board.

    The ChArUco board is mounted on (or held by) the gripper. As the arm
    moves through diverse poses, the fixed camera observes the board from
    different positions. Combined with FK-derived gripper poses, this
    provides enough information to solve for the camera's extrinsic
    transform relative to the robot base.

    Primary method: OpenCV's calibrateRobotWorldHandEye() which solves
    the AX=ZB formulation jointly.

    Fallback: Global anchor method — use a single best observation to
    chain transforms: T_cam_base = T_cam_board @ inv(T_base_board)
    where T_base_board = T_base_ee @ T_ee_board (board on gripper).
    """

    def __init__(
        self,
        camera_id: int = 0,
        charuco_detector: Optional[ChArUcoDetector] = None,
    ):
        self.camera_id = camera_id
        self.charuco = charuco_detector or ChArUcoDetector()

        # Collected observations
        self._T_base_ee_list: list[np.ndarray] = []  # FK poses (4x4)
        self._R_board_cam_list: list[np.ndarray] = []  # board-in-camera rotations
        self._t_board_cam_list: list[np.ndarray] = []  # board-in-camera translations

        # Intrinsics
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self._load_intrinsics()

        # Board-to-gripper transform (if board is rigidly attached to gripper)
        # Identity assumes board origin = gripper origin (adjust as needed)
        self.T_ee_board: np.ndarray = np.eye(4, dtype=np.float64)

    def _load_intrinsics(self):
        """Load camera intrinsics from shared file."""
        from .camera_model import load_intrinsics

        self.K, self.dist = load_intrinsics(self.camera_id)
        if self.K is not None:
            logger.info("FixedCam: loaded intrinsics for cam%d", self.camera_id)

    def set_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        """Manually set camera intrinsics."""
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)

    def set_board_to_gripper(self, T_ee_board: np.ndarray):
        """Set the transform from gripper frame to board frame.

        Required if the ChArUco board is not centered on the gripper.
        """
        self.T_ee_board = np.array(T_ee_board, dtype=np.float64)

    def add_observation(
        self,
        T_base_ee: np.ndarray,
        frame: np.ndarray,
    ) -> bool:
        """Add a calibration observation.

        Args:
            T_base_ee: 4x4 FK transform (end-effector in base frame)
            frame: BGR image from the fixed camera at this pose

        Returns:
            True if ChArUco board was detected and observation added.
        """
        if self.K is None:
            logger.error("No intrinsics for cam%d", self.camera_id)
            return False

        ok, rvec, tvec = self.charuco.estimate_pose(frame, self.K, self.dist)
        if not ok:
            logger.warning("FixedCam%d: no ChArUco board detected", self.camera_id)
            return False

        R_board_cam, _ = cv2.Rodrigues(rvec)

        self._T_base_ee_list.append(np.array(T_base_ee, dtype=np.float64))
        self._R_board_cam_list.append(R_board_cam)
        self._t_board_cam_list.append(tvec.reshape(3, 1))

        logger.info(
            "FixedCam%d: observation %d added",
            self.camera_id,
            len(self._T_base_ee_list),
        )
        return True

    @property
    def num_observations(self) -> int:
        return len(self._T_base_ee_list)

    def solve_robot_world(self) -> Optional[FixedCameraResult]:
        """Solve using OpenCV's calibrateRobotWorldHandEye().

        This solves the AX=ZB problem:
          A = T_base_ee (robot FK)
          B = T_board_cam (from ChArUco detection)
          X = T_base_world (robot base to world)
          Z = T_gripper_cam (gripper to camera — for eye-in-hand sub-problem)

        For our fixed camera case, we extract T_cam_base from the result.
        """
        n = self.num_observations
        if n < 3:
            logger.error("Need at least 3 observations, have %d", n)
            return None

        # Prepare inputs
        R_world2cam = []
        t_world2cam = []
        R_base2gripper = []
        t_base2gripper = []

        for i in range(n):
            # Board-in-camera (what the camera sees)
            R_world2cam.append(self._R_board_cam_list[i])
            t_world2cam.append(self._t_board_cam_list[i])

            # Gripper-in-base with board offset
            T_base_board = self._T_base_ee_list[i] @ self.T_ee_board
            R_base2gripper.append(T_base_board[:3, :3])
            t_base2gripper.append(T_base_board[:3, 3].reshape(3, 1))

        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_world2cam,
            t_world2cam,
            R_base2gripper,
            t_base2gripper,
        )

        # Build T_base_world
        T_base_world = np.eye(4, dtype=np.float64)
        T_base_world[:3, :3] = R_base2world
        T_base_world[:3, 3] = t_base2world.flatten()

        # T_cam_base: we need to derive this
        # The "world" in this solver is the camera's world, so
        # T_base_world here is really T_base_cam (base in camera frame)
        # T_cam_base = inv(T_base_cam)
        T_cam_base = np.linalg.inv(T_base_world)

        det, orth_err, is_valid = _validate_rotation(T_cam_base[:3, :3])
        if not is_valid:
            logger.warning(
                "FixedCam%d: rotation failed validation (det=%.4f, orth=%.4f)",
                self.camera_id,
                det,
                orth_err,
            )

        logger.info(
            "FixedCam%d solved (RobotWorldHandEye): "
            "T_cam_base t=(%.4f, %.4f, %.4f)m, det(R)=%.6f",
            self.camera_id,
            T_cam_base[0, 3],
            T_cam_base[1, 3],
            T_cam_base[2, 3],
            det,
        )

        return FixedCameraResult(
            camera_id=self.camera_id,
            T_cam_base=T_cam_base,
            T_base_world=T_base_world,
            method="calibrateRobotWorldHandEye",
            num_observations=n,
            rotation_det=det,
        )

    def solve_global_anchor(self) -> Optional[FixedCameraResult]:
        """Solve using the global anchor method (Gemini 3 Pro approach).

        Simpler fallback: for each observation, compute T_cam_base via
        the transform chain, then average across all observations.

        T_cam_base = T_cam_board @ inv(T_board_base)
        T_board_base = T_ee_base^(-1) @ T_ee_board^(-1)  [board on gripper]

        Actually: T_base_board = T_base_ee @ T_ee_board
        So: T_cam_base = T_cam_board @ inv(T_base_board)
        """
        n = self.num_observations
        if n < 1:
            logger.error("Need at least 1 observation, have %d", n)
            return None

        cam_base_candidates = []

        for i in range(n):
            # T_cam_board from detection
            T_cam_board = np.eye(4, dtype=np.float64)
            T_cam_board[:3, :3] = self._R_board_cam_list[i]
            T_cam_board[:3, 3] = self._t_board_cam_list[i].flatten()

            # T_base_board from FK + board offset
            T_base_board = self._T_base_ee_list[i] @ self.T_ee_board

            # T_cam_base = T_cam_board @ inv(T_base_board)
            T_cam_base = T_cam_board @ np.linalg.inv(T_base_board)
            cam_base_candidates.append(T_cam_base)

        # Average translations, use first rotation as reference
        # (proper rotation averaging would use quaternion mean, but for
        # a sanity check / fallback this is sufficient)
        t_avg = np.mean([T[:3, 3] for T in cam_base_candidates], axis=0)

        # Use the median observation (least affected by outliers)
        dists = [np.linalg.norm(T[:3, 3] - t_avg) for T in cam_base_candidates]
        median_idx = int(np.argmin(dists))
        T_cam_base = cam_base_candidates[median_idx].copy()

        # Use averaged translation with median rotation
        T_cam_base[:3, 3] = t_avg

        det, orth_err, is_valid = _validate_rotation(T_cam_base[:3, :3])

        logger.info(
            "FixedCam%d solved (GlobalAnchor, %d candidates): "
            "T_cam_base t=(%.4f, %.4f, %.4f)m, det(R)=%.6f",
            self.camera_id,
            n,
            T_cam_base[0, 3],
            T_cam_base[1, 3],
            T_cam_base[2, 3],
            det,
        )

        # Report consistency across observations
        t_spread = np.ptp([T[:3, 3] for T in cam_base_candidates], axis=0)
        logger.info(
            "  Translation spread across observations: " "dx=%.1fmm, dy=%.1fmm, dz=%.1fmm",
            t_spread[0] * 1000,
            t_spread[1] * 1000,
            t_spread[2] * 1000,
        )

        return FixedCameraResult(
            camera_id=self.camera_id,
            T_cam_base=T_cam_base,
            T_base_world=np.linalg.inv(T_cam_base),
            method="global_anchor",
            num_observations=n,
            rotation_det=det,
        )

    def solve(self) -> Optional[FixedCameraResult]:
        """Solve with primary method, falling back to global anchor.

        Uses calibrateRobotWorldHandEye() if >= 3 observations,
        otherwise falls back to global anchor method.
        """
        if self.num_observations >= 3:
            result = self.solve_robot_world()
            if result is not None:
                return result
            logger.warning("calibrateRobotWorldHandEye failed, trying global anchor")

        return self.solve_global_anchor()

    def solve_and_save(self) -> Optional[FixedCameraResult]:
        """Solve and save result to JSON."""
        result = self.solve()
        if result is None:
            return None

        path = CALIBRATION_DIR / f"camera{self.camera_id}_fixed_extrinsics.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "camera_id": result.camera_id,
            "T_cam_base": result.T_cam_base.tolist(),
            "T_base_world": result.T_base_world.tolist(),
            "method": result.method,
            "num_observations": result.num_observations,
            "rotation_det": result.rotation_det,
            "description": (
                "Fixed camera extrinsics. T_cam_base transforms points "
                "from robot base frame to camera frame. "
                f"Solved using {result.method} with {result.num_observations} "
                "observations of ChArUco board on gripper."
            ),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved fixed camera extrinsics to %s", path)
        return result

    def clear(self):
        """Clear all collected observations."""
        self._T_base_ee_list.clear()
        self._R_board_cam_list.clear()
        self._t_board_cam_list.clear()
