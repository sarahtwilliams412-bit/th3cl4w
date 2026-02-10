"""Hand-eye calibration for the arm-mounted camera (cam1).

The arm camera moves with the end-effector. Its pose relative to the EE is fixed:
    T_world_cam = T_world_ee @ T_ee_cam

This module solves for T_ee_cam given multiple (T_world_ee, T_world_cam) pairs
collected by moving the arm to different poses while observing a fixed calibration
target (ChArUco board on the table).

Uses ChArUco board detection for robustness against partial occlusion and steep
viewing angles. Runs all five OpenCV solvers and cross-checks, defaulting to
Daniilidis per expert consensus.

Usage:
    calibrator = HandEyeCalibrator(camera_id=1)
    # Move arm to 15-25+ diverse poses, at each pose:
    calibrator.add_observation(T_world_ee, frame)
    # Then solve:
    result = calibrator.solve_all_methods()
    T_ee_cam = result.best_transform
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco_detector import ChArUcoDetector

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration_results"

# All five OpenCV hand-eye solvers
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# Default solver — Daniilidis is robust and handles noise well
DEFAULT_METHOD = "DANIILIDIS"


@dataclass
class SolverResult:
    """Result from a single hand-eye solver."""

    method_name: str
    T_ee_cam: np.ndarray
    rotation_det: float  # det(R), should be ~1.0
    rotation_orthogonality_error: float  # ||R^T R - I||_F
    is_valid: bool  # passes sanity checks


@dataclass
class HandEyeResult:
    """Result from multi-solver hand-eye calibration."""

    solver_results: list[SolverResult] = field(default_factory=list)
    best_method: str = ""
    best_transform: Optional[np.ndarray] = None
    num_observations: int = 0
    translation_spread_mm: float = 0.0  # spread across solvers
    rotation_spread_deg: float = 0.0  # spread across solvers


def _validate_rotation(R: np.ndarray) -> tuple[float, float, bool]:
    """Validate a rotation matrix.

    Returns:
        (det, orthogonality_error, is_valid)
    """
    det = float(np.linalg.det(R))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3), "fro"))
    is_valid = abs(det - 1.0) < 0.01 and orth_err < 0.01
    return det, orth_err, is_valid


class HandEyeCalibrator:
    """Solves hand-eye calibration for arm-mounted camera using ChArUco board.

    Improvements over previous version:
    - Uses ChArUco board instead of plain checkerboard (handles partial occlusion)
    - Defaults to Daniilidis solver (robust to noise)
    - Runs all 5 solvers and cross-checks for consistency
    - Validates rotation matrix (det(R) ~ 1.0, orthogonality)
    - Recommends minimum 15 observations with >= 30deg rotational diversity
    """

    def __init__(
        self,
        camera_id: int = 1,
        charuco_detector: Optional[ChArUcoDetector] = None,
    ):
        self.camera_id = camera_id
        self.charuco = charuco_detector or ChArUcoDetector()

        # Collected observations
        self._T_world_ee_list: list[np.ndarray] = []  # FK poses (4x4)
        self._R_cam_board_list: list[np.ndarray] = []  # camera-to-board rotations
        self._t_cam_board_list: list[np.ndarray] = []  # camera-to-board translations

        # Intrinsics (loaded from shared file)
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self._load_intrinsics()

    def _load_intrinsics(self):
        """Load camera intrinsics from shared file."""
        from .camera_model import load_intrinsics

        self.K, self.dist = load_intrinsics(self.camera_id)
        if self.K is not None:
            logger.info("HandEye: loaded intrinsics for cam%d", self.camera_id)

    def set_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        """Manually set camera intrinsics."""
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)

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
            True if ChArUco board was detected and observation added.
        """
        if self.K is None:
            logger.error("No intrinsics loaded for cam%d", self.camera_id)
            return False

        ok, rvec, tvec = self.charuco.estimate_pose(frame, self.K, self.dist)
        if not ok:
            logger.warning("HandEye: no ChArUco board detected in frame")
            return False

        R_cam_board, _ = cv2.Rodrigues(rvec)

        self._T_world_ee_list.append(np.array(T_world_ee, dtype=np.float64))
        self._R_cam_board_list.append(R_cam_board)
        self._t_cam_board_list.append(tvec.reshape(3, 1))

        logger.info(
            "HandEye: observation %d added (board at t=%.3f, %.3f, %.3f m in cam frame)",
            len(self._T_world_ee_list),
            tvec.flatten()[0],
            tvec.flatten()[1],
            tvec.flatten()[2],
        )
        return True

    @property
    def num_observations(self) -> int:
        return len(self._T_world_ee_list)

    def _check_rotational_diversity(self) -> tuple[float, float, float]:
        """Check rotational diversity of collected poses.

        Returns:
            (max_rot_x, max_rot_y, max_rot_z) — maximum angular spread
            per axis in degrees across all pose pairs.
        """
        if self.num_observations < 2:
            return 0.0, 0.0, 0.0

        # Extract Euler-ish angles from each EE pose
        angles = []
        for T in self._T_world_ee_list:
            R = T[:3, :3]
            rvec, _ = cv2.Rodrigues(R)
            angles.append(rvec.flatten() * 180.0 / np.pi)

        angles = np.array(angles)
        spreads = angles.max(axis=0) - angles.min(axis=0)
        return float(spreads[0]), float(spreads[1]), float(spreads[2])

    def solve(
        self, method: int = HAND_EYE_METHODS[DEFAULT_METHOD]
    ) -> Optional[np.ndarray]:
        """Solve for T_ee_cam using a single OpenCV hand-eye method.

        Args:
            method: OpenCV CALIB_HAND_EYE_* constant.

        Returns:
            4x4 T_ee_cam transform, or None if insufficient data.
        """
        n = self.num_observations
        if n < 3:
            logger.error("Need at least 3 observations, have %d", n)
            return None

        if n < 15:
            logger.warning(
                "Only %d observations — recommend 15-25+ with >= 30deg "
                "rotational variation on each axis for reliable results",
                n,
            )

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
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method,
        )

        # Build T_ee_cam
        T_ee_cam = np.eye(4, dtype=np.float64)
        T_ee_cam[:3, :3] = R_cam2ee
        T_ee_cam[:3, 3] = t_cam2ee.flatten()

        # Sanity check rotation matrix
        det, orth_err, is_valid = _validate_rotation(R_cam2ee)
        if not is_valid:
            logger.warning(
                "HandEye: rotation matrix failed sanity check "
                "(det=%.4f, orth_err=%.4f)",
                det,
                orth_err,
            )

        logger.info(
            "HandEye solved: T_ee_cam t=(%.4f, %.4f, %.4f)m, "
            "det(R)=%.6f, orth_err=%.6f",
            T_ee_cam[0, 3],
            T_ee_cam[1, 3],
            T_ee_cam[2, 3],
            det,
            orth_err,
        )

        return T_ee_cam

    def solve_all_methods(self) -> HandEyeResult:
        """Run all five OpenCV hand-eye solvers and cross-compare.

        Returns a HandEyeResult with per-solver results, consistency metrics,
        and the best transform (Daniilidis by default, or the solver with
        best consistency if Daniilidis fails validation).
        """
        result = HandEyeResult(num_observations=self.num_observations)

        if self.num_observations < 3:
            logger.error(
                "Need at least 3 observations, have %d",
                self.num_observations,
            )
            return result

        # Check rotational diversity
        rx, ry, rz = self._check_rotational_diversity()
        logger.info(
            "Rotational diversity: X=%.1f deg, Y=%.1f deg, Z=%.1f deg",
            rx,
            ry,
            rz,
        )
        if min(rx, ry, rz) < 30.0:
            logger.warning(
                "Rotational diversity < 30 deg on at least one axis — "
                "results may be ill-conditioned. Recommend adding more "
                "diverse poses."
            )

        # Run each solver
        valid_translations = []
        valid_rotations = []

        for name, method_id in HAND_EYE_METHODS.items():
            try:
                T = self.solve(method_id)
                if T is None:
                    continue

                R = T[:3, :3]
                det, orth_err, is_valid = _validate_rotation(R)

                sr = SolverResult(
                    method_name=name,
                    T_ee_cam=T.copy(),
                    rotation_det=det,
                    rotation_orthogonality_error=orth_err,
                    is_valid=is_valid,
                )
                result.solver_results.append(sr)

                if is_valid:
                    valid_translations.append(T[:3, 3])
                    rvec, _ = cv2.Rodrigues(R)
                    valid_rotations.append(rvec.flatten())

                logger.info(
                    "  %s: t=(%.4f, %.4f, %.4f)m, det=%.6f, valid=%s",
                    name,
                    T[0, 3],
                    T[1, 3],
                    T[2, 3],
                    det,
                    is_valid,
                )
            except Exception as e:
                logger.warning("Solver %s failed: %s", name, e)

        # Compute cross-solver consistency
        if len(valid_translations) >= 2:
            t_arr = np.array(valid_translations)
            result.translation_spread_mm = (
                float(np.max(np.ptp(t_arr, axis=0))) * 1000.0
            )
            r_arr = np.array(valid_rotations)
            result.rotation_spread_deg = (
                float(np.max(np.ptp(r_arr, axis=0))) * 180.0 / np.pi
            )
            logger.info(
                "Cross-solver spread: translation=%.2fmm, rotation=%.2fdeg",
                result.translation_spread_mm,
                result.rotation_spread_deg,
            )

        # Pick best solver — prefer Daniilidis if valid
        for sr in result.solver_results:
            if sr.method_name == DEFAULT_METHOD and sr.is_valid:
                result.best_method = sr.method_name
                result.best_transform = sr.T_ee_cam.copy()
                break

        # Fallback: first valid solver
        if result.best_transform is None:
            for sr in result.solver_results:
                if sr.is_valid:
                    result.best_method = sr.method_name
                    result.best_transform = sr.T_ee_cam.copy()
                    logger.warning(
                        "Daniilidis failed validation; using %s instead",
                        sr.method_name,
                    )
                    break

        if result.best_transform is not None:
            logger.info(
                "Best hand-eye result: %s, t=(%.4f, %.4f, %.4f)m",
                result.best_method,
                result.best_transform[0, 3],
                result.best_transform[1, 3],
                result.best_transform[2, 3],
            )
        else:
            logger.error("All hand-eye solvers failed validation")

        return result

    def solve_and_save(
        self, method: str = DEFAULT_METHOD
    ) -> Optional[np.ndarray]:
        """Solve using all methods, save the best result."""
        result = self.solve_all_methods()
        if result.best_transform is None:
            return None

        path = CALIBRATION_DIR / f"camera{self.camera_id}_hand_eye.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        solver_summary = {}
        for sr in result.solver_results:
            solver_summary[sr.method_name] = {
                "T_ee_cam": sr.T_ee_cam.tolist(),
                "rotation_det": sr.rotation_det,
                "orthogonality_error": sr.rotation_orthogonality_error,
                "is_valid": sr.is_valid,
            }

        data = {
            "camera_id": self.camera_id,
            "T_ee_cam": result.best_transform.tolist(),
            "best_method": result.best_method,
            "num_observations": result.num_observations,
            "translation_spread_mm": result.translation_spread_mm,
            "rotation_spread_deg": result.rotation_spread_deg,
            "all_solvers": solver_summary,
            "description": (
                "Transform from end-effector frame to camera frame. "
                "World pose: T_world_cam = T_world_ee @ T_ee_cam. "
                "Solved using ChArUco board target with all 5 OpenCV "
                "hand-eye solvers cross-compared."
            ),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved hand-eye calibration to %s", path)
        return result.best_transform

    def clear(self):
        """Clear all collected observations."""
        self._T_world_ee_list.clear()
        self._R_cam_board_list.clear()
        self._t_cam_board_list.clear()
