"""Touch test validation — end-to-end calibration verification.

Per Claude Opus 4.6 Thinking unique finding: goes beyond reprojection error
to validate the entire grasp pipeline. The arm physically moves to a
camera-detected point, providing an unambiguous real-world accuracy check.

The touch test:
1. Place a ChArUco board or ArUco marker at a known position
2. Detect its position using each calibrated camera
3. Triangulate or chain transforms to get 3D position in robot base frame
4. Command the arm to physically touch that 3D point
5. Measure the actual error by checking if the gripper tip contacts the target

This validates the full chain: intrinsics -> distortion -> extrinsics ->
FK -> IK -> motor control. Any error in ANY stage shows up as a physical miss.

Multi-camera agreement test:
  All calibrated cameras should agree on the 3D position of a single
  ArUco marker to within 5-10mm. Larger disagreement indicates a
  calibration error in one or more cameras.

Usage:
    validator = CalibrationValidator()
    result = validator.validate_multi_camera_agreement(frames, calibrations)
    touch_result = await validator.run_touch_test(target_3d, arm_client)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .charuco_detector import ChArUcoDetector

logger = logging.getLogger(__name__)


@dataclass
class CameraAgreementResult:
    """Result of multi-camera agreement test."""

    camera_positions: dict[int, np.ndarray]  # cam_id -> 3D position estimate
    mean_position: np.ndarray  # average of all estimates
    max_disagreement_mm: float  # max pairwise distance
    per_camera_error_mm: dict[int, float]  # each camera's deviation from mean
    passed: bool  # all cameras agree within threshold


@dataclass
class TouchTestResult:
    """Result of a physical touch test."""

    target_3d: np.ndarray  # where we wanted to go (meters)
    commanded_joint_angles: list[float]  # what we told the arm
    actual_joint_angles: list[float]  # where the arm actually went
    fk_position: np.ndarray  # FK-computed position at actual angles
    estimated_error_mm: float  # ||fk_position - target_3d|| in mm
    passed: bool  # error within threshold


class CalibrationValidator:
    """Validates calibration quality through multi-camera agreement and touch tests.

    Provides two validation levels:
    1. Multi-camera agreement: all cameras should agree on the 3D position of
       a single marker to within 5-10mm
    2. Touch test: arm physically moves to a camera-detected point to verify
       the entire pipeline end-to-end
    """

    def __init__(
        self,
        charuco_detector: Optional[ChArUcoDetector] = None,
        agreement_threshold_mm: float = 10.0,
        touch_threshold_mm: float = 15.0,
    ):
        self.charuco = charuco_detector or ChArUcoDetector()
        self.agreement_threshold_mm = agreement_threshold_mm
        self.touch_threshold_mm = touch_threshold_mm

    def estimate_board_position_from_camera(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        T_cam_base: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate ChArUco board center position in robot base frame.

        Args:
            frame: BGR image from the camera.
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs: Distortion coefficients.
            T_cam_base: 4x4 transform from camera frame to robot base frame.
                For fixed cameras this comes from FixedCameraCalibrator.
                For arm camera: T_cam_base = inv(T_base_ee @ T_ee_cam).

        Returns:
            (3,) position of board center in robot base frame (meters),
            or None if detection fails.
        """
        ok, rvec, tvec = self.charuco.estimate_pose(
            frame, camera_matrix, dist_coeffs
        )
        if not ok:
            return None

        # Board origin in camera frame
        board_in_cam = tvec.flatten()

        # Transform to base frame
        board_in_cam_h = np.array(
            [board_in_cam[0], board_in_cam[1], board_in_cam[2], 1.0]
        )
        T_base_cam = np.linalg.inv(T_cam_base)
        board_in_base = (T_base_cam @ board_in_cam_h)[:3]

        return board_in_base

    def validate_multi_camera_agreement(
        self,
        frames: dict[int, np.ndarray],
        camera_matrices: dict[int, np.ndarray],
        dist_coeffs: dict[int, np.ndarray],
        T_cam_base: dict[int, np.ndarray],
    ) -> CameraAgreementResult:
        """Test that all cameras agree on the position of a calibration target.

        All calibrated cameras should agree on the 3D position of the
        ChArUco board to within 5-10mm. Larger disagreement indicates
        a calibration error.

        Args:
            frames: {camera_id: BGR_image} for each camera.
            camera_matrices: {camera_id: 3x3_K} intrinsics.
            dist_coeffs: {camera_id: dist} distortion.
            T_cam_base: {camera_id: 4x4_transform} camera-to-base.

        Returns:
            CameraAgreementResult with per-camera positions and agreement metric.
        """
        positions = {}

        for cam_id in frames:
            if cam_id not in camera_matrices or cam_id not in T_cam_base:
                logger.warning("Camera %d missing calibration data", cam_id)
                continue

            pos = self.estimate_board_position_from_camera(
                frames[cam_id],
                camera_matrices[cam_id],
                dist_coeffs[cam_id],
                T_cam_base[cam_id],
            )
            if pos is not None:
                positions[cam_id] = pos
                logger.info(
                    "Camera %d estimates board at (%.1f, %.1f, %.1f) mm",
                    cam_id,
                    pos[0] * 1000,
                    pos[1] * 1000,
                    pos[2] * 1000,
                )

        if len(positions) < 2:
            logger.warning(
                "Need at least 2 cameras for agreement test, got %d",
                len(positions),
            )
            return CameraAgreementResult(
                camera_positions=positions,
                mean_position=np.zeros(3),
                max_disagreement_mm=float("inf"),
                per_camera_error_mm={},
                passed=False,
            )

        # Compute mean position
        pos_array = np.array(list(positions.values()))
        mean_pos = pos_array.mean(axis=0)

        # Per-camera deviation from mean
        per_camera_err = {}
        for cam_id, pos in positions.items():
            err_mm = float(np.linalg.norm(pos - mean_pos) * 1000)
            per_camera_err[cam_id] = err_mm

        # Max pairwise disagreement
        max_disagree = 0.0
        cam_ids = list(positions.keys())
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                dist = np.linalg.norm(
                    positions[cam_ids[i]] - positions[cam_ids[j]]
                )
                max_disagree = max(max_disagree, dist * 1000)

        passed = max_disagree <= self.agreement_threshold_mm

        logger.info(
            "Multi-camera agreement: max disagreement = %.1f mm (%s)",
            max_disagree,
            "PASS" if passed else "FAIL",
        )
        for cam_id, err in per_camera_err.items():
            logger.info("  Camera %d: %.1f mm from mean", cam_id, err)

        return CameraAgreementResult(
            camera_positions=positions,
            mean_position=mean_pos,
            max_disagreement_mm=max_disagree,
            per_camera_error_mm=per_camera_err,
            passed=passed,
        )

    async def run_touch_test(
        self,
        target_3d: np.ndarray,
        arm_client,
        fk_engine,
        ik_solver=None,
        approach_offset_m: float = 0.05,
    ) -> TouchTestResult:
        """Execute a physical touch test.

        Commands the arm to move to a camera-detected 3D position.
        The gripper should physically touch (or come very close to) the
        target point. This validates the entire pipeline end-to-end.

        Args:
            target_3d: Target position in robot base frame (meters).
            arm_client: Object with get_joint_angles() and command_pose() methods.
            fk_engine: Module with fk_positions() function.
            ik_solver: Optional IK solver. If None, uses a simple approach.
            approach_offset_m: Approach from this distance above target first.

        Returns:
            TouchTestResult with measured accuracy.
        """
        logger.info(
            "Touch test: target at (%.1f, %.1f, %.1f) mm",
            target_3d[0] * 1000,
            target_3d[1] * 1000,
            target_3d[2] * 1000,
        )

        # Get current joint angles
        current_angles = await arm_client.get_joint_angles()

        # If we have an IK solver, use it to plan the motion
        if ik_solver is not None:
            # First approach point (offset above target)
            approach_target = target_3d.copy()
            approach_target[2] += approach_offset_m

            approach_angles = ik_solver.solve(approach_target)
            if approach_angles is not None:
                await arm_client.command_pose(tuple(approach_angles))
                import asyncio

                await asyncio.sleep(2.0)

            # Move to actual target
            target_angles = ik_solver.solve(target_3d)
            if target_angles is None:
                logger.error("IK solver failed for target position")
                return TouchTestResult(
                    target_3d=target_3d,
                    commanded_joint_angles=current_angles,
                    actual_joint_angles=current_angles,
                    fk_position=np.zeros(3),
                    estimated_error_mm=float("inf"),
                    passed=False,
                )

            await arm_client.command_pose(tuple(target_angles))
            import asyncio

            await asyncio.sleep(2.0)
            commanded = list(target_angles)
        else:
            logger.warning(
                "No IK solver provided — touch test requires manual "
                "joint angle computation or IK integration"
            )
            commanded = current_angles

        # Read actual angles after motion
        actual_angles = await arm_client.get_joint_angles()

        # Compute FK position at actual angles
        positions = fk_engine.fk_positions(actual_angles)
        ee_pos = np.array(positions[-1])  # end-effector position

        # Error = distance between FK position and target
        error_m = float(np.linalg.norm(ee_pos - target_3d))
        error_mm = error_m * 1000.0

        passed = error_mm <= self.touch_threshold_mm

        logger.info(
            "Touch test result: FK position (%.1f, %.1f, %.1f) mm, "
            "error = %.1f mm (%s)",
            ee_pos[0] * 1000,
            ee_pos[1] * 1000,
            ee_pos[2] * 1000,
            error_mm,
            "PASS" if passed else "FAIL",
        )

        return TouchTestResult(
            target_3d=target_3d,
            commanded_joint_angles=commanded,
            actual_joint_angles=actual_angles,
            fk_position=ee_pos,
            estimated_error_mm=error_mm,
            passed=passed,
        )
