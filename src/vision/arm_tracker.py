"""
Dual-Camera Arm Tracker — Locates the robotic arm and objects in 3D space.

Uses both cameras simultaneously to:
1. Detect objects (e.g. Red Bull can) via color segmentation
2. Triangulate object positions using stereo depth
3. Transform camera-frame coordinates into arm-base-frame coordinates
4. Track the arm's end-effector position visually for verification

Requires stereo calibration data and an extrinsic transform from
camera frame to arm base frame (camera-to-arm transform).
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .calibration import StereoCalibrator
from .stereo_depth import StereoDepthEstimator
from .object_detection import ObjectDetector, DetectedObject, ColorRange, COLOR_PRESETS

logger = logging.getLogger("th3cl4w.vision.arm_tracker")


@dataclass
class TrackedObject:
    """An object tracked in 3D space from dual cameras."""

    label: str
    position_mm: np.ndarray  # (3,) XYZ in arm-base frame (mm)
    position_cam_mm: np.ndarray  # (3,) XYZ in camera frame (mm)
    size_mm: tuple[float, float, float]  # estimated (width, height, depth) in mm
    confidence: float  # 0-1
    bbox_left: tuple[int, int, int, int]  # bounding box in left camera
    bbox_right: Optional[tuple[int, int, int, int]]  # bounding box in right camera
    centroid_left: tuple[int, int]  # pixel centroid in left camera
    centroid_right: Optional[tuple[int, int]]  # pixel centroid in right camera
    depth_mm: float  # stereo depth estimate
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ArmTrackingResult:
    """Result of a dual-camera tracking pass."""

    objects: list[TrackedObject]
    depth_map: Optional[np.ndarray]  # depth map from stereo pair
    left_frame: Optional[np.ndarray]  # annotated left frame
    right_frame: Optional[np.ndarray]  # annotated right frame
    elapsed_ms: float
    status: str  # "ok", "no_calibration", "no_frames", "error"
    message: str = ""


class DualCameraArmTracker:
    """Track arm and objects using synchronized dual camera views.

    The tracker uses the left camera as the reference view for stereo depth,
    and cross-validates detections against the right camera for robustness.
    """

    # Default camera-to-arm extrinsic: identity (cameras aligned with arm base).
    # In practice this needs to be calibrated for the specific setup.
    # This transform converts from camera frame (Z forward, X right, Y down)
    # to arm base frame (X forward, Y left, Z up).
    DEFAULT_CAM_TO_ARM = np.array([
        [0.0, 0.0, 1.0, 0.0],   # arm X = cam Z (forward)
        [-1.0, 0.0, 0.0, 0.0],  # arm Y = -cam X (left)
        [0.0, -1.0, 0.0, 0.0],  # arm Z = -cam Y (up)
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    def __init__(
        self,
        calibrator: StereoCalibrator,
        depth_estimator: Optional[StereoDepthEstimator] = None,
        cam_to_arm: Optional[np.ndarray] = None,
        target_labels: Optional[list[str]] = None,
    ):
        self.calibrator = calibrator
        self._depth_est = depth_estimator
        self.cam_to_arm = cam_to_arm if cam_to_arm is not None else self.DEFAULT_CAM_TO_ARM.copy()

        # Build object detectors for specific targets
        self._build_detectors(target_labels)

        # Tracking state
        self._lock = threading.Lock()
        self._last_result: Optional[ArmTrackingResult] = None
        self._tracking_count = 0

    def _build_detectors(self, target_labels: Optional[list[str]] = None):
        """Set up object detectors for specified targets.

        Default targets include Red Bull can detection (red + blue/silver).
        """
        if target_labels is None:
            target_labels = ["redbull", "red", "blue"]

        self._detectors: dict[str, ObjectDetector] = {}

        # Red Bull can: primarily red with blue/silver accents
        # The can has distinctive red body and blue/silver top
        redbull_colors = [
            ColorRange("redbull", np.array([0, 120, 80]), np.array([10, 255, 255])),
            ColorRange("redbull", np.array([160, 120, 80]), np.array([180, 255, 255])),
        ]
        self._detectors["redbull"] = ObjectDetector(
            color_ranges=redbull_colors,
            min_area=800,
            max_area=150000,
            morph_iterations=2,
        )

        # Generic red object detector
        self._detectors["red"] = ObjectDetector(
            color_ranges=[COLOR_PRESETS["red_low"], COLOR_PRESETS["red_high"]],
            min_area=500,
            max_area=150000,
        )

        # Generic blue object detector
        self._detectors["blue"] = ObjectDetector(
            color_ranges=[COLOR_PRESETS["blue"]],
            min_area=500,
            max_area=150000,
        )

        # All-color detector for general scene understanding
        self._detectors["all"] = ObjectDetector(min_area=500)

        self._active_labels = target_labels

    def _ensure_depth_estimator(self):
        """Lazy-create depth estimator from calibrator."""
        if self._depth_est is None and self.calibrator.is_calibrated:
            self._depth_est = StereoDepthEstimator(
                self.calibrator,
                num_disparities=128,
                block_size=5,
            )

    def cam_point_to_arm_frame(self, point_cam_mm: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to arm base frame.

        Args:
            point_cam_mm: (3,) point in camera frame [X_right, Y_down, Z_forward] in mm.

        Returns:
            (3,) point in arm base frame in mm.
        """
        p_hom = np.array([point_cam_mm[0], point_cam_mm[1], point_cam_mm[2], 1.0])
        p_arm = self.cam_to_arm @ p_hom
        return p_arm[:3]

    def set_cam_to_arm_transform(self, transform: np.ndarray):
        """Update the camera-to-arm extrinsic transform."""
        self.cam_to_arm = np.asarray(transform, dtype=np.float64).reshape(4, 4)

    def calibrate_cam_to_arm_from_known_point(
        self,
        cam_point_mm: np.ndarray,
        arm_point_mm: np.ndarray,
    ):
        """Rough calibration: given a known point in both frames, compute translation offset.

        This assumes the rotation part of cam_to_arm is already approximately correct
        (from mounting knowledge) and only adjusts the translation.
        """
        p_cam_hom = np.array([cam_point_mm[0], cam_point_mm[1], cam_point_mm[2], 1.0])
        p_arm_from_cam = self.cam_to_arm @ p_cam_hom
        translation_correction = arm_point_mm - p_arm_from_cam[:3]
        self.cam_to_arm[:3, 3] += translation_correction
        logger.info(
            "Camera-to-arm translation updated: offset=%s",
            [round(float(x), 1) for x in translation_correction],
        )

    def track(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        target_label: str = "redbull",
        annotate: bool = True,
    ) -> ArmTrackingResult:
        """Run a full tracking pass on a stereo frame pair.

        Args:
            left_frame: BGR image from left camera.
            right_frame: BGR image from right camera.
            target_label: Which detector to use ("redbull", "red", "blue", "all").
            annotate: Whether to draw annotations on the frames.

        Returns:
            ArmTrackingResult with detected and localized objects.
        """
        t0 = time.monotonic()

        if not self.calibrator.is_calibrated:
            return ArmTrackingResult(
                objects=[],
                depth_map=None,
                left_frame=left_frame if annotate else None,
                right_frame=right_frame if annotate else None,
                elapsed_ms=0.0,
                status="no_calibration",
                message="Stereo calibration required before tracking",
            )

        self._ensure_depth_estimator()
        if self._depth_est is None:
            return ArmTrackingResult(
                objects=[],
                depth_map=None,
                left_frame=left_frame,
                right_frame=right_frame,
                elapsed_ms=0.0,
                status="error",
                message="Depth estimator could not be initialized",
            )

        # Compute stereo depth
        disparity, depth_map = self._depth_est.compute_depth(left_frame, right_frame, rectify=True)

        # Select detector
        detector = self._detectors.get(target_label, self._detectors.get("all"))
        if detector is None:
            detector = self._detectors["all"]

        # Detect in left camera (reference view for depth)
        left_detections = detector.detect(
            left_frame, depth_map=depth_map, Q=self.calibrator.Q
        )

        # Detect in right camera (for cross-validation)
        right_detections = detector.detect(right_frame)

        # Build tracked objects with 3D positions
        tracked: list[TrackedObject] = []
        for det in left_detections:
            if det.centroid_3d is None or det.depth_mm <= 0:
                continue

            cam_pos = np.array(det.centroid_3d, dtype=np.float64)
            arm_pos = self.cam_point_to_arm_frame(cam_pos)

            # Estimate object size from bounding box and depth
            x, y, w, h = det.bbox
            size_mm = self._estimate_object_size(w, h, det.depth_mm)

            # Find matching detection in right camera
            right_match = self._find_matching_detection(det, right_detections)

            confidence = det.confidence
            if right_match is not None:
                # Cross-validated: boost confidence
                confidence = min(1.0, confidence * 1.3)

            obj = TrackedObject(
                label=det.label,
                position_mm=arm_pos,
                position_cam_mm=cam_pos,
                size_mm=size_mm,
                confidence=confidence,
                bbox_left=det.bbox,
                bbox_right=right_match.bbox if right_match else None,
                centroid_left=det.centroid_2d,
                centroid_right=right_match.centroid_2d if right_match else None,
                depth_mm=det.depth_mm,
            )
            tracked.append(obj)

        # Annotate frames
        vis_left = left_frame
        vis_right = right_frame
        if annotate and tracked:
            vis_left = self._annotate_frame(left_frame.copy(), tracked, "L")
            vis_right = self._annotate_frame(right_frame.copy(), tracked, "R")

        elapsed_ms = (time.monotonic() - t0) * 1000

        result = ArmTrackingResult(
            objects=tracked,
            depth_map=depth_map,
            left_frame=vis_left if annotate else None,
            right_frame=vis_right if annotate else None,
            elapsed_ms=round(elapsed_ms, 1),
            status="ok",
            message=f"Tracked {len(tracked)} object(s)",
        )

        with self._lock:
            self._last_result = result
            self._tracking_count += 1

        return result

    def _estimate_object_size(
        self, bbox_w_px: int, bbox_h_px: int, depth_mm: float
    ) -> tuple[float, float, float]:
        """Estimate real-world object size from bounding box and depth.

        Uses pinhole camera model: real_size = pixel_size * depth / focal_length
        """
        if self.calibrator.camera_matrix_left is not None:
            fx = self.calibrator.camera_matrix_left[0, 0]
            fy = self.calibrator.camera_matrix_left[1, 1]
        else:
            fx = fy = 500.0  # fallback

        width_mm = bbox_w_px * depth_mm / fx
        height_mm = bbox_h_px * depth_mm / fy
        # Estimate depth dimension as average of width and height (rough)
        depth_dim = (width_mm + height_mm) / 2.0

        return (round(width_mm, 1), round(height_mm, 1), round(depth_dim, 1))

    def _find_matching_detection(
        self,
        target: DetectedObject,
        candidates: list[DetectedObject],
        max_y_diff: int = 50,
        min_area_ratio: float = 0.3,
    ) -> Optional[DetectedObject]:
        """Find a matching detection in the other camera view.

        Matching criteria:
        - Same label
        - Similar vertical position (epipolar constraint)
        - Similar area (within ratio bounds)
        """
        best = None
        best_score = float("inf")

        for cand in candidates:
            if cand.label != target.label:
                continue

            # Epipolar: y coordinates should be similar in rectified images
            y_diff = abs(cand.centroid_2d[1] - target.centroid_2d[1])
            if y_diff > max_y_diff:
                continue

            # Area similarity
            area_ratio = min(cand.area, target.area) / max(cand.area, target.area)
            if area_ratio < min_area_ratio:
                continue

            # Score: lower is better (prefer close y + similar area)
            score = y_diff + (1.0 - area_ratio) * 100
            if score < best_score:
                best_score = score
                best = cand

        return best

    def _annotate_frame(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        cam_label: str,
    ) -> np.ndarray:
        """Draw tracking annotations on a camera frame."""
        for obj in objects:
            if cam_label == "L":
                bbox = obj.bbox_left
                centroid = obj.centroid_left
            else:
                bbox = obj.bbox_right
                centroid = obj.centroid_right
                if bbox is None or centroid is None:
                    continue

            x, y, w, h = bbox
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Centroid marker
            cv2.drawMarker(frame, centroid, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

            # Label with 3D position
            pos = obj.position_mm
            label = f"{obj.label} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})mm"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

            # Depth info
            depth_label = f"d={obj.depth_mm:.0f}mm conf={obj.confidence:.2f}"
            cv2.putText(
                frame, depth_label, (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1, cv2.LINE_AA,
            )

        # Camera label
        cv2.putText(
            frame, f"CAM {cam_label}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return frame

    def get_last_result(self) -> Optional[ArmTrackingResult]:
        """Get the most recent tracking result."""
        with self._lock:
            return self._last_result

    @property
    def tracking_count(self) -> int:
        with self._lock:
            return self._tracking_count
