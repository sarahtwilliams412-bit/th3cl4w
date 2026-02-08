"""
Claw Position Predictor — Visual 3D position estimation of the claw/gripper.

Uses both stereo cameras and scale calibration from measuring devices
(checkerboard or tape measure) to detect the claw in both images,
triangulate its 3D position, and compare against the FK-derived position.

This module provides a toggleable prediction system that:
1. Detects the gripper/end-effector in left and right camera frames
2. Uses stereo triangulation to compute 3D position in camera space
3. Applies the scale factor from measuring device calibration
4. Reports predicted world coordinates with confidence metrics
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

logger = logging.getLogger("th3cl4w.vision.claw_position")


@dataclass
class ClawPrediction:
    """Result of a single claw position prediction."""

    # Detected position in camera pixel space (left camera)
    pixel_left: Optional[tuple[int, int]] = None
    # Detected position in camera pixel space (right camera)
    pixel_right: Optional[tuple[int, int]] = None
    # Predicted 3D position in world coordinates (mm)
    position_mm: Optional[list[float]] = None
    # FK-computed position for comparison (mm)
    fk_position_mm: Optional[list[float]] = None
    # Euclidean error between predicted and FK position (mm)
    error_mm: float = 0.0
    # Detection confidence (0-1)
    confidence: float = 0.0
    # Depth at the detected claw location (mm)
    depth_mm: float = 0.0
    # Was the claw detected in both views
    detected: bool = False
    # Timestamp of the prediction
    timestamp: float = 0.0
    # Processing time in ms
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_left": list(self.pixel_left) if self.pixel_left else None,
            "pixel_right": list(self.pixel_right) if self.pixel_right else None,
            "position_mm": [round(v, 1) for v in self.position_mm] if self.position_mm else None,
            "fk_position_mm": (
                [round(v, 1) for v in self.fk_position_mm] if self.fk_position_mm else None
            ),
            "error_mm": round(self.error_mm, 1),
            "confidence": round(self.confidence, 3),
            "depth_mm": round(self.depth_mm, 1),
            "detected": self.detected,
            "timestamp": self.timestamp,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class ClawPositionPredictor:
    """Predicts the claw's real-world position using stereo vision.

    Detection strategy:
    - Convert both camera frames to HSV color space
    - Apply adaptive color thresholding to isolate the gripper
    - Use morphological operations to clean up the mask
    - Find the largest contour matching the expected shape/size
    - Compute centroid in both images
    - Triangulate using stereo geometry
    - Apply scale correction from calibration
    """

    # Default HSV range for gripper detection (metallic/gray tone)
    # These can be tuned based on the actual gripper appearance
    DEFAULT_HSV_LOWER = np.array([0, 0, 80])
    DEFAULT_HSV_UPPER = np.array([180, 60, 220])

    # Secondary detection: look for the distinctive end-effector shape
    # using edge-based detection as a fallback
    MIN_CONTOUR_AREA = 200  # minimum pixel area for a valid detection
    MAX_CONTOUR_AREA = 50000  # maximum pixel area

    def __init__(
        self,
        calibrator: StereoCalibrator,
        depth_estimator: Optional[StereoDepthEstimator] = None,
        scale_factor: float = 1.0,
    ):
        self.calibrator = calibrator
        self._depth_est = depth_estimator
        self._scale_factor = scale_factor

        # Detection parameters (can be tuned via API)
        self._hsv_lower = self.DEFAULT_HSV_LOWER.copy()
        self._hsv_upper = self.DEFAULT_HSV_UPPER.copy()
        self._use_edge_detection = True
        self._detection_roi: Optional[tuple[int, int, int, int]] = None  # x, y, w, h

        # State
        self._enabled = False
        self._lock = threading.Lock()
        self._last_prediction: Optional[ClawPrediction] = None
        self._prediction_count = 0
        self._detection_history: list[ClawPrediction] = []
        self._max_history = 20

        # Smoothing: exponential moving average on position
        self._smooth_position: Optional[np.ndarray] = None
        self._smooth_alpha = 0.4  # higher = more responsive, lower = smoother

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        logger.info("Claw position predictor enabled")

    def disable(self):
        self._enabled = False
        self._smooth_position = None
        logger.info("Claw position predictor disabled")

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        if not self._enabled:
            self._smooth_position = None
        logger.info("Claw position predictor %s", "enabled" if self._enabled else "disabled")
        return self._enabled

    def set_scale_factor(self, factor: float):
        """Update scale factor from workspace mapper calibration."""
        self._scale_factor = factor

    def set_hsv_range(
        self,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ):
        """Set custom HSV range for gripper color detection."""
        self._hsv_lower = np.array(lower)
        self._hsv_upper = np.array(upper)

    def set_detection_roi(self, x: int, y: int, w: int, h: int):
        """Set a region of interest to limit detection area."""
        self._detection_roi = (x, y, w, h)

    def clear_detection_roi(self):
        """Clear the detection ROI (search entire frame)."""
        self._detection_roi = None

    def _ensure_depth_estimator(self):
        """Lazy-create depth estimator from calibrator."""
        if self._depth_est is None and self.calibrator.is_calibrated:
            self._depth_est = StereoDepthEstimator(
                self.calibrator,
                num_disparities=64,
                block_size=7,
            )

    def _detect_claw_in_frame(
        self, frame: np.ndarray
    ) -> tuple[Optional[tuple[int, int]], float, Optional[np.ndarray]]:
        """Detect the claw/gripper in a single camera frame.

        Returns:
            (centroid, confidence, mask) where centroid is (x, y) pixel coords,
            confidence is 0-1, and mask is the binary detection mask.
        """
        h, w = frame.shape[:2]

        # Apply ROI if set
        roi_offset = (0, 0)
        if self._detection_roi is not None:
            rx, ry, rw, rh = self._detection_roi
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            frame = frame[ry : ry + rh, rx : rx + rw]
            roi_offset = (rx, ry)

        # Convert to HSV and threshold
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Edge-based detection enhancement
        if self._use_edge_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # Dilate edges to connect nearby edge segments
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel_edge, iterations=1)
            # Combine color mask with edge info: keep color detections near edges
            edge_dilated = cv2.dilate(edges, kernel_edge, iterations=3)
            mask = cv2.bitwise_and(mask, edge_dilated)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0, mask

        # Filter contours by area
        valid_contours = [
            c
            for c in contours
            if self.MIN_CONTOUR_AREA <= cv2.contourArea(c) <= self.MAX_CONTOUR_AREA
        ]

        if not valid_contours:
            return None, 0.0, mask

        # Score contours: prefer larger, more circular, centered contours
        best_contour = None
        best_score = 0.0

        frame_cx, frame_cy = frame.shape[1] / 2, frame.shape[0] / 2

        for contour in valid_contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Circularity (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

            # Compactness score based on bounding rect aspect ratio
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect = min(w_c, h_c) / (max(w_c, h_c) + 1e-6)

            # Center proximity (prefer detections toward frame center for the end effector)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_to_center = np.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
                max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
                center_score = 1.0 - (dist_to_center / max_dist)
            else:
                center_score = 0.0

            # Combined score
            area_score = min(1.0, area / 5000)  # normalize area
            score = (
                area_score * 0.35 + circularity * 0.25 + aspect * 0.15 + center_score * 0.25
            )

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return None, 0.0, mask

        # Compute centroid
        M = cv2.moments(best_contour)
        if M["m00"] <= 0:
            return None, 0.0, mask

        cx = int(M["m10"] / M["m00"]) + roi_offset[0]
        cy = int(M["m01"] / M["m00"]) + roi_offset[1]

        return (cx, cy), float(best_score), mask

    def predict(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        fk_position_mm: Optional[list[float]] = None,
    ) -> ClawPrediction:
        """Run claw position prediction on a stereo frame pair.

        Args:
            left_frame: Left camera image (BGR).
            right_frame: Right camera image (BGR).
            fk_position_mm: Optional FK-computed end-effector position [x, y, z]
                for comparison/error computation.

        Returns:
            ClawPrediction with results.
        """
        t0 = time.monotonic()
        prediction = ClawPrediction(timestamp=time.time())

        if not self._enabled:
            prediction.elapsed_ms = (time.monotonic() - t0) * 1000
            return prediction

        # Detect claw in both frames
        left_centroid, left_conf, _ = self._detect_claw_in_frame(left_frame)
        right_centroid, right_conf, _ = self._detect_claw_in_frame(right_frame)

        prediction.pixel_left = left_centroid
        prediction.pixel_right = right_centroid
        prediction.confidence = (left_conf + right_conf) / 2

        if left_centroid is None or right_centroid is None:
            prediction.elapsed_ms = (time.monotonic() - t0) * 1000
            self._update_state(prediction)
            return prediction

        prediction.detected = True

        # Compute 3D position via stereo triangulation
        position_3d = self._triangulate(
            left_centroid, right_centroid, left_frame, right_frame
        )

        if position_3d is not None:
            # Apply scale correction
            scaled = position_3d * self._scale_factor

            # Apply exponential smoothing
            if self._smooth_position is not None:
                self._smooth_position = (
                    self._smooth_alpha * scaled + (1 - self._smooth_alpha) * self._smooth_position
                )
            else:
                self._smooth_position = scaled.copy()

            prediction.position_mm = self._smooth_position.tolist()
            prediction.depth_mm = float(scaled[2])

            # Compare with FK position
            if fk_position_mm is not None:
                prediction.fk_position_mm = fk_position_mm
                fk_arr = np.array(fk_position_mm)
                prediction.error_mm = float(np.linalg.norm(self._smooth_position - fk_arr))

        prediction.elapsed_ms = (time.monotonic() - t0) * 1000
        self._update_state(prediction)
        return prediction

    def _triangulate(
        self,
        left_pt: tuple[int, int],
        right_pt: tuple[int, int],
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute 3D position from stereo correspondences.

        Uses the stereo depth estimator for disparity-based depth at the
        detected points, then back-projects to 3D.
        """
        self._ensure_depth_estimator()
        if self._depth_est is None:
            return None

        # Compute depth map
        _, depth_map = self._depth_est.compute_depth(left_frame, right_frame, rectify=True)

        # Get depth at detected claw position (left camera, averaged over window)
        depth = self._depth_est.get_depth_at(depth_map, left_pt[0], left_pt[1], window=7)
        if depth <= 0:
            # Fallback: try disparity-based triangulation directly
            disparity = abs(left_pt[0] - right_pt[0])
            if disparity > 0 and self.calibrator.Q is not None:
                focal = abs(self.calibrator.Q[2, 3])
                baseline_inv = abs(self.calibrator.Q[3, 2])
                if baseline_inv > 0:
                    depth = focal / (baseline_inv * disparity)

        if depth <= 0 or depth > 10000:
            return None

        # Back-project to 3D using camera intrinsics
        if self.calibrator.camera_matrix_left is not None:
            fx = self.calibrator.camera_matrix_left[0, 0]
            fy = self.calibrator.camera_matrix_left[1, 1]
            cx = self.calibrator.camera_matrix_left[0, 2]
            cy = self.calibrator.camera_matrix_left[1, 2]
        else:
            return None

        x_3d = (left_pt[0] - cx) * depth / fx
        y_3d = (left_pt[1] - cy) * depth / fy
        z_3d = depth

        return np.array([x_3d, y_3d, z_3d], dtype=np.float64)

    def _update_state(self, prediction: ClawPrediction):
        """Update internal state with latest prediction."""
        with self._lock:
            self._last_prediction = prediction
            self._prediction_count += 1
            self._detection_history.append(prediction)
            if len(self._detection_history) > self._max_history:
                self._detection_history.pop(0)

    def get_last_prediction(self) -> Optional[ClawPrediction]:
        """Return the most recent prediction."""
        with self._lock:
            return self._last_prediction

    def get_status(self) -> dict:
        """Return current status for the API."""
        with self._lock:
            last = self._last_prediction
            recent_detections = sum(
                1 for p in self._detection_history if p.detected
            )
            total_recent = len(self._detection_history)

            status = {
                "enabled": self._enabled,
                "prediction_count": self._prediction_count,
                "detection_rate": (
                    round(recent_detections / max(total_recent, 1), 2)
                ),
                "scale_factor": round(self._scale_factor, 4),
                "hsv_lower": self._hsv_lower.tolist(),
                "hsv_upper": self._hsv_upper.tolist(),
                "has_roi": self._detection_roi is not None,
                "has_calibration": self.calibrator.is_calibrated,
            }

            if last is not None:
                status["last_prediction"] = last.to_dict()

            return status

    def get_annotated_frame(self, frame: np.ndarray, is_left: bool = True) -> np.ndarray:
        """Draw detection annotations on a camera frame.

        Overlays the detected claw position, bounding info, and
        coordinate readout onto the frame for visualization.
        """
        annotated = frame.copy()
        prediction = self._last_prediction

        if prediction is None or not prediction.detected:
            # Draw "NO DETECTION" indicator
            cv2.putText(
                annotated,
                "CLAW: NO DETECTION",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 200),
                1,
                cv2.LINE_AA,
            )
            return annotated

        pixel = prediction.pixel_left if is_left else prediction.pixel_right
        if pixel is None:
            return annotated

        px, py = pixel

        # Draw crosshair at detected position
        color = (0, 220, 100)  # green
        cv2.drawMarker(
            annotated,
            (px, py),
            color,
            cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
        )

        # Draw circle around detection
        cv2.circle(annotated, (px, py), 20, color, 2)

        # Draw confidence and position text
        conf_text = f"Conf: {prediction.confidence:.0%}"
        cv2.putText(
            annotated,
            conf_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        if prediction.position_mm:
            pos = prediction.position_mm
            pos_text = f"X:{pos[0]:.0f} Y:{pos[1]:.0f} Z:{pos[2]:.0f} mm"
            cv2.putText(
                annotated,
                pos_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 50),
                1,
                cv2.LINE_AA,
            )

        if prediction.error_mm > 0:
            err_color = (0, 200, 0) if prediction.error_mm < 30 else (0, 140, 255)
            err_text = f"FK err: {prediction.error_mm:.0f}mm"
            cv2.putText(
                annotated,
                err_text,
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                err_color,
                1,
                cv2.LINE_AA,
            )

        return annotated
