"""
Claw Position Predictor — Visual position estimation of the claw/gripper.

Uses independent cameras (cam0 overhead, cam1 front/side) to detect the
claw in each view and estimate its workspace position. No stereo pair
or stereo calibration is required.

Detection strategy:
1. Detect the gripper/end-effector in each camera frame via HSV color + edge detection
2. cam0 (overhead) provides X/Y position in workspace
3. cam1 (front/side) provides Z (height) information
4. Combine into an estimated 3D position
5. Compare against FK-derived position for validation
"""

import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.claw_position")


@dataclass
class ClawPrediction:
    """Result of a single claw position prediction."""

    # Detected position in camera pixel space (cam0 / overhead)
    pixel_cam0: Optional[tuple[int, int]] = None
    # Detected position in camera pixel space (cam1 / front-side)
    pixel_cam1: Optional[tuple[int, int]] = None
    # Predicted 3D position in world coordinates (mm)
    position_mm: Optional[list[float]] = None
    # FK-computed position for comparison (mm)
    fk_position_mm: Optional[list[float]] = None
    # Euclidean error between predicted and FK position (mm)
    error_mm: float = 0.0
    # Detection confidence (0-1)
    confidence: float = 0.0
    # Was the claw detected in at least one view
    detected: bool = False
    # Which cameras detected the claw
    detected_cam0: bool = False
    detected_cam1: bool = False
    # Timestamp of the prediction
    timestamp: float = 0.0
    # Processing time in ms
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_cam0": list(self.pixel_cam0) if self.pixel_cam0 else None,
            "pixel_cam1": list(self.pixel_cam1) if self.pixel_cam1 else None,
            "position_mm": [round(v, 1) for v in self.position_mm] if self.position_mm else None,
            "fk_position_mm": (
                [round(v, 1) for v in self.fk_position_mm] if self.fk_position_mm else None
            ),
            "error_mm": round(self.error_mm, 1),
            "confidence": round(self.confidence, 3),
            "detected": self.detected,
            "detected_cam0": self.detected_cam0,
            "detected_cam1": self.detected_cam1,
            "timestamp": self.timestamp,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class ClawPositionPredictor:
    """Predicts the claw's real-world position using independent cameras.

    cam0 (overhead): provides X/Y position via top-down view
    cam1 (front/side): provides Z (height) via side view

    Detection uses HSV color thresholding + edge detection to find the
    gripper in each camera frame independently.
    """

    # Default HSV range for gripper detection (metallic/gray tone)
    DEFAULT_HSV_LOWER = np.array([0, 0, 80])
    DEFAULT_HSV_UPPER = np.array([180, 60, 220])

    MIN_CONTOUR_AREA = 200
    MAX_CONTOUR_AREA = 50000

    def __init__(
        self,
        scale_factor: float = 1.0,
        cam0_pixels_per_mm: float = 2.0,
        cam1_pixels_per_mm: float = 2.0,
    ):
        self._scale_factor = scale_factor
        self._cam0_px_per_mm = cam0_pixels_per_mm
        self._cam1_px_per_mm = cam1_pixels_per_mm

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
        self._smooth_alpha = 0.4

        # Workspace origin offsets (cam center maps to these arm-frame coords)
        self._cam0_origin_mm = np.array([0.0, 0.0])  # X, Y offset
        self._cam1_z_origin_mm = 0.0  # Z offset

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
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel_edge, iterations=1)
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
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect = min(w_c, h_c) / (max(w_c, h_c) + 1e-6)

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_to_center = np.sqrt((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2)
                max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
                center_score = 1.0 - (dist_to_center / max_dist)
            else:
                center_score = 0.0

            area_score = min(1.0, area / 5000)
            score = area_score * 0.35 + circularity * 0.25 + aspect * 0.15 + center_score * 0.25

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return None, 0.0, mask

        M = cv2.moments(best_contour)
        if M["m00"] <= 0:
            return None, 0.0, mask

        cx = int(M["m10"] / M["m00"]) + roi_offset[0]
        cy = int(M["m01"] / M["m00"]) + roi_offset[1]

        return (cx, cy), float(best_score), mask

    def predict(
        self,
        cam0_frame: np.ndarray,
        cam1_frame: np.ndarray,
        fk_position_mm: Optional[list[float]] = None,
    ) -> ClawPrediction:
        """Run claw position prediction using independent camera frames.

        Args:
            cam0_frame: Overhead camera image (BGR) — provides X/Y.
            cam1_frame: Front/side camera image (BGR) — provides Z.
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

        # Detect claw in both cameras independently
        cam0_centroid, cam0_conf, _ = self._detect_claw_in_frame(cam0_frame)
        cam1_centroid, cam1_conf, _ = self._detect_claw_in_frame(cam1_frame)

        prediction.pixel_cam0 = cam0_centroid
        prediction.pixel_cam1 = cam1_centroid
        prediction.detected_cam0 = cam0_centroid is not None
        prediction.detected_cam1 = cam1_centroid is not None
        prediction.detected = prediction.detected_cam0 or prediction.detected_cam1
        prediction.confidence = (cam0_conf + cam1_conf) / 2

        if not prediction.detected:
            prediction.elapsed_ms = (time.monotonic() - t0) * 1000
            self._update_state(prediction)
            return prediction

        # Estimate 3D position from independent cameras
        position_3d = self._estimate_position(cam0_centroid, cam1_centroid, cam0_frame, cam1_frame)

        if position_3d is not None:
            scaled = position_3d * self._scale_factor

            # Apply exponential smoothing
            if self._smooth_position is not None:
                self._smooth_position = (
                    self._smooth_alpha * scaled + (1 - self._smooth_alpha) * self._smooth_position
                )
            else:
                self._smooth_position = scaled.copy()

            prediction.position_mm = self._smooth_position.tolist()

            # Compare with FK position
            if fk_position_mm is not None:
                prediction.fk_position_mm = fk_position_mm
                fk_arr = np.array(fk_position_mm)
                prediction.error_mm = float(np.linalg.norm(self._smooth_position - fk_arr))

        prediction.elapsed_ms = (time.monotonic() - t0) * 1000
        self._update_state(prediction)
        return prediction

    def _estimate_position(
        self,
        cam0_centroid: Optional[tuple[int, int]],
        cam1_centroid: Optional[tuple[int, int]],
        cam0_frame: np.ndarray,
        cam1_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate 3D position from independent camera detections.

        cam0 (overhead) → X, Y workspace coordinates
        cam1 (front/side) → Z (height) coordinate
        """
        x_mm = 0.0
        y_mm = 0.0
        z_mm = 0.0

        if cam0_centroid is not None:
            # Overhead camera: pixel position → X, Y in workspace
            h0, w0 = cam0_frame.shape[:2]
            cx0, cy0 = w0 / 2, h0 / 2
            x_mm = (cam0_centroid[0] - cx0) / self._cam0_px_per_mm + self._cam0_origin_mm[0]
            y_mm = (cam0_centroid[1] - cy0) / self._cam0_px_per_mm + self._cam0_origin_mm[1]

        if cam1_centroid is not None:
            # Front/side camera: vertical pixel position → Z (height)
            h1, w1 = cam1_frame.shape[:2]
            cy1 = h1 / 2
            # Invert Y axis: higher in image = higher in world
            z_mm = (cy1 - cam1_centroid[1]) / self._cam1_px_per_mm + self._cam1_z_origin_mm

        if cam0_centroid is None and cam1_centroid is None:
            return None

        return np.array([x_mm, y_mm, z_mm], dtype=np.float64)

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
            recent_detections = sum(1 for p in self._detection_history if p.detected)
            total_recent = len(self._detection_history)

            status = {
                "enabled": self._enabled,
                "prediction_count": self._prediction_count,
                "detection_rate": round(recent_detections / max(total_recent, 1), 2),
                "scale_factor": round(self._scale_factor, 4),
                "hsv_lower": self._hsv_lower.tolist(),
                "hsv_upper": self._hsv_upper.tolist(),
                "has_roi": self._detection_roi is not None,
            }

            if last is not None:
                status["last_prediction"] = last.to_dict()

            return status

    def get_annotated_frame(self, frame: np.ndarray, camera_id: int = 0) -> np.ndarray:
        """Draw detection annotations on a camera frame."""
        annotated: np.ndarray = frame.copy()
        prediction = self._last_prediction

        if prediction is None or not prediction.detected:
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

        pixel = prediction.pixel_cam0 if camera_id == 0 else prediction.pixel_cam1
        if pixel is None:
            return annotated

        px, py = pixel
        color = (0, 220, 100)

        cv2.drawMarker(annotated, (px, py), color, cv2.MARKER_CROSS, markerSize=30, thickness=2)
        cv2.circle(annotated, (px, py), 20, color, 2)

        conf_text = f"Conf: {prediction.confidence:.0%}"
        cv2.putText(
            annotated, conf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
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
