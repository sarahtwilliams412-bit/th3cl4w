"""
Visual joint detection: match segmented arm features to FK-predicted joint positions.

Combines HSV marker detection, gold centroid proximity, contour inflection points,
and silhouette width minima to refine FK pixel predictions with visual evidence.

Primary mode: neon-colored marker detection via HSV thresholding.
Fallback: background subtraction + gold/contour/width features.
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
from .gpu_preprocess import to_hsv
import numpy as np

from .arm_segmenter import ArmSegmentation

logger = logging.getLogger("th3cl4w.vision.joint_detector")

# Joint names for the 5 key points
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "end_effector"]

# Default HSV marker color definitions
# Each entry: name -> (lower_hsv, upper_hsv)
DEFAULT_MARKER_COLORS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "neon_green": (
        np.array([35, 100, 100], dtype=np.uint8),
        np.array([85, 255, 255], dtype=np.uint8),
    ),
    "neon_orange": (
        np.array([5, 150, 150], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8),
    ),
    "hot_pink": (
        np.array([140, 100, 100], dtype=np.uint8),
        np.array([170, 255, 255], dtype=np.uint8),
    ),
}


class DetectionSource(Enum):
    MARKER = "marker"
    GOLD = "gold"
    CONTOUR = "contour"
    WIDTH = "width"
    FK_ONLY = "fk_only"


@dataclass
class JointDetection:
    """Single joint detection result."""
    joint_index: int
    pixel_pos: tuple[float, float]
    confidence: float  # 0.0 - 1.0
    source: DetectionSource

    @property
    def name(self) -> str:
        return JOINT_NAMES[self.joint_index] if self.joint_index < len(JOINT_NAMES) else f"joint_{self.joint_index}"


@dataclass
class SmoothedJoint:
    """Temporally smoothed joint position."""
    joint_index: int
    pixel_pos: tuple[float, float]
    confidence: float
    frames_tracked: int


# Per-camera visibility: which joints are best observed from each camera
_CAMERA_VISIBLE_JOINTS: dict[int, list[int]] = {
    0: [1, 2, 4],     # cam0 front: shoulder pitch, elbow pitch, wrist pitch
    1: [0, 1, 2, 4],  # cam1 overhead: base yaw + reach (all pitch joints visible too)
}


@dataclass
class MarkerDetection:
    """A detected color marker blob."""
    color_name: str
    centroid: tuple[float, float]  # (x, y)
    area: float
    confidence: float


class JointDetector:
    """Detect arm joints by fusing FK predictions with visual features.

    Primary mode: HSV marker detection (neon colored markers on joints).
    Fallback: background subtraction + gold/contour/width features.
    """

    def __init__(
        self,
        gold_search_radius: float = 40.0,
        contour_search_radius: float = 30.0,
        width_search_radius: float = 25.0,
        min_inflection_angle: float = 20.0,  # degrees
        marker_colors: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
        marker_min_area: float = 30.0,
        marker_search_radius: float = 50.0,
        blur_kernel: int = 5,
    ):
        self.gold_search_radius = gold_search_radius
        self.contour_search_radius = contour_search_radius
        self.width_search_radius = width_search_radius
        self.min_inflection_angle = min_inflection_angle
        self.marker_colors = marker_colors if marker_colors is not None else dict(DEFAULT_MARKER_COLORS)
        self.marker_min_area = marker_min_area
        self.marker_search_radius = marker_search_radius
        self.blur_kernel = blur_kernel
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_markers(self, frame: np.ndarray) -> list[MarkerDetection]:
        """Detect all colored markers in frame via HSV thresholding.

        Returns list of MarkerDetection with centroid, color name, area.
        """
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        hsv = to_hsv(blurred)

        markers: list[MarkerDetection] = []
        for color_name, (lower, upper) in self.marker_colors.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.marker_min_area:
                    continue
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Confidence based on area (larger = more confident, capped at 1.0)
                conf = min(1.0, 0.7 + area / 5000.0)
                markers.append(MarkerDetection(color_name, (cx, cy), area, conf))

        # Sort by area descending (largest markers first)
        markers.sort(key=lambda m: m.area, reverse=True)
        return markers

    def detect_joints(
        self,
        segmentation: ArmSegmentation,
        fk_pixels: list[tuple[float, float]],
        frame: Optional[np.ndarray] = None,
    ) -> list[JointDetection]:
        """Detect joints — marker-based (primary) with fallback to visual features.

        If frame is provided, attempts HSV marker detection first.
        Falls back to gold/contour/width matching per joint.
        """
        # Try marker detection first if frame available
        marker_centroids: list[tuple[float, float]] = []
        if frame is not None:
            markers = self.detect_markers(frame)
            marker_centroids = [(m.centroid[0], m.centroid[1]) for m in markers]
            if markers:
                logger.debug("Detected %d markers: %s", len(markers),
                             [(m.color_name, m.centroid) for m in markers])

        detections: list[JointDetection] = []
        inflection_pts = self._find_contour_inflections(segmentation)
        width_minima = self._find_width_minima(segmentation)

        for i, fk_pos in enumerate(fk_pixels):
            det = self._detect_single_joint(
                i, fk_pos, segmentation.gold_centroids, inflection_pts, width_minima,
                marker_centroids=marker_centroids,
            )
            detections.append(det)

        return detections

    def _detect_single_joint(
        self,
        joint_idx: int,
        fk_pos: tuple[float, float],
        gold_centroids: list[tuple[int, int]],
        inflection_pts: list[tuple[float, float]],
        width_minima: list[tuple[float, float]],
        marker_centroids: Optional[list[tuple[float, float]]] = None,
    ) -> JointDetection:
        # 0. Marker match (PRIMARY — highest confidence)
        if marker_centroids:
            best_marker, marker_dist = self._nearest(fk_pos, marker_centroids)
            if best_marker is not None and marker_dist <= self.marker_search_radius:
                conf = max(0.7, 1.0 - marker_dist / self.marker_search_radius * 0.3)
                # Remove used marker to prevent double-assignment
                marker_centroids.remove(best_marker)
                return JointDetection(joint_idx, best_marker, conf, DetectionSource.MARKER)

        # 1. Gold centroid match
        best_gold, gold_dist = self._nearest(fk_pos, gold_centroids)
        if best_gold is not None and gold_dist <= self.gold_search_radius:
            conf = max(0.5, 1.0 - gold_dist / self.gold_search_radius)
            return JointDetection(joint_idx, (float(best_gold[0]), float(best_gold[1])), conf, DetectionSource.GOLD)

        # 2. Contour inflection match
        best_infl, infl_dist = self._nearest(fk_pos, inflection_pts)
        if best_infl is not None and infl_dist <= self.contour_search_radius:
            conf = max(0.3, 0.7 - infl_dist / self.contour_search_radius * 0.4)
            return JointDetection(joint_idx, best_infl, conf, DetectionSource.CONTOUR)

        # 3. Width minima match
        best_wid, wid_dist = self._nearest(fk_pos, width_minima)
        if best_wid is not None and wid_dist <= self.width_search_radius:
            conf = max(0.2, 0.5 - wid_dist / self.width_search_radius * 0.3)
            return JointDetection(joint_idx, best_wid, conf, DetectionSource.WIDTH)

        # 4. FK-only fallback
        return JointDetection(joint_idx, fk_pos, 0.2, DetectionSource.FK_ONLY)

    @staticmethod
    def _nearest(
        target: tuple[float, float],
        candidates: list,
    ) -> tuple[Optional[tuple[float, float]], float]:
        if not candidates:
            return None, float("inf")
        best = None
        best_dist = float("inf")
        for c in candidates:
            d = math.hypot(c[0] - target[0], c[1] - target[1])
            if d < best_dist:
                best_dist = d
                best = (float(c[0]), float(c[1]))
        return best, best_dist

    def _find_contour_inflections(
        self, segmentation: ArmSegmentation
    ) -> list[tuple[float, float]]:
        """Find inflection points along the arm contour (direction changes)."""
        if segmentation.contour is None or len(segmentation.contour) < 10:
            return []

        pts = segmentation.contour.reshape(-1, 2).astype(np.float64)
        step = max(1, len(pts) // 50)  # sample ~50 points
        inflections: list[tuple[float, float]] = []

        for i in range(step, len(pts) - step, step):
            v1 = pts[i] - pts[i - step]
            v2 = pts[i + step] - pts[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_a))
            if angle_deg >= self.min_inflection_angle:
                inflections.append((float(pts[i][0]), float(pts[i][1])))

        return inflections

    def _find_width_minima(
        self, segmentation: ArmSegmentation
    ) -> list[tuple[float, float]]:
        """Find local width minima in the arm silhouette (joints are narrower)."""
        if segmentation.silhouette_mask is None:
            return []
        mask = segmentation.silhouette_mask
        if segmentation.bounding_box is None:
            return []

        bx, by, bw, bh = segmentation.bounding_box
        if bh < 10:
            return []

        # Measure width at each row within bounding box
        widths: list[tuple[int, float, float]] = []  # (row, width, center_x)
        for row in range(by, by + bh):
            if row < 0 or row >= mask.shape[0]:
                continue
            row_slice = mask[row, max(0, bx): min(mask.shape[1], bx + bw)]
            nonzero = np.nonzero(row_slice)[0]
            if len(nonzero) >= 2:
                w = float(nonzero[-1] - nonzero[0])
                cx = float(bx + (nonzero[0] + nonzero[-1]) / 2.0)
                widths.append((row, w, cx))

        if len(widths) < 5:
            return []

        # Find local minima in width profile
        minima: list[tuple[float, float]] = []
        for i in range(1, len(widths) - 1):
            _, w_prev, _ = widths[i - 1]
            row, w_cur, cx = widths[i]
            _, w_next, _ = widths[i + 1]
            if w_cur < w_prev and w_cur < w_next:
                minima.append((cx, float(row)))

        return minima

    @staticmethod
    def get_visible_joints(camera_id: int) -> list[int]:
        """Return joint indices best observable from given camera."""
        return list(_CAMERA_VISIBLE_JOINTS.get(camera_id, []))


class JointTracker:
    """Temporal smoothing of joint detections via exponential moving average."""

    def __init__(
        self,
        alpha: float = 0.3,
        outlier_threshold: float = 50.0,
        outlier_alpha: float = 0.05,
        confidence_decay: float = 0.9,
        num_joints: int = 5,
    ):
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.outlier_alpha = outlier_alpha
        self.confidence_decay = confidence_decay
        self.num_joints = num_joints

        self._positions: list[Optional[tuple[float, float]]] = [None] * num_joints
        self._confidences: list[float] = [0.0] * num_joints
        self._frames: list[int] = [0] * num_joints

    def update(self, detections: list[JointDetection]) -> list[SmoothedJoint]:
        """Update tracker with new detections, return smoothed positions."""
        # Index detections by joint
        det_map: dict[int, JointDetection] = {d.joint_index: d for d in detections}

        results: list[SmoothedJoint] = []
        for i in range(self.num_joints):
            if i in det_map:
                det = det_map[i]
                new_pos = det.pixel_pos

                if self._positions[i] is None:
                    # First detection
                    self._positions[i] = new_pos
                    self._confidences[i] = det.confidence
                    self._frames[i] = 1
                else:
                    prev = self._positions[i]
                    dist = math.hypot(new_pos[0] - prev[0], new_pos[1] - prev[1])

                    # Choose alpha based on outlier check
                    a = self.outlier_alpha if dist > self.outlier_threshold else self.alpha

                    self._positions[i] = (
                        prev[0] + a * (new_pos[0] - prev[0]),
                        prev[1] + a * (new_pos[1] - prev[1]),
                    )
                    self._confidences[i] = (1 - a) * self._confidences[i] + a * det.confidence
                    self._frames[i] += 1
            else:
                # No detection — decay confidence
                self._confidences[i] *= self.confidence_decay

            pos = self._positions[i] or (0.0, 0.0)
            results.append(SmoothedJoint(
                joint_index=i,
                pixel_pos=pos,
                confidence=self._confidences[i],
                frames_tracked=self._frames[i],
            ))

        return results

    def reset(self) -> None:
        self._positions = [None] * self.num_joints
        self._confidences = [0.0] * self.num_joints
        self._frames = [0] * self.num_joints
