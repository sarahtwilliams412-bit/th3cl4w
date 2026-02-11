"""
Object Dimension Estimator — Estimates physical dimensions from camera views.

Uses dual independent cameras to estimate object dimensions (width, height, depth)
from bounding boxes and known scale factors. Employs aggressive confidence grading
and multi-frame re-assessment to converge on accurate estimates quickly.

Camera layout:
  cam0 (front/side): height from bounding box vertical extent
  cam1 (overhead):   width/depth from bounding box horizontal extent

Grading philosophy:
  - Single-camera estimates start at low confidence (0.2-0.4)
  - Cross-camera corroboration boosts confidence significantly
  - Multi-frame consistency is required for high confidence (>0.7)
  - Estimates that vary wildly between frames are penalized heavily
  - Only estimates above a threshold are promoted to the world model
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .object_detection import ObjectDetector, DetectedObject, COLOR_PRESETS
from .calibration import CameraCalibration

logger = logging.getLogger("th3cl4w.vision.dimension_estimator")

# Arm workspace constants (Unitree D1)
ARM_MAX_REACH_MM = 550.0
ARM_BASE_HEIGHT_MM = 0.0

# Default scale: pixels to mm when no calibration is available.
# Assumes overhead camera at ~600mm above a 400x400mm workspace viewed at 1920x1080.
DEFAULT_OVERHEAD_MM_PER_PIXEL = 0.35  # ~670mm / 1920px
DEFAULT_FRONT_MM_PER_PIXEL = 0.40  # rough vertical scale


@dataclass
class DimensionEstimate:
    """Estimated physical dimensions for a single detected object."""

    label: str
    width_mm: float  # extent along the table X axis
    height_mm: float  # vertical extent (Z)
    depth_mm: float  # extent along the table Y axis

    # Pixel-level measurements used to produce these estimates
    bbox_cam0: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) front
    bbox_cam1: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) overhead

    # Confidence & grading
    confidence: float = 0.0  # 0-1, aggressively graded
    grade: str = "F"  # A-F letter grade
    grade_reasons: list[str] = field(default_factory=list)

    # Which cameras contributed
    source: str = "unknown"  # "cam0", "cam1", "both"

    # Frame tracking
    frame_count: int = 1  # how many frames contributed
    variance_mm: float = 0.0  # standard deviation across frames

    @property
    def volume_mm3(self) -> float:
        """Estimated volume in cubic millimeters."""
        return self.width_mm * self.height_mm * self.depth_mm

    @property
    def graspable(self) -> bool:
        """Whether the gripper can physically close around this object."""
        min_dim = min(self.width_mm, self.depth_mm)
        return min_dim <= 65.0 and min_dim > 3.0  # gripper range

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "width_mm": round(self.width_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "depth_mm": round(self.depth_mm, 1),
            "volume_mm3": round(self.volume_mm3, 0),
            "confidence": round(self.confidence, 3),
            "grade": self.grade,
            "grade_reasons": self.grade_reasons,
            "graspable": self.graspable,
            "source": self.source,
            "frame_count": self.frame_count,
            "variance_mm": round(self.variance_mm, 1),
        }


@dataclass
class _EstimateHistory:
    """Internal: accumulates per-frame measurements for a tracked object."""

    label: str
    widths: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    depths: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    bboxes_cam0: list[Optional[tuple[int, int, int, int]]] = field(default_factory=list)
    bboxes_cam1: list[Optional[tuple[int, int, int, int]]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


class ObjectDimensionEstimator:
    """Estimates physical object dimensions from dual-camera views.

    Pipeline:
    1. Run detection on both camera frames.
    2. For each detected object, convert bounding-box pixels to mm using
       camera calibration or default scale factors.
    3. Cross-validate: overhead gives width/depth, front gives height.
    4. Grade the estimate aggressively — penalize single-camera, high variance,
       implausible sizes.
    5. Accumulate across frames and re-grade as consistency data grows.
    """

    # Grading thresholds (aggressive)
    GRADE_THRESHOLDS = {
        "A": 0.80,  # both cameras, multi-frame consistent, plausible
        "B": 0.60,  # both cameras, some variance
        "C": 0.40,  # single camera or moderate issues
        "D": 0.20,  # low confidence, high variance
        "F": 0.00,  # unreliable
    }

    # Plausibility bounds: objects we can reasonably interact with (mm)
    MIN_OBJECT_DIM_MM = 5.0
    MAX_OBJECT_DIM_MM = 400.0
    MIN_OBJECT_HEIGHT_MM = 3.0
    MAX_OBJECT_HEIGHT_MM = 350.0

    def __init__(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
        overhead_mm_per_pixel: float = DEFAULT_OVERHEAD_MM_PER_PIXEL,
        front_mm_per_pixel: float = DEFAULT_FRONT_MM_PER_PIXEL,
        consistency_frames: int = 3,
        max_history: int = 10,
    ):
        self.cal_cam0 = cal_cam0
        self.cal_cam1 = cal_cam1
        self.overhead_scale = overhead_mm_per_pixel
        self.front_scale = front_mm_per_pixel
        self.consistency_frames = consistency_frames
        self.max_history = max_history

        self._detector = ObjectDetector(min_area=300)
        self._history: dict[str, _EstimateHistory] = {}

    def set_calibration(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
    ):
        if cal_cam0 is not None:
            self.cal_cam0 = cal_cam0
        if cal_cam1 is not None:
            self.cal_cam1 = cal_cam1

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def estimate_from_frames(
        self,
        cam0_frame: Optional[np.ndarray] = None,
        cam1_frame: Optional[np.ndarray] = None,
    ) -> list[DimensionEstimate]:
        """Estimate object dimensions from one or both camera frames.

        Returns a list of DimensionEstimate, one per detected object,
        with aggressive confidence grading.
        """
        dets_cam0: list[DetectedObject] = []
        dets_cam1: list[DetectedObject] = []

        if cam0_frame is not None:
            dets_cam0 = self._detector.detect(cam0_frame)
        if cam1_frame is not None:
            dets_cam1 = self._detector.detect(cam1_frame)

        if not dets_cam0 and not dets_cam1:
            return []

        # Match detections by label across cameras
        estimates = self._fuse_and_estimate(dets_cam0, dets_cam1, cam0_frame, cam1_frame)

        # Accumulate into history and re-grade
        now = time.monotonic()
        for est in estimates:
            self._accumulate(est, now)

        # Re-assess all with history
        reassessed = []
        for est in estimates:
            reassessed.append(self._reassess(est))

        return reassessed

    def _fuse_and_estimate(
        self,
        dets_cam0: list[DetectedObject],
        dets_cam1: list[DetectedObject],
        cam0_frame: Optional[np.ndarray],
        cam1_frame: Optional[np.ndarray],
    ) -> list[DimensionEstimate]:
        """Match detections across cameras and produce raw estimates."""
        estimates: list[DimensionEstimate] = []

        # Index cam0 detections by label
        cam0_by_label: dict[str, list[DetectedObject]] = {}
        for d in dets_cam0:
            cam0_by_label.setdefault(d.label, []).append(d)

        cam1_used_labels: set[str] = set()

        # Process overhead detections (primary for width/depth)
        for d1 in dets_cam1:
            x1, y1, w1, h1 = d1.bbox
            width_mm = self._pixels_to_mm_overhead(w1, cam1_frame)
            depth_mm = self._pixels_to_mm_overhead(h1, cam1_frame)

            # Try to find matching front camera detection for height
            height_mm = 0.0
            bbox_cam0 = None
            source = "cam1"
            base_confidence = 0.3  # single camera baseline

            if d1.label in cam0_by_label and cam0_by_label[d1.label]:
                d0 = cam0_by_label[d1.label].pop(0)
                _, _, _, h0 = d0.bbox
                height_mm = self._pixels_to_mm_front(h0, cam0_frame)
                bbox_cam0 = d0.bbox
                source = "both"
                base_confidence = 0.55  # dual camera boost

            cam1_used_labels.add(d1.label)

            est = DimensionEstimate(
                label=d1.label,
                width_mm=width_mm,
                height_mm=height_mm,
                depth_mm=depth_mm,
                bbox_cam0=bbox_cam0,
                bbox_cam1=d1.bbox,
                confidence=base_confidence,
                source=source,
            )
            self._grade(est)
            estimates.append(est)

        # Objects only seen in front camera
        for label, remaining in cam0_by_label.items():
            for d0 in remaining:
                x0, y0, w0, h0 = d0.bbox
                # From front view: width is approximate X extent, height is Z
                width_mm = self._pixels_to_mm_front(w0, cam0_frame)
                height_mm = self._pixels_to_mm_front(h0, cam0_frame)
                depth_mm = width_mm  # assume roughly symmetric (no overhead view)

                est = DimensionEstimate(
                    label=d0.label,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    depth_mm=depth_mm,
                    bbox_cam0=d0.bbox,
                    confidence=0.2,  # front-only is weakest
                    source="cam0",
                )
                self._grade(est)
                estimates.append(est)

        return estimates

    def _pixels_to_mm_overhead(self, pixel_extent: int, frame: Optional[np.ndarray]) -> float:
        """Convert pixel extent in overhead view to mm."""
        if self.cal_cam1 is not None and self.cal_cam1.is_calibrated and frame is not None:
            # Use focal length + assumed height for more accurate conversion
            # At known distance Z, 1 pixel ≈ Z / fy mm
            # But without depth, use the calibrated image scale
            h, w = frame.shape[:2]
            # Approximate: workspace is ~800mm across, image is w pixels
            scale = 800.0 / max(w, 1)
            return pixel_extent * scale
        return pixel_extent * self.overhead_scale

    def _pixels_to_mm_front(self, pixel_extent: int, frame: Optional[np.ndarray]) -> float:
        """Convert pixel extent in front view to mm."""
        if self.cal_cam0 is not None and self.cal_cam0.is_calibrated and frame is not None:
            h, w = frame.shape[:2]
            scale = 600.0 / max(h, 1)
            return pixel_extent * scale
        return pixel_extent * self.front_scale

    # ------------------------------------------------------------------
    # Aggressive grading
    # ------------------------------------------------------------------

    def _grade(self, est: DimensionEstimate):
        """Apply aggressive grading to a single-frame estimate.

        Penalizes:
        - Single camera source
        - Implausible dimensions
        - Very small or very large bounding boxes
        - Missing height information
        """
        reasons: list[str] = []
        confidence = est.confidence

        # Source penalty
        if est.source == "cam0":
            confidence *= 0.5
            reasons.append("front-camera-only: depth/width unreliable")
        elif est.source == "cam1":
            confidence *= 0.7
            reasons.append("overhead-only: height unknown")

        # Plausibility checks — penalize heavily for impossible dimensions
        for dim_name, dim_val, lo, hi in [
            ("width", est.width_mm, self.MIN_OBJECT_DIM_MM, self.MAX_OBJECT_DIM_MM),
            ("depth", est.depth_mm, self.MIN_OBJECT_DIM_MM, self.MAX_OBJECT_DIM_MM),
            ("height", est.height_mm, self.MIN_OBJECT_HEIGHT_MM, self.MAX_OBJECT_HEIGHT_MM),
        ]:
            if dim_val < lo:
                confidence *= 0.3
                reasons.append(f"{dim_name}={dim_val:.0f}mm below minimum {lo:.0f}mm")
            elif dim_val > hi:
                confidence *= 0.3
                reasons.append(f"{dim_name}={dim_val:.0f}mm above maximum {hi:.0f}mm")

        # Missing height is a big deal — can't plan grasps without it
        if est.height_mm <= 0:
            confidence *= 0.5
            reasons.append("no height estimate available")

        # Aspect ratio sanity: objects taller than 6x their width are suspect
        if est.width_mm > 0 and est.height_mm > 0:
            aspect = est.height_mm / est.width_mm
            if aspect > 6.0 or aspect < 0.05:
                confidence *= 0.4
                reasons.append(f"extreme aspect ratio {aspect:.1f}")

        # Gripper feasibility bonus
        min_graspable = min(est.width_mm, est.depth_mm)
        if 10.0 <= min_graspable <= 65.0:
            confidence *= 1.1  # slight boost for graspable objects
            reasons.append("within gripper range")
        elif min_graspable > 65.0:
            reasons.append("too wide for gripper")

        confidence = max(0.0, min(1.0, confidence))
        est.confidence = confidence
        est.grade = self._confidence_to_grade(confidence)
        est.grade_reasons = reasons

    def _confidence_to_grade(self, confidence: float) -> str:
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if confidence >= threshold:
                return grade
        return "F"

    # ------------------------------------------------------------------
    # Multi-frame accumulation and re-assessment
    # ------------------------------------------------------------------

    def _accumulate(self, est: DimensionEstimate, timestamp: float):
        """Add a single-frame estimate to the history for its label."""
        key = est.label
        if key not in self._history:
            self._history[key] = _EstimateHistory(label=key)

        hist = self._history[key]
        hist.widths.append(est.width_mm)
        hist.heights.append(est.height_mm)
        hist.depths.append(est.depth_mm)
        hist.confidences.append(est.confidence)
        hist.sources.append(est.source)
        hist.bboxes_cam0.append(est.bbox_cam0)
        hist.bboxes_cam1.append(est.bbox_cam1)
        hist.timestamps.append(timestamp)

        # Trim to max history
        if len(hist.widths) > self.max_history:
            for lst in [
                hist.widths,
                hist.heights,
                hist.depths,
                hist.confidences,
                hist.sources,
                hist.bboxes_cam0,
                hist.bboxes_cam1,
                hist.timestamps,
            ]:
                del lst[0]

    def _reassess(self, est: DimensionEstimate) -> DimensionEstimate:
        """Re-assess an estimate using accumulated history.

        Multi-frame consistency dramatically affects the grade:
        - Consistent measurements across N frames → confidence boost
        - High variance across frames → confidence penalty
        - Weighted average of measurements replaces single-frame value
        """
        key = est.label
        hist = self._history.get(key)
        if hist is None or len(hist.widths) < 2:
            return est  # not enough data to reassess

        n = len(hist.widths)

        # Compute weighted averages (recent frames weighted more)
        weights = np.array([0.5 + 0.5 * (i / max(n - 1, 1)) for i in range(n)])
        weights /= weights.sum()

        avg_w = float(np.average(hist.widths, weights=weights))
        avg_h = float(np.average(hist.heights, weights=weights))
        avg_d = float(np.average(hist.depths, weights=weights))

        # Variance (penalize inconsistency)
        std_w = float(np.std(hist.widths)) if n >= 2 else 0.0
        std_h = float(np.std(hist.heights)) if n >= 2 else 0.0
        std_d = float(np.std(hist.depths)) if n >= 2 else 0.0
        max_std = max(std_w, std_h, std_d)

        # Update the estimate with averaged values
        est.width_mm = avg_w
        est.height_mm = avg_h
        est.depth_mm = avg_d
        est.frame_count = n
        est.variance_mm = max_std

        # Re-grade with multi-frame data
        reasons = list(est.grade_reasons)
        confidence = est.confidence

        # Consistency bonus: low variance across multiple frames
        if n >= self.consistency_frames:
            if max_std < 5.0:
                confidence *= 1.4
                reasons.append(f"consistent across {n} frames (std={max_std:.1f}mm)")
            elif max_std < 15.0:
                confidence *= 1.15
                reasons.append(f"moderately consistent across {n} frames")
            elif max_std < 30.0:
                # No change
                reasons.append(f"some variance across {n} frames (std={max_std:.1f}mm)")
            else:
                confidence *= 0.6
                reasons.append(f"HIGH variance across {n} frames (std={max_std:.1f}mm)")

        # Multi-camera corroboration across frames
        both_count = sum(1 for s in hist.sources if s == "both")
        if both_count >= 2:
            confidence *= 1.2
            reasons.append(f"dual-camera corroboration in {both_count}/{n} frames")

        confidence = max(0.0, min(1.0, confidence))
        est.confidence = confidence
        est.grade = self._confidence_to_grade(confidence)
        est.grade_reasons = reasons

        return est

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_best_estimates(self) -> list[DimensionEstimate]:
        """Return the current best estimate for each tracked object.

        Uses accumulated history to produce the most accurate estimate
        available, re-graded.
        """
        results = []
        for key, hist in self._history.items():
            if not hist.widths:
                continue

            n = len(hist.widths)
            weights = np.array([0.5 + 0.5 * (i / max(n - 1, 1)) for i in range(n)])
            weights /= weights.sum()

            # Determine best source
            both_count = sum(1 for s in hist.sources if s == "both")
            if both_count > 0:
                source = "both"
                base_conf = 0.55
            elif any(s == "cam1" for s in hist.sources):
                source = "cam1"
                base_conf = 0.3
            else:
                source = "cam0"
                base_conf = 0.2

            est = DimensionEstimate(
                label=key,
                width_mm=float(np.average(hist.widths, weights=weights)),
                height_mm=float(np.average(hist.heights, weights=weights)),
                depth_mm=float(np.average(hist.depths, weights=weights)),
                bbox_cam0=hist.bboxes_cam0[-1] if hist.bboxes_cam0 else None,
                bbox_cam1=hist.bboxes_cam1[-1] if hist.bboxes_cam1 else None,
                confidence=base_conf,
                source=source,
                frame_count=n,
                variance_mm=float(
                    max(
                        np.std(hist.widths) if n >= 2 else 0.0,
                        np.std(hist.heights) if n >= 2 else 0.0,
                        np.std(hist.depths) if n >= 2 else 0.0,
                    )
                ),
            )
            self._grade(est)
            est = self._reassess(est)
            results.append(est)

        results.sort(key=lambda e: e.confidence, reverse=True)
        return results

    def clear_history(self):
        """Reset all accumulated estimates."""
        self._history.clear()

    def get_history_summary(self) -> dict:
        """Summarize estimation history for diagnostics."""
        summary = {}
        for key, hist in self._history.items():
            summary[key] = {
                "frames": len(hist.widths),
                "sources": list(set(hist.sources)),
                "latest_width": round(hist.widths[-1], 1) if hist.widths else 0,
                "latest_height": round(hist.heights[-1], 1) if hist.heights else 0,
                "latest_depth": round(hist.depths[-1], 1) if hist.depths else 0,
            }
        return summary
