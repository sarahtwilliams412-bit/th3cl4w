"""
Object Detector — Detects objects from camera frames and maps them to 3D workspace.

Uses the dual-camera setup:
  cam0 (overhead, video0): provides object X/Y position on workspace table
  cam2 (side view, video6): provides object height (Z) estimation

Detected objects are checked against the arm's reachable workspace (550mm radius)
and exposed for 3D simulator visualization so users can practice pick operations.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from shared.safety.limits import MAX_WORKSPACE_RADIUS_MM

from .object_labeler import ObjectLabeler

logger = logging.getLogger("th3cl4w.vision.object_detector")

# Default HSV ranges for common object colors
_COLOR_RANGES = {
    "red_low": {"lower": np.array([0, 100, 80]), "upper": np.array([10, 255, 255])},
    "red_high": {"lower": np.array([160, 100, 80]), "upper": np.array([180, 255, 255])},
    "blue": {"lower": np.array([100, 80, 60]), "upper": np.array([130, 255, 255])},
    "green": {"lower": np.array([35, 80, 60]), "upper": np.array([85, 255, 255])},
    "yellow": {"lower": np.array([20, 80, 80]), "upper": np.array([35, 255, 255])},
    "orange": {"lower": np.array([10, 100, 80]), "upper": np.array([22, 255, 255])},
}

# Workspace bounds (mm) relative to arm base — matches workspace_mapper.py
_WS_MIN = (-400.0, -400.0)
_WS_MAX = (400.0, 400.0)

# Min/max contour area (pixels) to filter noise and overly large blobs
_MIN_CONTOUR_AREA = 800
_MAX_CONTOUR_AREA = 200000

# Estimated workspace table dimensions in overhead camera FOV
_TABLE_WIDTH_MM = 800.0  # X extent
_TABLE_DEPTH_MM = 800.0  # Y extent

# Default object height when front camera not available (mm)
_DEFAULT_OBJECT_HEIGHT_MM = 50.0

# Height estimation parameters for front camera
_FRONT_CAM_TABLE_Y_FRAC = 0.85  # table surface is ~85% down in front cam view
_FRONT_CAM_MAX_HEIGHT_MM = 300.0  # max expected object height


@dataclass
class DetectedObject:
    """A single detected object with workspace position and metadata."""

    # Unique ID for tracking across frames
    obj_id: int

    # Label / color name
    label: str

    # Workspace position in mm relative to arm base
    x_mm: float  # left/right
    y_mm: float  # forward/back (depth from arm)
    z_mm: float  # height above table

    # Bounding box in overhead image (for annotation)
    bbox_overhead: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    # Estimated size
    width_mm: float = 0.0
    depth_mm: float = 0.0
    height_mm: float = 0.0

    # Distance from arm base (2D, on table plane)
    distance_from_base_mm: float = 0.0

    # Reachability
    within_reach: bool = False

    # Dominant color (BGR)
    color_bgr: tuple[int, int, int] = (128, 128, 128)

    # Detection confidence (0-1)
    confidence: float = 0.0

    # Timestamp of detection
    timestamp: float = 0.0

    # Shape classification: "cylinder", "box", "sphere", "irregular"
    shape: str = "box"

    # Rotation angle from minAreaRect (degrees)
    rotation_deg: float = 0.0

    # LLM vision labeling fields
    category: str = ""
    llm_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.obj_id,
            "label": self.label,
            "x_mm": round(self.x_mm, 1),
            "y_mm": round(self.y_mm, 1),
            "z_mm": round(self.z_mm, 1),
            "x_m": round(self.x_mm / 1000.0, 4),
            "y_m": round(self.y_mm / 1000.0, 4),
            "z_m": round(self.z_mm / 1000.0, 4),
            "width_mm": round(self.width_mm, 1),
            "depth_mm": round(self.depth_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "distance_mm": round(self.distance_from_base_mm, 1),
            "within_reach": self.within_reach,
            "color_bgr": list(self.color_bgr),
            "color_hex": "#{:02x}{:02x}{:02x}".format(
                self.color_bgr[2], self.color_bgr[1], self.color_bgr[0]
            ),
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox_overhead),
            "timestamp": round(self.timestamp, 3),
            "shape": self.shape,
            "rotation_deg": round(self.rotation_deg, 1),
            "category": self.category,
            "llm_confidence": round(self.llm_confidence, 3),
        }


class ObjectDetector:
    """Detects objects from camera frames and maps them into the arm's workspace.

    Uses color-based segmentation on the overhead camera (cam1) to find objects,
    maps pixel coordinates to workspace mm, and checks reachability against the
    arm's 550mm workspace radius.

    Objects within reach are exposed for 3D simulator visualization.
    """

    def __init__(
        self,
        max_reach_mm: float = MAX_WORKSPACE_RADIUS_MM,
        min_reach_mm: float = 80.0,
        min_contour_area: int = _MIN_CONTOUR_AREA,
        max_objects: int = 20,
        scale_mm_per_pixel: Optional[float] = None,
    ):
        self._max_reach = max_reach_mm
        self._min_reach = min_reach_mm
        self._min_contour_area = min_contour_area
        self._max_objects = max_objects
        self._scale = scale_mm_per_pixel  # if None, auto-estimate from frame size

        # State
        self._lock = threading.Lock()
        self._enabled = False
        self._objects: list[DetectedObject] = []
        self._next_id = 1
        self._last_update: float = 0.0
        self._update_count: int = 0
        self._frame_count: int = 0

        # Vision labeler for LLM-based object identification
        self._labeler = ObjectLabeler()

        # Background model for adaptive detection
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )

        # Overhead camera ROI (normalized 0-1, set during calibration)
        self._roi_x = 0.05
        self._roi_y = 0.05
        self._roi_w = 0.90
        self._roi_h = 0.90

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        logger.info("Object detector enabled")

    def disable(self):
        self._enabled = False
        logger.info("Object detector disabled")

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        logger.info("Object detector %s", "enabled" if self._enabled else "disabled")
        return self._enabled

    def get_objects(self) -> list[DetectedObject]:
        """Get current detected objects (thread-safe)."""
        with self._lock:
            return list(self._objects)

    def get_reachable_objects(self) -> list[DetectedObject]:
        """Get only objects within the arm's reach."""
        with self._lock:
            return [o for o in self._objects if o.within_reach]

    def detect_from_overhead(
        self,
        overhead_frame: np.ndarray,
        side_frame: Optional[np.ndarray] = None,
    ) -> dict:
        """Run object detection on overhead camera frame.

        Args:
            overhead_frame: BGR frame from overhead camera (cam0=video0).
            side_frame: Optional BGR frame from side camera (cam2=video6) for height estimation.

        Returns:
            Detection results summary dict.
        """
        if not self._enabled:
            return {"status": "disabled"}

        t0 = time.monotonic()
        self._frame_count += 1

        h, w = overhead_frame.shape[:2]

        # Apply ROI
        rx = int(self._roi_x * w)
        ry = int(self._roi_y * h)
        rw = int(self._roi_w * w)
        rh = int(self._roi_h * h)
        roi = overhead_frame[ry : ry + rh, rx : rx + rw]

        # Auto-estimate scale if not calibrated
        if self._scale is None:
            self._scale = _TABLE_WIDTH_MM / max(rw, 1)

        # Detect objects using combined approach:
        # 1. Color segmentation (good for known colored objects)
        # 2. Background subtraction (good for any object on table)
        color_objects = self._detect_by_color(roi, rx, ry)
        bg_objects = self._detect_by_background(roi, rx, ry)

        # Merge detections (prefer color-labeled ones, add bg-only ones)
        merged = self._merge_detections(color_objects, bg_objects)

        # Estimate height from front camera if available
        if side_frame is not None:
            self._estimate_heights(merged, side_frame)

        # Label objects using LLM vision (both camera views + ontology)
        try:
            self._labeler.label_objects(overhead_frame, side_frame, merged)
        except Exception as e:
            logger.debug("Object labeler error: %s", e)

        # Compute workspace position and reachability
        for obj in merged:
            obj.distance_from_base_mm = math.sqrt(obj.x_mm**2 + obj.y_mm**2)
            obj.within_reach = self._min_reach <= obj.distance_from_base_mm <= self._max_reach
            obj.timestamp = time.monotonic()

        # Sort by distance (closest first) and limit count
        merged.sort(key=lambda o: o.distance_from_base_mm)
        merged = merged[: self._max_objects]

        with self._lock:
            self._objects = merged
            self._last_update = time.monotonic()
            self._update_count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        n_reach = sum(1 for o in merged if o.within_reach)

        return {
            "status": "ok",
            "total_objects": len(merged),
            "reachable_objects": n_reach,
            "elapsed_ms": round(elapsed_ms, 1),
            "frame_count": self._frame_count,
            "update_count": self._update_count,
        }

    def _detect_by_color(
        self, roi: np.ndarray, offset_x: int, offset_y: int
    ) -> list[DetectedObject]:
        """Detect objects by color segmentation in HSV space."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh, rw = roi.shape[:2]
        objects: list[DetectedObject] = []

        color_groups = {
            "red": ["red_low", "red_high"],
            "blue": ["blue"],
            "green": ["green"],
            "yellow": ["yellow"],
            "orange": ["orange"],
        }

        for label, range_keys in color_groups.items():
            # Combine masks for multi-range colors (e.g., red wraps around HSV)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for key in range_keys:
                cr = _COLOR_RANGES[key]
                m = cv2.inRange(hsv, cr["lower"], cr["upper"])
                mask = cv2.bitwise_or(mask, m)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._min_contour_area or area > _MAX_CONTOUR_AREA:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # Get dominant color from the object region
                obj_region = roi[y : y + bh, x : x + bw]
                obj_mask = mask[y : y + bh, x : x + bw]
                color_bgr = self._get_dominant_color(obj_region, obj_mask)

                # Use contour moments for better centroid
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx_px = M["m10"] / M["m00"]
                    cy_px = M["m01"] / M["m00"]
                else:
                    cx_px = x + bw / 2.0
                    cy_px = y + bh / 2.0
                ws_x, ws_y = self._pixel_to_workspace(cx_px, cy_px, rw, rh)

                # Use minAreaRect for true width/depth and rotation
                rect = cv2.minAreaRect(cnt)
                _, (rect_w, rect_h), angle = rect
                # Ensure width >= depth (swap if needed)
                if rect_h > rect_w:
                    rect_w, rect_h = rect_h, rect_w
                    angle = angle + 90.0

                width_mm = rect_w * self._scale if self._scale else 0
                depth_mm = rect_h * self._scale if self._scale else 0

                # Classify shape
                shape, rotation = self._classify_shape(cnt, area, rect_w, rect_h)

                obj = DetectedObject(
                    obj_id=self._next_id,
                    label=f"{label} {shape}",
                    x_mm=ws_x,
                    y_mm=ws_y,
                    z_mm=_DEFAULT_OBJECT_HEIGHT_MM / 2.0,
                    bbox_overhead=(x + offset_x, y + offset_y, bw, bh),
                    width_mm=width_mm,
                    depth_mm=depth_mm,
                    height_mm=_DEFAULT_OBJECT_HEIGHT_MM,
                    color_bgr=color_bgr,
                    confidence=min(1.0, area / 5000.0),
                    shape=shape,
                    rotation_deg=angle,
                )
                self._next_id += 1
                objects.append(obj)

        return objects

    def _detect_by_background(
        self, roi: np.ndarray, offset_x: int, offset_y: int
    ) -> list[DetectedObject]:
        """Detect objects using background subtraction (any object on table)."""
        # Apply background subtractor
        fg_mask = self._bg_subtractor.apply(roi, learningRate=0.002)

        # Remove shadows (shadow pixels have value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Heavy morphological filtering to get clean blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        rh, rw = roi.shape[:2]
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects: list[DetectedObject] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_contour_area or area > _MAX_CONTOUR_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            obj_region = roi[y : y + bh, x : x + bw]
            obj_mask = fg_mask[y : y + bh, x : x + bw]
            color_bgr = self._get_dominant_color(obj_region, obj_mask)

            # Use contour moments for better centroid
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
            else:
                cx_px = x + bw / 2.0
                cy_px = y + bh / 2.0
            ws_x, ws_y = self._pixel_to_workspace(cx_px, cy_px, rw, rh)

            # Use minAreaRect for true dimensions
            rect = cv2.minAreaRect(cnt)
            _, (rect_w, rect_h), angle = rect
            if rect_h > rect_w:
                rect_w, rect_h = rect_h, rect_w
                angle = angle + 90.0

            width_mm = rect_w * self._scale if self._scale else 0
            depth_mm = rect_h * self._scale if self._scale else 0

            shape, _ = self._classify_shape(cnt, area, rect_w, rect_h)

            # Derive color name from dominant color for background-detected objects
            bg_color_name = self._color_name_from_bgr(color_bgr)

            obj = DetectedObject(
                obj_id=self._next_id,
                label=f"{bg_color_name} {shape}",
                x_mm=ws_x,
                y_mm=ws_y,
                z_mm=_DEFAULT_OBJECT_HEIGHT_MM / 2.0,
                bbox_overhead=(x + offset_x, y + offset_y, bw, bh),
                width_mm=width_mm,
                depth_mm=depth_mm,
                height_mm=_DEFAULT_OBJECT_HEIGHT_MM,
                color_bgr=color_bgr,
                confidence=min(1.0, area / 8000.0) * 0.7,
                shape=shape,
                rotation_deg=angle,
            )
            self._next_id += 1
            objects.append(obj)

        return objects

    @staticmethod
    def _classify_shape(
        contour: np.ndarray, area: float, rect_w: float, rect_h: float
    ) -> tuple[str, float]:
        """Classify contour shape based on geometry.

        Returns (shape, circularity) where shape is one of:
        "cylinder", "box", "sphere", "irregular".
        """
        # Circularity: 4*pi*area / perimeter^2  (1.0 = perfect circle)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            return "irregular", 0.0
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)

        # Aspect ratio from rotated rect
        aspect = rect_w / max(rect_h, 1e-6)

        # Extent: contour area / bounding rect area
        rect_area = rect_w * rect_h
        extent = area / max(rect_area, 1e-6)

        if circularity > 0.82 and aspect < 1.25:
            # Nearly circular top-down → could be cylinder or sphere
            # We can't distinguish sphere vs cylinder from overhead alone,
            # so default to cylinder (more common: cans, cups, bottles)
            return "cylinder", circularity
        elif circularity > 0.65 and aspect < 1.4:
            # Somewhat circular — likely cylinder
            return "cylinder", circularity
        elif extent > 0.85 and aspect < 1.6:
            # High extent + roughly square → box
            return "box", circularity
        elif extent > 0.75:
            # Rectangular with higher aspect ratio → still box
            return "box", circularity
        else:
            return "irregular", circularity

    def _merge_detections(
        self,
        color_objs: list[DetectedObject],
        bg_objs: list[DetectedObject],
    ) -> list[DetectedObject]:
        """Merge color and background detections, avoiding duplicates."""
        merged = list(color_objs)

        for bg_obj in bg_objs:
            is_duplicate = False
            for existing in merged:
                # Check if centers are close (within 40mm)
                dx = bg_obj.x_mm - existing.x_mm
                dy = bg_obj.y_mm - existing.y_mm
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 40.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(bg_obj)

        return merged

    def _estimate_heights(self, objects: list[DetectedObject], side_frame: np.ndarray):
        """Estimate object heights from the side camera (cam2=video6).

        Uses contour detection on the side camera to find object silhouettes,
        then matches them to overhead-detected objects by horizontal position.
        The contour's vertical extent gives a much better height estimate than
        simple edge scanning.
        """
        h, w = side_frame.shape[:2]
        table_y = int(h * _FRONT_CAM_TABLE_Y_FRAC)

        # Find object contours in the front camera using color + edge detection
        hsv = cv2.cvtColor(side_frame, cv2.COLOR_BGR2HSV)

        # Build a combined mask of all detectable colors in front view
        front_mask = np.zeros((h, w), dtype=np.uint8)
        for key, cr in _COLOR_RANGES.items():
            m = cv2.inRange(hsv, cr["lower"], cr["upper"])
            front_mask = cv2.bitwise_or(front_mask, m)

        # Also add edge-based detection for objects that don't match color ranges
        gray = cv2.cvtColor(side_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)
        # Dilate edges to close gaps, then fill
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, edge_kernel, iterations=2)
        front_mask = cv2.bitwise_or(front_mask, edges_dilated)

        # Only look above the table line
        front_mask[table_y:, :] = 0

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        front_contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build list of front-view silhouettes with their horizontal center and height
        front_silhouettes: list[tuple[float, float, float]] = []  # (cx_norm, top_y, bot_y)
        for cnt in front_contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # skip noise
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 8:  # too short
                continue
            cx_norm = (x + bw / 2.0) / w  # normalized horizontal center
            top_y = float(y)
            bot_y = float(y + bh)
            front_silhouettes.append((cx_norm, top_y, bot_y))

        # Match each overhead object to the closest front silhouette by horizontal position
        for obj in objects:
            norm_x = (obj.x_mm - _WS_MIN[0]) / (_WS_MAX[0] - _WS_MIN[0])

            best_match = None
            best_dist = 0.15  # max horizontal distance threshold (normalized)

            for cx_norm, top_y, bot_y in front_silhouettes:
                dist = abs(cx_norm - norm_x)
                if dist < best_dist:
                    best_dist = dist
                    best_match = (top_y, bot_y)

            if best_match is not None:
                top_y, bot_y = best_match
                # Use the contour's vertical extent relative to table line
                pixel_height = table_y - top_y
                if pixel_height > 5:
                    height_mm = (pixel_height / table_y) * _FRONT_CAM_MAX_HEIGHT_MM
                    height_mm = max(10.0, min(_FRONT_CAM_MAX_HEIGHT_MM, height_mm))
                    obj.height_mm = height_mm
                    obj.z_mm = height_mm / 2.0

                    # If object is cylindrical and tall relative to width,
                    # it might actually be a sphere if height ≈ width
                    if obj.shape == "cylinder":
                        diameter = max(obj.width_mm, obj.depth_mm)
                        if diameter > 0 and 0.8 < (height_mm / diameter) < 1.2:
                            obj.shape = "sphere"

    def _pixel_to_workspace(
        self, px: float, py: float, img_w: int, img_h: int
    ) -> tuple[float, float]:
        """Convert overhead camera pixel to workspace coordinates (mm).

        Assumes overhead camera is centered over the workspace.
        Origin (0,0) = arm base position = center of overhead view.
        """
        # Normalize to 0-1
        norm_x = px / max(img_w, 1)
        norm_y = py / max(img_h, 1)

        # Map to workspace bounds (centered on arm base)
        ws_x = _WS_MIN[0] + norm_x * (_WS_MAX[0] - _WS_MIN[0])
        ws_y = _WS_MIN[1] + norm_y * (_WS_MAX[1] - _WS_MIN[1])

        return ws_x, ws_y

    def _get_dominant_color(
        self, region: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> tuple[int, int, int]:
        """Get the dominant color of a region (median of masked pixels)."""
        if region.size == 0:
            return (128, 128, 128)

        if mask is not None and mask.shape[:2] == region.shape[:2]:
            pixels = region[mask > 0]
        else:
            pixels = region.reshape(-1, 3)

        if len(pixels) == 0:
            return (128, 128, 128)

        # Use median for robustness
        median = np.median(pixels, axis=0).astype(int)
        return (int(median[0]), int(median[1]), int(median[2]))

    @staticmethod
    def _color_name_from_bgr(bgr: tuple[int, int, int]) -> str:
        """Derive a human-readable color name from a BGR tuple using HSV."""
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        if v < 40:
            return "dark"
        if s < 40:
            return "white" if v > 180 else "gray"
        # Hue-based naming (OpenCV hue range 0-179)
        if h < 8 or h >= 165:
            return "red"
        if h < 22:
            return "orange"
        if h < 35:
            return "yellow"
        if h < 80:
            return "green"
        if h < 100:
            return "teal"
        if h < 130:
            return "blue"
        if h < 150:
            return "purple"
        return "pink"

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection overlays on a camera frame for visualization.

        Draws bounding boxes, enclosing circles, crosshairs, labels with
        object name, position, and reachability status.
        """
        annotated = frame.copy()
        with self._lock:
            objects = list(self._objects)

        for obj in objects:
            bx, by, bw, bh = obj.bbox_overhead

            # Color based on reachability
            if obj.within_reach:
                box_color = (0, 255, 100)  # green
                label_tag = "REACHABLE"
            else:
                box_color = (0, 140, 255)  # orange
                label_tag = "OUT OF REACH"

            # Semi-transparent fill for bounding box
            overlay = annotated.copy()
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), box_color, -1)
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

            # Draw bounding box border
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

            # Draw enclosing circle around the object
            cx = bx + bw // 2
            cy = by + bh // 2
            radius = int(max(bw, bh) * 0.65)
            cv2.circle(annotated, (cx, cy), radius, box_color, 2, cv2.LINE_AA)

            # Draw crosshair at center
            cross_size = 8
            cv2.line(
                annotated, (cx - cross_size, cy), (cx + cross_size, cy), box_color, 1, cv2.LINE_AA
            )
            cv2.line(
                annotated, (cx, cy - cross_size), (cx, cy + cross_size), box_color, 1, cv2.LINE_AA
            )

            # Draw a small filled circle at center
            cv2.circle(annotated, (cx, cy), 3, box_color, -1, cv2.LINE_AA)

            # Label background for readability
            name_text = obj.label.upper()
            pos_text = f"({obj.x_mm:.0f}, {obj.y_mm:.0f}) mm  d={obj.distance_from_base_mm:.0f}mm"
            tag_text = label_tag

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_name = 0.5
            font_scale_info = 0.38
            thickness = 1

            # Measure text sizes
            (nw, nh), _ = cv2.getTextSize(name_text, font, font_scale_name, thickness)
            (pw, ph), _ = cv2.getTextSize(pos_text, font, font_scale_info, thickness)
            (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale_info, thickness)

            label_w = max(nw, pw, tw) + 10
            label_h = nh + ph + th + 18
            label_x = bx
            label_y = max(0, by - label_h - 4)

            # Dark background behind label
            cv2.rectangle(
                annotated, (label_x, label_y), (label_x + label_w, label_y + label_h), (0, 0, 0), -1
            )
            cv2.rectangle(
                annotated, (label_x, label_y), (label_x + label_w, label_y + label_h), box_color, 1
            )

            # Draw label text
            y_off = label_y + nh + 4
            cv2.putText(
                annotated,
                name_text,
                (label_x + 5, y_off),
                font,
                font_scale_name,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y_off += ph + 6
            cv2.putText(
                annotated,
                pos_text,
                (label_x + 5, y_off),
                font,
                font_scale_info,
                (200, 200, 200),
                thickness,
                cv2.LINE_AA,
            )
            y_off += th + 4
            cv2.putText(
                annotated,
                tag_text,
                (label_x + 5, y_off),
                font,
                font_scale_info,
                box_color,
                thickness,
                cv2.LINE_AA,
            )

        # Draw legend in corner
        if objects:
            legend_y = 20
            cv2.putText(
                annotated,
                f"Detected: {len(objects)} objects",
                (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            n_reach = sum(1 for o in objects if o.within_reach)
            legend_y += 20
            cv2.putText(
                annotated,
                f"Reachable: {n_reach}",
                (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 100),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def get_config(self) -> dict:
        """Return current detection configuration as a JSON-serializable dict."""
        color_ranges = {}
        for key, cr in _COLOR_RANGES.items():
            color_ranges[key] = {
                "lower": cr["lower"].tolist(),
                "upper": cr["upper"].tolist(),
            }
        return {
            "color_ranges": color_ranges,
            "min_contour_area": self._min_contour_area,
            "max_contour_area": _MAX_CONTOUR_AREA,
            "workspace_bounds": {
                "x_min": _WS_MIN[0],
                "y_min": _WS_MIN[1],
                "x_max": _WS_MAX[0],
                "y_max": _WS_MAX[1],
            },
            "table_dimensions": {
                "width_mm": _TABLE_WIDTH_MM,
                "depth_mm": _TABLE_DEPTH_MM,
            },
            "height_estimation": {
                "table_y_fraction": _FRONT_CAM_TABLE_Y_FRAC,
                "max_height_mm": _FRONT_CAM_MAX_HEIGHT_MM,
            },
            "labeler_enabled": self._labeler is not None,
            "confidence_threshold": 0.0,
            "max_objects": self._max_objects,
        }

    def set_config(self, config: dict):
        """Update detection configuration in-place from a dict."""
        global _COLOR_RANGES, _WS_MIN, _WS_MAX, _TABLE_WIDTH_MM, _TABLE_DEPTH_MM
        global _MIN_CONTOUR_AREA, _MAX_CONTOUR_AREA
        global _FRONT_CAM_TABLE_Y_FRAC, _FRONT_CAM_MAX_HEIGHT_MM

        if "color_ranges" in config:
            for key, cr in config["color_ranges"].items():
                if key in _COLOR_RANGES:
                    _COLOR_RANGES[key] = {
                        "lower": np.array(cr["lower"], dtype=np.uint8),
                        "upper": np.array(cr["upper"], dtype=np.uint8),
                    }

        if "min_contour_area" in config:
            self._min_contour_area = int(config["min_contour_area"])
            _MIN_CONTOUR_AREA = self._min_contour_area
        if "max_contour_area" in config:
            _MAX_CONTOUR_AREA = int(config["max_contour_area"])

        if "workspace_bounds" in config:
            wb = config["workspace_bounds"]
            _WS_MIN = (float(wb.get("x_min", _WS_MIN[0])), float(wb.get("y_min", _WS_MIN[1])))
            _WS_MAX = (float(wb.get("x_max", _WS_MAX[0])), float(wb.get("y_max", _WS_MAX[1])))

        if "table_dimensions" in config:
            td = config["table_dimensions"]
            _TABLE_WIDTH_MM = float(td.get("width_mm", _TABLE_WIDTH_MM))
            _TABLE_DEPTH_MM = float(td.get("depth_mm", _TABLE_DEPTH_MM))

        if "height_estimation" in config:
            he = config["height_estimation"]
            _FRONT_CAM_TABLE_Y_FRAC = float(he.get("table_y_fraction", _FRONT_CAM_TABLE_Y_FRAC))
            _FRONT_CAM_MAX_HEIGHT_MM = float(he.get("max_height_mm", _FRONT_CAM_MAX_HEIGHT_MM))

        if "max_objects" in config:
            self._max_objects = int(config["max_objects"])

        # Reset scale so it recalculates with new table dimensions
        self._scale = None

        logger.info("Object detector config updated")

    def clear(self):
        """Clear all detected objects."""
        with self._lock:
            self._objects = []
            self._update_count = 0
        logger.info("Object detector cleared")

    def get_status(self) -> dict:
        """Get detector status summary."""
        with self._lock:
            n_total = len(self._objects)
            n_reach = sum(1 for o in self._objects if o.within_reach)
        return {
            "enabled": self._enabled,
            "total_objects": n_total,
            "reachable_objects": n_reach,
            "update_count": self._update_count,
            "frame_count": self._frame_count,
            "last_update": round(self._last_update, 3),
            "max_reach_mm": self._max_reach,
            "min_reach_mm": self._min_reach,
            "scale_mm_per_pixel": round(self._scale, 4) if self._scale else None,
        }
