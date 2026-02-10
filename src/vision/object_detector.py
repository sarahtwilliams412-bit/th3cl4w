"""
Object Detector — Detects objects from camera frames and maps them to 3D workspace.

Uses the dual-camera setup:
  cam0 (front/side): provides object height (Z) estimation
  cam1 (overhead):   provides object X/Y position on workspace table

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

from src.safety.limits import MAX_WORKSPACE_RADIUS_MM

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
_TABLE_WIDTH_MM = 800.0   # X extent
_TABLE_DEPTH_MM = 800.0   # Y extent

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
        cam1_frame: np.ndarray,
        cam0_frame: Optional[np.ndarray] = None,
    ) -> dict:
        """Run object detection on overhead camera frame.

        Args:
            cam1_frame: BGR frame from overhead camera (cam1).
            cam0_frame: Optional BGR frame from front camera (cam0) for height estimation.

        Returns:
            Detection results summary dict.
        """
        if not self._enabled:
            return {"status": "disabled"}

        t0 = time.monotonic()
        self._frame_count += 1

        h, w = cam1_frame.shape[:2]

        # Apply ROI
        rx = int(self._roi_x * w)
        ry = int(self._roi_y * h)
        rw = int(self._roi_w * w)
        rh = int(self._roi_h * h)
        roi = cam1_frame[ry:ry+rh, rx:rx+rw]

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
        if cam0_frame is not None:
            self._estimate_heights(merged, cam0_frame)

        # Compute workspace position and reachability
        for obj in merged:
            obj.distance_from_base_mm = math.sqrt(obj.x_mm**2 + obj.y_mm**2)
            obj.within_reach = (
                self._min_reach <= obj.distance_from_base_mm <= self._max_reach
            )
            obj.timestamp = time.monotonic()

        # Sort by distance (closest first) and limit count
        merged.sort(key=lambda o: o.distance_from_base_mm)
        merged = merged[:self._max_objects]

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
                obj_region = roi[y:y+bh, x:x+bw]
                obj_mask = mask[y:y+bh, x:x+bw]
                color_bgr = self._get_dominant_color(obj_region, obj_mask)

                # Map pixel position to workspace mm
                cx_px = x + bw / 2.0
                cy_px = y + bh / 2.0
                ws_x, ws_y = self._pixel_to_workspace(cx_px, cy_px, rw, rh)

                # Determine shape descriptor from aspect ratio
                aspect = bw / max(bh, 1)
                if aspect > 1.5:
                    shape = "bar"
                elif aspect < 0.67:
                    shape = "tall object"
                elif min(bw, bh) / max(bw, bh) > 0.85:
                    shape = "cube"
                else:
                    shape = "block"

                obj = DetectedObject(
                    obj_id=self._next_id,
                    label=f"{label} {shape}",
                    x_mm=ws_x,
                    y_mm=ws_y,
                    z_mm=_DEFAULT_OBJECT_HEIGHT_MM / 2.0,
                    bbox_overhead=(x + offset_x, y + offset_y, bw, bh),
                    width_mm=bw * self._scale if self._scale else 0,
                    depth_mm=bh * self._scale if self._scale else 0,
                    height_mm=_DEFAULT_OBJECT_HEIGHT_MM,
                    color_bgr=color_bgr,
                    confidence=min(1.0, area / 5000.0),
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
            obj_region = roi[y:y+bh, x:x+bw]
            obj_mask = fg_mask[y:y+bh, x:x+bw]
            color_bgr = self._get_dominant_color(obj_region, obj_mask)

            # Derive a color name from the dominant BGR color
            bg_label = self._color_name_from_bgr(color_bgr)
            # Determine shape descriptor from aspect ratio
            aspect = bw / max(bh, 1)
            if aspect > 1.5:
                shape = "bar"
            elif aspect < 0.67:
                shape = "tall object"
            elif min(bw, bh) / max(bw, bh) > 0.85:
                shape = "cube"
            else:
                shape = "block"

            cx_px = x + bw / 2.0
            cy_px = y + bh / 2.0
            ws_x, ws_y = self._pixel_to_workspace(cx_px, cy_px, rw, rh)

            obj = DetectedObject(
                obj_id=self._next_id,
                label=f"{bg_label} {shape}",
                x_mm=ws_x,
                y_mm=ws_y,
                z_mm=_DEFAULT_OBJECT_HEIGHT_MM / 2.0,
                bbox_overhead=(x + offset_x, y + offset_y, bw, bh),
                width_mm=bw * self._scale if self._scale else 0,
                depth_mm=bh * self._scale if self._scale else 0,
                height_mm=_DEFAULT_OBJECT_HEIGHT_MM,
                color_bgr=color_bgr,
                confidence=min(1.0, area / 8000.0) * 0.7,  # lower confidence for bg method
            )
            self._next_id += 1
            objects.append(obj)

        return objects

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

    def _estimate_heights(
        self, objects: list[DetectedObject], cam0_frame: np.ndarray
    ):
        """Estimate object heights from the front camera (cam0).

        Uses vertical position in the front camera to estimate height above table.
        Objects higher in the image are taller (assuming perspective from front).
        """
        h, w = cam0_frame.shape[:2]
        gray = cv2.cvtColor(cam0_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Simple edge-based height estimation:
        # Find vertical extent of objects in the front camera
        edges = cv2.Canny(blurred, 30, 100)
        table_y = int(h * _FRONT_CAM_TABLE_Y_FRAC)

        # For each detected object, try to estimate height from front view
        for obj in objects:
            # Scan a vertical column at the approximate horizontal position
            # Map workspace X to front camera horizontal pixel
            norm_x = (obj.x_mm - _WS_MIN[0]) / (_WS_MAX[0] - _WS_MIN[0])
            col = int(norm_x * w)
            col = max(0, min(w - 1, col))

            # Find the topmost edge above the table line in a band around the column
            col_lo = max(0, col - 20)
            col_hi = min(w, col + 20)
            edge_band = edges[:table_y, col_lo:col_hi]

            if edge_band.size > 0:
                rows_with_edges = np.where(edge_band.max(axis=1) > 0)[0]
                if len(rows_with_edges) > 0:
                    top_edge_row = rows_with_edges[0]
                    pixel_height = table_y - top_edge_row
                    # Convert pixel height to mm (linear approximation)
                    height_mm = (pixel_height / table_y) * _FRONT_CAM_MAX_HEIGHT_MM
                    height_mm = max(10.0, min(_FRONT_CAM_MAX_HEIGHT_MM, height_mm))
                    obj.height_mm = height_mm
                    obj.z_mm = height_mm / 2.0

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
            cv2.line(annotated, (cx - cross_size, cy), (cx + cross_size, cy), box_color, 1, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - cross_size), (cx, cy + cross_size), box_color, 1, cv2.LINE_AA)

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
                annotated,
                (label_x, label_y),
                (label_x + label_w, label_y + label_h),
                (0, 0, 0), -1
            )
            cv2.rectangle(
                annotated,
                (label_x, label_y),
                (label_x + label_w, label_y + label_h),
                box_color, 1
            )

            # Draw label text
            y_off = label_y + nh + 4
            cv2.putText(annotated, name_text, (label_x + 5, y_off), font, font_scale_name, (255, 255, 255), thickness, cv2.LINE_AA)
            y_off += ph + 6
            cv2.putText(annotated, pos_text, (label_x + 5, y_off), font, font_scale_info, (200, 200, 200), thickness, cv2.LINE_AA)
            y_off += th + 4
            cv2.putText(annotated, tag_text, (label_x + 5, y_off), font, font_scale_info, box_color, thickness, cv2.LINE_AA)

        # Draw legend in corner
        if objects:
            legend_y = 20
            cv2.putText(annotated, f"Detected: {len(objects)} objects", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            n_reach = sum(1 for o in objects if o.within_reach)
            legend_y += 20
            cv2.putText(annotated, f"Reachable: {n_reach}", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

        return annotated

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
