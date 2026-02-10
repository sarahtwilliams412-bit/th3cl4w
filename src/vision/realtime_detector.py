"""Real-time object detection using HSV color segmentation.

Pure OpenCV, no ML models. Runs <30ms per frame on CPU.
Primary target: Red Bull can (red body + blue/silver accents).
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV color ranges for known targets
# ---------------------------------------------------------------------------

# Red Bull can: dominant red color (two ranges because red wraps in HSV)
_REDBULL_HSV_RANGES = [
    # Lower red range
    {"lower": np.array([0, 100, 80]), "upper": np.array([10, 255, 255])},
    # Upper red range
    {"lower": np.array([160, 100, 80]), "upper": np.array([180, 255, 255])},
]

# Blue accent on Red Bull (secondary detection)
_REDBULL_BLUE_HSV = {
    "lower": np.array([100, 120, 60]),
    "upper": np.array([130, 255, 255]),
}

# Generic color presets
COLOR_PRESETS = {
    "red": [
        {"lower": np.array([0, 100, 80]), "upper": np.array([10, 255, 255])},
        {"lower": np.array([160, 100, 80]), "upper": np.array([180, 255, 255])},
    ],
    "blue": [
        {"lower": np.array([100, 120, 60]), "upper": np.array([130, 255, 255])},
    ],
    "green": [
        {"lower": np.array([35, 80, 60]), "upper": np.array([85, 255, 255])},
    ],
    "yellow": [
        {"lower": np.array([20, 100, 100]), "upper": np.array([35, 255, 255])},
    ],
}

# Minimum contour area in pixels to be considered a valid detection
MIN_CONTOUR_AREA = 500

# Morphological kernel for noise removal
_MORPH_KERNEL = np.ones((5, 5), np.uint8)


@dataclass
class Detection:
    """Single object detection result."""

    found: bool
    centroid_px: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 0.0
    label: str = ""
    mask_area: int = 0


def _create_hsv_mask(hsv: np.ndarray, ranges: list[dict]) -> np.ndarray:
    """Create a combined mask from multiple HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, r["lower"], r["upper"])
    return mask


def _find_best_contour(
    mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA
) -> tuple[Optional[np.ndarray], float]:
    """Find the largest contour above min_area. Returns (contour, area) or (None, 0)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < min_area:
        return None, 0.0
    return best, area


def detect_object(
    frame: np.ndarray,
    target: str = "redbull",
    min_area: int = MIN_CONTOUR_AREA,
) -> Detection:
    """Detect an object in frame using HSV color segmentation.

    Parameters
    ----------
    frame : BGR image (numpy array).
    target : Target identifier. "redbull" uses Red Bull-specific detection.
             Other values look up COLOR_PRESETS.
    min_area : Minimum contour area in pixels.

    Returns
    -------
    Detection with found, centroid_px, bbox, confidence, etc.
    """
    if frame is None or frame.size == 0:
        return Detection(found=False)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if target == "redbull":
        return _detect_redbull(hsv, min_area)

    # Generic color detection
    ranges = COLOR_PRESETS.get(target)
    if ranges is None:
        logger.warning("Unknown target '%s', falling back to red", target)
        ranges = COLOR_PRESETS["red"]

    mask = _create_hsv_mask(hsv, ranges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contour, area = _find_best_contour(mask, min_area)
    if contour is None:
        return Detection(found=False, label=target)

    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2

    # Confidence based on area relative to frame
    frame_area = frame.shape[0] * frame.shape[1]
    # Reasonable object: 0.1%-5% of frame → confidence 0.5-0.95
    area_ratio = area / frame_area
    confidence = min(0.95, 0.5 + area_ratio * 10)

    return Detection(
        found=True,
        centroid_px=(cx, cy),
        bbox=(x, y, w, h),
        confidence=confidence,
        label=target,
        mask_area=int(area),
    )


def _detect_redbull(hsv: np.ndarray, min_area: int) -> Detection:
    """Red Bull specific detection: find red region, optionally validate with blue nearby."""
    # Primary: detect red body
    red_mask = _create_hsv_mask(hsv, _REDBULL_HSV_RANGES)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contour, area = _find_best_contour(red_mask, min_area)
    if contour is None:
        return Detection(found=False, label="redbull")

    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2

    # Check for blue in the vicinity (Red Bull has blue band)
    # Expand ROI slightly for blue search
    h_img, w_img = hsv.shape[:2]
    pad = max(w, h) // 2
    roi_x1 = max(0, x - pad)
    roi_y1 = max(0, y - pad)
    roi_x2 = min(w_img, x + w + pad)
    roi_y2 = min(h_img, y + h + pad)

    blue_roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
    blue_mask = cv2.inRange(blue_roi, _REDBULL_BLUE_HSV["lower"], _REDBULL_BLUE_HSV["upper"])
    blue_area = cv2.countNonZero(blue_mask)

    # Confidence: higher if blue is present near red
    frame_area = h_img * w_img
    base_conf = min(0.85, 0.5 + (area / frame_area) * 10)
    if blue_area > 100:
        confidence = min(0.95, base_conf + 0.1)
    else:
        confidence = base_conf

    return Detection(
        found=True,
        centroid_px=(cx, cy),
        bbox=(x, y, w, h),
        confidence=confidence,
        label="redbull",
        mask_area=int(area),
    )


def detect_gripper(
    frame: np.ndarray,
    hsv_lower: np.ndarray = np.array([0, 0, 180]),
    hsv_upper: np.ndarray = np.array([180, 40, 255]),
    min_area: int = 300,
) -> Detection:
    """Detect the gripper in overhead camera view.

    Default HSV range targets bright/white metallic gripper.
    Adjust ranges based on actual gripper appearance.
    """
    if frame is None or frame.size == 0:
        return Detection(found=False, label="gripper")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)

    contour, area = _find_best_contour(mask, min_area)
    if contour is None:
        return Detection(found=False, label="gripper")

    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2

    return Detection(
        found=True,
        centroid_px=(cx, cy),
        bbox=(x, y, w, h),
        confidence=0.8,
        label="gripper",
        mask_area=int(area),
    )
