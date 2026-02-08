"""
Arm segmentation pipeline for the D1 robotic arm.

Uses background subtraction (frame differencing) for the matte-black arm body
and HSV filtering for gold accent segments at joints. Designed to work
independently per camera (cam0=front, cam1=overhead).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.arm_segmenter")

# Gold accent HSV range (BGR→HSV)
GOLD_HSV_LOWER = np.array([20, 100, 100], dtype=np.uint8)
GOLD_HSV_UPPER = np.array([40, 255, 255], dtype=np.uint8)


@dataclass
class ArmSegmentation:
    """Result of arm segmentation on a single frame."""

    silhouette_mask: np.ndarray  # binary mask of entire arm
    gold_centroids: list[tuple[int, int]]  # (x, y) centers of gold segments
    contour: Optional[np.ndarray] = None  # largest contour
    bounding_box: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)


class ArmSegmenter:
    """Detect and segment the D1 robotic arm in camera frames.

    The arm is matte-black (hard to segment by color alone), so we rely on
    background subtraction via frame differencing. Gold accents at joints
    provide additional landmarks via HSV filtering.
    """

    def __init__(
        self,
        bg_threshold: int = 30,
        morph_kernel_size: int = 5,
        morph_open_iter: int = 2,
        morph_close_iter: int = 2,
        min_contour_area: float = 500.0,
        blur_kernel: int = 5,
        gold_min_area: float = 50.0,
    ):
        self.bg_threshold = bg_threshold
        self.morph_kernel_size = morph_kernel_size
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.min_contour_area = min_contour_area
        self.blur_kernel = blur_kernel
        self.gold_min_area = gold_min_area

        self._background: Optional[np.ndarray] = None  # float64 running average
        self._roi: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    # ── Background modeling ───────────────────────────────────────────

    def capture_background(self, frames: list[np.ndarray]) -> None:
        """Build running average background from N frames.

        The arm should be at a known position or out of frame during capture.
        """
        if not frames:
            logger.warning("No frames provided for background capture")
            return

        acc = np.zeros_like(frames[0], dtype=np.float64)
        for frame in frames:
            acc += frame.astype(np.float64)
        self._background = acc / len(frames)
        logger.info("Background captured from %d frames", len(frames))

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        """Return binary foreground mask via frame differencing.

        Uses adaptive thresholding on the absolute difference between
        the current frame and the stored background model.
        """
        if self._background is None:
            # No background — return empty mask
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        diff = cv2.absdiff(frame.astype(np.float64), self._background)
        # Convert to grayscale difference magnitude
        if len(diff.shape) == 3:
            diff_gray = np.max(diff, axis=2)  # max channel difference
        else:
            diff_gray = diff

        mask = (diff_gray > self.bg_threshold).astype(np.uint8) * 255

        # Morphological cleanup: open (remove noise) then close (fill gaps)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=self.morph_open_iter
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=self.morph_close_iter
        )
        return mask

    # ── Gold segment detection ────────────────────────────────────────

    def detect_gold_segments(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Detect gold accents on the arm via HSV filtering.

        Returns list of (x, y) centroids of gold-colored regions.
        """
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GOLD_HSV_LOWER, GOLD_HSV_UPPER)

        # Cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids: list[tuple[int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.gold_min_area:
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

        return centroids

    # ── Combined segmentation ─────────────────────────────────────────

    def segment_arm(self, frame: np.ndarray) -> ArmSegmentation:
        """Full arm segmentation combining background subtraction and gold detection.

        Works independently per camera view.
        """
        # Apply ROI if set
        roi_offset_x, roi_offset_y = 0, 0
        proc_frame = frame
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            h, w = frame.shape[:2]
            # Clamp to frame bounds
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            proc_frame = frame[ry : ry + rh, rx : rx + rw]
            roi_offset_x, roi_offset_y = rx, ry

        # Background subtraction — crop background to match ROI if needed
        saved_bg = self._background
        if self._roi is not None and self._background is not None:
            self._background = self._background[roi_offset_y : roi_offset_y + proc_frame.shape[0],
                                                  roi_offset_x : roi_offset_x + proc_frame.shape[1]]
        fg_mask = self.subtract_background(proc_frame)
        self._background = saved_bg

        # Gold detection (union with foreground for better coverage)
        gold_centroids_local = self.detect_gold_segments(proc_frame)

        # Find largest contour in foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a > best_area and a >= self.min_contour_area:
                best_area = a
                best_contour = c

        bbox: Optional[tuple[int, int, int, int]] = None
        out_contour: Optional[np.ndarray] = None

        if best_contour is not None:
            # Offset contour back to full-frame coordinates
            if self._roi is not None:
                best_contour = best_contour + np.array([roi_offset_x, roi_offset_y])
            out_contour = best_contour
            x, y, w, h = cv2.boundingRect(best_contour)
            bbox = (x, y, w, h)

        # Build full-frame silhouette mask
        if self._roi is not None:
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            fh, fw = fg_mask.shape[:2]
            full_mask[roi_offset_y : roi_offset_y + fh, roi_offset_x : roi_offset_x + fw] = fg_mask
        else:
            full_mask = fg_mask

        # Offset gold centroids to full-frame coords
        gold_centroids = [
            (cx + roi_offset_x, cy + roi_offset_y) for cx, cy in gold_centroids_local
        ]

        return ArmSegmentation(
            silhouette_mask=full_mask,
            gold_centroids=gold_centroids,
            contour=out_contour,
            bounding_box=bbox,
        )

    # ── ROI optimization ──────────────────────────────────────────────

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Restrict processing to a region of interest.

        When FK predictions are available, call this to only process
        near the predicted arm location for faster segmentation.
        """
        self._roi = (x, y, w, h)
        logger.debug("ROI set to (%d, %d, %d, %d)", x, y, w, h)

    def clear_roi(self) -> None:
        """Clear ROI restriction, process full frame."""
        self._roi = None

    @property
    def has_background(self) -> bool:
        return self._background is not None
