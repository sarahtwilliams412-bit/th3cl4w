"""
Basic object detection for robotic grasping.

Uses color-based segmentation, contour detection, bounding boxes,
and centroid estimation in 3D using depth data.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
from .gpu_preprocess import to_hsv
import numpy as np

logger = logging.getLogger("th3cl4w.vision.object_detection")


@dataclass
class DetectedObject:
    """A detected object with 2D and optional 3D information."""

    label: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    centroid_2d: tuple[int, int]  # (cx, cy) in pixels
    area: float  # contour area in pixels
    contour: np.ndarray  # raw contour points
    centroid_3d: Optional[tuple[float, float, float]] = None  # (X, Y, Z) in mm
    depth_mm: float = 0.0
    confidence: float = 0.0


@dataclass
class ColorRange:
    """HSV color range for segmentation."""

    name: str
    lower: np.ndarray  # [H, S, V] lower bound
    upper: np.ndarray  # [H, S, V] upper bound

    def __post_init__(self):
        self.lower = np.array(self.lower, dtype=np.uint8)
        self.upper = np.array(self.upper, dtype=np.uint8)


# Common color presets
COLOR_PRESETS: dict[str, ColorRange] = {
    "red_low": ColorRange("red", [0, 100, 100], [10, 255, 255]),
    "red_high": ColorRange("red", [160, 100, 100], [180, 255, 255]),
    "green": ColorRange("green", [35, 100, 100], [85, 255, 255]),
    "blue": ColorRange("blue", [100, 100, 100], [130, 255, 255]),
    "yellow": ColorRange("yellow", [20, 100, 100], [35, 255, 255]),
    "orange": ColorRange("orange", [10, 100, 100], [20, 255, 255]),
}


class ObjectDetector:
    """Detect graspable objects via color segmentation."""

    def __init__(
        self,
        color_ranges: Optional[list[ColorRange]] = None,
        min_area: float = 500.0,
        max_area: float = 100000.0,
        blur_kernel: int = 5,
        morph_kernel: int = 5,
        morph_iterations: int = 2,
    ):
        if color_ranges is None:
            # Default: detect red, green, blue objects
            self.color_ranges = [
                COLOR_PRESETS["red_low"],
                COLOR_PRESETS["red_high"],
                COLOR_PRESETS["green"],
                COLOR_PRESETS["blue"],
            ]
        else:
            self.color_ranges = color_ranges

        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.morph_kernel_size = morph_kernel
        self.morph_iterations = morph_iterations

    def detect(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ) -> list[DetectedObject]:
        """Detect objects in a BGR image.

        Args:
            image: BGR image.
            depth_map: Optional depth map (mm) for 3D centroid estimation.
            Q: Optional disparity-to-depth matrix for 3D projection.

        Returns:
            List of DetectedObject sorted by area (largest first).
        """
        blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
        hsv = to_hsv(blurred)

        detections: list[DetectedObject] = []

        for color_range in self.color_ranges:
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_kernel_size, self.morph_kernel_size),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > self.max_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                obj = DetectedObject(
                    label=color_range.name,
                    bbox=(x, y, w, h),
                    centroid_2d=(cx, cy),
                    area=area,
                    contour=contour,
                    confidence=min(1.0, area / self.max_area),
                )

                # 3D centroid from depth map
                if depth_map is not None:
                    obj.depth_mm = self._get_depth(depth_map, cx, cy)
                    if obj.depth_mm > 0 and Q is not None:
                        obj.centroid_3d = self._pixel_to_3d(cx, cy, obj.depth_mm, Q)

                detections.append(obj)

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def _get_depth(self, depth_map: np.ndarray, x: int, y: int, window: int = 5) -> float:
        """Get median depth around a pixel."""
        h, w = depth_map.shape[:2]
        half = window // 2
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        region = depth_map[y0:y1, x0:x1]
        valid = region[region > 0]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def _pixel_to_3d(
        self, x: int, y: int, depth_mm: float, Q: np.ndarray
    ) -> tuple[float, float, float]:
        """Convert pixel + depth to 3D coordinates using Q matrix.

        Uses the principal point and focal length from Q to back-project.
        """
        # Q matrix layout:
        # [1  0  0    -cx     ]
        # [0  1  0    -cy     ]
        # [0  0  0     f      ]
        # [0  0 -1/Tx  (cx-cx')/Tx]
        cx = -Q[0, 3]
        cy = -Q[1, 3]
        f = Q[2, 3]

        Z = depth_mm
        X = (x - cx) * Z / f if f != 0 else 0
        Y = (y - cy) * Z / f if f != 0 else 0

        return (float(X), float(Y), float(Z))

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[DetectedObject],
        draw_contours: bool = True,
    ) -> np.ndarray:
        """Draw detection results on an image (returns a copy)."""
        vis = image.copy()

        for det in detections:
            x, y, w, h = det.bbox
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # Centroid
            cv2.circle(vis, det.centroid_2d, 5, (0, 0, 255), -1)

            # Label
            label = f"{det.label}"
            if det.depth_mm > 0:
                label += f" {det.depth_mm:.0f}mm"
            cv2.putText(
                vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

            if draw_contours:
                cv2.drawContours(vis, [det.contour], -1, color, 1)

        return vis
