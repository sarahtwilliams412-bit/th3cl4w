"""Arm-mounted camera alignment for fine-grained gripper positioning.

Uses the arm camera (cam1) to detect objects and compute centering error,
providing correction vectors to align the gripper over the target.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.config.pick_config import get_pick_config as _get_pick_config

logger = logging.getLogger(__name__)

# Default image dimensions for arm camera
ARM_CAM_WIDTH = 1920
ARM_CAM_HEIGHT = 1080


def _arm_cam_hsv():
    ranges = _get_pick_config().get("arm_camera", "redbull_hsv_ranges")
    return [(np.array(r["lower"]), np.array(r["upper"])) for r in ranges]


# Legacy aliases
REDBULL_HSV_RANGES = [
    (np.array([0, 80, 60]), np.array([12, 255, 255])),
    (np.array([158, 80, 60]), np.array([180, 255, 255])),
    (np.array([100, 80, 60]), np.array([130, 255, 255])),
]
DEFAULT_PX_TO_MM = 0.18
DEFAULT_TOLERANCE_PX = 30


class ArmCameraAligner:
    """Computes alignment corrections using the arm-mounted camera."""

    def __init__(
        self,
        image_width: int = ARM_CAM_WIDTH,
        image_height: int = ARM_CAM_HEIGHT,
        px_to_mm: float = None,
        tolerance_px: int = None,
    ):
        cfg = _get_pick_config()
        self.image_width = image_width
        self.image_height = image_height
        self.cx = image_width // 2
        self.cy = image_height // 2
        self.px_to_mm = px_to_mm if px_to_mm is not None else cfg.get("arm_camera", "px_to_mm")
        self.tolerance_px = tolerance_px if tolerance_px is not None else cfg.get("arm_camera", "tolerance_px")

    def detect_object(
        self, frame: np.ndarray, target: str = "redbull"
    ) -> Optional[dict]:
        """Detect the target object in the arm camera frame.

        Args:
            frame: BGR image from arm camera.
            target: Object type to detect.

        Returns:
            Detection dict with 'centroid', 'bbox', 'area', or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if target == "redbull":
            mask = self._detect_redbull(hsv)
        else:
            # Generic: try red-ish objects
            mask = self._detect_redbull(hsv)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 200:
            return None

        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None

        obj_cx = int(M["m10"] / M["m00"])
        obj_cy = int(M["m01"] / M["m00"])

        return {
            "centroid": (obj_cx, obj_cy),
            "bbox": (x, y, w, h),
            "area": area,
        }

    def _detect_redbull(self, hsv: np.ndarray) -> np.ndarray:
        """Detect Red Bull can colors in HSV."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in _arm_cam_hsv():
            mask |= cv2.inRange(hsv, lower, upper)
        return mask

    def compute_alignment(
        self, frame: np.ndarray, target: str = "redbull"
    ) -> dict:
        """Compute alignment correction to center object under gripper.

        Args:
            frame: BGR image from arm camera.
            target: Object type to detect.

        Returns:
            Dict with:
                - centered: bool — True if object is within tolerance
                - detected: bool — True if object was found
                - error_px: (du, dv) pixel error from center
                - correction_mm: (dx, dy) correction in mm
                    dx > 0 means move arm right, dy > 0 means move arm forward
                - distance_px: scalar distance from center
        """
        detection = self.detect_object(frame, target)

        if detection is None:
            return {
                "centered": False,
                "detected": False,
                "error_px": (0, 0),
                "correction_mm": (0.0, 0.0),
                "distance_px": 0.0,
            }

        obj_cx, obj_cy = detection["centroid"]

        # Error: object position relative to image center
        du = obj_cx - self.cx  # positive = object is right of center
        dv = obj_cy - self.cy  # positive = object is below center

        distance_px = np.sqrt(du**2 + dv**2)
        centered = distance_px < self.tolerance_px

        # Convert pixel error to mm correction
        # The arm camera looks down, so:
        #   image +X (right) → arm +X (right) — but mirrored depending on mounting
        #   image +Y (down) → arm +Y (forward) — depends on camera orientation
        # Using negative because we want to move the arm TO the object:
        #   if object is right of center, move arm right (positive dx)
        # Camera is behind gripper looking forward/down, so:
        #   image X maps to arm X (same direction — camera faces same way)
        #   image Y maps to arm Y (same direction)
        dx_mm = du * self.px_to_mm
        dy_mm = dv * self.px_to_mm

        return {
            "centered": centered,
            "detected": True,
            "error_px": (du, dv),
            "correction_mm": (dx_mm, dy_mm),
            "distance_px": float(distance_px),
        }
