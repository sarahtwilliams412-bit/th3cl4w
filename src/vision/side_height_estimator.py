"""Side camera height estimator for Z-axis positioning.

Uses the side-view camera (cam2, /dev/video6) to estimate the Z height
(mm above table) of the gripper and objects via pixel-Y → Z mapping.

Calibration is stored in data/side_calibration.json as a set of
(pixel_y, z_mm) reference points, fitted with a linear model.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CALIBRATION_FILE = DATA_DIR / "side_calibration.json"

# Default HSV ranges for object detection in side view
# Red Bull can: red + blue/silver
REDBULL_HSV_RANGES = [
    # Red (two ranges for hue wrap)
    (np.array([0, 100, 80]), np.array([10, 255, 255])),
    (np.array([160, 100, 80]), np.array([180, 255, 255])),
]

# Gripper/arm: dark segments or neon tape
GRIPPER_HSV_RANGES = [
    # Dark arm segments (low saturation, low value)
    (np.array([0, 0, 0]), np.array([180, 80, 60])),
]

# Neon green tape markers (if present)
NEON_TAPE_HSV = (np.array([35, 100, 100]), np.array([85, 255, 255]))


class SideHeightEstimator:
    """Estimates Z height from the side camera using pixel-Y calibration."""

    def __init__(self, calibration_path: Optional[str] = None):
        self.cal_path = Path(calibration_path) if calibration_path else CALIBRATION_FILE
        self.reference_points: list[tuple[float, float]] = []  # (pixel_y, z_mm)
        # Linear model: z_mm = slope * pixel_y + intercept
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.calibrated: bool = False
        self.image_height: int = 1080
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration from JSON file if it exists."""
        if self.cal_path.exists():
            try:
                data = json.loads(self.cal_path.read_text())
                self.reference_points = [
                    (p["pixel_y"], p["z_mm"]) for p in data.get("points", [])
                ]
                self.image_height = data.get("image_height", 1080)
                if len(self.reference_points) >= 2:
                    self._fit_model()
                logger.info(
                    "Loaded side calibration: %d points, calibrated=%s",
                    len(self.reference_points),
                    self.calibrated,
                )
            except Exception as e:
                logger.warning("Failed to load side calibration: %s", e)

    def _fit_model(self):
        """Fit linear model from reference points."""
        if len(self.reference_points) < 2:
            self.calibrated = False
            return
        pts = np.array(self.reference_points)
        pixel_ys = pts[:, 0]
        z_mms = pts[:, 1]
        # Linear fit: z_mm = slope * pixel_y + intercept
        # In side view, higher pixel_y = lower in image = lower Z
        coeffs = np.polyfit(pixel_ys, z_mms, 1)
        self.slope = float(coeffs[0])
        self.intercept = float(coeffs[1])
        self.calibrated = True
        logger.info(
            "Side calibration fit: z_mm = %.4f * pixel_y + %.2f",
            self.slope,
            self.intercept,
        )

    def calibrate(self, points: list[tuple[float, float]], image_height: int = 1080):
        """Set calibration reference points and save.

        Args:
            points: List of (pixel_y, z_mm) pairs.
            image_height: Image height in pixels.
        """
        self.reference_points = list(points)
        self.image_height = image_height
        self._fit_model()
        self._save_calibration()

    def _save_calibration(self):
        """Save calibration to JSON."""
        self.cal_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image_height": self.image_height,
            "slope": self.slope,
            "intercept": self.intercept,
            "calibrated": self.calibrated,
            "points": [
                {"pixel_y": py, "z_mm": zm} for py, zm in self.reference_points
            ],
        }
        self.cal_path.write_text(json.dumps(data, indent=2))
        logger.info("Saved side calibration to %s", self.cal_path)

    def pixel_y_to_z_mm(self, pixel_y: float) -> float:
        """Convert a pixel Y coordinate to Z height in mm."""
        if not self.calibrated:
            raise RuntimeError("Side camera not calibrated")
        return self.slope * pixel_y + self.intercept

    def detect_target(
        self, frame: np.ndarray, target: str = "redbull"
    ) -> Optional[dict]:
        """Detect a target in the side-view frame.

        Args:
            frame: BGR image from side camera.
            target: "gripper", "redbull", or "neon_tape".

        Returns:
            Dict with 'bbox', 'centroid', 'bottom_y', 'top_y' or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if target == "redbull":
            mask = self._detect_redbull(hsv)
        elif target == "gripper":
            mask = self._detect_gripper(hsv)
        elif target == "neon_tape":
            mask = self._detect_neon_tape(hsv)
        else:
            logger.warning("Unknown target type: %s", target)
            return None

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 300:
            return None

        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return {
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "bottom_y": y + h,  # lowest point in image = closest to table
            "top_y": y,  # highest point in image = furthest from table
            "area": area,
        }

    def _detect_redbull(self, hsv: np.ndarray) -> np.ndarray:
        """Detect Red Bull can in HSV image (red regions)."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in REDBULL_HSV_RANGES:
            mask |= cv2.inRange(hsv, lower, upper)
        return mask

    def _detect_gripper(self, hsv: np.ndarray) -> np.ndarray:
        """Detect gripper/arm in HSV image (dark segments)."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in GRIPPER_HSV_RANGES:
            mask |= cv2.inRange(hsv, lower, upper)
        return mask

    def _detect_neon_tape(self, hsv: np.ndarray) -> np.ndarray:
        """Detect neon tape markers on gripper."""
        return cv2.inRange(hsv, NEON_TAPE_HSV[0], NEON_TAPE_HSV[1])

    def estimate_height(
        self, frame: np.ndarray, target: str = "gripper"
    ) -> Optional[float]:
        """Estimate the Z height (mm above table) of a target.

        Args:
            frame: BGR image from side camera.
            target: "gripper" or "redbull".

        Returns:
            Z height in mm, or None if target not detected or not calibrated.
        """
        if not self.calibrated:
            logger.warning("Side camera not calibrated — cannot estimate height")
            return None

        detection = self.detect_target(frame, target)
        if detection is None:
            return None

        # Use bottom edge for objects (height of base),
        # centroid for gripper (center of gripper)
        if target == "gripper" or target == "neon_tape":
            ref_y = detection["centroid"][1]
        else:
            # For objects, bottom edge = where it sits on table
            ref_y = detection["bottom_y"]

        z_mm = self.pixel_y_to_z_mm(ref_y)
        return z_mm

    def get_calibration_info(self) -> dict:
        """Return current calibration state."""
        return {
            "calibrated": self.calibrated,
            "num_points": len(self.reference_points),
            "slope": self.slope,
            "intercept": self.intercept,
            "image_height": self.image_height,
            "points": [
                {"pixel_y": py, "z_mm": zm} for py, zm in self.reference_points
            ],
        }
