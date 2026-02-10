"""Overhead camera calibrator: pixel ↔ workspace coordinate mapping.

Uses 4-point (or more) homography to map between overhead camera pixel
coordinates and workspace XY coordinates in mm from arm base.

This is SPECIFIC to the overhead camera (cam0, /dev/video0) which looks
straight down at the workspace. It uses a 2D homography (no Z component)
since the overhead view projects the workspace as a flat plane.

For side camera (cam2) height estimation, see side_height_estimator.py.
For arm-mounted camera (cam1), see hand_eye_calibrator.py and camera_model.py.
For full 3D extrinsics (any camera), see camera_model.py.

Calibration data stored in data/overhead_calibration.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default calibration file path
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_CALIBRATION_PATH = _DATA_DIR / "overhead_calibration.json"


class OverheadCalibrator:
    """Maps pixel coordinates from overhead camera to workspace XY (mm)."""

    def __init__(self, calibration_path: Optional[Path] = None):
        self.calibration_path = calibration_path or DEFAULT_CALIBRATION_PATH
        self._H: Optional[np.ndarray] = None  # pixel → world homography
        self._H_inv: Optional[np.ndarray] = None  # world → pixel homography
        self._pixel_points: list[list[float]] = []
        self._world_points: list[list[float]] = []
        self._reprojection_error: float = 0.0

        # Try to load existing calibration
        self._load()

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    @property
    def reprojection_error(self) -> float:
        return self._reprojection_error

    def calibrate(
        self,
        pixel_points: list[list[float]],
        world_points: list[list[float]],
    ) -> dict:
        """Compute homography from point correspondences.

        Parameters
        ----------
        pixel_points : List of [u, v] pixel coordinates (at least 4).
        world_points : List of [x_mm, y_mm] workspace coordinates (at least 4).

        Returns
        -------
        dict with calibration status, reprojection_error, etc.
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError("Need at least 4 point pairs for homography")
        if len(pixel_points) != len(world_points):
            raise ValueError("pixel_points and world_points must have same length")

        px = np.array(pixel_points, dtype=np.float32)
        wld = np.array(world_points, dtype=np.float32)

        H, mask = cv2.findHomography(px, wld, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Homography computation failed — points may be degenerate")

        H_inv, _ = cv2.findHomography(wld, px, cv2.RANSAC, 5.0)

        # Compute reprojection error
        errors = []
        for p, w in zip(px, wld):
            projected = self._apply_homography(H, float(p[0]), float(p[1]))
            err = np.sqrt((projected[0] - w[0]) ** 2 + (projected[1] - w[1]) ** 2)
            errors.append(err)
        reproj_err = float(np.mean(errors))

        self._H = H
        self._H_inv = H_inv
        self._pixel_points = [p.tolist() for p in px]
        self._world_points = [w.tolist() for w in wld]
        self._reprojection_error = reproj_err

        self._save()

        logger.info(
            "Overhead calibration complete: %d points, reprojection error %.2f mm",
            len(pixel_points), reproj_err,
        )

        return {
            "calibrated": True,
            "num_points": len(pixel_points),
            "reprojection_error_mm": round(reproj_err, 3),
        }

    def pixel_to_workspace(self, u: float, v: float) -> tuple[float, float]:
        """Convert pixel coordinates to workspace XY in mm.

        Parameters
        ----------
        u, v : Pixel coordinates in overhead camera image.

        Returns
        -------
        (x_mm, y_mm) in workspace frame.
        """
        if self._H is None:
            raise RuntimeError("Calibration not loaded. Call calibrate() first.")
        return self._apply_homography(self._H, u, v)

    def workspace_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Convert workspace XY (mm) to pixel coordinates.

        Parameters
        ----------
        x_mm, y_mm : Workspace coordinates in mm from arm base.

        Returns
        -------
        (u, v) pixel coordinates in overhead camera image.
        """
        if self._H_inv is None:
            raise RuntimeError("Calibration not loaded. Call calibrate() first.")
        return self._apply_homography(self._H_inv, x_mm, y_mm)

    def get_status(self) -> dict:
        """Return calibration status info."""
        return {
            "calibrated": self.is_calibrated,
            "num_points": len(self._pixel_points),
            "reprojection_error_mm": round(self._reprojection_error, 3),
            "pixel_points": self._pixel_points,
            "world_points": self._world_points,
        }

    @staticmethod
    def _apply_homography(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
        pt = H @ np.array([x, y, 1.0])
        return float(pt[0] / pt[2]), float(pt[1] / pt[2])

    def _save(self) -> None:
        """Persist calibration to JSON."""
        if self._H is None:
            return
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "homography": self._H.tolist(),
            "homography_inv": self._H_inv.tolist() if self._H_inv is not None else None,
            "pixel_points": self._pixel_points,
            "world_points": self._world_points,
            "reprojection_error_mm": self._reprojection_error,
        }
        self.calibration_path.write_text(json.dumps(data, indent=2))
        logger.info("Calibration saved to %s", self.calibration_path)

    def _load(self) -> None:
        """Load calibration from JSON if it exists."""
        if not self.calibration_path.exists():
            return
        try:
            data = json.loads(self.calibration_path.read_text())
            self._H = np.array(data["homography"], dtype=np.float64)
            if data.get("homography_inv"):
                self._H_inv = np.array(data["homography_inv"], dtype=np.float64)
            else:
                self._H_inv, _ = cv2.findHomography(
                    np.array(data["world_points"], dtype=np.float32),
                    np.array(data["pixel_points"], dtype=np.float32),
                )
            self._pixel_points = data.get("pixel_points", [])
            self._world_points = data.get("world_points", [])
            self._reprojection_error = data.get("reprojection_error_mm", 0.0)
            logger.info("Loaded overhead calibration (%d points)", len(self._pixel_points))
        except Exception as e:
            logger.warning("Failed to load calibration: %s", e)
            self._H = None
