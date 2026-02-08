"""
Stereo depth estimation using Semi-Global Block Matching (SGBM).

Computes disparity maps, converts to depth, and generates point clouds.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .calibration import StereoCalibrator

logger = logging.getLogger("th3cl4w.vision.stereo_depth")


class StereoDepthEstimator:
    """Compute depth from rectified stereo image pairs using SGBM."""

    def __init__(
        self,
        calibrator: StereoCalibrator,
        min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        disp12_max_diff: int = 1,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
    ):
        self.calibrator = calibrator

        # SGBM parameters
        cn = 3  # number of channels
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.p1 = p1 if p1 is not None else 8 * cn * block_size**2
        self.p2 = p2 if p2 is not None else 32 * cn * block_size**2

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
        rectify: bool = True,
    ) -> np.ndarray:
        """Compute disparity map from a stereo pair.

        Args:
            left: Left image (BGR).
            right: Right image (BGR).
            rectify: Whether to apply rectification first.

        Returns:
            Disparity map as float32 (in pixels). Invalid pixels are <= 0.
        """
        if rectify and self.calibrator.is_calibrated:
            left, right = self.calibrator.rectify(left, right)

        disparity = self.stereo.compute(left, right)
        # SGBM returns fixed-point disparity (multiplied by 16)
        return disparity.astype(np.float32) / 16.0

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity map to depth map (mm).

        Uses the Q matrix from stereo calibration.
        Invalid disparities (<=0) produce depth of 0.

        Returns:
            Depth map in mm, same shape as disparity.
        """
        if self.calibrator.Q is None:
            raise RuntimeError("Q matrix not available — calibration required")

        # Q matrix maps (x, y, disparity, 1) -> (X, Y, Z, W) in homogeneous coords
        # Q[2][3] is focal length, Q[3][2] is -1/baseline
        # depth = focal * baseline / disparity = -Q[2][3] / Q[3][2] / disparity
        # But we can also use reprojectImageTo3D for the full thing.
        # For just depth, direct computation is faster:
        mask = disparity > 0
        depth = np.zeros_like(disparity)

        focal = self.calibrator.Q[2, 3]
        baseline_inv = self.calibrator.Q[3, 2]  # -1/Tx

        if baseline_inv != 0:
            depth[mask] = focal / (baseline_inv * disparity[mask])
            # Clamp unreasonable depths
            depth[depth < 0] = 0
            depth[depth > 10000] = 0  # >10m is noise
        return depth

    def compute_depth(
        self,
        left: np.ndarray,
        right: np.ndarray,
        rectify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both disparity and depth maps.

        Returns:
            (disparity, depth) tuple.
        """
        disparity = self.compute_disparity(left, right, rectify=rectify)
        depth = self.disparity_to_depth(disparity)
        return disparity, depth

    def compute_point_cloud(
        self,
        disparity: np.ndarray,
        left_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a 3D point cloud from a disparity map.

        Args:
            disparity: Disparity map (float32).
            left_image: Optional BGR image for coloring points.

        Returns:
            Nx3 or Nx6 array of points (X, Y, Z [, B, G, R]) in mm.
            Only valid (finite, non-zero depth) points are included.
        """
        if self.calibrator.Q is None:
            raise RuntimeError("Q matrix not available — calibration required")

        points_3d = cv2.reprojectImageTo3D(disparity, self.calibrator.Q, handleMissingValues=True)

        # Filter invalid points
        mask = (
            np.isfinite(points_3d[:, :, 2])
            & (points_3d[:, :, 2] > 0)
            & (points_3d[:, :, 2] < 10000)
        )

        points = points_3d[mask]

        if left_image is not None and left_image.shape[:2] == disparity.shape[:2]:
            colors = left_image[mask]
            points = np.hstack([points, colors.astype(np.float32)])

        return points

    def get_depth_at(self, depth_map: np.ndarray, x: int, y: int, window: int = 3) -> float:
        """Get depth at a pixel location, averaging over a small window.

        Returns depth in mm, or 0 if invalid.
        """
        h, w = depth_map.shape[:2]
        half = window // 2
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)

        region = depth_map[y0:y1, x0:x1]
        valid = region[region > 0]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))
