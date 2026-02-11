"""Environment point cloud and voxel grid manager.

Ingests depth frames, back-projects to 3D, merges incrementally.
Wraps pointcloud_generator and depth_estimator for the map pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.vision.pointcloud_generator import (
    backproject_depth,
    compute_camera_pose_from_joints,
    merge_point_clouds,
    voxel_downsample,
    save_ply,
)

logger = logging.getLogger(__name__)

SCAN_DIR = Path(__file__).parent.parent.parent / "data" / "scans"


class EnvMapConfig:
    """Configuration for the environment mapper."""

    def __init__(self):
        self.voxel_size_m: float = 0.01
        self.max_points: int = 200_000
        self.depth_min_m: float = 0.05
        self.depth_max_m: float = 2.0
        self.camera_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_size_m": self.voxel_size_m,
            "max_points": self.max_points,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "camera_id": self.camera_id,
        }

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))


class EnvMap:
    """Manages the 3D environment point cloud and voxel grid."""

    def __init__(self, config: Optional[EnvMapConfig] = None):
        self.config = config or EnvMapConfig()
        self._cloud: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self._voxel_centers: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._last_update: float = 0.0
        self._depth_estimator = None
        self._depth_available: Optional[bool] = None

    def _ensure_depth_estimator(self) -> bool:
        """Lazy-load depth estimator."""
        if self._depth_available is not None:
            return self._depth_available
        try:
            from src.vision import depth_estimator

            self._depth_estimator = depth_estimator
            self._depth_available = depth_estimator.is_available()
            if not self._depth_available:
                # Try loading
                self._depth_available = depth_estimator._load_model()
            return self._depth_available
        except Exception as e:
            logger.warning("Depth estimator not available: %s", e)
            self._depth_available = False
            return False

    def ingest_depth_frame(
        self,
        rgb_frame: np.ndarray,
        joint_angles_deg: List[float],
        depth_map: Optional[np.ndarray] = None,
    ) -> int:
        """Ingest a single frame: estimate depth, back-project, merge.

        Args:
            rgb_frame: HxWx3 BGR frame from camera.
            joint_angles_deg: Current arm joint angles for camera pose.
            depth_map: Pre-computed depth map. If None, estimates depth.

        Returns:
            Number of new points added.
        """
        import cv2

        if depth_map is None:
            if not self._ensure_depth_estimator():
                return 0
            rel_depth = self._depth_estimator.estimate_depth(rgb_frame)
            if rel_depth is None:
                return 0
            # Scale relative depth to approximate metric
            camera_pose = compute_camera_pose_from_joints(joint_angles_deg)
            camera_height = max(abs(camera_pose[2, 3]), 0.2)
            depth_map = rel_depth * camera_height * 2
        else:
            camera_pose = compute_camera_pose_from_joints(joint_angles_deg)

        # Convert BGR to RGB for backprojection
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_frame

        new_points = backproject_depth(
            depth_map,
            rgb,
            camera_pose=camera_pose,
            max_depth=self.config.depth_max_m,
            min_depth=self.config.depth_min_m,
            subsample=2,
        )

        if len(new_points) == 0:
            return 0

        # Merge into existing cloud
        self._cloud = merge_point_clouds([self._cloud, new_points])

        # Downsample if too many points
        if len(self._cloud) > self.config.max_points:
            self._cloud = voxel_downsample(self._cloud, self.config.voxel_size_m)

        # Update voxel centers
        self._update_voxel_centers()
        self._last_update = time.time()

        return len(new_points)

    def _update_voxel_centers(self) -> None:
        """Compute voxel grid centers from current point cloud."""
        if len(self._cloud) == 0:
            self._voxel_centers = np.zeros((0, 3), dtype=np.float32)
            return

        vs = self.config.voxel_size_m
        voxel_keys = (self._cloud[:, :3] / vs).astype(np.int32)
        unique_keys, idx = np.unique(
            voxel_keys[:, 0] * 1_000_000 + voxel_keys[:, 1] * 1_000 + voxel_keys[:, 2],
            return_index=True,
        )
        self._voxel_centers = (voxel_keys[idx].astype(np.float32) + 0.5) * vs

    def get_cloud(self) -> np.ndarray:
        return self._cloud.copy()

    def get_voxel_centers(self) -> np.ndarray:
        return self._voxel_centers.copy()

    def get_stats(self) -> Dict[str, Any]:
        bounds_min = self._cloud[:, :3].min(axis=0).tolist() if len(self._cloud) > 0 else [0, 0, 0]
        bounds_max = self._cloud[:, :3].max(axis=0).tolist() if len(self._cloud) > 0 else [0, 0, 0]
        return {
            "total_points": len(self._cloud),
            "voxel_count": len(self._voxel_centers),
            "bounds": {"min": bounds_min, "max": bounds_max},
            "last_update": self._last_update,
            "voxel_size_m": self.config.voxel_size_m,
        }

    def clear(self) -> None:
        self._cloud = np.zeros((0, 6), dtype=np.float32)
        self._voxel_centers = np.zeros((0, 3), dtype=np.float32)
        self._last_update = 0.0

    def save_ply(self, filepath: str) -> bool:
        return save_ply(self._cloud, filepath)


class MapScanManager:
    """Wraps ScanManager for map-integrated arm-sweep scans.

    Feeds scan point clouds directly into the live EnvMap instead of
    just saving to PLY.
    """

    def __init__(
        self,
        env_map: EnvMap,
        arm_command_fn=None,
        get_state_fn=None,
        camera_url: str = None,
    ):
        from src.vision.scan_manager import ScanManager, ScanStatus
        from src.config.camera_config import CAMERA_SERVER_URL
        camera_url = camera_url or CAMERA_SERVER_URL

        self._env_map = env_map
        self._scanner = ScanManager(
            arm_command_fn=arm_command_fn,
            get_state_fn=get_state_fn,
            camera_url=camera_url,
        )
        self.status = self._scanner.status

    async def start_scan(self) -> Dict[str, Any]:
        return await self._scanner.start_scan()

    async def stop_scan(self) -> Dict[str, Any]:
        return await self._scanner.stop_scan()

    def get_status(self) -> Dict[str, Any]:
        return self._scanner.status.to_dict()

    @staticmethod
    def list_scans() -> List[Dict[str, Any]]:
        from src.vision.scan_manager import ScanManager

        return ScanManager.list_scans()

    @staticmethod
    def get_scan_ply(scan_id: Optional[str] = None) -> Optional[str]:
        from src.vision.scan_manager import ScanManager

        return ScanManager.get_scan_ply(scan_id)
