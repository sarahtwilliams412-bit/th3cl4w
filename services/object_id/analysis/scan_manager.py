"""Workspace scan manager — orchestrates multi-pose 3D scanning.

Moves the arm through predefined viewpoints, captures frames, estimates depth,
generates and merges point clouds into a PLY file.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import httpx

from shared.config.camera_config import CAM_OVERHEAD, CAMERA_SERVER_URL as _DEFAULT_CAMERA_URL

logger = logging.getLogger(__name__)

# Scan data directory
SCAN_DIR = Path(__file__).parent.parent.parent / "data" / "scans"

# Predefined scan viewpoints (joint angles in degrees)
# 8 viewpoints: circle around workspace at 2 heights
SCAN_POSES: List[Dict[str, float]] = [
    {"J0": angle, "J1": 30, "J2": 40, "J3": 0, "J4": 50, "J5": 0} for angle in [-60, -30, 0, 30, 60]
] + [{"J0": angle, "J1": 50, "J2": 30, "J3": 0, "J4": 40, "J5": 0} for angle in [-45, 0, 45]]

# Joint limits for safety
JOINT_LIMITS = {
    0: (-135, 135),
    1: (-80, 80),
    2: (-80, 80),
    3: (-135, 135),
    4: (-80, 80),
    5: (-135, 135),
}


def _pose_to_angles(pose: Dict[str, float]) -> List[float]:
    """Convert pose dict to list of 6 joint angles."""
    return [pose.get(f"J{i}", 0.0) for i in range(6)]


def _validate_pose(pose: Dict[str, float]) -> bool:
    """Check that all joints are within safe limits."""
    angles = _pose_to_angles(pose)
    for i, a in enumerate(angles):
        lo, hi = JOINT_LIMITS.get(i, (-135, 135))
        if not (lo <= a <= hi):
            logger.warning("Pose rejected: J%d=%.1f outside [%.1f, %.1f]", i, a, lo, hi)
            return False
    return True


class ScanStatus:
    """Thread-safe scan status tracker."""

    def __init__(self):
        self.running = False
        self.current_pose = 0
        self.total_poses = 0
        self.phase = "idle"  # idle, moving, capturing, processing, merging, done, error, aborted
        self.error: Optional[str] = None
        self.result_path: Optional[str] = None
        self.start_time: Optional[float] = None
        self.points_total = 0
        self.scan_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "running": self.running,
            "current_pose": self.current_pose,
            "total_poses": self.total_poses,
            "phase": self.phase,
            "error": self.error,
            "result_path": self.result_path,
            "elapsed_seconds": round(elapsed, 1),
            "points_total": self.points_total,
            "scan_id": self.scan_id,
        }


class ScanManager:
    """Orchestrates workspace scanning via arm movement + depth estimation."""

    def __init__(
        self,
        arm_command_fn=None,
        get_state_fn=None,
        camera_url: str = None,
        poses: Optional[List[Dict[str, float]]] = None,
    ):
        """
        Parameters
        ----------
        arm_command_fn : async callable(angles: list[float]) -> bool
            Function to move arm to target joint angles.
        get_state_fn : callable() -> dict
            Returns current arm state (with 'joints' key).
        camera_url : URL of camera server for frame capture.
        poses : list of scan poses (default: SCAN_POSES).
        """
        self.arm_command_fn = arm_command_fn
        self.get_state_fn = get_state_fn
        self.camera_url = camera_url or _DEFAULT_CAMERA_URL
        self.poses = poses or SCAN_POSES
        self.status = ScanStatus()
        self._abort = False
        self._task: Optional[asyncio.Task] = None

    async def _wait_for_position(
        self, target_angles: List[float], tolerance: float = 2.0, timeout: float = 5.0
    ) -> bool:
        """Poll arm state until joints reach target angles (within tolerance degrees)."""
        elapsed = 0.0
        while elapsed < timeout:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get("http://localhost:8080/api/state")
                    state = resp.json()
                    current = state.get("joint_angles", [])
                    if len(current) >= 6 and all(
                        abs(current[i] - target_angles[i]) < tolerance
                        for i in range(min(len(target_angles), 6))
                    ):
                        return True
            except Exception:
                pass  # connection error — retry
            await asyncio.sleep(0.1)
            elapsed += 0.1
        logger.warning("Motion timeout after %.1fs — continuing scan", timeout)
        return False

    async def start_scan(self) -> Dict[str, Any]:
        """Start a workspace scan. Returns immediately."""
        if self.status.running:
            return {"ok": False, "error": "Scan already in progress"}

        # Validate all poses
        valid_poses = [p for p in self.poses if _validate_pose(p)]
        if not valid_poses:
            return {"ok": False, "error": "No valid scan poses"}

        self._abort = False
        self.status = ScanStatus()
        self.status.running = True
        self.status.total_poses = len(valid_poses)
        self.status.start_time = time.time()
        self.status.scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.status.phase = "starting"

        self._task = asyncio.create_task(self._run_scan(valid_poses))
        return {"ok": True, "scan_id": self.status.scan_id, "total_poses": len(valid_poses)}

    async def stop_scan(self) -> Dict[str, Any]:
        """Abort the current scan."""
        if not self.status.running:
            return {"ok": False, "error": "No scan in progress"}
        self._abort = True
        self.status.phase = "aborted"
        self.status.running = False
        if self._task:
            self._task.cancel()
        return {"ok": True}

    async def _run_scan(self, poses: List[Dict[str, float]]):
        """Execute the scan sequence."""
        from services.positioning.depth import depth_estimator
        from services.positioning.depth.pointcloud_gen import (
            backproject_depth,
            compute_camera_pose_from_joints,
            merge_point_clouds,
            voxel_downsample,
            save_ply,
        )

        clouds: List[np.ndarray] = []
        scan_dir = SCAN_DIR / self.status.scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "scan_id": self.status.scan_id,
            "start_time": datetime.now().isoformat(),
            "poses": [],
        }

        try:
            for i, pose in enumerate(poses):
                if self._abort:
                    break

                self.status.current_pose = i + 1
                self.status.phase = "moving"
                angles = _pose_to_angles(pose)

                # Move arm to pose
                if self.arm_command_fn:
                    try:
                        ok = await self.arm_command_fn(angles)
                        if not ok:
                            logger.warning("Failed to move to pose %d, skipping", i)
                            continue
                    except Exception as e:
                        logger.warning("Arm move error at pose %d: %s", i, e)
                        continue

                    # Wait for arm to reach target position
                    await self._wait_for_position(angles)

                # Capture frame from overhead camera
                self.status.phase = "capturing"
                frame = await self._capture_frame(camera_id=CAM_OVERHEAD)
                if frame is None:
                    logger.warning("No frame at pose %d, skipping", i)
                    continue

                # Save raw frame
                frame_path = scan_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                # Estimate depth
                self.status.phase = "processing"
                if not depth_estimator.is_available():
                    logger.error("Depth estimator not available")
                    self.status.error = "Depth model not available"
                    self.status.phase = "error"
                    self.status.running = False
                    return

                # Get current joint angles for FK
                current_angles = angles
                if self.get_state_fn:
                    try:
                        state = self.get_state_fn()
                        if state and "joints" in state:
                            current_angles = state["joints"]
                    except Exception:
                        pass

                # Estimate depth (relative, then scale to approximate metric)
                rel_depth = depth_estimator.estimate_depth(frame)
                if rel_depth is None:
                    logger.warning("Depth estimation failed at pose %d", i)
                    continue

                # Scale relative depth to approximate metric using arm height
                # At typical scan pose, camera is ~0.3-0.5m from workspace
                camera_pose = compute_camera_pose_from_joints(current_angles)
                camera_height = max(abs(camera_pose[2, 3]), 0.2)  # z position as reference
                metric_depth = rel_depth * camera_height * 2  # rough scale

                # Generate point cloud
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cloud = backproject_depth(
                    metric_depth,
                    frame_rgb,
                    camera_pose=camera_pose,
                    subsample=2,
                )

                if len(cloud) > 0:
                    clouds.append(cloud)
                    self.status.points_total += len(cloud)

                metadata["poses"].append(
                    {
                        "index": i,
                        "target_angles": angles,
                        "actual_angles": current_angles,
                        "points": len(cloud),
                        "camera_pose": camera_pose.tolist(),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                logger.info("Pose %d/%d: %d points", i + 1, len(poses), len(cloud))

            if self._abort:
                self.status.phase = "aborted"
                self.status.running = False
                return

            # Merge and downsample
            self.status.phase = "merging"
            if clouds:
                merged = merge_point_clouds(clouds)
                logger.info("Merged: %d points", len(merged))

                # Voxel downsample
                merged = voxel_downsample(merged, voxel_size=0.005)
                logger.info("After downsampling: %d points", len(merged))
                self.status.points_total = len(merged)

                # Save PLY
                ply_path = scan_dir / "scan.ply"
                save_ply(merged, str(ply_path))
                self.status.result_path = str(ply_path)

                # Also save to latest
                latest_path = SCAN_DIR / "latest.ply"
                save_ply(merged, str(latest_path))
            else:
                self.status.error = "No point clouds generated"

            # Save metadata
            metadata["end_time"] = datetime.now().isoformat()
            metadata["total_points"] = self.status.points_total
            with open(scan_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.status.phase = "done"
            self.status.running = False
            logger.info(
                "Scan complete: %s (%d points)", self.status.scan_id, self.status.points_total
            )

        except asyncio.CancelledError:
            self.status.phase = "aborted"
            self.status.running = False
        except Exception as e:
            logger.error("Scan error: %s", e, exc_info=True)
            self.status.error = str(e)
            self.status.phase = "error"
            self.status.running = False

    async def _capture_frame(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """Capture a frame from the camera server."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.camera_url}/snap/{camera_id}")
                if resp.status_code == 200:
                    img_array = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return frame
        except ImportError:
            # Try urllib
            import urllib.request

            try:
                with urllib.request.urlopen(
                    f"{self.camera_url}/snap/{camera_id}", timeout=5
                ) as resp:
                    img_array = np.frombuffer(resp.read(), dtype=np.uint8)
                    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.warning("Frame capture failed (urllib): %s", e)
        except Exception as e:
            logger.warning("Frame capture failed: %s", e)

        return None

    @staticmethod
    def list_scans() -> List[Dict[str, Any]]:
        """List all previous scans."""
        scans = []
        if not SCAN_DIR.exists():
            return scans

        for d in sorted(SCAN_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d / "metadata.json").exists():
                try:
                    with open(d / "metadata.json") as f:
                        meta = json.load(f)
                    scans.append(
                        {
                            "scan_id": d.name,
                            "start_time": meta.get("start_time"),
                            "end_time": meta.get("end_time"),
                            "total_points": meta.get("total_points", 0),
                            "num_poses": len(meta.get("poses", [])),
                            "has_ply": (d / "scan.ply").exists(),
                        }
                    )
                except Exception:
                    scans.append({"scan_id": d.name, "has_ply": (d / "scan.ply").exists()})

        return scans

    @staticmethod
    def get_scan_ply(scan_id: Optional[str] = None) -> Optional[str]:
        """Get path to a scan's PLY file. None = latest."""
        if scan_id:
            path = SCAN_DIR / scan_id / "scan.ply"
        else:
            path = SCAN_DIR / "latest.ply"

        if path.exists():
            return str(path)
        return None
