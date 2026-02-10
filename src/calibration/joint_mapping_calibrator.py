"""
Joint Mapping Calibration — moves one joint at a time and uses
camera frame differencing to detect which physical part moved.

This identifies the actual DDS-index-to-physical-joint correspondence.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_SERVER = "http://localhost:8081"
MOVE_DELTA_DEG = 10.0
SETTLE_TIME_S = 2.0

# Image regions for movement detection (approximate zones in normalized coords)
# These map detected movement regions to suggested physical joint labels.
REGION_LABELS = {
    "bottom-center": "base yaw (J0)",
    "upper": "shoulder pitch (J1)",
    "middle": "elbow pitch (J2)",
    "mid-rotate": "elbow roll (J3)",
    "lower-tip": "wrist pitch (J4)",
    "tip-rotate": "wrist roll (J5)",
}


class CalibrationState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class JointTestResult:
    """Result of testing a single DDS joint index."""
    dds_index: int
    moved: bool
    diff_score: float  # total frame difference magnitude
    diff_region: str  # which image region had the most change
    diff_center_y: float  # normalized Y center of change (0=top, 1=bottom)
    suggested_label: str
    skipped: bool = False
    skip_reason: str = ""
    cam0_diff_score: float = 0.0
    cam2_diff_score: float = 0.0


@dataclass
class CalibrationResult:
    """Full calibration result."""
    state: CalibrationState = CalibrationState.IDLE
    current_joint: int = -1
    total_joints: int = 6
    results: list[JointTestResult] = field(default_factory=list)
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0


class JointMappingCalibrator:
    """Orchestrates the joint mapping calibration sequence."""

    def __init__(self):
        self.result = CalibrationResult()
        self._abort = False

    @property
    def progress(self) -> dict:
        return {
            "state": self.result.state.value,
            "current_joint": self.result.current_joint,
            "total_joints": self.result.total_joints,
            "results": [
                {
                    "dds_index": r.dds_index,
                    "moved": r.moved,
                    "diff_score": round(r.diff_score, 1),
                    "diff_region": r.diff_region,
                    "diff_center_y": round(r.diff_center_y, 3),
                    "suggested_label": r.suggested_label,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in self.result.results
            ],
            "error": self.result.error,
        }

    def abort(self):
        self._abort = True

    async def run(
        self,
        get_angles_fn,
        set_joint_fn,
        is_enabled_fn,
        joint_limits: dict,
    ) -> CalibrationResult:
        """Run the full 6-joint calibration sequence.

        Args:
            get_angles_fn: callable returning list of 6 current joint angles (degrees)
            set_joint_fn: async callable(joint_id, angle_deg) to move a joint
            is_enabled_fn: callable returning bool (arm enabled?)
            joint_limits: dict of {joint_id: (min_deg, max_deg)}
        """
        import httpx

        self.result = CalibrationResult(
            state=CalibrationState.RUNNING,
            start_time=time.time(),
        )
        self._abort = False

        try:
            if not is_enabled_fn():
                self.result.state = CalibrationState.ERROR
                self.result.error = "Arm must be enabled before calibration"
                return self.result

            for dds_idx in range(6):
                if self._abort:
                    self.result.state = CalibrationState.ABORTED
                    return self.result

                self.result.current_joint = dds_idx
                logger.info("Testing DDS joint %d...", dds_idx)

                # Check if enabled (emergency stop check)
                if not is_enabled_fn():
                    self.result.state = CalibrationState.ERROR
                    self.result.error = f"Arm disabled during joint {dds_idx} test"
                    return self.result

                # Get current angles
                current_angles = get_angles_fn()
                current_angle = current_angles[dds_idx]
                lo, hi = joint_limits.get(dds_idx, (-135.0, 135.0))

                # Check if near limits — skip if can't move ±10°
                if current_angle + MOVE_DELTA_DEG > hi and current_angle - MOVE_DELTA_DEG < lo:
                    self.result.results.append(JointTestResult(
                        dds_index=dds_idx,
                        moved=False,
                        diff_score=0.0,
                        diff_region="none",
                        diff_center_y=0.5,
                        suggested_label="unknown (skipped)",
                        skipped=True,
                        skip_reason=f"Joint at {current_angle:.1f}°, too close to limits [{lo}, {hi}]",
                    ))
                    continue

                # Decide direction: prefer positive, fall back to negative
                target_angle = current_angle + MOVE_DELTA_DEG
                if target_angle > hi:
                    target_angle = current_angle - MOVE_DELTA_DEG

                # Capture BEFORE snapshots
                before_cam0 = await self._grab_snapshot(0)
                before_cam2 = await self._grab_snapshot(2)

                # Move joint
                await set_joint_fn(dds_idx, target_angle)
                await asyncio.sleep(SETTLE_TIME_S)

                # Capture AFTER snapshots
                after_cam0 = await self._grab_snapshot(0)
                after_cam2 = await self._grab_snapshot(2)

                # Return to original position
                await set_joint_fn(dds_idx, current_angle)
                await asyncio.sleep(SETTLE_TIME_S)

                # Analyze frame differences
                result = self._analyze_diff(dds_idx, before_cam0, after_cam0, before_cam2, after_cam2)
                self.result.results.append(result)
                logger.info(
                    "DDS joint %d: diff=%.1f, region=%s, label=%s",
                    dds_idx, result.diff_score, result.diff_region, result.suggested_label,
                )

            self.result.state = CalibrationState.COMPLETE
            self.result.end_time = time.time()
            return self.result

        except Exception as e:
            self.result.state = CalibrationState.ERROR
            self.result.error = str(e)
            logger.exception("Joint mapping calibration failed")
            return self.result

    async def _grab_snapshot(self, cam_id: int) -> Optional[np.ndarray]:
        """Fetch a JPEG snapshot from the camera server and decode it."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{CAMERA_SERVER}/snap/{cam_id}")
                if resp.status_code != 200:
                    return None
                return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("Failed to grab snapshot from cam %d: %s", cam_id, e)
            return None

    def _analyze_diff(
        self,
        dds_idx: int,
        before0: Optional[np.ndarray],
        after0: Optional[np.ndarray],
        before2: Optional[np.ndarray],
        after2: Optional[np.ndarray],
    ) -> JointTestResult:
        """Analyze frame differences to detect where movement occurred."""
        cam0_score = 0.0
        cam2_score = 0.0
        best_region = "unknown"
        best_center_y = 0.5

        # Process each camera pair
        for label, before, after in [("cam0", before0, after0), ("cam2", before2, after2)]:
            if before is None or after is None:
                continue

            # Convert to grayscale and blur to reduce noise
            gray_before = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            gray_after = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (21, 21), 0)

            # Absolute difference
            diff = cv2.absdiff(gray_before, gray_after)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            score = float(np.sum(thresh) / 255.0)  # number of changed pixels

            if label == "cam0":
                cam0_score = score
            else:
                cam2_score = score

            # Find center of mass of the difference
            if score > 50:  # minimum threshold for meaningful movement
                moments = cv2.moments(thresh)
                if moments["m00"] > 0:
                    cy = moments["m01"] / moments["m00"]
                    cx = moments["m10"] / moments["m00"]
                    h, w = thresh.shape
                    norm_y = cy / h  # 0=top, 1=bottom
                    norm_x = cx / w

                    # Determine region based on vertical position
                    if norm_y < 0.25:
                        region = "upper"
                    elif norm_y < 0.45:
                        region = "middle"
                    elif norm_y < 0.65:
                        region = "mid-rotate"
                    elif norm_y < 0.85:
                        region = "lower-tip"
                    else:
                        region = "bottom-center"

                    # Use the camera with more movement
                    if label == "cam0" and cam0_score >= cam2_score:
                        best_region = region
                        best_center_y = norm_y
                    elif label == "cam2" and cam2_score > cam0_score:
                        best_region = region
                        best_center_y = norm_y

        total_score = cam0_score + cam2_score
        moved = total_score > 100  # threshold: at least 100 changed pixels total

        # Suggest label based on region
        suggested = REGION_LABELS.get(best_region, f"unknown region ({best_region})")

        return JointTestResult(
            dds_index=dds_idx,
            moved=moved,
            diff_score=total_score,
            diff_region=best_region,
            diff_center_y=best_center_y,
            suggested_label=suggested,
            cam0_diff_score=cam0_score,
            cam2_diff_score=cam2_score,
        )
