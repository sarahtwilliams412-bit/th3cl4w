"""
Calibration Runner — orchestrates 20-pose calibration sequence with dual CV+LLM pipelines.

Safety-critical: uses set_joint (funcode 1) ONLY, moves ≤10° increments,
verifies feedback between moves, enforces 5° margin from hardware limits.
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import httpx

logger = logging.getLogger("th3cl4w.calibration")

# Hardware joint limits (degrees) with 5° safety margin
JOINT_LIMITS_SAFE = {
    0: (-130.0, 130.0),   # J0 base yaw (hw ±135)
    1: (-85.0, 85.0),     # J1 shoulder pitch (hw ±90)
    2: (-85.0, 85.0),     # J2 elbow pitch (hw ±90)
    3: (-130.0, 130.0),   # J3 wrist roll (hw ±135)
    4: (-85.0, 85.0),     # J4 wrist pitch (hw ±90)
    5: (-130.0, 130.0),   # J5 wrist roll (hw ±135)
}

MAX_INCREMENT_DEG = 10.0
MAX_TRACKING_ERROR_DEG = 20.0

CALIBRATION_POSES = [
    (0, 0, 0, 0, 0, 0),           # home
    (30, 0, 0, 0, 0, 0),          # yaw right
    (-30, 0, 0, 0, 0, 0),         # yaw left
    (0, -30, 0, 0, 0, 0),         # lean forward
    (0, -60, 0, 0, 0, 0),         # lean far forward
    (0, 0, 45, 0, 0, 0),          # elbow out
    (0, 0, -45, 0, 0, 0),         # elbow in
    (0, -30, 30, 0, 0, 0),        # forward + elbow
    (0, -45, 45, 0, -30, 0),      # reaching forward-down
    (30, -30, 30, 0, 0, 0),       # right + forward + elbow
    (-30, -30, 30, 0, 0, 0),      # left + forward + elbow
    (0, 30, 0, 0, 0, 0),          # lean back
    (0, -30, 45, 0, 45, 0),       # forward + elbow + wrist down
    (45, -45, 30, 0, 0, 0),       # diagonal reach
    (-45, -45, 30, 0, 0, 0),      # opposite diagonal
    (0, 0, 0, 0, -45, 0),         # wrist up only
    (0, -30, 0, 0, 45, 0),        # forward + wrist down
    (60, 0, 30, 0, 0, 0),         # wide yaw + elbow
    (-60, 0, 30, 0, 0, 0),        # opposite wide yaw
    (0, -60, 60, 0, -45, 0),      # maximum forward reach
]


@dataclass
class PoseCapture:
    pose_index: int
    commanded_angles: tuple
    actual_angles: list[float]
    cam0_jpeg: bytes = field(repr=False)
    cam1_jpeg: bytes = field(repr=False)
    timestamp: float = 0.0
    comparison: Optional[Any] = None  # ComparisonResult when available


@dataclass
class CalibrationSession:
    captures: list[PoseCapture] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_poses: int = 0
    comparison_report: Optional[Any] = None  # ComparisonReport when available


class CalibrationError(Exception):
    """Raised on safety violations during calibration."""
    pass


class CalibrationRunner:
    """Orchestrates the 20-pose calibration sequence."""

    def __init__(
        self,
        arm_host: str = 'localhost',
        arm_port: int = 8080,
        cam_host: str = 'localhost',
        cam_port: int = 8081,
        settle_time: float = 2.5,
    ):
        self.arm_base = f"http://{arm_host}:{arm_port}"
        self.cam_base = f"http://{cam_host}:{cam_port}"
        self.settle_time = settle_time
        self._abort = False
        self._current_pose: int = -1
        self._total_poses: int = len(CALIBRATION_POSES)
        self._session_id: Optional[str] = None
        self._running = False

    @property
    def progress(self) -> dict:
        return {
            "running": self._running,
            "current_pose": self._current_pose,
            "total_poses": self._total_poses,
            "session_id": self._session_id,
            "aborted": self._abort,
        }

    def abort(self):
        """Signal the runner to stop after the current pose."""
        self._abort = True

    async def get_joint_angles(self) -> list[float]:
        """Read actual joint angles from /api/state."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.arm_base}/api/state")
            resp.raise_for_status()
            data = resp.json()
            return [float(j) for j in data["joints"]]

    async def set_single_joint(self, joint_id: int, angle: float) -> bool:
        """Send a single set-joint command (funcode 1)."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.arm_base}/api/command/set-joint",
                json={"id": joint_id, "angle": round(angle, 2)},
            )
            if resp.status_code == 409:
                raise CalibrationError("Arm not enabled")
            data = resp.json()
            return data.get("ok", False)

    async def capture_frames(self) -> tuple[bytes, bytes]:
        """Capture JPEG frames from both cameras."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r0 = await client.get(f"{self.cam_base}/snap/0")
                cam0 = r0.content if r0.status_code == 200 else b""
            except Exception:
                cam0 = b""
            try:
                r1 = await client.get(f"{self.cam_base}/snap/1")
                cam1 = r1.content if r1.status_code == 200 else b""
            except Exception:
                cam1 = b""
        return cam0, cam1

    async def command_pose(self, angles: tuple) -> None:
        """
        Command arm to target pose using set_joint (funcode 1) one joint at a time.
        
        Moves in ≤10° increments, verifies feedback between moves.
        Raises CalibrationError on safety violations.
        """
        current = await self.get_joint_angles()

        for joint_id in range(6):
            target = float(angles[joint_id])
            
            # Safety: check against limits with 5° margin
            lo, hi = JOINT_LIMITS_SAFE[joint_id]
            if not (lo <= target <= hi):
                raise CalibrationError(
                    f"J{joint_id} target {target}° outside safe limits [{lo}, {hi}]"
                )

            current_angle = current[joint_id]
            diff = target - current_angle

            if abs(diff) < 0.5:
                continue  # Already there

            # Break into ≤10° increments
            steps = _compute_increments(current_angle, target)

            for step_angle in steps:
                if self._abort:
                    raise CalibrationError("Calibration aborted by user")

                ok = await self.set_single_joint(joint_id, step_angle)
                if not ok:
                    raise CalibrationError(
                        f"set_joint failed for J{joint_id} -> {step_angle}°"
                    )
                # Brief pause for arm to move
                await asyncio.sleep(0.5)

            # Verify feedback after moving this joint — retry reads due to stale DDS feedback
            best_error = 999.0
            for _retry in range(5):
                await asyncio.sleep(0.5)
                feedback = await self.get_joint_angles()
                error = abs(feedback[joint_id] - target)
                best_error = min(best_error, error)
                if best_error <= MAX_TRACKING_ERROR_DEG:
                    break
            if best_error > MAX_TRACKING_ERROR_DEG:
                raise CalibrationError(
                    f"SAFETY: J{joint_id} tracking error {best_error:.1f}° > {MAX_TRACKING_ERROR_DEG}° "
                    f"(commanded={target}°, actual={feedback[joint_id]:.1f}°)"
                )

    async def run_single_pose(
        self, pose_index: int, angles: tuple, comparator=None
    ) -> PoseCapture:
        """Run a single calibration pose: move, settle, capture."""
        logger.info(f"Pose {pose_index + 1}/{self._total_poses}: {angles}")
        self._current_pose = pose_index

        # 1. Command arm to pose
        await self.command_pose(angles)

        # 2. Wait for settling
        await asyncio.sleep(self.settle_time)

        # 3. Read actual joint angles
        actual = await self.get_joint_angles()

        # 4. Capture frames
        cam0, cam1 = await self.capture_frames()

        capture = PoseCapture(
            pose_index=pose_index,
            commanded_angles=angles,
            actual_angles=actual,
            cam0_jpeg=cam0,
            cam1_jpeg=cam1,
            timestamp=time.time(),
        )

        # 5. If comparator provided, fire CV+LLM (LLM async)
        if comparator is not None:
            try:
                capture.comparison = await comparator.compare(
                    cam0_jpeg=cam0,
                    cam1_jpeg=cam1,
                    joint_angles=actual,
                    pose_index=pose_index,
                )
            except Exception as e:
                logger.warning(f"Comparator failed for pose {pose_index}: {e}")

        return capture

    async def run_full_calibration(self, comparator=None) -> CalibrationSession:
        """
        Run the full 20-pose calibration sequence.
        
        For each pose:
        1. Command arm via set_joint (funcode 1, ≤10° increments)
        2. Wait settle_time
        3. Read actual angles, capture frames
        4. Optionally run CV+LLM comparison
        
        Returns CalibrationSession with all captures.
        """
        self._running = True
        self._abort = False
        self._session_id = f"cal_{int(time.time())}"
        session = CalibrationSession(
            start_time=time.time(),
            total_poses=len(CALIBRATION_POSES),
        )

        try:
            for i, pose in enumerate(CALIBRATION_POSES):
                if self._abort:
                    logger.warning("Calibration aborted by user")
                    break

                capture = await self.run_single_pose(i, pose, comparator)
                session.captures.append(capture)

        except CalibrationError as e:
            logger.error(f"Calibration stopped: {e}")
            raise
        finally:
            session.end_time = time.time()
            self._running = False

        logger.info(
            f"Calibration complete: {len(session.captures)}/{session.total_poses} poses "
            f"in {session.end_time - session.start_time:.1f}s"
        )
        return session

    def save_session(self, session: CalibrationSession, path: str) -> None:
        """Save calibration session to JSON file (images as base64)."""
        data = {
            "start_time": session.start_time,
            "end_time": session.end_time,
            "total_poses": session.total_poses,
            "captures": [],
        }
        for cap in session.captures:
            data["captures"].append({
                "pose_index": cap.pose_index,
                "commanded_angles": list(cap.commanded_angles),
                "actual_angles": cap.actual_angles,
                "cam0_jpeg_b64": base64.b64encode(cap.cam0_jpeg).decode() if cap.cam0_jpeg else "",
                "cam1_jpeg_b64": base64.b64encode(cap.cam1_jpeg).decode() if cap.cam1_jpeg else "",
                "timestamp": cap.timestamp,
            })

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session saved to {path}")

    def load_session(self, path: str) -> CalibrationSession:
        """Load a saved calibration session from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        session = CalibrationSession(
            start_time=data["start_time"],
            end_time=data["end_time"],
            total_poses=data["total_poses"],
        )
        for cap_data in data["captures"]:
            session.captures.append(PoseCapture(
                pose_index=cap_data["pose_index"],
                commanded_angles=tuple(cap_data["commanded_angles"]),
                actual_angles=cap_data["actual_angles"],
                cam0_jpeg=base64.b64decode(cap_data.get("cam0_jpeg_b64", "")),
                cam1_jpeg=base64.b64decode(cap_data.get("cam1_jpeg_b64", "")),
                timestamp=cap_data["timestamp"],
            ))
        return session


def _compute_increments(current: float, target: float) -> list[float]:
    """
    Break a move from current to target into ≤10° steps.
    Returns list of intermediate + final angle values.
    """
    diff = target - current
    if abs(diff) <= MAX_INCREMENT_DEG:
        return [target]

    n_steps = int(abs(diff) / MAX_INCREMENT_DEG) + 1
    step_size = diff / n_steps
    steps = []
    for i in range(1, n_steps):
        steps.append(round(current + step_size * i, 2))
    steps.append(target)  # Always end exactly at target
    return steps
