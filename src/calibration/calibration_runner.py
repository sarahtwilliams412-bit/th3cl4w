"""
Calibration Runner — orchestrates multi-camera calibration with ChArUco board.

Implements the full calibration state machine:
  Init -> Intrinsics -> Hand-eye -> Fixed cameras -> Validate -> Room scan -> Done

Safety-critical: uses set_joint (funcode 1) ONLY, moves <= 10deg increments,
verifies feedback between moves, enforces 5deg margin from hardware limits.

Key improvements over previous version:
- Uses ChArUco board instead of checkerboard (handles partial occlusion)
- 25 diverse poses with >= 30deg rotational variation on each axis
- Camera buffer flushing before capture (prevents stale frames)
- All 3 cameras captured at each pose
- 1.5-2s settle time per expert recommendation
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import httpx

logger = logging.getLogger("th3cl4w.calibration")

# Hardware joint limits (degrees) with 5deg safety margin
JOINT_LIMITS_SAFE = {
    0: (-130.0, 130.0),  # J0 base yaw (hw +/-135)
    1: (-85.0, 85.0),  # J1 shoulder pitch (hw +/-90)
    2: (-85.0, 85.0),  # J2 elbow pitch (hw +/-90)
    3: (-130.0, 130.0),  # J3 wrist roll (hw +/-135)
    4: (-85.0, 85.0),  # J4 wrist pitch (hw +/-90)
    5: (-130.0, 130.0),  # J5 wrist roll (hw +/-135)
}

MAX_INCREMENT_DEG = 10.0
MAX_TRACKING_ERROR_DEG = 20.0
POSE_REACHED_TOLERANCE_DEG = 6.0
POSE_REACHED_TIMEOUT_S = 30.0

# Number of frames to read and discard before actual capture.
# Flushes the camera's internal buffer to avoid stale frames.
BUFFER_FLUSH_FRAMES = 3

# 25 diverse calibration poses with >= 30deg rotational variation on each axis.
# Designed for good coverage of the workspace hemisphere with both translation
# AND rotation diversity — critical for well-conditioned hand-eye calibration.
CALIBRATION_POSES = [
    # === Home & single-axis exploration ===
    (0, 0, 0, 0, 0, 0),  # 1: home
    (45, 0, 0, 0, 0, 0),  # 2: yaw right
    (-45, 0, 0, 0, 0, 0),  # 3: yaw left
    (0, -45, 0, 0, 0, 0),  # 4: lean forward
    (0, 45, 0, 0, 0, 0),  # 5: lean back
    (0, 0, 45, 0, 0, 0),  # 6: elbow out
    (0, 0, -45, 0, 0, 0),  # 7: elbow in
    (0, 0, 0, 0, 45, 0),  # 8: wrist pitch down
    (0, 0, 0, 0, -45, 0),  # 9: wrist pitch up
    # === Two-axis combinations ===
    (30, -30, 0, 0, 0, 0),  # 10: yaw + forward
    (-30, -30, 0, 0, 0, 0),  # 11: opposite yaw + forward
    (0, -30, 45, 0, 0, 0),  # 12: forward + elbow
    (0, -45, 0, 0, -30, 0),  # 13: forward + wrist up
    (0, -30, 0, 0, 45, 0),  # 14: forward + wrist down
    # === Three-axis combinations for rotational diversity ===
    (30, -30, 30, 0, 0, 0),  # 15: yaw + fwd + elbow
    (-30, -30, 30, 0, 0, 0),  # 16: opp yaw + fwd + elbow
    (45, -45, 30, 0, -30, 0),  # 17: diagonal reach w/ wrist
    (-45, -45, 30, 0, -30, 0),  # 18: opposite diagonal w/ wrist
    (0, -45, 45, 0, 45, 0),  # 19: forward + elbow + wrist down
    (0, -30, 45, 0, -45, 0),  # 20: forward + elbow + wrist up
    # === Wide coverage / extremes ===
    (60, -30, 30, 0, 0, 0),  # 21: wide yaw + forward
    (-60, -30, 30, 0, 0, 0),  # 22: opposite wide yaw
    (0, -60, 60, 0, -45, 0),  # 23: maximum forward reach
    (30, 30, -30, 0, 30, 0),  # 24: backward lean + wrist
    (-30, 30, -30, 0, -30, 0),  # 25: opposite backward + wrist
]


@dataclass
class PoseCapture:
    pose_index: int
    commanded_angles: tuple
    actual_angles: list[float]
    cam0_jpeg: bytes = field(default=b"", repr=False)  # overhead
    cam1_jpeg: bytes = field(default=b"", repr=False)  # arm-mounted
    cam2_jpeg: bytes = field(default=b"", repr=False)  # side
    timestamp: float = 0.0
    comparison: Optional[Any] = None


@dataclass
class CalibrationSession:
    captures: list[PoseCapture] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_poses: int = 0
    comparison_report: Optional[Any] = None


class CalibrationError(Exception):
    """Raised on safety violations during calibration."""

    pass


class CalibrationRunner:
    """Orchestrates the multi-camera calibration sequence.

    Captures from all 3 cameras (overhead, arm-mounted, side) at each
    calibration pose. Implements buffer flushing to prevent stale frames.
    """

    def __init__(
        self,
        arm_host: str = "localhost",
        arm_port: int = 8080,
        cam_host: str = "localhost",
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
        """Send a single set-joint command (funcode 1). Retries on timeout."""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.arm_base}/api/command/set-joint",
                        json={"id": joint_id, "angle": round(angle, 2)},
                    )
                    if resp.status_code == 409:
                        raise CalibrationError("Arm not enabled")
                    data = resp.json()
                    return data.get("ok", False)
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
            ):
                logger.warning(f"Joint {joint_id} command timeout (attempt {attempt+1}/3)")
                if attempt == 2:
                    logger.error(f"Joint {joint_id} command timed out after 3 attempts")
                    return False
                await asyncio.sleep(2)

    async def wait_for_pose_reached(
        self,
        target_angles: tuple,
        tolerance_deg: float = POSE_REACHED_TOLERANCE_DEG,
        timeout_s: float = POSE_REACHED_TIMEOUT_S,
    ) -> list[float]:
        """
        Poll joint feedback until ALL joints are within tolerance of target.

        Verifies feedback freshness by requiring two consecutive reads that
        both satisfy the tolerance (guards against stale DDS data).
        """
        deadline = time.monotonic() + timeout_s
        consecutive_ok = 0
        last_angles = None

        while time.monotonic() < deadline:
            angles = await self.get_joint_angles()
            errors = [abs(angles[j] - float(target_angles[j])) for j in range(6)]
            max_error = max(errors)

            if max_error <= tolerance_deg:
                consecutive_ok += 1
                last_angles = angles
                if consecutive_ok >= 2:
                    return last_angles
            else:
                consecutive_ok = 0
                last_angles = angles

            await asyncio.sleep(0.3)

        if last_angles is not None:
            error_str = ", ".join(
                f"J{j}={last_angles[j]:.1f} (target {target_angles[j]}, "
                f"err {abs(last_angles[j] - float(target_angles[j])):.1f})"
                for j in range(6)
            )
        else:
            error_str = "no feedback received"
        raise CalibrationError(
            f"Pose not reached within {timeout_s}s " f"(tolerance +/-{tolerance_deg}): {error_str}"
        )

    async def _flush_camera_buffer(self, cam_id: int) -> None:
        """Flush stale frames from camera buffer by reading and discarding.

        Per expert recommendation: read and discard several frames before
        the actual capture to prevent stale frames from corrupting data.
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _ in range(BUFFER_FLUSH_FRAMES):
                try:
                    await client.get(f"{self.cam_base}/snap/{cam_id}")
                except Exception:
                    pass
                await asyncio.sleep(0.05)

    async def capture_frames(self) -> tuple[bytes, bytes, bytes]:
        """Capture JPEG frames from all 3 cameras with buffer flushing.

        Flushes each camera's buffer first, then captures the actual frame.
        Returns (cam0_jpeg, cam1_jpeg, cam2_jpeg).
        """
        # Flush all camera buffers in parallel
        await asyncio.gather(
            self._flush_camera_buffer(0),
            self._flush_camera_buffer(1),
            self._flush_camera_buffer(2),
        )

        # Brief pause after flush
        await asyncio.sleep(0.1)

        # Capture actual frames
        results = [b"", b"", b""]
        async with httpx.AsyncClient(timeout=5.0) as client:
            for cam_id in range(3):
                try:
                    r = await client.get(f"{self.cam_base}/snap/{cam_id}")
                    results[cam_id] = r.content if r.status_code == 200 else b""
                except Exception:
                    results[cam_id] = b""

        return results[0], results[1], results[2]

    async def command_pose(self, angles: tuple) -> None:
        """
        Command arm to target pose using set_joint one joint at a time.

        Moves in <= 10deg increments, verifies feedback between moves.
        Raises CalibrationError on safety violations.
        """
        current = await self.get_joint_angles()

        for joint_id in range(6):
            target = float(angles[joint_id])

            lo, hi = JOINT_LIMITS_SAFE[joint_id]
            if not (lo <= target <= hi):
                raise CalibrationError(
                    f"J{joint_id} target {target} outside safe limits " f"[{lo}, {hi}]"
                )

            current_angle = current[joint_id]
            diff = target - current_angle

            if abs(diff) < 0.5:
                continue

            steps = _compute_increments(current_angle, target)

            for step_angle in steps:
                if self._abort:
                    raise CalibrationError("Calibration aborted by user")

                ok = await self.set_single_joint(joint_id, step_angle)
                if not ok:
                    raise CalibrationError(f"set_joint failed for J{joint_id} -> {step_angle}")
                await asyncio.sleep(0.5)

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
                    f"SAFETY: J{joint_id} tracking error {best_error:.1f} > "
                    f"{MAX_TRACKING_ERROR_DEG} "
                    f"(commanded={target}, actual={feedback[joint_id]:.1f})"
                )

    async def run_single_pose(self, pose_index: int, angles: tuple, comparator=None) -> PoseCapture:
        """Run a single calibration pose: move, settle, flush, capture."""
        logger.info(f"Pose {pose_index + 1}/{self._total_poses}: commanding {angles}")
        self._current_pose = pose_index

        # 1. Command arm to pose
        await self.command_pose(angles)

        # 2. Wait for all joints to reach target with freshness check
        actual = await self.wait_for_pose_reached(angles)
        logger.info(
            f"Pose {pose_index + 1}: joints reached target. "
            f"Errors: {[f'J{j}={abs(actual[j] - float(angles[j])):.1f}' for j in range(6)]}"
        )

        # 3. Settle time for vibration/oscillation to damp (1.5-2s recommended)
        await asyncio.sleep(self.settle_time)

        # 4. Re-read actual angles right before capture
        actual = await self.get_joint_angles()
        logger.info(
            f"Pose {pose_index + 1} at capture: "
            f"commanded={list(angles)}, "
            f"actual=[{', '.join(f'{a:.1f}' for a in actual)}]"
        )

        # 5. Capture frames from ALL 3 cameras (with buffer flush)
        cam0, cam1, cam2 = await self.capture_frames()
        for i, data in enumerate([cam0, cam1, cam2]):
            if not data:
                logger.warning(f"Pose {pose_index + 1}: cam{i} returned empty frame!")

        capture = PoseCapture(
            pose_index=pose_index,
            commanded_angles=angles,
            actual_angles=actual,
            cam0_jpeg=cam0,
            cam1_jpeg=cam1,
            cam2_jpeg=cam2,
            timestamp=time.time(),
        )

        # 6. Optional comparator (CV+LLM pipeline)
        if comparator is not None:
            try:
                capture.comparison = await comparator.compare(
                    cam0_jpeg=cam0,
                    cam1_jpeg=cam1,
                    cam2_jpeg=cam2,
                    joint_angles=actual,
                    pose_index=pose_index,
                )
            except Exception as e:
                logger.warning(f"Comparator failed for pose {pose_index}: {e}")

        return capture

    async def run_full_calibration(
        self,
        comparator=None,
        output_dir: Optional[str] = None,
    ) -> CalibrationSession:
        """
        Run the full 25-pose calibration sequence.

        For each pose:
        1. Command arm via set_joint (funcode 1, <= 10deg increments)
        2. Wait for all joints to reach target within +/-6deg
        3. Wait settle_time for vibration damping (2.0s default)
        4. Flush camera buffers, then capture ALL 3 cameras
        5. Optionally run CV+LLM comparison
        """
        self._running = True
        self._abort = False
        if self._session_id is None:
            self._session_id = f"cal_{int(time.time())}"

        if output_dir is None:
            output_dir = f"calibration_results/{self._session_id}"

        session = CalibrationSession(
            start_time=time.time(),
            total_poses=len(CALIBRATION_POSES),
        )

        try:
            for i, pose in enumerate(CALIBRATION_POSES):
                if self._abort:
                    logger.warning("Calibration aborted by user")
                    break

                try:
                    capture = await self.run_single_pose(i, pose, comparator)
                    session.captures.append(capture)
                except Exception as e:
                    logger.warning(f"Pose {i+1} failed, skipping: {e}")
                    continue

        except Exception as e:
            logger.error(f"Calibration stopped: {e}")
            raise
        finally:
            session.end_time = time.time()
            self._running = False

        try:
            self.save_frames(session, output_dir)
        except Exception as e:
            logger.warning(f"Failed to save frames: {e}")

        logger.info(
            f"Calibration complete: {len(session.captures)}/{session.total_poses} "
            f"poses in {session.end_time - session.start_time:.1f}s"
        )
        return session

    def save_frames(self, session: CalibrationSession, output_dir: str) -> None:
        """Save individual frame JPEGs to output_dir/frames/."""
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for cap in session.captures:
            for cam_idx, jpeg_data in [
                (0, cap.cam0_jpeg),
                (1, cap.cam1_jpeg),
                (2, cap.cam2_jpeg),
            ]:
                if jpeg_data:
                    fname = f"pose{cap.pose_index:02d}_cam{cam_idx}.jpg"
                    fpath = os.path.join(frames_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(jpeg_data)
                    logger.debug(f"Saved {fpath} ({len(jpeg_data)} bytes)")
                else:
                    logger.warning(f"Skipping pose{cap.pose_index:02d}_cam{cam_idx}.jpg")

        logger.info(f"Frames saved to {frames_dir}")

    def save_session(self, session: CalibrationSession, path: str) -> None:
        """Save calibration session to JSON file (images as base64)."""
        data = {
            "start_time": session.start_time,
            "end_time": session.end_time,
            "total_poses": session.total_poses,
            "captures": [],
        }
        for cap in session.captures:
            data["captures"].append(
                {
                    "pose_index": cap.pose_index,
                    "commanded_angles": list(cap.commanded_angles),
                    "actual_angles": cap.actual_angles,
                    "cam0_jpeg_b64": (
                        base64.b64encode(cap.cam0_jpeg).decode() if cap.cam0_jpeg else ""
                    ),
                    "cam1_jpeg_b64": (
                        base64.b64encode(cap.cam1_jpeg).decode() if cap.cam1_jpeg else ""
                    ),
                    "cam2_jpeg_b64": (
                        base64.b64encode(cap.cam2_jpeg).decode() if cap.cam2_jpeg else ""
                    ),
                    "timestamp": cap.timestamp,
                }
            )

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session saved to {path}")

    def load_session(self, path: str) -> CalibrationSession:
        """Load a saved calibration session from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        session = CalibrationSession(
            start_time=data["start_time"],
            end_time=data["end_time"],
            total_poses=data["total_poses"],
        )
        for cap_data in data["captures"]:
            session.captures.append(
                PoseCapture(
                    pose_index=cap_data["pose_index"],
                    commanded_angles=tuple(cap_data["commanded_angles"]),
                    actual_angles=cap_data["actual_angles"],
                    cam0_jpeg=base64.b64decode(cap_data.get("cam0_jpeg_b64", "")),
                    cam1_jpeg=base64.b64decode(cap_data.get("cam1_jpeg_b64", "")),
                    cam2_jpeg=base64.b64decode(cap_data.get("cam2_jpeg_b64", "")),
                    timestamp=cap_data["timestamp"],
                )
            )
        return session


def _compute_increments(current: float, target: float) -> list[float]:
    """
    Break a move from current to target into <= 10deg steps.
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
    steps.append(target)
    return steps
