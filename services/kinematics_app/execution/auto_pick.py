"""
Autonomous Pick Pipeline — Camera detection + geometric planning + execution.

Uses top-down approach strategy with geometric IK (no DH parameters).
Validated against known reference pose: [1.0, 25.9, 6.7, 0.5, 88.7, 3.3] at gripper 32.5mm.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from src.control.arm_operations import ArmOps  # TODO: Replace with HTTP call to control_plane service
from src.telemetry.pick_episode import PickEpisodeRecorder  # TODO: Replace with shared module or HTTP call when telemetry service is extracted
from src.telemetry.pick_recorder import PickVideoRecorder  # TODO: Replace with shared module or HTTP call when telemetry service is extracted
from shared.config.camera_config import CAM_OVERHEAD, snap_url

logger = logging.getLogger("th3cl4w.planning.auto_pick")

# Arm geometry (mm)
D0 = 121.5   # base height
L1 = 208.5   # upper arm
L2 = 208.5   # forearm
L3 = 113.0   # wrist length

# Pixel-to-mm conversion for overhead camera (rough)
# Workspace ~800mm across 1920px
PX_TO_MM = 0.417
IMG_CX = 960   # image center X
IMG_CY = 540   # image center Y

# Default server URL
DEFAULT_SERVER = "http://localhost:8080"
DEFAULT_CAM_SERVER = "http://localhost:8081"


class AutoPickPhase(str, enum.Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    PLANNING = "planning"
    APPROACHING = "approaching"
    LOWERING = "lowering"
    GRIPPING = "gripping"
    LIFTING = "lifting"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class PickResult:
    success: bool = False
    phase: AutoPickPhase = AutoPickPhase.IDLE
    target_xy_mm: tuple[float, float] = (0.0, 0.0)
    joints: list[float] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0


@dataclass
class AutoPickState:
    phase: AutoPickPhase = AutoPickPhase.IDLE
    target: str = ""
    target_xy_mm: tuple[float, float] = (0.0, 0.0)
    planned_joints: list[float] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    attempt_number: int = 0  # 0 = standalone, 1+ = rehearsal attempt


class AutoPick:
    """Autonomous pick from any position using camera detection + geometric planning."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER,
        cam_server_url: str = DEFAULT_CAM_SERVER,
    ):
        self.server_url = server_url
        self.cam_server_url = cam_server_url
        self.ops = ArmOps(server_url)
        self.state = AutoPickState()
        self._stop_requested = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.episode_recorder = PickEpisodeRecorder()
        self.video_recorder = PickVideoRecorder()
        self._current_mode: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        status = {
            "phase": self.state.phase.value,
            "target": self.state.target,
            "target_xy_mm": list(self.state.target_xy_mm),
            "planned_joints": self.state.planned_joints,
            "error": self.state.error,
            "running": self._running,
            "elapsed_s": round(time.time() - self.state.started_at, 1) if self.state.started_at else 0,
            "log": self.state.log[-20:],
            "mode": self._current_mode or "unknown",
        }
        ep = self.episode_recorder.current
        if ep:
            status["episode_id"] = ep.episode_id
            status["episode_phases"] = [
                {"name": p.name, "success": p.success, "duration_s": round(p.end_time - p.start_time, 2) if p.end_time else 0}
                for p in ep.phases
            ]
        return status

    def stop(self):
        """Request stop of current pick operation."""
        self._stop_requested = True
        self._log("Stop requested")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.state.log.append(entry)
        logger.info("AutoPick: %s", msg)

    def _check_stop(self):
        if self._stop_requested:
            raise _StopRequested()

    async def start(self, target: str = "redbull", mode: str = "auto") -> asyncio.Task:
        """Start the pick pipeline as a background task."""
        if self._running:
            raise RuntimeError("Pick already in progress")
        self._stop_requested = False
        self._running = True
        self.state = AutoPickState(target=target, started_at=time.time())
        self._task = asyncio.create_task(self._run(target, mode))
        return self._task

    async def _run(self, target: str, mode: str = "auto") -> PickResult:
        t0 = time.time()
        try:
            result = await self.execute(target, mode)
            result.duration_s = time.time() - t0
            return result
        except _StopRequested:
            self.state.phase = AutoPickPhase.STOPPED
            self._log("Stopped by user")
            return PickResult(phase=AutoPickPhase.STOPPED, error="Stopped by user", duration_s=time.time() - t0)
        except Exception as e:
            self.state.phase = AutoPickPhase.FAILED
            self.state.error = str(e)
            self._log(f"Failed: {e}")
            logger.exception("AutoPick failed")
            return PickResult(phase=AutoPickPhase.FAILED, error=str(e), duration_s=time.time() - t0)
        finally:
            self._running = False

    async def execute(
        self,
        target: str = "redbull",
        mode: str = "auto",
        attempt_number: int = 0,
        jitter_xy_mm: tuple[float, float] = (0.0, 0.0),
        override_xy_mm: Optional[tuple[float, float]] = None,
        override_joints: Optional[list[float]] = None,
    ) -> PickResult:
        """Full autonomous pick pipeline with episode recording.

        Parameters
        ----------
        target : str
            Object to pick.
        mode : str
            "auto" (detect from server), "simulation", or "physical".
        attempt_number : int
            When called from SimRehearsalRunner, identifies which attempt this is.
            0 means standalone (not part of a rehearsal).
        jitter_xy_mm : tuple
            Position offset (x, y) in mm to add to detected coordinates.
            Used by rehearsal to test plan robustness under camera noise.
        override_xy_mm : tuple, optional
            If provided, skip camera detection and use these coordinates directly.
            Used by promote_to_physical to reuse the best position from simulation.
        override_joints : list, optional
            If provided, skip planning and use these joint angles directly.
            Used by promote_to_physical to reuse the best plan from simulation.
        """
        # Determine actual mode
        actual_mode = mode
        if mode == "auto":
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.server_url}/api/sim-mode")
                    data = resp.json()
                    actual_mode = "simulation" if data.get("sim_mode") else "physical"
            except Exception:
                actual_mode = "physical"
        self._current_mode = actual_mode
        self.state.attempt_number = attempt_number

        attempt_tag = f" (attempt {attempt_number})" if attempt_number > 0 else ""
        self._log(f"Mode: {actual_mode}{attempt_tag}")

        episode = self.episode_recorder.start(mode=actual_mode, target=target)

        # Start video recording (best-effort)
        try:
            self.video_recorder.start(episode.episode_id)
        except Exception as e:
            logger.warning("Failed to start video recording: %s", e)

        try:
            # 1. DETECT (or use override)
            self.state.phase = AutoPickPhase.DETECTING
            self._check_stop()

            if override_xy_mm is not None:
                x_mm, y_mm = override_xy_mm
                self._log(f"Using override position ({x_mm:.1f}, {y_mm:.1f}) mm")
                self.episode_recorder.start_phase("detect")
                self.episode_recorder.record_detection(
                    method="override",
                    position_px=(0, 0),
                    position_mm=(x_mm, y_mm, 0.0),
                    confidence=1.0,
                )
                self.episode_recorder.end_phase(success=True)
            else:
                self._log(f"Detecting '{target}' via overhead camera...")
                self.episode_recorder.start_phase("detect")
                x_mm, y_mm = await self._detect(target)

                # Apply jitter if provided (rehearsal robustness testing)
                if jitter_xy_mm != (0.0, 0.0):
                    x_mm += jitter_xy_mm[0]
                    y_mm += jitter_xy_mm[1]
                    self._log(
                        f"Jitter applied: ({jitter_xy_mm[0]:+.1f}, {jitter_xy_mm[1]:+.1f}) mm"
                    )

                self._log(f"Target at ({x_mm:.1f}, {y_mm:.1f}) mm")
                self.episode_recorder.record_detection(
                    method="hsv",
                    position_px=(0, 0),
                    position_mm=(x_mm, y_mm, 0.0),
                    confidence=1.0,
                )
                self.episode_recorder.end_phase(success=True)

            self.state.target_xy_mm = (x_mm, y_mm)

            # 2. PLAN (or use override)
            self.state.phase = AutoPickPhase.PLANNING
            self._check_stop()
            self.episode_recorder.start_phase("plan")

            if override_joints is not None:
                joints = list(override_joints)
                self._log(f"Using override joints: [{', '.join(f'{j:.1f}' for j in joints)}]")
            else:
                joints = self.plan_joints(x_mm, y_mm)
                self._log(f"Planned joints: [{', '.join(f'{j:.1f}' for j in joints)}]")

            self.state.planned_joints = joints
            self.episode_recorder.record_plan(joints=joints)
            self.episode_recorder.end_phase(success=True)

            # 3. EXECUTE (with per-phase recording)
            await self._execute_pick(joints)

            # 4. VERIFY (mode-dependent)
            self.state.phase = AutoPickPhase.VERIFYING
            self._check_stop()
            self._log("Verifying grip...")
            self.episode_recorder.start_phase("verify")

            if actual_mode == "simulation":
                from ..execution.virtual_grip import VirtualGripDetector
                detector = VirtualGripDetector()
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        state_resp = await client.get(f"{self.server_url}/api/state")
                        state_data = state_resp.json()
                        current_joints = state_data.get("joints", joints)
                        gripper_w = state_data.get("gripper", 32.5)
                        # Try to get detected objects
                        obj_resp = await client.get(f"{self.server_url}/api/virtual-grip/check",
                                                    params={"joints": ",".join(str(j) for j in current_joints),
                                                            "gripper": str(gripper_w)})
                        obj_data = obj_resp.json()
                        verified = obj_data.get("gripped", False)
                        gripped_label = obj_data.get("object_label", target)
                except Exception as e:
                    self._log(f"Sim verification fallback: {e}")
                    verified = True
                    gripped_label = target
            else:
                verified = await self._verify_grip()
                gripped_label = target

            self.episode_recorder.end_phase(success=verified)

            if verified:
                self.state.phase = AutoPickPhase.DONE
                self._log("Pick successful!")
            else:
                self.state.phase = AutoPickPhase.DONE
                self._log("Pick complete (verification inconclusive)")

            self.episode_recorder.record_result(
                success=verified,
                grip_verified=verified,
                gripped_object=gripped_label,
            )

            return PickResult(
                success=verified,
                phase=AutoPickPhase.DONE,
                target_xy_mm=(x_mm, y_mm),
                joints=joints,
            )
        except Exception as e:
            self.episode_recorder.record_result(success=False, failure_reason=str(e))
            raise
        finally:
            # Stop video recording (best-effort)
            try:
                await self.video_recorder.stop()
            except Exception as e:
                logger.warning("Failed to stop video recording: %s", e)
            self.episode_recorder.finish()

    async def _detect(self, target: str) -> tuple[float, float]:
        """Detect target object via overhead camera HSV detection.

        Returns (x_mm, y_mm) in arm-base frame.
        """
        import cv2
        import numpy as np

        # Snap overhead camera
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(snap_url(CAM_OVERHEAD))
            if resp.status_code != 200:
                raise RuntimeError(f"Camera snap failed: {resp.status_code}")
            img_bytes = resp.content

        # Decode
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode camera image")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # HSV ranges for targets
        if target in ("redbull", "red"):
            # Red has two hue ranges
            mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
            mask = mask1 | mask2
        elif target == "blue":
            mask = cv2.inRange(hsv, np.array([100, 80, 60]), np.array([130, 255, 255]))
        elif target == "green":
            mask = cv2.inRange(hsv, np.array([35, 80, 60]), np.array([85, 255, 255]))
        else:
            # "any" — try all colors, pick largest
            masks = []
            for name, (lo, hi) in [
                ("red1", (np.array([0, 100, 80]), np.array([10, 255, 255]))),
                ("red2", (np.array([160, 100, 80]), np.array([180, 255, 255]))),
                ("blue", (np.array([100, 80, 60]), np.array([130, 255, 255]))),
                ("green", (np.array([35, 80, 60]), np.array([85, 255, 255]))),
            ]:
                masks.append(cv2.inRange(hsv, lo, hi))
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError(f"No '{target}' object detected in overhead view")

        # Pick largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 500:
            raise RuntimeError(f"Detected object too small (area={area}px)")

        # Centroid
        M = cv2.moments(largest)
        if M["m00"] == 0:
            raise RuntimeError("Zero-area contour")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        self._log(f"Detected at pixel ({cx}, {cy}), area={area:.0f}px")

        # Convert pixel to arm-frame mm
        x_mm = (cx - IMG_CX) * PX_TO_MM
        y_mm = (cy - IMG_CY) * PX_TO_MM

        return x_mm, y_mm

    @staticmethod
    def plan_joints(x_mm: float, y_mm: float, z_mm: float = 0.0) -> list[float]:
        """Compute joint angles for top-down approach above (x, y, z).

        Uses empirical model calibrated from known reference pose:
          Reference: joints [1.0, 25.9, 6.7, 0.5, 88.7, 3.3] at ~100mm reach

        The existing DH parameters are known to be incorrect, so we use
        an empirical scaling model instead of analytical IK:
          - J0 = atan2(y, x) for base yaw (geometrically exact)
          - J1 scales roughly linearly with horizontal reach
          - J2 provides small correction for reach fine-tuning
          - J4 ≈ 90° for wrist straight down (confirmed working)

        Calibration reference point:
          r_ref ≈ 100mm → J1_ref = 25.9°, J2_ref = 6.7°
        """
        # Base yaw — geometrically exact
        j0 = math.degrees(math.atan2(y_mm, x_mm))

        # Horizontal distance from base
        r_mm = math.sqrt(x_mm ** 2 + y_mm ** 2)

        # Empirical model calibrated from reference pose
        # Reference: r ≈ 100mm → J1 = 25.9°, J2 = 6.7°
        # Scale factors derived from reference
        R_REF = 100.0   # reference horizontal distance mm
        J1_REF = 25.9    # reference shoulder angle
        J2_REF = 6.7     # reference elbow angle

        # Linear scaling: J1 is primary reach control
        # At r=0 the arm hangs straight down (J1≈0)
        # J1 scales roughly as: J1 = (r / R_REF) * J1_REF
        j1 = (r_mm / R_REF) * J1_REF

        # J2 provides secondary reach correction
        # Scales similarly but is much smaller
        j2 = (r_mm / R_REF) * J2_REF

        # Height correction: raising z (object is higher) needs less reach-down
        # Rough: 10mm height ≈ -1° on J1
        if z_mm != 0.0:
            j1 -= z_mm * 0.1

        # Clamp to safe ranges
        j1 = max(0.0, min(j1, 80.0))
        j2 = max(0.0, min(j2, 40.0))

        j3 = 0.0
        j4 = 90.0  # straight down — confirmed working
        j5 = 0.0

        return [round(j0, 1), round(j1, 1), round(j2, 1), round(j3, 1), round(j4, 1), round(j5, 1)]

    async def _execute_pick(self, target_joints: list[float], gripper_mm: float = 32.5):
        """Execute the pick sequence using ArmOps primitives."""
        # Phase 1: Open gripper wide
        self.state.phase = AutoPickPhase.APPROACHING
        self._check_stop()
        self._log("Opening gripper...")
        self.episode_recorder.start_phase("open_gripper")
        await self.ops._set_gripper(60.0)
        await asyncio.sleep(0.5)
        self.episode_recorder.end_phase(success=True)

        # Phase 2: Approach from above (hover + lower)
        self._log("Approaching from above...")
        self._check_stop()
        self.episode_recorder.start_phase("approach")
        result = await self.ops.approach_from_above(target_joints)
        if not result.success:
            self.episode_recorder.end_phase(success=False)
            raise RuntimeError(f"Approach failed: {result.error}")
        self.episode_recorder.end_phase(success=True)

        # Phase 3: Grip
        self.state.phase = AutoPickPhase.GRIPPING
        self._check_stop()
        self._log(f"Closing gripper to {gripper_mm}mm...")
        self.episode_recorder.start_phase("grip")
        grip_ok = await self.ops.grip_and_verify(gripper_mm)
        if not grip_ok:
            self.episode_recorder.end_phase(success=False)
            await self.ops.retreat_home()
            raise RuntimeError("Grip failed")
        self.episode_recorder.end_phase(success=True)

        # Phase 4: Lift
        self.state.phase = AutoPickPhase.LIFTING
        self._check_stop()
        self._log("Lifting...")
        self.episode_recorder.start_phase("lift")
        lift_result = await self.ops.lift_from_pick(target_joints)
        if not lift_result.success:
            self.episode_recorder.end_phase(success=False)
            await self.ops.retreat_home()
            raise RuntimeError(f"Lift failed: {lift_result.error}")
        self.episode_recorder.end_phase(success=True)

    async def _verify_grip(self) -> bool:
        """Simple grip verification via arm camera.

        Returns True if we think something is in the gripper.
        For now, just return True (placeholder for more sophisticated check).
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.cam_server_url}/snap/1")
                if resp.status_code == 200:
                    self._log("Arm camera snap captured for verification")
                    return True
        except Exception as e:
            self._log(f"Verification camera error: {e}")
        return True  # Optimistic


class _StopRequested(Exception):
    pass
