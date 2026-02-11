"""Multi-view fusion controller for autonomous pick sequences.

Orchestrates 3 cameras (overhead, side, arm) through a state machine
to pick objects with increasing precision at each phase:
  A: Overhead XY positioning
  B: Side cam Z approach
  C: Arm cam fine XY alignment
  D: Side cam descend to grasp height
  E: Arm cam verify + grasp
  F: Side cam verify lift
"""

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

import cv2
import numpy as np
import httpx

from src.vision.side_height_estimator import SideHeightEstimator
from src.vision.arm_camera_aligner import ArmCameraAligner

logger = logging.getLogger(__name__)

from src.config.camera_config import CAMERA_SERVER_URL as CAM_API

ARM_API = "http://localhost:8080"

# Phase timeouts in seconds
PHASE_TIMEOUTS = {
    "A": 15.0,
    "B": 10.0,
    "C": 10.0,
    "D": 10.0,
    "E": 5.0,
    "F": 8.0,
}

# Z heights in mm
APPROACH_HEIGHT_MM = 120.0  # safe approach altitude above table
GRASP_HEIGHT_OFFSET_MM = 10.0  # mm above object top to start grasp
LIFT_HEIGHT_MM = 80.0  # mm above grasp point after lifting
Z_STEP_MM = 5.0  # conservative Z step size
Z_TOLERANCE_MM = 5.0


class PickPhase(enum.Enum):
    IDLE = "idle"
    A_OVERHEAD_XY = "A_overhead_xy"
    B_SIDE_Z_APPROACH = "B_side_z_approach"
    C_ARM_FINE_ALIGN = "C_arm_fine_align"
    D_SIDE_DESCEND = "D_side_descend"
    E_VERIFY_GRASP = "E_verify_grasp"
    F_LIFT_VERIFY = "F_lift_verify"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class PickState:
    """Current state of the pick sequence."""

    phase: PickPhase = PickPhase.IDLE
    target: str = "redbull"
    target_xy_mm: Optional[tuple[float, float]] = None
    target_z_mm: Optional[float] = None
    gripper_z_mm: Optional[float] = None
    phase_start_time: float = 0.0
    total_start_time: float = 0.0
    error_message: str = ""
    steps_in_phase: int = 0
    max_steps_per_phase: int = 30
    alignment_history: list[float] = field(default_factory=list)


class MultiviewController:
    """State machine orchestrating multi-camera pick sequences."""

    def __init__(
        self,
        side_estimator: Optional[SideHeightEstimator] = None,
        arm_aligner: Optional[ArmCameraAligner] = None,
        overhead_move_xy: Optional[Callable] = None,
    ):
        self.side = side_estimator or SideHeightEstimator()
        self.arm_aligner = arm_aligner or ArmCameraAligner()
        self.state = PickState()
        self._abort_event = asyncio.Event()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # overhead_move_xy: async callable (x_mm, y_mm) -> bool
        # Stub if the overhead servo module isn't ready yet
        self._overhead_move_xy = overhead_move_xy or self._stub_overhead_move

    async def _stub_overhead_move(self, x_mm: float, y_mm: float) -> bool:
        """Stub for overhead XY servo — just logs and succeeds."""
        logger.info("STUB: overhead_move_xy(%.1f, %.1f) — assuming success", x_mm, y_mm)
        await asyncio.sleep(0.5)
        return True

    async def _grab_frame(self, cam_id: int) -> Optional[np.ndarray]:
        """Fetch a frame from the camera server."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{CAM_API}/snap/{cam_id}")
            if resp.status_code != 200:
                logger.warning("Camera %d returned status %d", cam_id, resp.status_code)
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error("Failed to grab frame from cam %d: %s", cam_id, e)
            return None

    async def _send_arm_delta(self, dx: float = 0, dy: float = 0, dz: float = 0) -> bool:
        """Send a cartesian delta move to the arm API.

        Uses the existing text command endpoint for now.
        """
        # Clamp movements for safety
        dx = np.clip(dx, -20, 20)
        dy = np.clip(dy, -20, 20)
        dz = np.clip(dz, -15, 15)  # conservative Z

        if abs(dx) < 0.1 and abs(dy) < 0.1 and abs(dz) < 0.1:
            return True

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Use text command for cartesian delta
                cmd = f"move delta x={dx:.1f} y={dy:.1f} z={dz:.1f}"
                resp = await client.post(
                    f"{ARM_API}/api/command/text",
                    json={"command": cmd},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("Arm delta move failed: %s", e)
            return False

    async def _close_gripper(self) -> bool:
        """Close the gripper."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{ARM_API}/api/gripper/adaptive-close")
                return resp.status_code == 200
        except Exception as e:
            logger.error("Gripper close failed: %s", e)
            return False

    def _check_timeout(self) -> bool:
        """Check if current phase has timed out."""
        phase_key = self.state.phase.value.split("_")[0]
        timeout = PHASE_TIMEOUTS.get(phase_key, 15.0)
        return (time.monotonic() - self.state.phase_start_time) > timeout

    def _check_abort(self) -> bool:
        """Check if abort was requested."""
        return self._abort_event.is_set()

    def _transition(self, new_phase: PickPhase):
        """Transition to a new phase."""
        old = self.state.phase
        self.state.phase = new_phase
        self.state.phase_start_time = time.monotonic()
        self.state.steps_in_phase = 0
        logger.info("Pick phase: %s → %s", old.value, new_phase.value)

    # ── Phase implementations ──

    async def _phase_a_overhead_xy(self) -> bool:
        """Phase A: Use overhead camera to move to target XY."""
        # Detect target in overhead cam
        frame = await self._grab_frame(0)  # cam0 = overhead
        if frame is None:
            self.state.error_message = "Cannot grab overhead camera frame"
            return False

        # For now, delegate to the overhead servo module
        # which handles its own detection + servo loop
        if self.state.target_xy_mm:
            x, y = self.state.target_xy_mm
            success = await self._overhead_move_xy(x, y)
            if not success:
                self.state.error_message = "Overhead XY servo failed"
                return False
            return True

        # If no pre-set target, try to detect it
        # (the overhead servo module should handle this)
        self.state.error_message = "No target XY position set and overhead servo not available"
        return False

    async def _phase_b_side_z_approach(self) -> bool:
        """Phase B: Use side camera to move to approach height."""
        while self.state.steps_in_phase < self.state.max_steps_per_phase:
            if self._check_abort():
                return False
            if self._check_timeout():
                self.state.error_message = "Phase B timed out"
                return False

            frame = await self._grab_frame(2)  # cam2 = side
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            gripper_z = self.side.estimate_height(frame, target="gripper")
            if gripper_z is None:
                # Try neon tape
                gripper_z = self.side.estimate_height(frame, target="neon_tape")
            if gripper_z is None:
                logger.warning("Cannot detect gripper in side cam")
                self.state.steps_in_phase += 1
                await asyncio.sleep(0.2)
                continue

            self.state.gripper_z_mm = gripper_z
            target_z = APPROACH_HEIGHT_MM

            error_z = target_z - gripper_z
            if abs(error_z) < Z_TOLERANCE_MM:
                logger.info("Phase B: approach height reached (%.1f mm)", gripper_z)
                return True

            # P-controller for Z
            dz = np.clip(0.3 * error_z, -Z_STEP_MM, Z_STEP_MM)
            await self._send_arm_delta(dz=dz)
            self.state.steps_in_phase += 1
            await asyncio.sleep(0.15)

        self.state.error_message = "Phase B: max steps exceeded"
        return False

    async def _phase_c_arm_fine_align(self) -> bool:
        """Phase C: Use arm camera for fine XY centering."""
        while self.state.steps_in_phase < self.state.max_steps_per_phase:
            if self._check_abort():
                return False
            if self._check_timeout():
                self.state.error_message = "Phase C timed out"
                return False

            frame = await self._grab_frame(1)  # cam1 = arm
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            alignment = self.arm_aligner.compute_alignment(frame, self.state.target)

            if not alignment["detected"]:
                logger.warning("Phase C: target not detected in arm cam")
                self.state.steps_in_phase += 1
                await asyncio.sleep(0.2)
                continue

            self.state.alignment_history.append(alignment["distance_px"])

            if alignment["centered"]:
                logger.info("Phase C: aligned (error=%.1f px)", alignment["distance_px"])
                return True

            # Apply correction with gain damping
            dx, dy = alignment["correction_mm"]
            gain = 0.5  # conservative
            await self._send_arm_delta(dx=gain * dx, dy=gain * dy)
            self.state.steps_in_phase += 1
            await asyncio.sleep(0.15)

        self.state.error_message = "Phase C: max steps exceeded"
        return False

    async def _phase_d_side_descend(self) -> bool:
        """Phase D: Use side camera to descend to grasp height."""
        # First, estimate object height
        frame = await self._grab_frame(2)
        if frame is None:
            self.state.error_message = "Cannot grab side cam for descent"
            return False

        object_z = self.side.estimate_height(frame, target=self.state.target)
        if object_z is None:
            # Use a default grasp height
            object_z = 30.0  # assume 30mm (Red Bull can height ~130mm, top at ~130)
            logger.warning("Cannot detect object height, using default %.1f mm", object_z)

        self.state.target_z_mm = object_z
        target_grasp_z = object_z + GRASP_HEIGHT_OFFSET_MM

        while self.state.steps_in_phase < self.state.max_steps_per_phase:
            if self._check_abort():
                return False
            if self._check_timeout():
                self.state.error_message = "Phase D timed out"
                return False

            frame = await self._grab_frame(2)
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            gripper_z = self.side.estimate_height(frame, target="gripper")
            if gripper_z is None:
                gripper_z = self.side.estimate_height(frame, target="neon_tape")
            if gripper_z is None:
                self.state.steps_in_phase += 1
                await asyncio.sleep(0.2)
                continue

            self.state.gripper_z_mm = gripper_z
            error_z = target_grasp_z - gripper_z

            if abs(error_z) < Z_TOLERANCE_MM:
                logger.info("Phase D: grasp height reached (%.1f mm)", gripper_z)
                return True

            # Descend slowly
            dz = np.clip(0.25 * error_z, -Z_STEP_MM, Z_STEP_MM)
            await self._send_arm_delta(dz=dz)
            self.state.steps_in_phase += 1
            await asyncio.sleep(0.2)

        self.state.error_message = "Phase D: max steps exceeded"
        return False

    async def _phase_e_verify_grasp(self) -> bool:
        """Phase E: Verify alignment with arm cam, then close gripper."""
        # Quick alignment check
        frame = await self._grab_frame(1)
        if frame is not None:
            alignment = self.arm_aligner.compute_alignment(frame, self.state.target)
            if alignment["detected"] and not alignment["centered"]:
                # Minor correction
                dx, dy = alignment["correction_mm"]
                await self._send_arm_delta(dx=0.3 * dx, dy=0.3 * dy)
                await asyncio.sleep(0.3)

        # Close gripper
        logger.info("Phase E: closing gripper")
        success = await self._close_gripper()
        if not success:
            self.state.error_message = "Gripper close failed"
            return False

        await asyncio.sleep(0.5)  # wait for gripper to close
        return True

    async def _phase_f_lift_verify(self) -> bool:
        """Phase F: Lift and verify with side camera."""
        # Lift
        lift_steps = int(LIFT_HEIGHT_MM / Z_STEP_MM)
        for _ in range(lift_steps):
            if self._check_abort():
                return False
            await self._send_arm_delta(dz=Z_STEP_MM)
            await asyncio.sleep(0.15)

        # Verify lift with side cam
        frame = await self._grab_frame(2)
        if frame is not None:
            gripper_z = self.side.estimate_height(frame, target="gripper")
            if gripper_z is not None:
                self.state.gripper_z_mm = gripper_z
                logger.info("Phase F: lifted to %.1f mm", gripper_z)

        return True

    # ── Main execution ──

    async def execute_pick(
        self, target: str = "redbull", target_xy_mm: Optional[tuple[float, float]] = None
    ):
        """Execute the full multi-view pick sequence.

        Args:
            target: Object type to pick.
            target_xy_mm: Pre-computed XY position (from overhead detection).
        """
        self.state = PickState(
            target=target,
            target_xy_mm=target_xy_mm,
            total_start_time=time.monotonic(),
        )
        self._abort_event.clear()
        self._running = True

        phases = [
            (PickPhase.A_OVERHEAD_XY, self._phase_a_overhead_xy),
            (PickPhase.B_SIDE_Z_APPROACH, self._phase_b_side_z_approach),
            (PickPhase.C_ARM_FINE_ALIGN, self._phase_c_arm_fine_align),
            (PickPhase.D_SIDE_DESCEND, self._phase_d_side_descend),
            (PickPhase.E_VERIFY_GRASP, self._phase_e_verify_grasp),
            (PickPhase.F_LIFT_VERIFY, self._phase_f_lift_verify),
        ]

        try:
            for phase_enum, phase_fn in phases:
                if self._check_abort():
                    self._transition(PickPhase.ABORTED)
                    return

                self._transition(phase_enum)
                success = await phase_fn()

                if not success:
                    if self._check_abort():
                        self._transition(PickPhase.ABORTED)
                    else:
                        self._transition(PickPhase.FAILED)
                    return

            self._transition(PickPhase.COMPLETE)
            elapsed = time.monotonic() - self.state.total_start_time
            logger.info("Pick sequence complete in %.1f s", elapsed)

        except Exception as e:
            logger.exception("Pick sequence error: %s", e)
            self.state.error_message = str(e)
            self._transition(PickPhase.FAILED)
        finally:
            self._running = False

    def start_pick(
        self, target: str = "redbull", target_xy_mm: Optional[tuple[float, float]] = None
    ):
        """Start pick sequence as background task."""
        if self._running:
            raise RuntimeError("Pick already in progress")
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.execute_pick(target, target_xy_mm))
        return self._task

    def abort(self):
        """Abort the current pick sequence."""
        self._abort_event.set()

    def get_status(self) -> dict:
        """Get current pick status."""
        elapsed = 0.0
        if self.state.total_start_time > 0:
            elapsed = time.monotonic() - self.state.total_start_time

        return {
            "phase": self.state.phase.value,
            "target": self.state.target,
            "running": self._running,
            "elapsed_s": round(elapsed, 1),
            "target_xy_mm": self.state.target_xy_mm,
            "target_z_mm": self.state.target_z_mm,
            "gripper_z_mm": self.state.gripper_z_mm,
            "error": self.state.error_message,
            "steps_in_phase": self.state.steps_in_phase,
            "side_calibrated": self.side.calibrated,
        }
