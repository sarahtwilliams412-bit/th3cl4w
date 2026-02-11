"""Visual servo controller: move gripper to detected object using overhead camera.

Move-snap-verify loop: command small move, wait, snap frame, check error, repeat.
Converges when pixel error < threshold. Pure proportional control.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx
import numpy as np

from src.vision.realtime_detector import detect_object, detect_gripper, Detection
from src.vision.overhead_calibrator import OverheadCalibrator
from shared.kinematics.kinematics import D1Kinematics

logger = logging.getLogger(__name__)

ARM_API = "http://localhost:8080"
from shared.config.camera_config import CAMERA_SERVER_URL as CAM_API


class ServoState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ServoConfig:
    """Configuration for the visual servo loop."""

    convergence_threshold_px: float = 20.0  # pixels
    max_iterations: int = 30
    kp: float = 0.3  # proportional gain (workspace mm per pixel error)
    max_step_mm: float = 30.0  # max single step in mm
    settle_time_s: float = 0.3  # wait after each move for arm to settle
    camera_id: int = 0
    frame_width: int = 1920
    frame_height: int = 1080


@dataclass
class ServoStepLog:
    iteration: int
    target_px: tuple[int, int]
    gripper_px: tuple[int, int]
    error_px: float
    move_mm: tuple[float, float] = (0.0, 0.0)
    note: str = ""


@dataclass
class ServoResult:
    state: ServoState
    message: str = ""
    iterations: int = 0
    final_error_px: float = 0.0
    total_time_s: float = 0.0
    steps: list[ServoStepLog] = field(default_factory=list)


class VisualServoController:
    """Overhead camera visual servo: move gripper XY to target object."""

    def __init__(
        self,
        calibrator: Optional[OverheadCalibrator] = None,
        config: Optional[ServoConfig] = None,
    ):
        self.calibrator = calibrator or OverheadCalibrator()
        self.config = config or ServoConfig()
        self.kinematics = D1Kinematics()

        self._state = ServoState.IDLE
        self._result: Optional[ServoResult] = None
        self._abort = False

    @property
    def state(self) -> ServoState:
        return self._state

    @property
    def result(self) -> Optional[ServoResult]:
        return self._result

    def abort(self) -> None:
        """Request abort of running servo."""
        self._abort = True

    async def run(self, target: str = "redbull", camera_id: int = 0) -> ServoResult:
        """Execute the visual servo loop.

        1. Snap frame from overhead camera
        2. Detect target object → target_px
        3. Detect gripper (or use FK + calibration) → gripper_px
        4. Compute pixel error
        5. If error < threshold → converged
        6. Convert error to workspace delta (mm)
        7. Command arm move via joint adjustments
        8. Wait for settle, repeat
        """
        self._state = ServoState.RUNNING
        self._abort = False
        self.config.camera_id = camera_id
        steps: list[ServoStepLog] = []
        t0 = time.monotonic()

        try:
            for i in range(self.config.max_iterations):
                if self._abort:
                    result = ServoResult(
                        state=ServoState.ABORTED,
                        message="Aborted by user",
                        iterations=i,
                        steps=steps,
                        total_time_s=time.monotonic() - t0,
                    )
                    self._state = ServoState.ABORTED
                    self._result = result
                    return result

                # 1. Grab frame
                frame = await self._grab_frame()
                if frame is None:
                    result = ServoResult(
                        state=ServoState.FAILED,
                        message="Failed to grab camera frame",
                        iterations=i,
                        steps=steps,
                        total_time_s=time.monotonic() - t0,
                    )
                    self._state = ServoState.FAILED
                    self._result = result
                    return result

                # 2. Detect target
                target_det = detect_object(frame, target=target)
                if not target_det.found:
                    result = ServoResult(
                        state=ServoState.FAILED,
                        message=f"Target '{target}' not detected in frame",
                        iterations=i,
                        steps=steps,
                        total_time_s=time.monotonic() - t0,
                    )
                    self._state = ServoState.FAILED
                    self._result = result
                    return result

                # 3. Detect gripper (try visual detection, fall back to FK)
                gripper_det = detect_gripper(frame)
                if gripper_det.found:
                    gripper_px = gripper_det.centroid_px
                else:
                    gripper_px = await self._gripper_px_from_fk()
                    if gripper_px is None:
                        result = ServoResult(
                            state=ServoState.FAILED,
                            message="Cannot determine gripper position",
                            iterations=i,
                            steps=steps,
                            total_time_s=time.monotonic() - t0,
                        )
                        self._state = ServoState.FAILED
                        self._result = result
                        return result

                target_px = target_det.centroid_px

                # 4. Compute error
                error_x = target_px[0] - gripper_px[0]
                error_y = target_px[1] - gripper_px[1]
                error_mag = np.sqrt(error_x**2 + error_y**2)

                step_log = ServoStepLog(
                    iteration=i,
                    target_px=target_px,
                    gripper_px=gripper_px,
                    error_px=round(error_mag, 1),
                )

                # 5. Check convergence
                if error_mag < self.config.convergence_threshold_px:
                    step_log.note = "CONVERGED"
                    steps.append(step_log)
                    result = ServoResult(
                        state=ServoState.CONVERGED,
                        message=f"Converged in {i+1} iterations, error={error_mag:.1f}px",
                        iterations=i + 1,
                        final_error_px=round(error_mag, 1),
                        steps=steps,
                        total_time_s=time.monotonic() - t0,
                    )
                    self._state = ServoState.CONVERGED
                    self._result = result
                    return result

                # 6. Convert pixel error to workspace delta
                if self.calibrator.is_calibrated:
                    # Use calibrated mapping for proper scale
                    target_ws = self.calibrator.pixel_to_workspace(*target_px)
                    gripper_ws = self.calibrator.pixel_to_workspace(*gripper_px)
                    dx_mm = target_ws[0] - gripper_ws[0]
                    dy_mm = target_ws[1] - gripper_ws[1]
                else:
                    # Fallback: approximate scale (0.3mm per pixel for 1080p overhead)
                    dx_mm = error_x * 0.3
                    dy_mm = error_y * 0.3

                # Apply proportional gain and clamp
                dx_mm = np.clip(
                    self.config.kp * dx_mm, -self.config.max_step_mm, self.config.max_step_mm
                )
                dy_mm = np.clip(
                    self.config.kp * dy_mm, -self.config.max_step_mm, self.config.max_step_mm
                )

                step_log.move_mm = (round(dx_mm, 2), round(dy_mm, 2))
                step_log.note = f"moving dx={dx_mm:.1f}mm dy={dy_mm:.1f}mm"
                steps.append(step_log)

                # 7. Check arm safety
                safe = await self._check_arm_safe()
                if not safe:
                    result = ServoResult(
                        state=ServoState.FAILED,
                        message="Arm safety check failed — aborting",
                        iterations=i + 1,
                        final_error_px=round(error_mag, 1),
                        steps=steps,
                        total_time_s=time.monotonic() - t0,
                    )
                    self._state = ServoState.FAILED
                    self._result = result
                    return result

                # 8. Command arm move (cartesian delta via IK)
                await self._move_arm_delta(dx_mm, dy_mm)

                # 9. Wait for settle
                await asyncio.sleep(self.config.settle_time_s)

                logger.info(
                    "Servo iter %d: error=%.1fpx, move=(%.1f, %.1f)mm",
                    i,
                    error_mag,
                    dx_mm,
                    dy_mm,
                )

            # Max iterations reached
            result = ServoResult(
                state=ServoState.FAILED,
                message=f"Did not converge after {self.config.max_iterations} iterations",
                iterations=self.config.max_iterations,
                final_error_px=round(error_mag, 1),
                steps=steps,
                total_time_s=time.monotonic() - t0,
            )
            self._state = ServoState.FAILED
            self._result = result
            return result

        except Exception as e:
            logger.error("Visual servo error: %s", e, exc_info=True)
            result = ServoResult(
                state=ServoState.FAILED,
                message=f"Error: {e}",
                iterations=len(steps),
                steps=steps,
                total_time_s=time.monotonic() - t0,
            )
            self._state = ServoState.FAILED
            self._result = result
            return result

    async def _grab_frame(self) -> Optional[np.ndarray]:
        """Grab a frame from the overhead camera via camera server."""
        import cv2 as _cv2

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{CAM_API}/snapshot",
                    params={"camera": self.config.camera_id},
                )
                if resp.status_code != 200:
                    logger.warning("Camera snapshot failed: %d", resp.status_code)
                    return None
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            logger.warning("Failed to grab frame: %s", e)
            return None

    async def _gripper_px_from_fk(self) -> Optional[tuple[int, int]]:
        """Compute gripper pixel position from FK + calibration."""
        if not self.calibrator.is_calibrated:
            return None
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{ARM_API}/api/state")
                if resp.status_code != 200:
                    return None
                state = resp.json()
                angles_deg = state.get("joints", [])
                if len(angles_deg) < 7:
                    return None

            angles_rad = np.radians(angles_deg[:7])
            T_ee = self.kinematics.forward_kinematics(angles_rad)
            ee_x_mm = T_ee[0, 3] * 1000  # m → mm
            ee_y_mm = T_ee[1, 3] * 1000

            u, v = self.calibrator.workspace_to_pixel(ee_x_mm, ee_y_mm)
            return (int(round(u)), int(round(v)))
        except Exception as e:
            logger.warning("FK gripper position failed: %s", e)
            return None

    async def _check_arm_safe(self) -> bool:
        """Check arm state for safety (overcurrent, errors)."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{ARM_API}/api/state")
                if resp.status_code != 200:
                    return False
                state = resp.json()
                # Check for error states
                if state.get("error"):
                    logger.warning("Arm error state: %s", state.get("error"))
                    return False
                return True
        except Exception:
            return False

    async def _move_arm_delta(self, dx_mm: float, dy_mm: float) -> None:
        """Move the arm by a small XY delta using joint-level commands.

        Strategy: Get current joint angles, compute current EE position via FK,
        add delta, solve IK, command new joint angles.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get current state
                resp = await client.get(f"{ARM_API}/api/state")
                if resp.status_code != 200:
                    logger.warning("Cannot get arm state for move")
                    return
                state = resp.json()
                angles_deg = state.get("joints", [])
                if len(angles_deg) < 7:
                    return

                angles_rad = np.radians(angles_deg[:7])

                # Current EE pose
                T_current = self.kinematics.forward_kinematics(angles_rad)

                # Apply delta (convert mm to m)
                T_target = T_current.copy()
                T_target[0, 3] += dx_mm / 1000.0
                T_target[1, 3] += dy_mm / 1000.0

                # Solve IK
                q_new = self.kinematics.inverse_kinematics(T_target, q_init=angles_rad)
                new_deg = np.degrees(q_new)

                # Command each joint
                for i, angle in enumerate(new_deg):
                    await client.post(
                        f"{ARM_API}/api/command/set-joint",
                        json={"id": i, "angle": round(float(angle), 2)},
                    )
                    # Small delay between joint commands
                    await asyncio.sleep(0.02)

        except Exception as e:
            logger.warning("Move arm delta failed: %s", e)

    def get_status(self) -> dict:
        """Return current servo status as dict."""
        result_dict = None
        if self._result:
            result_dict = {
                "state": self._result.state.value,
                "message": self._result.message,
                "iterations": self._result.iterations,
                "final_error_px": self._result.final_error_px,
                "total_time_s": round(self._result.total_time_s, 2),
                "steps": [
                    {
                        "iteration": s.iteration,
                        "target_px": s.target_px,
                        "gripper_px": s.gripper_px,
                        "error_px": s.error_px,
                        "move_mm": s.move_mm,
                        "note": s.note,
                    }
                    for s in self._result.steps
                ],
            }
        return {
            "state": self._state.value,
            "result": result_dict,
        }
