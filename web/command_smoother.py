"""
Command Smoother for D1 Arm

Buffers rapid joint commands and interpolates smoothly toward target positions
at the D1's native 10Hz DDS cycle rate. Instead of sending each slider change
as an instant position jump, the smoother ramps from current to target using
exponential smoothing with configurable speed.

Usage:
    smoother = CommandSmoother(arm, rate_hz=10, smoothing_factor=0.3)
    await smoother.start()
    smoother.set_joint_target(0, 45.0)   # non-blocking, updates target
    smoother.set_gripper_target(30.0)
    ...
    await smoother.stop()
"""

import asyncio
import logging
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class ArmInterface(Protocol):
    """Minimal arm interface needed by the smoother."""

    def set_joint(self, joint_id: int, angle_deg: float, **kwargs) -> bool: ...
    def set_all_joints(self, angles_deg: list, **kwargs) -> bool: ...
    def set_gripper(self, position_mm: float, **kwargs) -> bool: ...
    def get_joint_angles(self) -> Any: ...
    def get_gripper_position(self) -> float: ...


class CommandSmoother:
    """
    Smooths joint commands by interpolating toward targets at a fixed rate.

    Each tick (default 10Hz = 100ms), the smoother moves the commanded position
    a fraction of the remaining distance toward the target. The smoothing_factor
    controls responsiveness:
      - 0.3 = smooth, ~3 ticks to reach 97% of target (~300ms lag)
      - 0.5 = moderate, ~2 ticks for 75%
      - 1.0 = no smoothing (pass-through)

    The max_step_deg parameter limits the maximum angular change per tick,
    providing a velocity cap for safety.
    """

    def __init__(
        self,
        arm: Any,
        rate_hz: float = 10.0,
        smoothing_factor: float = 0.35,
        max_step_deg: float = 15.0,
        max_gripper_step_mm: float = 5.0,
        num_joints: int = 6,
        collector: Any = None,
    ):
        self._arm = arm
        self._rate_hz = rate_hz
        self._interval = 1.0 / rate_hz
        self._alpha = smoothing_factor
        self._max_step = max_step_deg
        self._max_grip_step = max_gripper_step_mm
        self._num_joints = num_joints
        self._collector = collector

        # Current commanded positions (what we last sent to the arm)
        self._current = [0.0] * num_joints
        self._current_gripper = 0.0

        # Target positions (set by UI)
        self._target = [None] * num_joints  # None = no pending target
        self._target_gripper: Optional[float] = None

        # Track which joints have active targets
        self._dirty_joints: set = set()
        self._dirty_gripper = False

        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._ticks = 0
        self._commands_sent = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "ticks": self._ticks,
            "commands_sent": self._commands_sent,
            "rate_hz": self._rate_hz,
            "smoothing_factor": self._alpha,
        }

    def sync_current_positions(self) -> None:
        """Read current positions from arm to initialize smoother state."""
        try:
            angles = self._arm.get_joint_angles()
            if angles is not None:
                for i in range(min(len(angles), self._num_joints)):
                    self._current[i] = float(angles[i])
            if hasattr(self._arm, "get_gripper_position"):
                self._current_gripper = float(self._arm.get_gripper_position())
        except Exception:
            logger.debug("Could not sync positions from arm")

    def set_joint_target(self, joint_id: int, angle_deg: float) -> None:
        """Set a target angle for a joint. Non-blocking."""
        if 0 <= joint_id < self._num_joints:
            self._target[joint_id] = angle_deg
            self._dirty_joints.add(joint_id)

    def set_all_joints_target(self, angles_deg: list) -> None:
        """Set targets for all joints at once."""
        for i, a in enumerate(angles_deg[:self._num_joints]):
            self._target[i] = a
            self._dirty_joints.add(i)

    def set_gripper_target(self, position_mm: float) -> None:
        """Set a target gripper position. Non-blocking."""
        self._target_gripper = position_mm
        self._dirty_gripper = True

    async def start(self) -> None:
        """Start the smoothing loop."""
        if self._running:
            return
        self.sync_current_positions()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Command smoother started at %.0fHz, alpha=%.2f", self._rate_hz, self._alpha)

    async def stop(self) -> None:
        """Stop the smoothing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Command smoother stopped")

    async def _loop(self) -> None:
        """Main smoothing loop running at the configured rate."""
        while self._running:
            t0 = time.monotonic()
            self._tick()
            self._ticks += 1
            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _tick(self) -> None:
        """One smoothing step: interpolate current toward target, send if changed."""
        if not self._dirty_joints and not self._dirty_gripper:
            return

        # Log smoother state every 5th tick (~2Hz)
        should_log = self._collector is not None and self._ticks % 5 == 0

        joints_changed = []

        for jid in list(self._dirty_joints):
            target = self._target[jid]
            if target is None:
                self._dirty_joints.discard(jid)
                continue

            current = self._current[jid]
            diff = target - current

            if abs(diff) < 0.05:  # close enough, snap to target
                self._current[jid] = target
                self._target[jid] = None
                self._dirty_joints.discard(jid)
                joints_changed.append(jid)
                continue

            # Exponential smoothing with velocity cap
            step = diff * self._alpha
            if abs(step) > self._max_step:
                step = self._max_step if step > 0 else -self._max_step

            self._current[jid] = current + step
            joints_changed.append(jid)

        # Send joint commands
        if joints_changed:
            if len(joints_changed) >= 3:
                # Batch as set_all_joints when multiple joints move
                try:
                    self._arm.set_all_joints(list(self._current))
                    self._commands_sent += 1
                except Exception:
                    logger.debug("Failed to send all-joints command")
            else:
                for jid in joints_changed:
                    try:
                        self._arm.set_joint(jid, self._current[jid])
                        self._commands_sent += 1
                    except Exception:
                        logger.debug("Failed to send joint %d command", jid)

        # Log smoother state
        if should_log and joints_changed:
            try:
                states = []
                for jid in range(self._num_joints):
                    states.append({
                        "joint_id": jid,
                        "target": self._target[jid] if self._target[jid] is not None else self._current[jid],
                        "current": self._current[jid],
                        "sent": self._current[jid],
                        "dirty": jid in self._dirty_joints,
                    })
                if self._collector.enabled:
                    self._collector.log_smoother_state(states)
            except Exception:
                pass

        # Gripper
        if self._dirty_gripper and self._target_gripper is not None:
            diff = self._target_gripper - self._current_gripper
            if abs(diff) < 0.1:
                self._current_gripper = self._target_gripper
                self._target_gripper = None
                self._dirty_gripper = False
            else:
                step = diff * self._alpha
                if abs(step) > self._max_grip_step:
                    step = self._max_grip_step if step > 0 else -self._max_grip_step
                self._current_gripper += step

            try:
                if hasattr(self._arm, "set_gripper"):
                    self._arm.set_gripper(self._current_gripper)
                    self._commands_sent += 1
            except Exception:
                logger.debug("Failed to send gripper command")
