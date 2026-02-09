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

    SAFETY: The smoother will NOT send any commands until positions have been
    synced from the arm (via sync_current_positions() or sync_from_feedback()).
    This prevents phantom commands that drive joints toward 0°.
    """

    def __init__(
        self,
        arm: Any,
        rate_hz: float = 10.0,
        smoothing_factor: float = 0.35,
        max_step_deg: float = 10.0,
        max_gripper_step_mm: float = 5.0,
        num_joints: int = 6,
        collector: Any = None,
        safety_monitor: Any = None,
    ):
        self._arm = arm
        self._rate_hz = rate_hz
        self._interval = 1.0 / rate_hz
        self._alpha = smoothing_factor
        self._max_step = max_step_deg
        self._max_grip_step = max_gripper_step_mm
        self._num_joints = num_joints
        self._collector = collector
        self._safety_monitor = safety_monitor

        # Feedback freshness tracking
        self._last_feedback_time: float = 0.0

        # SAFETY: Initialize to None — unknown positions until synced
        self._current: list[Optional[float]] = [None] * num_joints
        self._current_gripper: Optional[float] = None

        # Target positions (set by UI)
        self._target = [None] * num_joints  # None = no pending target
        self._target_gripper: Optional[float] = None

        # Track which joints have active targets
        self._dirty_joints: set = set()
        self._dirty_gripper = False

        # SAFETY: Must be True before any commands are sent
        self._synced = False

        # SAFETY: Must be True (arm powered + enabled) before sending commands
        self._arm_enabled = False

        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._ticks = 0
        self._commands_sent = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def arm_enabled(self) -> bool:
        return self._arm_enabled

    @property
    def stats(self) -> dict:
        return {
            "ticks": self._ticks,
            "commands_sent": self._commands_sent,
            "rate_hz": self._rate_hz,
            "smoothing_factor": self._alpha,
        }

    def sync_current_positions(self) -> bool:
        """Read current positions from arm to initialize smoother state.

        Returns True if sync succeeded, False otherwise.
        The smoother will NOT send commands until this succeeds.
        """
        try:
            angles = self._arm.get_joint_angles()
            if angles is not None and len(angles) >= self._num_joints:
                for i in range(self._num_joints):
                    self._current[i] = float(angles[i])
            else:
                logger.warning("sync_current_positions: got invalid angles: %s", angles)
                return False
            if hasattr(self._arm, "get_gripper_position"):
                self._current_gripper = float(self._arm.get_gripper_position())
            else:
                self._current_gripper = 0.0
            self._synced = True
            self._last_feedback_time = time.time()
            logger.info("Smoother synced to arm positions: %s", self._current)
            return True
        except Exception as e:
            logger.warning("Could not sync positions from arm: %s", e)
            return False

    def sync_from_feedback(self, angles: list, gripper: Optional[float] = None) -> None:
        """Sync smoother state from arm feedback data (e.g., WebSocket state).

        This is called by the server when the first arm state arrives,
        ensuring the smoother knows the real joint positions before
        sending any commands.
        """
        if len(angles) >= self._num_joints:
            for i in range(self._num_joints):
                self._current[i] = float(angles[i])
            if gripper is not None:
                self._current_gripper = float(gripper)
            self._synced = True
            self._last_feedback_time = time.time()
            logger.info("Smoother synced from feedback: %s", self._current)

    def set_arm_enabled(self, enabled: bool) -> None:
        """Set whether the arm is powered and enabled.

        SAFETY: The smoother will not send any commands unless this is True.
        """
        was_enabled = self._arm_enabled
        self._arm_enabled = enabled
        if was_enabled and not enabled:
            # Arm just became disabled — clear all pending targets
            self._clear_targets()
            logger.info("Arm disabled — smoother targets cleared")
        elif not was_enabled and enabled:
            logger.info("Arm enabled — smoother will accept commands")

    def emergency_stop(self) -> None:
        """SAFETY: Immediately clear all targets and stop sending commands.

        Called from the e-stop handler. Clears all state so no further
        commands are sent until new targets are explicitly set.
        """
        self._dirty_joints.clear()
        self._dirty_gripper = False
        self._target = [None] * self._num_joints
        self._target_gripper = None
        self._arm_enabled = False
        logger.warning("EMERGENCY STOP — smoother halted, all targets cleared")

    def _clear_targets(self) -> None:
        """Clear all pending targets and dirty flags."""
        self._dirty_joints.clear()
        self._dirty_gripper = False
        self._target = [None] * self._num_joints
        self._target_gripper = None

    def set_joint_target(self, joint_id: int, angle_deg: float) -> None:
        """Set a target angle for a joint. Non-blocking."""
        if 0 <= joint_id < self._num_joints:
            self._target[joint_id] = angle_deg
            self._dirty_joints.add(joint_id)

    def set_all_joints_target(self, angles_deg: list) -> None:
        """Set targets for all joints at once."""
        for i, a in enumerate(angles_deg[: self._num_joints]):
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
        logger.info(
            "Command smoother started at %.0fHz, alpha=%.2f, synced=%s",
            self._rate_hz,
            self._alpha,
            self._synced,
        )

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
        """One smoothing step: interpolate current toward target, send if changed.

        SAFETY GUARDS:
        - No-op if positions haven't been synced from the arm
        - No-op if arm is not enabled (powered + motors enabled)
        - Skips any joint where current position is None (unknown)
        """
        # SAFETY: Don't send anything until we know the real arm positions
        if not self._synced:
            return

        # SAFETY: Don't send commands if arm is not enabled
        if not self._arm_enabled:
            return

        # SAFETY: Check feedback freshness — refuse commands if feedback is stale
        # Use feedback monitor if available (filters zero-reads), else wall-clock
        _fb_monitor = getattr(self._arm, '_feedback_monitor', None) if self._arm else None
        if _fb_monitor is not None:
            if not _fb_monitor.is_feedback_fresh(max_age_s=0.5):
                if self._ticks % 50 == 0:
                    health = _fb_monitor.get_health()
                    logger.error(
                        "Reliable feedback stale (last good %.1fs ago, zero_rate=%.0f%%) — refusing commands",
                        health.last_good_age_s, health.zero_rate * 100,
                    )
                return
        elif self._last_feedback_time > 0:
            feedback_age = time.time() - self._last_feedback_time
            if feedback_age > 0.5:  # 500ms staleness threshold
                if self._ticks % 50 == 0:  # Log every ~5s to avoid spam
                    logger.error(
                        "Feedback stale (%.1fs old) — refusing commands", feedback_age
                    )
                return

        # SAFETY: Check e-stop on safety monitor
        if self._safety_monitor is not None and self._safety_monitor.estop_active:
            if self._ticks % 50 == 0:
                logger.error("E-STOP active — refusing commands")
            return

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
            # SAFETY: Skip joints with unknown current position
            if current is None:
                logger.warning("Joint %d has no known position — skipping", jid)
                self._dirty_joints.discard(jid)
                self._target[jid] = None
                continue

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
            # Build the angles list, using current values (all must be non-None at this point due to _synced check)
            send_angles = []
            for i in range(self._num_joints):
                v = self._current[i]
                if v is None:
                    # This shouldn't happen if _synced is True, but be safe
                    logger.error(
                        "Joint %d position is None despite being synced — aborting send", i
                    )
                    return
                send_angles.append(v)

            # SAFETY: Validate command through SafetyMonitor before sending
            if self._safety_monitor is not None:
                from src.safety.limits import JOINT_LIMITS_DEG as _LIM
                for i, angle in enumerate(send_angles):
                    lo, hi = float(_LIM[i, 0]), float(_LIM[i, 1])
                    if angle < lo or angle > hi:
                        logger.warning(
                            "Safety: J%d=%.2f° outside [%.1f, %.1f] — blocking command",
                            i, angle, lo, hi,
                        )
                        return

            if len(joints_changed) >= 3:
                # Batch as set_all_joints when multiple joints move
                try:
                    self._arm.set_all_joints(send_angles)
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
                    states.append(
                        {
                            "joint_id": jid,
                            "target": (
                                self._target[jid]
                                if self._target[jid] is not None
                                else self._current[jid]
                            ),
                            "current": self._current[jid],
                            "sent": self._current[jid],
                            "dirty": jid in self._dirty_joints,
                        }
                    )
                if self._collector is not None:
                    self._collector.log_smoother_state(states)
            except Exception:
                pass

        # Gripper
        if self._dirty_gripper and self._target_gripper is not None:
            current_grip = self._current_gripper
            # SAFETY: Skip if gripper position unknown
            if current_grip is None:
                logger.warning("Gripper has no known position — skipping")
                self._dirty_gripper = False
                self._target_gripper = None
                return

            diff = self._target_gripper - current_grip
            if abs(diff) < 0.1:
                self._current_gripper = self._target_gripper
                self._target_gripper = None
                self._dirty_gripper = False
            else:
                step = diff * self._alpha
                if abs(step) > self._max_grip_step:
                    step = self._max_grip_step if step > 0 else -self._max_grip_step
                self._current_gripper = current_grip + step

            try:
                if hasattr(self._arm, "set_gripper"):
                    self._arm.set_gripper(self._current_gripper)
                    self._commands_sent += 1
            except Exception:
                logger.debug("Failed to send gripper command")
