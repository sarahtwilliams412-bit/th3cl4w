"""
Autonomous Place Pipeline — transport held object to target location and release.

Mirrors AutoPick pattern. Uses geometric planner from AutoPick for joint planning.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.control.arm_operations import ArmOps
from src.planning.auto_pick import AutoPick
from src.telemetry.pick_episode import PickEpisodeRecorder

logger = logging.getLogger("th3cl4w.planning.auto_place")

DEFAULT_SERVER = "http://localhost:8080"


class AutoPlacePhase(str, enum.Enum):
    IDLE = "idle"
    PLANNING = "planning"
    TRANSPORTING = "transporting"
    LOWERING = "lowering"
    RELEASING = "releasing"
    RETRACTING = "retracting"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class PlaceResult:
    success: bool = False
    phase: AutoPlacePhase = AutoPlacePhase.IDLE
    target_xy_mm: tuple[float, float] = (0.0, 0.0)
    joints: list[float] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0


@dataclass
class AutoPlaceState:
    phase: AutoPlacePhase = AutoPlacePhase.IDLE
    target_xy_mm: tuple[float, float] = (0.0, 0.0)
    planned_joints: list[float] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)


class AutoPlace:
    """Autonomous place — transport held object to target and release."""

    def __init__(self, server_url: str = DEFAULT_SERVER):
        self.server_url = server_url
        self.ops = ArmOps(server_url)
        self.state = AutoPlaceState()
        self._stop_requested = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.episode_recorder = PickEpisodeRecorder()

    @property
    def running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        status = {
            "phase": self.state.phase.value,
            "target_xy_mm": list(self.state.target_xy_mm),
            "planned_joints": self.state.planned_joints,
            "error": self.state.error,
            "running": self._running,
            "elapsed_s": round(time.time() - self.state.started_at, 1) if self.state.started_at else 0,
            "log": self.state.log[-20:],
        }
        ep = self.episode_recorder.current
        if ep:
            status["episode_id"] = ep.episode_id
            status["episode_phases"] = [
                {
                    "name": p.name,
                    "success": p.success,
                    "duration_s": round(p.end_time - p.start_time, 2) if p.end_time else 0,
                }
                for p in ep.phases
            ]
        return status

    def stop(self):
        self._stop_requested = True
        self._log("Stop requested")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.state.log.append(entry)
        logger.info("AutoPlace: %s", msg)

    def _check_stop(self):
        if self._stop_requested:
            raise _StopRequested()

    @staticmethod
    def plan_place_joints(x_mm: float, y_mm: float, z_mm: float = 0.0) -> list[float]:
        """Compute joint angles for placing at (x_mm, y_mm).

        Reuses AutoPick.plan_joints — same geometric planner, same top-down approach.
        """
        return AutoPick.plan_joints(x_mm, y_mm, z_mm)

    async def start(self, x_mm: float, y_mm: float) -> asyncio.Task:
        """Start the place pipeline as a background task."""
        if self._running:
            raise RuntimeError("Place already in progress")
        self._stop_requested = False
        self._running = True
        self.state = AutoPlaceState(
            target_xy_mm=(x_mm, y_mm),
            started_at=time.time(),
        )
        self._task = asyncio.create_task(self._run(x_mm, y_mm))
        return self._task

    async def _run(self, x_mm: float, y_mm: float) -> PlaceResult:
        t0 = time.time()
        try:
            result = await self.execute(x_mm, y_mm)
            result.duration_s = time.time() - t0
            return result
        except _StopRequested:
            self.state.phase = AutoPlacePhase.STOPPED
            self._log("Stopped by user")
            return PlaceResult(
                phase=AutoPlacePhase.STOPPED,
                error="Stopped by user",
                duration_s=time.time() - t0,
            )
        except Exception as e:
            self.state.phase = AutoPlacePhase.FAILED
            self.state.error = str(e)
            self._log(f"Failed: {e}")
            logger.exception("AutoPlace failed")
            return PlaceResult(
                phase=AutoPlacePhase.FAILED,
                error=str(e),
                duration_s=time.time() - t0,
            )
        finally:
            self._running = False

    async def execute(self, x_mm: float, y_mm: float) -> PlaceResult:
        """Full autonomous place pipeline with episode recording."""
        episode = self.episode_recorder.start(mode="physical", target=f"place@({x_mm:.0f},{y_mm:.0f})")

        try:
            # 1. PLAN
            self.state.phase = AutoPlacePhase.PLANNING
            self._check_stop()
            self._log(f"Planning place at ({x_mm:.1f}, {y_mm:.1f}) mm")
            self.episode_recorder.start_phase("plan_place")
            joints = self.plan_place_joints(x_mm, y_mm)
            self.state.planned_joints = joints
            self._log(f"Planned joints: [{', '.join(f'{j:.1f}' for j in joints)}]")
            self.episode_recorder.record_plan(joints=joints)
            self.episode_recorder.end_phase(success=True)

            # 2. TRANSPORT — staged_reach to above place target
            self.state.phase = AutoPlacePhase.TRANSPORTING
            self._check_stop()
            self._log("Transporting to place target (hover above)...")
            self.episode_recorder.start_phase("transport")
            hover_joints = list(joints)
            hover_joints[1] = max(0.0, joints[1] - 15.0)  # higher position
            hover_joints[4] = min(joints[4], 70.0)  # partial wrist tilt
            result = await self.ops.staged_reach(hover_joints)
            if not result.success:
                self.episode_recorder.end_phase(success=False)
                raise RuntimeError(f"Transport failed: {result.error}")
            self.episode_recorder.end_phase(success=True)
            await asyncio.sleep(0.5)

            # 3. LOWER — staged_reach to place position
            self.state.phase = AutoPlacePhase.LOWERING
            self._check_stop()
            self._log("Lowering to place position...")
            self.episode_recorder.start_phase("lower")
            lower_result = await self.ops.staged_reach(joints, step_deg=5.0, step_delay=0.5)
            if not lower_result.success:
                self.episode_recorder.end_phase(success=False)
                raise RuntimeError(f"Lower failed: {lower_result.error}")
            self.episode_recorder.end_phase(success=True)
            await asyncio.sleep(0.3)

            # 4. RELEASE — open gripper
            self.state.phase = AutoPlacePhase.RELEASING
            self._check_stop()
            self._log("Releasing gripper...")
            self.episode_recorder.start_phase("release")
            await self.ops._set_gripper(60.0)
            await asyncio.sleep(0.5)
            self.episode_recorder.end_phase(success=True)

            # 5. RETRACT — lift and retreat
            self.state.phase = AutoPlacePhase.RETRACTING
            self._check_stop()
            self._log("Retracting...")
            self.episode_recorder.start_phase("retract")
            retract_result = await self.ops.lift_from_pick(joints, lift_deg=20.0)
            if not retract_result.success:
                self.episode_recorder.end_phase(success=False)
                self._log("Retract lift failed, retreating home")
                await self.ops.retreat_home()
            else:
                await self.ops.retreat_home()
                self.episode_recorder.end_phase(success=True)

            self.state.phase = AutoPlacePhase.DONE
            self._log("Place successful!")
            self.episode_recorder.record_result(success=True)

            return PlaceResult(
                success=True,
                phase=AutoPlacePhase.DONE,
                target_xy_mm=(x_mm, y_mm),
                joints=joints,
            )
        except Exception as e:
            self.episode_recorder.record_result(success=False, failure_reason=str(e))
            raise
        finally:
            self.episode_recorder.finish()


class _StopRequested(Exception):
    pass
