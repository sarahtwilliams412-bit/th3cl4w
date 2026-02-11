"""
Simulation Rehearsal Runner — run N simulated pick attempts before physical execution.

Validates that the claw can reliably detect and pick up an object in simulation
(with optional position jitter to test robustness) before allowing a real-world attempt.

Usage:
    runner = SimRehearsalRunner(server_url="http://localhost:8080")
    report = await runner.run_rehearsal("redbull", max_attempts=5, required_successes=3)
    if report.ready_for_physical:
        result = await runner.promote_to_physical(report)
"""

from __future__ import annotations

import asyncio
import enum
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from services.kinematics_app.execution.auto_pick import AutoPick, AutoPickPhase, PickResult

logger = logging.getLogger("th3cl4w.planning.sim_rehearsal")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RehearsalPhase(str, enum.Enum):
    IDLE = "idle"
    SIMULATING = "simulating"
    ANALYSING = "analysing"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING_PHYSICAL = "executing_physical"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class AttemptResult:
    """Result of a single simulation attempt."""

    attempt_number: int
    success: bool
    phase_reached: str  # last AutoPickPhase before completion or failure
    joints: list[float] = field(default_factory=list)
    target_xy_mm: tuple[float, float] = (0.0, 0.0)
    jitter_applied_mm: tuple[float, float] = (0.0, 0.0)
    grip_distance_mm: float = float("inf")
    error: str = ""
    duration_s: float = 0.0
    episode_id: str = ""


@dataclass
class RehearsalReport:
    """Aggregate results from all simulation attempts."""

    target: str = ""
    total_attempts: int = 0
    successful_attempts: int = 0
    success_rate: float = 0.0
    required_successes: int = 0
    phase_failure_counts: dict[str, int] = field(default_factory=dict)
    best_joints: list[float] = field(default_factory=list)
    best_target_xy_mm: tuple[float, float] = (0.0, 0.0)
    avg_grip_distance_mm: float = float("inf")
    attempts: list[AttemptResult] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    recommendation: str = "no-go"  # "go" | "no-go" | "marginal"
    ready_for_physical: bool = False
    total_duration_s: float = 0.0

    # Physical attempt result (populated after promote_to_physical)
    physical_result: Optional[PickResult] = None


@dataclass
class RehearsalState:
    """Live state exposed to the API for progress tracking."""

    phase: RehearsalPhase = RehearsalPhase.IDLE
    target: str = ""
    current_attempt: int = 0
    max_attempts: int = 0
    successes_so_far: int = 0
    required_successes: int = 0
    jitter_mm: float = 0.0
    started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SimRehearsalRunner:
    """Orchestrates multiple simulation pick attempts before physical execution."""

    # Thresholds for go/no-go recommendation
    GO_THRESHOLD = 0.8
    MARGINAL_THRESHOLD = 0.6

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        cam_server_url: str = "http://localhost:8081",
    ):
        self.server_url = server_url
        self.cam_server_url = cam_server_url
        self.state = RehearsalState()
        self._stop_requested = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_report: Optional[RehearsalReport] = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_report(self) -> Optional[RehearsalReport]:
        return self._last_report

    def get_status(self) -> dict:
        """Return live rehearsal status for the API."""
        return {
            "phase": self.state.phase.value,
            "target": self.state.target,
            "current_attempt": self.state.current_attempt,
            "max_attempts": self.state.max_attempts,
            "successes_so_far": self.state.successes_so_far,
            "required_successes": self.state.required_successes,
            "jitter_mm": self.state.jitter_mm,
            "running": self._running,
            "elapsed_s": round(time.time() - self.state.started_at, 1) if self.state.started_at else 0,
            "log": self.state.log[-30:],
            "error": self.state.error,
            "report": self._report_to_dict(self._last_report) if self._last_report else None,
        }

    def stop(self):
        """Request stop of the rehearsal loop."""
        self._stop_requested = True
        self._log("Stop requested")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def start(
        self,
        target: str = "redbull",
        max_attempts: int = 5,
        required_successes: int = 3,
        jitter_mm: float = 3.0,
        auto_promote: bool = False,
    ) -> asyncio.Task:
        """Start the rehearsal as a background task.

        Parameters
        ----------
        target : str
            Object to pick (e.g. "redbull", "blue", "any").
        max_attempts : int
            Maximum number of simulation attempts to run.
        required_successes : int
            Number of successful sim picks needed before recommending physical.
        jitter_mm : float
            Random position offset (±mm) applied to each attempt to test robustness.
            Set to 0.0 for identical (deterministic) retries.
        auto_promote : bool
            If True, automatically execute the physical pick when the success
            threshold is met. If False, stop at "awaiting_approval" phase.
        """
        if self._running:
            raise RuntimeError("Rehearsal already in progress")

        if required_successes > max_attempts:
            raise ValueError(
                f"required_successes ({required_successes}) cannot exceed "
                f"max_attempts ({max_attempts})"
            )

        self._stop_requested = False
        self._running = True
        self._last_report = None
        self.state = RehearsalState(
            target=target,
            max_attempts=max_attempts,
            required_successes=required_successes,
            jitter_mm=jitter_mm,
            started_at=time.time(),
        )
        self._task = asyncio.create_task(
            self._run(target, max_attempts, required_successes, jitter_mm, auto_promote)
        )
        return self._task

    async def promote_to_physical(self, report: Optional[RehearsalReport] = None) -> PickResult:
        """Execute the physical pick using the best plan from simulation rehearsals.

        Can be called manually after reviewing the RehearsalReport, or is called
        automatically when auto_promote=True.
        """
        report = report or self._last_report
        if report is None:
            raise RuntimeError("No rehearsal report available — run rehearsal first")
        if not report.ready_for_physical:
            raise RuntimeError(
                f"Rehearsal not ready for physical (recommendation={report.recommendation}, "
                f"success_rate={report.success_rate:.0%})"
            )

        self.state.phase = RehearsalPhase.EXECUTING_PHYSICAL
        self._log("Promoting to physical pick with best sim joints")

        picker = AutoPick(self.server_url, self.cam_server_url)
        result = await picker.execute(
            target=report.target,
            mode="physical",
            override_xy_mm=report.best_target_xy_mm,
            override_joints=report.best_joints,
        )

        report.physical_result = result
        self._last_report = report

        if result.success:
            self._log(f"Physical pick SUCCEEDED — joints {result.joints}")
            self.state.phase = RehearsalPhase.DONE
        else:
            self._log(f"Physical pick FAILED — {result.error}")
            self.state.phase = RehearsalPhase.FAILED
            self.state.error = result.error

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(
        self,
        target: str,
        max_attempts: int,
        required_successes: int,
        jitter_mm: float,
        auto_promote: bool,
    ) -> RehearsalReport:
        t0 = time.time()
        try:
            report = await self._rehearsal_loop(
                target, max_attempts, required_successes, jitter_mm
            )
            report.total_duration_s = time.time() - t0
            self._last_report = report

            if report.ready_for_physical and auto_promote:
                self._log("Auto-promoting to physical pick...")
                await self.promote_to_physical(report)
            elif report.ready_for_physical:
                self.state.phase = RehearsalPhase.AWAITING_APPROVAL
                self._log(
                    f"Rehearsal complete — {report.recommendation.upper()} "
                    f"({report.success_rate:.0%} success rate). "
                    f"Call promote_to_physical to execute."
                )
            else:
                self.state.phase = RehearsalPhase.FAILED
                self._log(
                    f"Rehearsal complete — {report.recommendation.upper()} "
                    f"({report.success_rate:.0%} success rate). "
                    f"Not ready for physical execution."
                )

            return report

        except _RehearsalStopped:
            self.state.phase = RehearsalPhase.STOPPED
            self._log("Rehearsal stopped by user")
            report = RehearsalReport(target=target, total_duration_s=time.time() - t0)
            self._last_report = report
            return report

        except Exception as e:
            self.state.phase = RehearsalPhase.FAILED
            self.state.error = str(e)
            self._log(f"Rehearsal failed: {e}")
            logger.exception("SimRehearsalRunner failed")
            report = RehearsalReport(target=target, total_duration_s=time.time() - t0)
            self._last_report = report
            return report

        finally:
            self._running = False

    async def _rehearsal_loop(
        self,
        target: str,
        max_attempts: int,
        required_successes: int,
        jitter_mm: float,
    ) -> RehearsalReport:
        """Run simulation attempts in a loop, collecting results."""
        self.state.phase = RehearsalPhase.SIMULATING
        attempts: list[AttemptResult] = []
        successes = 0

        for attempt_num in range(1, max_attempts + 1):
            self._check_stop()

            self.state.current_attempt = attempt_num
            self._log(f"--- Simulation attempt {attempt_num}/{max_attempts} ---")

            # Generate jitter for this attempt
            jx = random.uniform(-jitter_mm, jitter_mm) if jitter_mm > 0 else 0.0
            jy = random.uniform(-jitter_mm, jitter_mm) if jitter_mm > 0 else 0.0

            if jitter_mm > 0:
                self._log(f"Position jitter: ({jx:+.1f}, {jy:+.1f}) mm")

            # Run a single sim attempt
            result = await self._run_single_attempt(
                target, attempt_num, jitter_xy_mm=(jx, jy)
            )
            attempts.append(result)

            if result.success:
                successes += 1
                self.state.successes_so_far = successes
                self._log(
                    f"Attempt {attempt_num}: SUCCESS "
                    f"(grip_dist={result.grip_distance_mm:.1f}mm, "
                    f"{successes}/{required_successes} needed)"
                )
            else:
                self._log(
                    f"Attempt {attempt_num}: FAILED at phase '{result.phase_reached}'"
                    f"{' — ' + result.error if result.error else ''}"
                )

            # Early exit if we already have enough successes
            if successes >= required_successes:
                self._log(
                    f"Required successes reached ({successes}/{required_successes}) "
                    f"after {attempt_num} attempts"
                )
                break

            # Early exit if success is now impossible
            remaining = max_attempts - attempt_num
            if successes + remaining < required_successes:
                self._log(
                    f"Cannot reach required successes "
                    f"({successes} + {remaining} remaining < {required_successes}). "
                    f"Stopping early."
                )
                break

        # Analyse results
        self.state.phase = RehearsalPhase.ANALYSING
        report = self._analyse(target, attempts, required_successes)
        return report

    async def _run_single_attempt(
        self,
        target: str,
        attempt_number: int,
        jitter_xy_mm: tuple[float, float] = (0.0, 0.0),
    ) -> AttemptResult:
        """Run one full simulation pick attempt."""
        t0 = time.time()
        picker = AutoPick(self.server_url, self.cam_server_url)

        try:
            # Reset arm to home before each attempt so state doesn't leak
            await picker.ops.retreat_home()
            await asyncio.sleep(0.3)

            result = await picker.execute(
                target=target,
                mode="simulation",
                attempt_number=attempt_number,
                jitter_xy_mm=jitter_xy_mm,
            )

            # Try to extract grip distance from the episode
            grip_dist = float("inf")
            ep = picker.episode_recorder.current
            episode_id = ep.episode_id if ep else ""

            # Query the virtual grip API for distance info
            try:
                import httpx

                async with httpx.AsyncClient(timeout=3.0) as client:
                    state_resp = await client.get(f"{self.server_url}/api/state")
                    state_data = state_resp.json()
                    current_joints = state_data.get("joints", result.joints)
                    gripper_w = state_data.get("gripper", 32.5)
                    obj_resp = await client.get(
                        f"{self.server_url}/api/virtual-grip/check",
                        params={
                            "joints": ",".join(str(j) for j in current_joints),
                            "gripper": str(gripper_w),
                        },
                    )
                    obj_data = obj_resp.json()
                    grip_dist = obj_data.get("distance_mm", float("inf"))
            except Exception:
                pass

            return AttemptResult(
                attempt_number=attempt_number,
                success=result.success,
                phase_reached=result.phase.value,
                joints=result.joints,
                target_xy_mm=result.target_xy_mm,
                jitter_applied_mm=jitter_xy_mm,
                grip_distance_mm=grip_dist,
                error=result.error,
                duration_s=time.time() - t0,
                episode_id=episode_id,
            )

        except Exception as e:
            return AttemptResult(
                attempt_number=attempt_number,
                success=False,
                phase_reached="failed",
                jitter_applied_mm=jitter_xy_mm,
                error=str(e),
                duration_s=time.time() - t0,
            )

    def _analyse(
        self,
        target: str,
        attempts: list[AttemptResult],
        required_successes: int,
    ) -> RehearsalReport:
        """Analyse attempt results and produce a report with recommendation."""
        total = len(attempts)
        successful = [a for a in attempts if a.success]
        failed = [a for a in attempts if not a.success]

        success_rate = len(successful) / total if total > 0 else 0.0

        # Count failures by phase
        phase_failures: dict[str, int] = {}
        for a in failed:
            phase_failures[a.phase_reached] = phase_failures.get(a.phase_reached, 0) + 1

        # Find best joints (successful attempt with lowest grip distance)
        best_joints: list[float] = []
        best_xy: tuple[float, float] = (0.0, 0.0)
        best_grip_dist = float("inf")

        for a in successful:
            if a.grip_distance_mm < best_grip_dist:
                best_grip_dist = a.grip_distance_mm
                best_joints = list(a.joints)
                best_xy = a.target_xy_mm

        # If no successful attempt had grip distance info, use the first success
        if not best_joints and successful:
            best_joints = list(successful[0].joints)
            best_xy = successful[0].target_xy_mm

        # Average grip distance across successful attempts
        grip_dists = [a.grip_distance_mm for a in successful if a.grip_distance_mm < float("inf")]
        avg_grip_dist = sum(grip_dists) / len(grip_dists) if grip_dists else float("inf")

        # Recommendation
        met_threshold = len(successful) >= required_successes
        if met_threshold and success_rate >= self.GO_THRESHOLD:
            recommendation = "go"
        elif met_threshold and success_rate >= self.MARGINAL_THRESHOLD:
            recommendation = "marginal"
        else:
            recommendation = "no-go"

        ready = recommendation in ("go", "marginal")

        return RehearsalReport(
            target=target,
            total_attempts=total,
            successful_attempts=len(successful),
            success_rate=round(success_rate, 3),
            required_successes=required_successes,
            phase_failure_counts=phase_failures,
            best_joints=best_joints,
            best_target_xy_mm=best_xy,
            avg_grip_distance_mm=round(avg_grip_dist, 2),
            attempts=attempts,
            episode_ids=[a.episode_id for a in attempts if a.episode_id],
            recommendation=recommendation,
            ready_for_physical=ready,
        )

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.state.log.append(entry)
        logger.info("SimRehearsal: %s", msg)

    def _check_stop(self):
        if self._stop_requested:
            raise _RehearsalStopped()

    @staticmethod
    def _report_to_dict(report: RehearsalReport) -> dict:
        """Serialize a RehearsalReport for JSON API responses."""
        return {
            "target": report.target,
            "total_attempts": report.total_attempts,
            "successful_attempts": report.successful_attempts,
            "success_rate": report.success_rate,
            "required_successes": report.required_successes,
            "phase_failure_counts": report.phase_failure_counts,
            "best_joints": report.best_joints,
            "best_target_xy_mm": list(report.best_target_xy_mm),
            "avg_grip_distance_mm": report.avg_grip_distance_mm,
            "episode_ids": report.episode_ids,
            "recommendation": report.recommendation,
            "ready_for_physical": report.ready_for_physical,
            "total_duration_s": round(report.total_duration_s, 1),
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "success": a.success,
                    "phase_reached": a.phase_reached,
                    "joints": a.joints,
                    "target_xy_mm": list(a.target_xy_mm),
                    "jitter_applied_mm": list(a.jitter_applied_mm),
                    "grip_distance_mm": a.grip_distance_mm,
                    "error": a.error,
                    "duration_s": round(a.duration_s, 1),
                    "episode_id": a.episode_id,
                }
                for a in report.attempts
            ],
            "physical_result": {
                "success": report.physical_result.success,
                "phase": report.physical_result.phase.value,
                "joints": report.physical_result.joints,
                "error": report.physical_result.error,
                "duration_s": round(report.physical_result.duration_s, 1),
            }
            if report.physical_result
            else None,
        }


class _RehearsalStopped(Exception):
    pass
