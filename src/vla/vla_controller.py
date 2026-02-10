"""VLA Controller — closed-loop vision-language-action control.

The main orchestrator: observe → plan → act → verify → repeat.

Usage:
    controller = VLAController()
    result = await controller.execute("pick up the red bull can")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from src.vla.vla_model import VLABackend, GeminiVLABackend, Observation, ActionPlan
from src.vla.action_decoder import ActionDecoder, ArmAction, ActionType
from src.control.contact_detector import GripperContactDetector

logger = logging.getLogger(__name__)

ARM_API = "http://localhost:8080"
from src.config.camera_config import CAMERA_SERVER_URL as CAM_API, CAM_SIDE, CAM_ARM, snap_url


class TaskState(Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ExecutionStep:
    """Record of one step in the execution."""

    step_num: int
    state: str
    action: Optional[str] = None
    observation_time_ms: float = 0.0
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    joints_before: Optional[List[float]] = None
    joints_after: Optional[List[float]] = None
    phase: str = ""
    confidence: float = 0.0
    notes: str = ""


@dataclass
class TaskResult:
    """Result of a VLA task execution."""

    success: bool
    task: str
    total_time_s: float = 0.0
    steps: List[ExecutionStep] = field(default_factory=list)
    actions_executed: int = 0
    observations_made: int = 0
    final_phase: str = ""
    message: str = ""
    error: Optional[str] = None


class VLAController:
    """Closed-loop VLA controller.

    Orchestrates the full pipeline:
    1. Observe: capture cameras + read joint state
    2. Plan: send to VLA model, get action plan
    3. Act: execute actions through arm API
    4. Verify: re-observe and check progress
    5. Repeat until done, failed, or max steps reached
    """

    def __init__(
        self,
        backend: Optional[VLABackend] = None,
        max_steps: int = 40,
        settle_time_s: float = 1.5,
        verify_every: int = 4,
    ):
        """
        Args:
            backend: VLA model backend (defaults to Gemini)
            max_steps: Maximum total actions before giving up
            settle_time_s: Time to wait after each joint move for arm to settle
            verify_every: Re-observe after this many actions even if plan doesn't say verify
        """
        self._backend = backend
        self._decoder = ActionDecoder()
        self._max_steps = max_steps
        self._settle_time = settle_time_s
        self._verify_every = verify_every
        self._state = TaskState.IDLE
        self._abort = False
        self._current_task: Optional[str] = None
        self._history: List[str] = []
        self._contact_detector = GripperContactDetector(api_base=ARM_API)
        self._use_adaptive_grip = True  # use contact detection for grips

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def is_busy(self) -> bool:
        return self._state not in (
            TaskState.IDLE,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.ABORTED,
        )

    def abort(self):
        """Signal the controller to abort the current task."""
        self._abort = True
        logger.warning("VLA controller: abort requested")

    def _ensure_backend(self):
        """Lazy-init the backend."""
        if self._backend is None:
            self._backend = GeminiVLABackend()

    async def _observe(self) -> Observation:
        """Capture cameras and read arm state."""
        async with httpx.AsyncClient(timeout=5.0) as c:
            # Parallel fetch: both cameras + arm state
            cam0_task = c.get(snap_url(CAM_SIDE))
            cam1_task = c.get(snap_url(CAM_ARM))
            state_task = c.get(f"{ARM_API}/api/state")

            cam0_resp, cam1_resp, state_resp = await asyncio.gather(
                cam0_task, cam1_task, state_task
            )

        state = state_resp.json()
        return Observation(
            cam0_jpeg=cam0_resp.content,
            cam1_jpeg=cam1_resp.content,
            joints=state["joints"],
            gripper_mm=state["gripper"],
            enabled=state.get("enabled", False),
        )

    async def _execute_joint(self, joint_id: int, angle: float) -> bool:
        """Send a joint command and wait for it to settle."""
        async with httpx.AsyncClient(timeout=15.0) as c:
            resp = await c.post(
                f"{ARM_API}/api/command/set-joint",
                json={"id": joint_id, "angle": angle},
            )
            if resp.status_code != 200:
                data = (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                logger.error("set-joint failed: %s %s", resp.status_code, data.get("error", ""))
                return False
        await asyncio.sleep(self._settle_time)
        return True

    async def _execute_gripper(self, position_mm: float) -> bool:
        """Send a gripper command with optional contact detection.

        For closing actions (position < 30mm), uses adaptive grip with
        contact detection. For opening actions, sends raw command.
        """
        is_closing = position_mm < 30.0

        if is_closing and self._use_adaptive_grip:
            logger.info("Using adaptive grip (target=%.0fmm)", position_mm)
            result = await self._contact_detector.adaptive_grip(
                initial_mm=position_mm,
                object_min_mm=25.0,
            )
            if result.contacted:
                logger.info(
                    "Contact detected at %.1fmm after %d steps",
                    result.final_mm,
                    result.steps_taken,
                )
                self._history.append(f"CONTACT at {result.final_mm:.1f}mm")
            else:
                logger.warning("No contact detected — gripper closed to %.1fmm", result.final_mm)
                self._history.append("NO_CONTACT — grip missed")
            return True

        # Raw command for opening or non-adaptive mode
        async with httpx.AsyncClient(timeout=15.0) as c:
            resp = await c.post(
                f"{ARM_API}/api/command/set-gripper",
                json={"position": position_mm},
            )
            if resp.status_code != 200:
                logger.error("set-gripper failed: %s", resp.status_code)
                return False
        await asyncio.sleep(self._settle_time * 0.5)
        return True

    async def _execute_action(self, action: ArmAction) -> bool:
        """Execute a single decoded action."""
        if not action.is_executable:
            return True

        if action.action_type == ActionType.JOINT:
            ok = await self._execute_joint(action.joint_id, action.target_angle)
            if ok:
                self._history.append(action.describe())
            return ok
        elif action.action_type == ActionType.GRIPPER:
            ok = await self._execute_gripper(action.gripper_mm)
            if ok:
                self._history.append(action.describe())
            return ok
        return True

    async def execute(self, task: str) -> TaskResult:
        """Execute a task end-to-end using the VLA pipeline.

        Args:
            task: Natural language task description (e.g., "pick up the red bull can")

        Returns:
            TaskResult with success/failure, steps taken, timing info
        """
        self._ensure_backend()
        self._state = TaskState.IDLE
        self._abort = False
        self._current_task = task
        self._history = []

        result = TaskResult(success=False, task=task)
        t0 = time.monotonic()
        total_actions = 0

        logger.info("VLA execute: task='%s', max_steps=%d", task, self._max_steps)

        try:
            while total_actions < self._max_steps and not self._abort:
                # 1. OBSERVE
                self._state = TaskState.OBSERVING
                obs_t0 = time.monotonic()
                try:
                    obs = await self._observe()
                except Exception as e:
                    logger.error("Observation failed: %s", e)
                    result.error = f"Observation failed: {e}"
                    self._state = TaskState.FAILED
                    break
                obs_ms = (time.monotonic() - obs_t0) * 1000
                result.observations_made += 1

                if not obs.enabled:
                    logger.error("Arm not enabled — cannot execute VLA task")
                    result.error = "Arm not enabled"
                    self._state = TaskState.FAILED
                    break

                # 2. PLAN
                self._state = TaskState.PLANNING
                plan_t0 = time.monotonic()
                if total_actions == 0 or not self._history:
                    plan = await self._backend.plan(obs, task, self._history)
                else:
                    plan = await self._backend.verify(obs, task, self._history)
                plan_ms = (time.monotonic() - plan_t0) * 1000

                if plan.error:
                    logger.error("Planning error: %s", plan.error)
                    # Retry once
                    plan = await self._backend.plan(obs, task, self._history)
                    if plan.error:
                        result.error = f"Planning failed: {plan.error}"
                        self._state = TaskState.FAILED
                        break

                logger.info(
                    "VLA plan: phase=%s, confidence=%.2f, %d actions, reasoning='%s'",
                    plan.phase,
                    plan.confidence,
                    len(plan.actions),
                    plan.reasoning[:100],
                )

                # Check if task is done
                if plan.is_done:
                    result.success = True
                    result.final_phase = plan.phase
                    result.message = plan.reasoning
                    self._state = TaskState.COMPLETED
                    break

                # 3. DECODE & VALIDATE
                decoded = self._decoder.decode(
                    plan.actions,
                    obs.joints,
                    obs.gripper_mm,
                )

                if not decoded:
                    logger.warning("No actions decoded from plan")
                    result.steps.append(
                        ExecutionStep(
                            step_num=total_actions,
                            state="no_actions",
                            planning_time_ms=plan_ms,
                            observation_time_ms=obs_ms,
                            phase=plan.phase,
                            confidence=plan.confidence,
                            notes="Model returned no valid actions",
                        )
                    )
                    continue

                # 4. EXECUTE actions until verify checkpoint
                self._state = TaskState.EXECUTING
                actions_this_round = 0

                for arm_action in decoded:
                    if self._abort:
                        break

                    if arm_action.action_type == ActionType.VERIFY:
                        logger.info("Verify checkpoint: %s", arm_action.reason)
                        break  # Exit to re-observe

                    if arm_action.action_type == ActionType.DONE:
                        result.success = True
                        result.message = arm_action.reason
                        self._state = TaskState.COMPLETED
                        break

                    if arm_action.rejected:
                        logger.info("Action rejected: %s", arm_action.reject_reason)
                        continue

                    # Execute
                    exec_t0 = time.monotonic()
                    ok = await self._execute_action(arm_action)
                    exec_ms = (time.monotonic() - exec_t0) * 1000

                    step = ExecutionStep(
                        step_num=total_actions,
                        state="executed" if ok else "failed",
                        action=arm_action.describe(),
                        execution_time_ms=exec_ms,
                        observation_time_ms=obs_ms if actions_this_round == 0 else 0,
                        planning_time_ms=plan_ms if actions_this_round == 0 else 0,
                        joints_before=obs.joints,
                        phase=plan.phase,
                        confidence=plan.confidence,
                    )
                    result.steps.append(step)
                    total_actions += 1
                    actions_this_round += 1
                    result.actions_executed += 1

                    if not ok:
                        logger.error("Action execution failed: %s", arm_action.describe())
                        # Don't abort the whole task, just skip to re-observe
                        break

                    # Force re-observe after N actions
                    if actions_this_round >= self._verify_every:
                        logger.info("Forcing verify after %d actions", actions_this_round)
                        break

                if result.success:
                    break

        except Exception as e:
            logger.exception("VLA execution error: %s", e)
            result.error = str(e)
            self._state = TaskState.FAILED

        if self._abort:
            self._state = TaskState.ABORTED
            result.message = "Task aborted by user"

        result.total_time_s = time.monotonic() - t0
        result.final_phase = result.final_phase or self._state.value

        if not result.success and not result.error:
            result.message = (
                f"Did not complete after {total_actions} actions "
                f"({result.observations_made} observations)"
            )

        self._state = TaskState.IDLE if result.success else TaskState.FAILED
        self._current_task = None

        logger.info(
            "VLA task %s: '%s' in %.1fs (%d actions, %d observations)",
            "SUCCEEDED" if result.success else "FAILED",
            task,
            result.total_time_s,
            result.actions_executed,
            result.observations_made,
        )

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            "state": self._state.value,
            "current_task": self._current_task,
            "history_length": len(self._history),
            "backend": self._backend.name if self._backend else None,
        }
