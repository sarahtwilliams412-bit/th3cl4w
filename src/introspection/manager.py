"""
Introspection Manager — orchestrates the full self-reflection loop.

This is the top-level coordinator that ties together:
  ReplayBuffer → WorldModel → EpisodeAnalyzer → FeedbackGenerator → CodeImprover

Usage:
    manager = IntrospectionManager()
    manager.start(telemetry_collector)  # begins recording

    # Before a task:
    manager.begin_task("pick_and_place", goal="pick red block", params={...})

    # After a task completes (or fails):
    report = manager.introspect()
    # report contains: episode, reconstruction, assessment, feedback, improvements

    # Or let it auto-introspect when a task ends:
    manager.end_task()  # triggers introspect() automatically

The manager also provides access to accumulated learning:
    manager.learning_summary()
    manager.get_improvements()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.introspection.replay_buffer import ReplayBuffer, TaskContext, Episode
from src.introspection.world_model import WorldModel, TrajectoryReconstruction
from src.introspection.episode_analyzer import EpisodeAnalyzer, EpisodeAssessment, Verdict
from src.introspection.feedback_generator import FeedbackGenerator, Feedback
from src.introspection.code_improver import CodeImprover, Improvement

logger = logging.getLogger("th3cl4w.introspection.manager")


@dataclass
class IntrospectionReport:
    """Complete result of one introspection cycle."""

    episode: Episode | None = None
    reconstruction: TrajectoryReconstruction | None = None
    assessment: EpisodeAssessment | None = None
    feedback: Feedback | None = None
    improvements: list[Improvement] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def verdict(self) -> str:
        if self.assessment:
            return self.assessment.verdict.value
        return "unknown"

    @property
    def narrative(self) -> str:
        if self.feedback:
            return self.feedback.narrative
        return ""

    def summary_dict(self) -> dict:
        return {
            "episode_id": self.episode.episode_id if self.episode else None,
            "task": self.episode.task.task_name if self.episode else None,
            "verdict": self.verdict,
            "overall_score": self.assessment.overall_score if self.assessment else 0.0,
            "confidence": self.assessment.confidence if self.assessment else 0.0,
            "criteria_passed": self.assessment.passed_count if self.assessment else 0,
            "criteria_total": self.assessment.total_criteria if self.assessment else 0,
            "improvements_applied": len(self.improvements),
            "introspection_duration_ms": self.duration_ms,
        }


class IntrospectionManager:
    """Orchestrates the arm's self-reflection pipeline.

    The full loop:
    1. ReplayBuffer continuously captures arm state via telemetry
    2. On introspect(): capture an Episode from the buffer
    3. WorldModel reconstructs what happened spatially
    4. EpisodeAnalyzer assesses success/failure
    5. FeedbackGenerator creates detailed self-feedback
    6. CodeImprover applies parameter adjustments

    The arm gets smarter every time it runs a task.
    """

    def __init__(
        self,
        replay_window_s: float = 15.0,
        auto_introspect: bool = True,
    ) -> None:
        self.replay_buffer = ReplayBuffer(window_seconds=replay_window_s)
        self.world_model = WorldModel()
        self.analyzer = EpisodeAnalyzer()
        self.feedback_gen = FeedbackGenerator()
        self.improver = CodeImprover()

        self.auto_introspect = auto_introspect
        self._task_active = False
        self._task_start_ts = 0.0
        self._reports: list[IntrospectionReport] = []
        self._started = False

    # -- Lifecycle --

    def start(self, telemetry_collector=None) -> None:
        """Start the introspection system.

        Parameters
        ----------
        telemetry_collector : optional TelemetryCollector to subscribe to.
            If provided, the replay buffer automatically captures all events.
            If None, use push_* methods on the replay buffer manually.
        """
        if self._started:
            logger.warning("IntrospectionManager already started")
            return

        if telemetry_collector is not None:
            telemetry_collector.subscribe(self.replay_buffer.on_telemetry_event)
            logger.info("Subscribed to telemetry collector for introspection")

        self._started = True
        logger.info(
            "IntrospectionManager started (window=%.1fs, auto_introspect=%s)",
            self.replay_buffer.window_seconds,
            self.auto_introspect,
        )

    def stop(self, telemetry_collector=None) -> None:
        if telemetry_collector is not None:
            telemetry_collector.unsubscribe(self.replay_buffer.on_telemetry_event)
        self._started = False
        logger.info("IntrospectionManager stopped")

    # -- Task lifecycle --

    def begin_task(
        self,
        task_name: str,
        goal: str = "",
        params: dict | None = None,
        target_position: np.ndarray | None = None,
        target_gripper_mm: float | None = None,
        trajectory_label: str = "",
    ) -> None:
        """Signal that a task is about to start.

        Call this before executing a task so the introspection system knows
        what the arm is trying to do.
        """
        ctx = TaskContext(
            task_name=task_name,
            task_params=params or {},
            goal_description=goal,
            target_position=target_position,
            target_gripper_mm=target_gripper_mm,
            started_at=time.monotonic(),
            planned_trajectory_label=trajectory_label,
        )
        self.replay_buffer.set_task_context(ctx)
        self._task_active = True
        self._task_start_ts = time.monotonic()

        logger.info("Task started: %s (goal: %s)", task_name, goal)

    def end_task(self, lookback_seconds: float | None = None) -> IntrospectionReport | None:
        """Signal that the current task has ended.

        If auto_introspect is True, runs the full introspection pipeline.
        Returns the report, or None if auto_introspect is off.
        """
        if not self._task_active:
            logger.warning("end_task called but no task was active")
            return None

        self._task_active = False

        # Update task context with end time
        ctx = self.replay_buffer.get_task_context()
        ctx.ended_at = time.monotonic()
        self.replay_buffer.set_task_context(ctx)

        # Use task duration as lookback if not specified
        if lookback_seconds is None:
            lookback_seconds = min(
                time.monotonic() - self._task_start_ts + 1.0,
                self.replay_buffer.window_seconds,
            )

        if self.auto_introspect:
            return self.introspect(lookback_seconds=lookback_seconds)

        return None

    # -- Core introspection pipeline --

    def introspect(self, lookback_seconds: float | None = None) -> IntrospectionReport:
        """Run the full introspection pipeline.

        1. Capture episode from replay buffer
        2. Reconstruct via world model
        3. Analyze success/failure
        4. Generate feedback
        5. Apply improvements

        Returns a complete IntrospectionReport.
        """
        t0 = time.monotonic()
        report = IntrospectionReport()

        # Step 1: Capture episode
        episode = self.replay_buffer.capture_episode(lookback_seconds)
        report.episode = episode
        logger.info(
            "Captured episode %s: %d states, %d commands, %.2fs",
            episode.episode_id,
            episode.n_states,
            len(episode.commands),
            episode.duration_s,
        )

        if episode.n_states == 0:
            logger.warning("Episode has no joint data — skipping analysis")
            report.duration_ms = (time.monotonic() - t0) * 1000
            self._reports.append(report)
            return report

        # Step 2: World model reconstruction
        reconstruction = self.world_model.reconstruct(episode)
        report.reconstruction = reconstruction

        motion_summary = self.world_model.summarize_motion(reconstruction)
        tracking_error = self.world_model.compute_tracking_error(episode, reconstruction)

        # Step 3: Analyze
        assessment = self.analyzer.analyze(
            episode, reconstruction, motion_summary, tracking_error
        )
        report.assessment = assessment
        logger.info(
            "Assessment: %s (score=%.2f, %d/%d criteria passed)",
            assessment.verdict.value,
            assessment.overall_score,
            assessment.passed_count,
            assessment.total_criteria,
        )

        # Step 4: Generate feedback
        feedback = self.feedback_gen.generate(assessment, motion_summary, episode)
        report.feedback = feedback

        # Step 5: Apply improvements
        improvements = self.improver.process_feedback(feedback)
        report.improvements = improvements

        report.duration_ms = (time.monotonic() - t0) * 1000

        self._reports.append(report)

        # Log the result
        logger.info(
            "Introspection complete in %.1fms: verdict=%s, %d improvements applied\n%s",
            report.duration_ms,
            report.verdict,
            len(improvements),
            feedback.narrative[:500] + "..." if len(feedback.narrative) > 500 else feedback.narrative,
        )

        return report

    # -- Query interface --

    def get_latest_report(self) -> IntrospectionReport | None:
        return self._reports[-1] if self._reports else None

    def get_reports(self, limit: int = 10) -> list[IntrospectionReport]:
        return self._reports[-limit:]

    def learning_summary(self) -> dict:
        """Get a summary of what the arm has learned."""
        feedback_summary = self.feedback_gen.summarize_learning()
        improvement_summary = self.improver.summary()

        n_reports = len(self._reports)
        verdicts = [r.verdict for r in self._reports]

        return {
            "total_introspections": n_reports,
            "session_verdicts": {
                v: verdicts.count(v) for v in set(verdicts)
            } if verdicts else {},
            "session_success_rate": (
                verdicts.count("success") / len(verdicts) if verdicts else 0.0
            ),
            "feedback_summary": feedback_summary,
            "improvement_summary": improvement_summary,
            "buffer_stats": self.replay_buffer.stats(),
        }

    def get_learned_parameter(self, target: str, param: str, default: float | None = None) -> float | None:
        """Get a learned parameter value from the code improver."""
        return self.improver.get_parameter(target, param, default)

    def get_best_speed_for_task(self, task_type: str) -> float:
        """Get the best known speed factor for a task type."""
        return self.improver.get_best_speed_factor(task_type)

    def get_pending_code_patches(self) -> list[dict]:
        """Get proposed code changes awaiting review."""
        return self.improver.get_pending_patches()

    # -- Convenience --

    def quick_introspect(self, lookback_seconds: float = 5.0) -> str:
        """Run introspection and return just the narrative text.

        Useful for quick "what just happened?" queries.
        """
        report = self.introspect(lookback_seconds=lookback_seconds)
        return report.narrative

    def is_improving(self) -> bool:
        """Check if the arm is getting better over time."""
        if len(self._reports) < 3:
            return False

        recent = self._reports[-5:]
        scores = [r.assessment.overall_score for r in recent if r.assessment]
        if len(scores) < 3:
            return False

        # Simple trend: is the average of the last half better than the first half?
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)
        return second_half > first_half
