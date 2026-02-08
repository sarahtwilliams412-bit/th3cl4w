"""
Feedback Generator — the arm's self-narration system.

Takes an EpisodeAssessment and produces detailed natural language feedback
that the arm can use to understand what it did:

- On SUCCESS: verbose documentation of the strategy, parameters, and
  execution that worked, so the arm can replicate it.
- On FAILURE: diagnosis of what went wrong, which criteria failed and why,
  and concrete suggestions for what to try differently.
- On PARTIAL: both — what worked and what didn't.

All feedback is stored in a persistent learning log so the arm accumulates
knowledge across sessions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.introspection.episode_analyzer import EpisodeAssessment, Verdict

logger = logging.getLogger("th3cl4w.introspection.feedback_generator")

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "learning_log.jsonl"


@dataclass
class Feedback:
    """Structured self-feedback for one episode."""

    episode_id: str = ""
    timestamp: float = 0.0
    verdict: str = ""
    task_name: str = ""

    # The core feedback text
    narrative: str = ""

    # Structured breakdown
    what_happened: str = ""
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # For the code improver
    parameter_adjustments: list[dict] = field(default_factory=list)
    strategy_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "task_name": self.task_name,
            "narrative": self.narrative,
            "what_happened": self.what_happened,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "root_causes": self.root_causes,
            "suggestions": self.suggestions,
            "parameter_adjustments": self.parameter_adjustments,
            "strategy_notes": self.strategy_notes,
        }


class FeedbackGenerator:
    """Generates self-feedback from episode assessments.

    The feedback is designed to be both human-readable and machine-parseable
    so the CodeImprover can act on it.
    """

    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> Feedback:
        """Generate comprehensive feedback for an episode.

        Parameters
        ----------
        assessment : the verdict and criteria from the analyzer
        motion_summary : spatial summary from the world model
        episode : the original episode data
        """
        fb = Feedback(
            episode_id=assessment.episode_id,
            timestamp=time.time(),
            verdict=assessment.verdict.value,
            task_name=assessment.task_name,
        )

        # Build the "what happened" summary
        fb.what_happened = self._describe_what_happened(assessment, motion_summary, episode)

        # Branch on verdict
        if assessment.verdict == Verdict.SUCCESS:
            self._generate_success_feedback(fb, assessment, motion_summary, episode)
        elif assessment.verdict == Verdict.FAILURE:
            self._generate_failure_feedback(fb, assessment, motion_summary, episode)
        elif assessment.verdict == Verdict.PARTIAL:
            self._generate_partial_feedback(fb, assessment, motion_summary, episode)
        else:
            fb.narrative = (
                "Unable to determine whether the task succeeded. "
                "Insufficient data was recorded during this episode. "
                "Ensure telemetry is active and the replay buffer is capturing data."
            )

        # Compose full narrative
        fb.narrative = self._compose_narrative(fb)

        # Persist to learning log
        self._persist(fb)

        return fb

    def _describe_what_happened(
        self,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> str:
        """Factual description of what the arm did."""
        parts = []

        task = episode.task
        if task.task_name:
            parts.append(f"Task: {task.task_name}")
            if task.goal_description:
                parts.append(f"Goal: {task.goal_description}")

        duration = motion_summary.get("duration_s", 0.0)
        path_len = motion_summary.get("ee_path_length_m", 0.0)
        displacement = motion_summary.get("ee_displacement_m", 0.0)
        max_speed = motion_summary.get("max_speed_ms", 0.0)

        parts.append(
            f"The arm operated for {duration:.2f} seconds. "
            f"The end-effector traveled {path_len * 1000:.1f}mm total path length "
            f"with {displacement * 1000:.1f}mm net displacement."
        )

        if max_speed > 0:
            parts.append(f"Peak end-effector speed: {max_speed * 1000:.1f}mm/s.")

        n_cmds = len(episode.commands)
        n_states = episode.n_states
        parts.append(f"Recorded {n_states} state snapshots and {n_cmds} commands.")

        efficiency = motion_summary.get("path_efficiency", 0.0)
        if path_len > 0.01:
            parts.append(f"Path efficiency (straight-line/actual): {efficiency:.2f}.")

        joint_rom = motion_summary.get("joint_range_of_motion_rad", [])
        if joint_rom:
            rom_deg = [f"J{i}:{np.degrees(r):.1f}°" for i, r in enumerate(joint_rom[:6])]
            parts.append(f"Joint range of motion: {', '.join(rom_deg)}.")

        n_cams = len(episode.camera_refs)
        connected_cams = set(c.camera_id for c in episode.camera_refs if c.connected)
        if n_cams:
            parts.append(
                f"Camera data: {n_cams} frames from {len(connected_cams)} active camera(s)."
            )
        else:
            parts.append("No camera data was available during this episode.")

        return " ".join(parts)

    def _generate_success_feedback(
        self,
        fb: Feedback,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> None:
        """Verbose documentation of what worked, so the arm can learn to repeat it."""

        # Document what worked
        for criterion in assessment.criteria:
            if criterion.passed:
                fb.what_worked.append(
                    f"{criterion.name}: {criterion.detail} (score={criterion.score:.2f})"
                )

        # Document the strategy
        task = episode.task
        fb.strategy_notes.append(
            f"Successfully completed '{task.task_name}' with the following parameters:"
        )
        if task.task_params:
            for k, v in task.task_params.items():
                fb.strategy_notes.append(f"  {k} = {v}")

        if task.planned_trajectory_label:
            fb.strategy_notes.append(f"Trajectory plan: {task.planned_trajectory_label}")

        # Document motion characteristics that worked
        path_efficiency = motion_summary.get("path_efficiency", 0.0)
        if path_efficiency > 0.6:
            fb.strategy_notes.append(
                f"Path was efficient ({path_efficiency:.2f}). The planned trajectory "
                "was close to optimal — maintain current planning parameters."
            )
        else:
            fb.strategy_notes.append(
                f"Path efficiency was {path_efficiency:.2f} (expected for multi-phase tasks). "
                "This is normal for pick-and-place or multi-waypoint sequences."
            )

        max_speed = motion_summary.get("max_speed_ms", 0.0)
        fb.strategy_notes.append(
            f"Peak speed was {max_speed * 1000:.1f}mm/s — within safe operating range. "
            "This speed setting produced a good result."
        )

        # Note the workspace margin for future reference
        margin = motion_summary.get("min_workspace_margin_m", 0.0)
        fb.strategy_notes.append(f"Minimum workspace margin: {margin * 1000:.1f}mm from boundary.")

        fb.suggestions.append(
            "This execution was successful. Store these parameters as a known-good "
            "configuration for future similar tasks."
        )

    def _generate_failure_feedback(
        self,
        fb: Feedback,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> None:
        """Diagnosis of what went wrong and concrete suggestions."""

        # Document failures
        for criterion in assessment.criteria:
            if not criterion.passed:
                fb.what_failed.append(
                    f"{criterion.name}: {criterion.detail} (score={criterion.score:.2f})"
                )

        # Root cause analysis
        self._diagnose_root_causes(fb, assessment, motion_summary, episode)

        # Generate specific suggestions
        self._generate_improvement_suggestions(fb, assessment, motion_summary, episode)

    def _generate_partial_feedback(
        self,
        fb: Feedback,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> None:
        """For partial success: document both sides."""
        for criterion in assessment.criteria:
            if criterion.passed:
                fb.what_worked.append(
                    f"{criterion.name}: {criterion.detail} (score={criterion.score:.2f})"
                )
            else:
                fb.what_failed.append(
                    f"{criterion.name}: {criterion.detail} (score={criterion.score:.2f})"
                )

        self._diagnose_root_causes(fb, assessment, motion_summary, episode)
        self._generate_improvement_suggestions(fb, assessment, motion_summary, episode)

        fb.strategy_notes.append(
            "This was a partial success. Some aspects worked correctly while others "
            "need adjustment. Focus improvements on the failed criteria while preserving "
            "the parameters that produced the successful aspects."
        )

    def _diagnose_root_causes(
        self,
        fb: Feedback,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> None:
        """Infer root causes from failed criteria patterns."""
        failed_names = {c.name for c in assessment.criteria if not c.passed}

        if "no_safety_violations" in failed_names:
            fb.root_causes.append(
                "Safety violations occurred — the arm may have been commanded outside "
                "safe limits or encountered an obstacle. Check collision memory and "
                "joint limit configuration."
            )

        if "tracking_accuracy" in failed_names:
            tracking = assessment.tracking_error
            max_err = tracking.get("max_error_rad", 0.0)
            fb.root_causes.append(
                f"Tracking error was excessive (max={np.degrees(max_err):.2f}°). "
                "Possible causes: command smoother step size too large, PID gains too low, "
                "mechanical resistance, or the arm couldn't keep up with the command rate."
            )

        if "motion_smoothness" in failed_names:
            fb.root_causes.append(
                "Motion was jerky. This could be caused by: discontinuous waypoints, "
                "command smoother not interpolating enough, or control loop instability."
            )

        if "workspace_bounds" in failed_names:
            fb.root_causes.append(
                "The arm went outside its workspace boundary. The planned trajectory "
                "may have targeted positions beyond the 550mm reach."
            )

        if "reached_target" in failed_names or "reached_target_pose" in failed_names:
            fb.root_causes.append(
                "The arm did not reach its target position. Possible causes: "
                "IK solution was unreachable, trajectory was interrupted by safety "
                "monitor, or insufficient time was allowed for the motion."
            )

        if "gripper_actuated" in failed_names:
            fb.root_causes.append(
                "The gripper did not actuate as expected. It may not have received "
                "the grasp/release commands, or the gripper range was too small to "
                "register as meaningful actuation."
            )

        if "camera_active" in failed_names:
            fb.root_causes.append(
                "No camera feed was active during the episode. Vision-based tasks "
                "cannot succeed without camera data. Check camera connections and "
                "the camera monitor health."
            )

        if not fb.root_causes:
            fb.root_causes.append(
                "No specific root cause identified. The failure may be due to "
                "a combination of minor issues. Review the individual criterion "
                "scores for clues."
            )

    def _generate_improvement_suggestions(
        self,
        fb: Feedback,
        assessment: EpisodeAssessment,
        motion_summary: dict,
        episode: "Episode",
    ) -> None:
        """Generate concrete, actionable suggestions."""
        failed_names = {c.name: c for c in assessment.criteria if not c.passed}

        if "tracking_accuracy" in failed_names:
            criterion = failed_names["tracking_accuracy"]
            err = criterion.value or 0.0
            fb.suggestions.append(
                "Reduce command smoother step size to improve tracking accuracy. "
                "Current error suggests the arm is lagging behind commands."
            )
            fb.parameter_adjustments.append(
                {
                    "target": "command_smoother",
                    "parameter": "max_step_deg",
                    "direction": "decrease",
                    "reason": f"tracking error {np.degrees(err):.2f}° exceeds limit",
                    "suggested_factor": 0.7,
                }
            )

        if "motion_smoothness" in failed_names:
            fb.suggestions.append(
                "Increase the trajectory interpolation density (more waypoints) "
                "or reduce the maximum speed factor to improve smoothness."
            )
            fb.parameter_adjustments.append(
                {
                    "target": "motion_planner",
                    "parameter": "speed_factor",
                    "direction": "decrease",
                    "reason": "motion was too jerky",
                    "suggested_factor": 0.8,
                }
            )

        if "reached_target" in failed_names or "reached_target_pose" in failed_names:
            fb.suggestions.append(
                "Verify the target position is within the arm's reachable workspace "
                "using forward kinematics. If the target is at the edge of the workspace, "
                "consider using a closer intermediate target."
            )
            fb.parameter_adjustments.append(
                {
                    "target": "task_planner",
                    "parameter": "position_tolerance_deg",
                    "direction": "increase",
                    "reason": "target may be barely reachable",
                    "suggested_factor": 1.5,
                }
            )

        if "workspace_bounds" in failed_names:
            fb.suggestions.append(
                "Reduce the reach of planned trajectories. Add workspace-boundary "
                "checking to the motion planner before executing."
            )

        if "no_safety_violations" in failed_names:
            fb.suggestions.append(
                "Review recent collision memory. If the workspace has changed, "
                "clear collision memory and re-learn. If limits are too tight, "
                "verify the safety monitor configuration."
            )

        if "gripper_actuated" in failed_names:
            fb.suggestions.append(
                "Verify gripper commands are being sent with sufficient open/close "
                "range. The gripper_open_mm and gripper_close_mm parameters may "
                "need wider separation."
            )
            fb.parameter_adjustments.append(
                {
                    "target": "task_planner",
                    "parameter": "gripper_open_mm",
                    "direction": "increase",
                    "reason": "gripper didn't actuate enough",
                    "suggested_factor": 1.3,
                }
            )

        if not fb.suggestions:
            fb.suggestions.append(
                "No specific improvement identified. Consider re-running the task "
                "with more conservative parameters (lower speed, wider tolerances) "
                "and compare the results."
            )

    def _compose_narrative(self, fb: Feedback) -> str:
        """Compose the full narrative text from structured feedback."""
        sections = []

        sections.append(f"=== INTROSPECTION REPORT: {fb.task_name or 'unknown task'} ===")
        sections.append(f"Episode: {fb.episode_id}")
        sections.append(f"Verdict: {fb.verdict.upper()}")
        sections.append("")

        sections.append("--- WHAT HAPPENED ---")
        sections.append(fb.what_happened)
        sections.append("")

        if fb.what_worked:
            sections.append("--- WHAT WORKED ---")
            for item in fb.what_worked:
                sections.append(f"  [+] {item}")
            sections.append("")

        if fb.what_failed:
            sections.append("--- WHAT FAILED ---")
            for item in fb.what_failed:
                sections.append(f"  [-] {item}")
            sections.append("")

        if fb.root_causes:
            sections.append("--- ROOT CAUSE ANALYSIS ---")
            for i, cause in enumerate(fb.root_causes, 1):
                sections.append(f"  {i}. {cause}")
            sections.append("")

        if fb.suggestions:
            sections.append("--- SUGGESTIONS FOR IMPROVEMENT ---")
            for i, suggestion in enumerate(fb.suggestions, 1):
                sections.append(f"  {i}. {suggestion}")
            sections.append("")

        if fb.strategy_notes:
            sections.append("--- STRATEGY NOTES ---")
            for note in fb.strategy_notes:
                sections.append(f"  > {note}")
            sections.append("")

        if fb.parameter_adjustments:
            sections.append("--- PARAMETER ADJUSTMENTS ---")
            for adj in fb.parameter_adjustments:
                sections.append(
                    f"  {adj['target']}.{adj['parameter']}: "
                    f"{adj['direction']} by factor {adj.get('suggested_factor', '?')} "
                    f"({adj['reason']})"
                )
            sections.append("")

        return "\n".join(sections)

    def _persist(self, fb: Feedback) -> None:
        """Append feedback to the learning log (JSONL format)."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(fb.to_dict()) + "\n")
            logger.info("Feedback persisted to %s", self.log_path)
        except Exception as e:
            logger.error("Failed to persist feedback: %s", e)

    def load_learning_log(self, limit: int = 100) -> list[dict]:
        """Load recent entries from the learning log."""
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to load learning log: %s", e)
        return entries[-limit:]

    def summarize_learning(self, limit: int = 50) -> dict:
        """Summarize what the arm has learned from recent episodes."""
        entries = self.load_learning_log(limit)
        if not entries:
            return {"total_episodes": 0, "summary": "No learning data available."}

        verdicts = [e.get("verdict", "unknown") for e in entries]
        success_rate = verdicts.count("success") / len(verdicts)

        all_suggestions = []
        all_adjustments = []
        recurring_failures = {}
        for e in entries:
            all_suggestions.extend(e.get("suggestions", []))
            all_adjustments.extend(e.get("parameter_adjustments", []))
            for f in e.get("what_failed", []):
                name = f.split(":")[0].strip()
                recurring_failures[name] = recurring_failures.get(name, 0) + 1

        # Most common failures
        top_failures = sorted(recurring_failures.items(), key=lambda x: -x[1])[:5]

        return {
            "total_episodes": len(entries),
            "success_rate": success_rate,
            "verdicts": {v: verdicts.count(v) for v in set(verdicts)},
            "top_recurring_failures": top_failures,
            "total_suggestions_generated": len(all_suggestions),
            "total_parameter_adjustments_proposed": len(all_adjustments),
        }
