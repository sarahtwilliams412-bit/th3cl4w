"""
Episode Analyzer — assess whether the arm succeeded at its task.

Takes an Episode and its WorldModel reconstruction and produces a
structured assessment with concrete metrics and a verdict.

Success criteria are task-dependent:
- pick_and_place: Did EE reach pick position? Did gripper close? Did EE
  reach place position? Did gripper open?
- pour: Did the wrist rotate to the pour angle? Was the motion smooth?
- go_home / go_ready: Did the arm reach the target pose within tolerance?
- generic: Did the arm move? Was it stable? Any safety events?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger("th3cl4w.introspection.episode_analyzer")


class Verdict(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class CriterionResult:
    """Result of checking a single success criterion."""

    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    detail: str
    value: float | None = None  # the measured value
    threshold: float | None = None  # the threshold it was compared against


@dataclass
class EpisodeAssessment:
    """Complete assessment of an episode."""

    episode_id: str = ""
    task_name: str = ""
    verdict: Verdict = Verdict.UNKNOWN
    confidence: float = 0.0  # 0-1, how confident the analyzer is
    overall_score: float = 0.0  # 0-1 composite score
    criteria: list[CriterionResult] = field(default_factory=list)
    motion_summary: dict = field(default_factory=dict)
    tracking_error: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.criteria if c.passed)

    @property
    def total_criteria(self) -> int:
        return len(self.criteria)


class EpisodeAnalyzer:
    """Analyzes episodes to determine success or failure.

    Uses the world model reconstruction and task context to evaluate
    task-specific and general success criteria.
    """

    # Thresholds (can be tuned by the CodeImprover)
    POSITION_TOLERANCE_RAD = 0.05  # ~2.9 degrees
    GRIPPER_TOLERANCE = 0.15  # normalized
    SMOOTHNESS_JERK_LIMIT = 50.0  # rad/s^3
    TRACKING_ERROR_LIMIT_RAD = 0.1  # ~5.7 degrees
    MIN_MOTION_THRESHOLD_M = 0.005  # must move at least 5mm to count as "moved"
    STALL_VELOCITY_THRESHOLD = 0.01  # rad/s, below this = stalled

    def analyze(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        motion_summary: dict,
        tracking_error: dict,
    ) -> EpisodeAssessment:
        """Run full analysis on an episode.

        Parameters
        ----------
        episode : captured Episode
        reconstruction : WorldModel reconstruction
        motion_summary : from WorldModel.summarize_motion()
        tracking_error : from WorldModel.compute_tracking_error()
        """
        from .replay_buffer import Episode
        from .world_model import TrajectoryReconstruction

        assessment = EpisodeAssessment(
            episode_id=episode.episode_id,
            task_name=episode.task.task_name,
            motion_summary=motion_summary,
            tracking_error=tracking_error,
        )

        # Always run general criteria
        self._check_general_criteria(
            episode, reconstruction, motion_summary, tracking_error, assessment
        )

        # Task-specific criteria
        task = episode.task.task_name.lower()
        if "pick" in task and "place" in task:
            self._check_pick_and_place(episode, reconstruction, assessment)
        elif "pour" in task:
            self._check_pour(episode, reconstruction, assessment)
        elif "home" in task or "ready" in task:
            self._check_go_to_pose(episode, reconstruction, assessment)
        elif "wave" in task:
            self._check_wave(episode, reconstruction, assessment)
        else:
            self._check_generic_motion(episode, reconstruction, motion_summary, assessment)

        # Compute verdict
        self._compute_verdict(assessment)

        return assessment

    # -- General criteria (applied to all tasks) --

    def _check_general_criteria(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        motion_summary: dict,
        tracking_error: dict,
        assessment: EpisodeAssessment,
    ) -> None:
        # 1. No safety events
        n_safety = len(episode.safety_events)
        assessment.criteria.append(
            CriterionResult(
                name="no_safety_violations",
                passed=n_safety == 0,
                score=1.0 if n_safety == 0 else 0.0,
                detail=(
                    f"{n_safety} safety events during episode"
                    if n_safety
                    else "No safety violations"
                ),
                value=float(n_safety),
                threshold=0.0,
            )
        )
        if n_safety > 0:
            assessment.anomalies.append(f"{n_safety} safety violations occurred")

        # 2. No workspace violations
        ws_violations = motion_summary.get("workspace_violations", 0)
        assessment.criteria.append(
            CriterionResult(
                name="workspace_bounds",
                passed=ws_violations == 0,
                score=1.0 if ws_violations == 0 else 0.0,
                detail=(
                    f"{ws_violations} workspace boundary violations"
                    if ws_violations
                    else "Stayed within workspace"
                ),
                value=float(ws_violations),
                threshold=0.0,
            )
        )

        # 3. No near-collisions
        near_collisions = motion_summary.get("near_collisions", 0)
        assessment.criteria.append(
            CriterionResult(
                name="no_near_collisions",
                passed=near_collisions == 0,
                score=1.0 if near_collisions == 0 else max(0.0, 1.0 - near_collisions * 0.2),
                detail=(
                    f"{near_collisions} near-collision events"
                    if near_collisions
                    else "No near-collisions"
                ),
                value=float(near_collisions),
                threshold=0.0,
            )
        )

        # 4. Motion smoothness (jerk)
        jerk_score = self._compute_smoothness(episode)
        assessment.criteria.append(
            CriterionResult(
                name="motion_smoothness",
                passed=jerk_score >= 0.5,
                score=jerk_score,
                detail=f"Smoothness score: {jerk_score:.2f}",
                value=jerk_score,
                threshold=0.5,
            )
        )

        # 5. Tracking accuracy
        mean_err = tracking_error.get("mean_error_rad", 0.0)
        tracking_ok = mean_err < self.TRACKING_ERROR_LIMIT_RAD
        assessment.criteria.append(
            CriterionResult(
                name="tracking_accuracy",
                passed=tracking_ok,
                score=max(0.0, 1.0 - mean_err / self.TRACKING_ERROR_LIMIT_RAD),
                detail=(
                    f"Mean tracking error: {np.degrees(mean_err):.2f}°"
                    if mean_err
                    else "No tracking data"
                ),
                value=mean_err,
                threshold=self.TRACKING_ERROR_LIMIT_RAD,
            )
        )

        # 6. Camera feed was active
        n_cam_refs = len(episode.camera_refs)
        cam_connected = (
            any(c.connected for c in episode.camera_refs) if episode.camera_refs else False
        )
        assessment.criteria.append(
            CriterionResult(
                name="camera_active",
                passed=cam_connected,
                score=1.0 if cam_connected else 0.0,
                detail=f"{n_cam_refs} camera frames, connected={cam_connected}",
                value=float(n_cam_refs),
            )
        )

    # -- Task-specific criteria --

    def _check_pick_and_place(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        assessment: EpisodeAssessment,
    ) -> None:
        task = episode.task

        # Check if the arm actually moved
        if reconstruction.n_frames < 2:
            assessment.criteria.append(
                CriterionResult(
                    name="arm_moved",
                    passed=False,
                    score=0.0,
                    detail="Arm did not move (no frames)",
                )
            )
            return

        positions = np.array([f.joint_positions_rad for f in reconstruction.frames])
        grippers = (
            np.array(
                [
                    episode.joint_snapshots[i].gripper
                    for i in range(min(len(episode.joint_snapshots), reconstruction.n_frames))
                ]
            )
            if episode.joint_snapshots
            else np.array([])
        )

        # Check gripper action: should close then open
        if len(grippers) > 5:
            gripper_min = float(np.min(grippers))
            gripper_max = float(np.max(grippers))
            gripper_range = gripper_max - gripper_min
            assessment.criteria.append(
                CriterionResult(
                    name="gripper_actuated",
                    passed=gripper_range > self.GRIPPER_TOLERANCE,
                    score=min(1.0, gripper_range / 0.5),
                    detail=f"Gripper range: {gripper_min:.2f} to {gripper_max:.2f}",
                    value=gripper_range,
                    threshold=self.GRIPPER_TOLERANCE,
                )
            )

        # Check if target position was reached (if specified)
        if task.target_position is not None:
            target_rad = np.deg2rad(task.target_position[:6])
            final_pos = positions[-1][:6]
            pos_error = float(np.linalg.norm(target_rad - final_pos))
            assessment.criteria.append(
                CriterionResult(
                    name="reached_target",
                    passed=pos_error < self.POSITION_TOLERANCE_RAD * 6,
                    score=max(0.0, 1.0 - pos_error / (self.POSITION_TOLERANCE_RAD * 12)),
                    detail=f"Position error from target: {np.degrees(pos_error):.2f}°",
                    value=pos_error,
                    threshold=self.POSITION_TOLERANCE_RAD * 6,
                )
            )

        # Check path efficiency (should be reasonably direct)
        ee_positions = np.array([f.ee_position for f in reconstruction.frames])
        total_dist = sum(
            np.linalg.norm(ee_positions[i] - ee_positions[i - 1])
            for i in range(1, len(ee_positions))
        )
        displacement = float(np.linalg.norm(ee_positions[-1] - ee_positions[0]))
        efficiency = displacement / max(float(total_dist), 1e-9) if total_dist > 0.01 else 1.0
        # For pick-and-place, efficiency should be moderate (not 1.0, since it goes out and back)
        assessment.criteria.append(
            CriterionResult(
                name="path_executed",
                passed=float(total_dist) > self.MIN_MOTION_THRESHOLD_M,
                score=min(1.0, float(total_dist) / 0.1),
                detail=f"EE traveled {total_dist * 1000:.1f}mm, displacement {displacement * 1000:.1f}mm",
                value=float(total_dist),
                threshold=self.MIN_MOTION_THRESHOLD_M,
            )
        )

    def _check_pour(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        assessment: EpisodeAssessment,
    ) -> None:
        if reconstruction.n_frames < 2:
            assessment.criteria.append(
                CriterionResult(
                    name="arm_moved",
                    passed=False,
                    score=0.0,
                    detail="Arm did not move",
                )
            )
            return

        # Pour requires wrist rotation
        positions = np.array([f.joint_positions_rad for f in reconstruction.frames])
        wrist_range = np.ptp(positions[:, 5])  # J5 = wrist roll
        wrist_threshold = np.deg2rad(30.0)  # expect at least 30° rotation

        assessment.criteria.append(
            CriterionResult(
                name="wrist_rotation",
                passed=wrist_range > wrist_threshold,
                score=min(1.0, wrist_range / wrist_threshold),
                detail=f"Wrist rotation range: {np.degrees(wrist_range):.1f}°",
                value=float(wrist_range),
                threshold=float(wrist_threshold),
            )
        )

        # Should return to approximate starting position
        start_pos = positions[0][:6]
        end_pos = positions[-1][:6]
        return_error = float(np.linalg.norm(start_pos - end_pos))
        assessment.criteria.append(
            CriterionResult(
                name="returned_to_start",
                passed=return_error < self.POSITION_TOLERANCE_RAD * 6,
                score=max(0.0, 1.0 - return_error / (self.POSITION_TOLERANCE_RAD * 12)),
                detail=f"Return error: {np.degrees(return_error):.2f}°",
                value=return_error,
                threshold=self.POSITION_TOLERANCE_RAD * 6,
            )
        )

    def _check_go_to_pose(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        assessment: EpisodeAssessment,
    ) -> None:
        if reconstruction.n_frames < 2:
            assessment.criteria.append(
                CriterionResult(
                    name="arm_moved",
                    passed=False,
                    score=0.0,
                    detail="Arm did not move",
                )
            )
            return

        # Check if we reached target pose
        task = episode.task
        if task.target_position is not None:
            target_rad = np.deg2rad(task.target_position[:6])
            final_pos = reconstruction.frames[-1].joint_positions_rad[:6]
            error = float(np.linalg.norm(target_rad - final_pos))
            assessment.criteria.append(
                CriterionResult(
                    name="reached_target_pose",
                    passed=error < self.POSITION_TOLERANCE_RAD * 3,
                    score=max(0.0, 1.0 - error / (self.POSITION_TOLERANCE_RAD * 6)),
                    detail=f"Final pose error: {np.degrees(error):.2f}°",
                    value=error,
                    threshold=self.POSITION_TOLERANCE_RAD * 3,
                )
            )
        else:
            # Without a target, just check it settled (low final velocity)
            final_vel = (
                episode.joint_snapshots[-1].velocities[:6]
                if episode.joint_snapshots
                else np.zeros(6)
            )
            vel_mag = float(np.linalg.norm(final_vel))
            assessment.criteria.append(
                CriterionResult(
                    name="settled",
                    passed=vel_mag < self.STALL_VELOCITY_THRESHOLD,
                    score=max(0.0, 1.0 - vel_mag / 0.1),
                    detail=f"Final velocity magnitude: {vel_mag:.4f} rad/s",
                    value=vel_mag,
                    threshold=self.STALL_VELOCITY_THRESHOLD,
                )
            )

    def _check_wave(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        assessment: EpisodeAssessment,
    ) -> None:
        if reconstruction.n_frames < 2:
            assessment.criteria.append(
                CriterionResult(
                    name="arm_moved",
                    passed=False,
                    score=0.0,
                    detail="Arm did not move",
                )
            )
            return

        positions = np.array([f.joint_positions_rad for f in reconstruction.frames])
        # Wave expects oscillation on J5
        j5 = positions[:, 5]
        # Count direction changes (peaks/valleys)
        direction_changes = 0
        for i in range(2, len(j5)):
            if (j5[i] - j5[i - 1]) * (j5[i - 1] - j5[i - 2]) < 0:
                direction_changes += 1

        assessment.criteria.append(
            CriterionResult(
                name="oscillation",
                passed=direction_changes >= 2,
                score=min(1.0, direction_changes / 4.0),
                detail=f"{direction_changes} direction changes detected on wave joint",
                value=float(direction_changes),
                threshold=2.0,
            )
        )

    def _check_generic_motion(
        self,
        episode: "Episode",
        reconstruction: "TrajectoryReconstruction",
        motion_summary: dict,
        assessment: EpisodeAssessment,
    ) -> None:
        """For unknown tasks, just check that the arm moved and didn't crash."""
        ee_path = motion_summary.get("ee_path_length_m", 0.0)
        assessment.criteria.append(
            CriterionResult(
                name="motion_executed",
                passed=ee_path > self.MIN_MOTION_THRESHOLD_M,
                score=min(1.0, ee_path / 0.05),
                detail=f"EE path length: {ee_path * 1000:.1f}mm",
                value=ee_path,
                threshold=self.MIN_MOTION_THRESHOLD_M,
            )
        )

    # -- Smoothness analysis --

    def _compute_smoothness(self, episode: "Episode") -> float:
        """Compute motion smoothness score based on jerk (third derivative of position).

        Returns a score from 0 (very jerky) to 1 (perfectly smooth).
        """
        if len(episode.joint_snapshots) < 4:
            return 1.0  # not enough data, assume smooth

        times = np.array([s.ts for s in episode.joint_snapshots])
        positions = np.array([s.positions[:6] for s in episode.joint_snapshots])

        # Numerical differentiation: velocity, acceleration, jerk
        dt = np.diff(times)
        dt = np.where(dt < 1e-9, 1e-9, dt)  # avoid division by zero

        vel = np.diff(positions, axis=0) / dt[:, np.newaxis]
        if len(vel) < 2:
            return 1.0

        dt2 = dt[:-1]
        acc = np.diff(vel, axis=0) / dt2[:, np.newaxis]
        if len(acc) < 2:
            return 1.0

        dt3 = dt2[:-1]
        jerk = np.diff(acc, axis=0) / dt3[:, np.newaxis]

        # RMS jerk across all joints
        rms_jerk = float(np.sqrt(np.mean(jerk**2)))

        # Map to 0-1 score (lower jerk = higher score)
        score = max(0.0, 1.0 - rms_jerk / self.SMOOTHNESS_JERK_LIMIT)
        return score

    # -- Verdict computation --

    def _compute_verdict(self, assessment: EpisodeAssessment) -> None:
        if not assessment.criteria:
            assessment.verdict = Verdict.UNKNOWN
            assessment.confidence = 0.0
            assessment.overall_score = 0.0
            return

        scores = [c.score for c in assessment.criteria]
        passed = [c.passed for c in assessment.criteria]

        assessment.overall_score = float(np.mean(scores))

        pass_rate = sum(passed) / len(passed)
        if pass_rate >= 0.8 and assessment.overall_score >= 0.7:
            assessment.verdict = Verdict.SUCCESS
        elif pass_rate >= 0.5 or assessment.overall_score >= 0.4:
            assessment.verdict = Verdict.PARTIAL
        else:
            assessment.verdict = Verdict.FAILURE

        # Confidence is higher when criteria are clear-cut (scores near 0 or 1)
        extremity = np.mean([abs(s - 0.5) * 2 for s in scores])
        assessment.confidence = float(extremity)

        # Build explanation
        failed_criteria = [c for c in assessment.criteria if not c.passed]
        passed_criteria = [c for c in assessment.criteria if c.passed]

        parts = [
            f"Verdict: {assessment.verdict.value} (score={assessment.overall_score:.2f}, confidence={assessment.confidence:.2f})"
        ]
        if passed_criteria:
            parts.append(f"Passed: {', '.join(c.name for c in passed_criteria)}")
        if failed_criteria:
            parts.append(
                f"Failed: {', '.join(c.name + ' (' + c.detail + ')' for c in failed_criteria)}"
            )
        if assessment.anomalies:
            parts.append(f"Anomalies: {'; '.join(assessment.anomalies)}")

        assessment.explanation = " | ".join(parts)
