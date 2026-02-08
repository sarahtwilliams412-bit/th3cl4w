"""Tests for the introspection subsystem."""

import time
import numpy as np
import pytest

from src.introspection.replay_buffer import (
    ReplayBuffer,
    TaskContext,
    Episode,
    JointSnapshot,
    CommandSnapshot,
)
from src.introspection.world_model import WorldModel, TrajectoryReconstruction
from src.introspection.episode_analyzer import EpisodeAnalyzer, Verdict
from src.introspection.feedback_generator import FeedbackGenerator
from src.introspection.code_improver import CodeImprover
from src.introspection.manager import IntrospectionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_joint_snapshot(ts: float, positions: np.ndarray | None = None) -> JointSnapshot:
    if positions is None:
        positions = np.zeros(7)
    return JointSnapshot(
        ts=ts,
        wall_ts=time.time(),
        positions=positions,
        velocities=np.zeros(7),
        torques=np.zeros(7),
        gripper=0.0,
    )


def _fill_buffer_with_motion(buf: ReplayBuffer, n_steps: int = 50, duration: float = 5.0) -> None:
    """Simulate arm moving J1 from 0 to 0.5 rad over a time window."""
    t0 = time.monotonic()
    for i in range(n_steps):
        frac = i / max(n_steps - 1, 1)
        positions = np.zeros(7)
        positions[1] = 0.5 * frac  # J1 sweeps 0 -> 0.5 rad
        buf.push_joint_state(
            positions=positions,
            velocities=np.full(7, 0.01),
            torques=np.zeros(7),
            gripper=0.0,
        )
        # Also push some commands
        if i % 5 == 0:
            buf.push_command(funcode=2, joint_id=1, target_value=0.5 * frac)


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_create_buffer(self):
        buf = ReplayBuffer(window_seconds=10.0)
        assert buf.window_seconds == 10.0
        assert buf.stats()["joint_snapshots"] == 0

    def test_push_and_capture(self):
        buf = ReplayBuffer(window_seconds=15.0)
        _fill_buffer_with_motion(buf, n_steps=20)

        episode = buf.capture_episode()
        assert episode.n_states == 20
        assert len(episode.commands) > 0
        assert episode.episode_id.startswith("ep-")

    def test_task_context(self):
        buf = ReplayBuffer()
        ctx = TaskContext(
            task_name="pick_and_place",
            goal_description="pick the red block",
            task_params={"speed": 0.5},
        )
        buf.set_task_context(ctx)

        episode = buf.capture_episode()
        assert episode.task.task_name == "pick_and_place"
        assert episode.task.goal_description == "pick the red block"

    def test_positions_array(self):
        buf = ReplayBuffer()
        _fill_buffer_with_motion(buf, n_steps=10)
        episode = buf.capture_episode()

        arr = episode.positions_array()
        assert arr.shape == (10, 7)
        # J1 should sweep from 0 to ~0.5
        assert arr[0, 1] == pytest.approx(0.0, abs=0.01)
        assert arr[-1, 1] == pytest.approx(0.5, abs=0.01)

    def test_telemetry_event_callback(self):
        buf = ReplayBuffer()
        # Simulate a DDS feedback event
        event = {
            "timestamp_ms": time.monotonic() * 1000,
            "wall_time_ms": time.time() * 1000,
            "event_type": "dds_receive",
            "payload": {
                "angles": {
                    "angle0": 0.1,
                    "angle1": 0.2,
                    "angle2": 0.3,
                    "angle3": 0.4,
                    "angle4": 0.5,
                    "angle5": 0.6,
                    "angle6": 0.0,
                },
            },
        }
        buf.on_telemetry_event(event)
        assert buf.stats()["joint_snapshots"] == 1


# ---------------------------------------------------------------------------
# WorldModel tests
# ---------------------------------------------------------------------------


class TestWorldModel:
    def test_reconstruct_static(self):
        """Arm at home position (all zeros) — should reconstruct without error."""
        wm = WorldModel()
        buf = ReplayBuffer()

        # Push a few snapshots at home
        for _ in range(5):
            buf.push_joint_state(np.zeros(7), np.zeros(7), np.zeros(7))

        episode = buf.capture_episode()
        recon = wm.reconstruct(episode)

        assert recon.n_frames == 5
        assert recon.total_ee_path_length < 0.01  # barely moved
        assert recon.workspace_violations == 0

    def test_reconstruct_motion(self):
        wm = WorldModel()
        buf = ReplayBuffer()
        _fill_buffer_with_motion(buf, n_steps=30)

        episode = buf.capture_episode()
        recon = wm.reconstruct(episode)

        assert recon.n_frames == 30
        assert recon.total_ee_path_length > 0  # arm moved
        assert recon.max_ee_speed >= 0

    def test_motion_summary(self):
        wm = WorldModel()
        buf = ReplayBuffer()
        _fill_buffer_with_motion(buf, n_steps=20)

        episode = buf.capture_episode()
        recon = wm.reconstruct(episode)
        summary = wm.summarize_motion(recon)

        assert "ee_path_length_m" in summary
        assert "max_speed_ms" in summary
        assert "joint_range_of_motion_rad" in summary
        assert "start_ee_pos" in summary


# ---------------------------------------------------------------------------
# EpisodeAnalyzer tests
# ---------------------------------------------------------------------------


class TestEpisodeAnalyzer:
    def test_analyze_generic_motion(self):
        buf = ReplayBuffer()
        _fill_buffer_with_motion(buf, n_steps=30)

        episode = buf.capture_episode()
        wm = WorldModel()
        recon = wm.reconstruct(episode)
        summary = wm.summarize_motion(recon)
        tracking = wm.compute_tracking_error(episode, recon)

        analyzer = EpisodeAnalyzer()
        assessment = analyzer.analyze(episode, recon, summary, tracking)

        assert assessment.verdict in (Verdict.SUCCESS, Verdict.PARTIAL, Verdict.FAILURE)
        assert len(assessment.criteria) > 0
        assert 0.0 <= assessment.overall_score <= 1.0
        assert assessment.explanation != ""

    def test_analyze_with_task_context(self):
        buf = ReplayBuffer()
        buf.set_task_context(
            TaskContext(
                task_name="go_home",
                target_position=np.zeros(6),
            )
        )
        # Arm starts slightly off and moves to zero
        for i in range(20):
            frac = i / 19.0
            positions = np.zeros(7)
            positions[1] = 0.3 * (1 - frac)  # converge to zero
            buf.push_joint_state(positions, np.zeros(7), np.zeros(7))

        episode = buf.capture_episode()
        wm = WorldModel()
        recon = wm.reconstruct(episode)
        summary = wm.summarize_motion(recon)
        tracking = wm.compute_tracking_error(episode, recon)

        analyzer = EpisodeAnalyzer()
        assessment = analyzer.analyze(episode, recon, summary, tracking)

        # Should find "reached_target_pose" criterion
        criteria_names = [c.name for c in assessment.criteria]
        assert "reached_target_pose" in criteria_names


# ---------------------------------------------------------------------------
# FeedbackGenerator tests
# ---------------------------------------------------------------------------


class TestFeedbackGenerator:
    def test_generate_feedback(self, tmp_path):
        log_path = tmp_path / "learning_log.jsonl"
        gen = FeedbackGenerator(log_path=log_path)

        buf = ReplayBuffer()
        _fill_buffer_with_motion(buf, n_steps=20)
        episode = buf.capture_episode()

        wm = WorldModel()
        recon = wm.reconstruct(episode)
        summary = wm.summarize_motion(recon)
        tracking = wm.compute_tracking_error(episode, recon)

        analyzer = EpisodeAnalyzer()
        assessment = analyzer.analyze(episode, recon, summary, tracking)

        feedback = gen.generate(assessment, summary, episode)

        assert feedback.narrative != ""
        assert feedback.what_happened != ""
        assert feedback.verdict in ("success", "partial", "failure", "unknown")

        # Check it was persisted
        assert log_path.exists()

    def test_learning_summary(self, tmp_path):
        log_path = tmp_path / "learning_log.jsonl"
        gen = FeedbackGenerator(log_path=log_path)

        summary = gen.summarize_learning()
        assert summary["total_episodes"] == 0


# ---------------------------------------------------------------------------
# CodeImprover tests
# ---------------------------------------------------------------------------


class TestCodeImprover:
    def test_process_feedback_adjusts_parameters(self, tmp_path):
        improver = CodeImprover(
            improvements_path=tmp_path / "improvements.json",
            history_path=tmp_path / "history.jsonl",
        )

        from src.introspection.feedback_generator import Feedback

        fb = Feedback(
            episode_id="test-001",
            verdict="failure",
            task_name="pick_and_place",
            parameter_adjustments=[
                {
                    "target": "motion_planner",
                    "parameter": "speed_factor",
                    "direction": "decrease",
                    "reason": "motion was too jerky",
                    "suggested_factor": 0.8,
                }
            ],
        )

        improvements = improver.process_feedback(fb)
        assert len(improvements) >= 1
        assert improvements[0].applied is True
        assert improvements[0].target == "motion_planner"

        # Check the parameter was stored
        val = improver.get_parameter("motion_planner", "speed_factor")
        assert val is not None

    def test_safety_bounds_enforced(self, tmp_path):
        improver = CodeImprover(
            improvements_path=tmp_path / "improvements.json",
            history_path=tmp_path / "history.jsonl",
        )

        from src.introspection.feedback_generator import Feedback

        fb = Feedback(
            episode_id="test-002",
            verdict="failure",
            parameter_adjustments=[
                {
                    "target": "motion_planner",
                    "parameter": "speed_factor",
                    "direction": "decrease",
                    "reason": "test",
                    "suggested_factor": 0.001,  # extreme reduction
                }
            ],
        )

        improvements = improver.process_feedback(fb)
        # Value should be clamped to safety bounds (min 0.1)
        val = improver.get_parameter("motion_planner", "speed_factor")
        assert val is not None
        assert 0.1 <= val <= 1.0

    def test_rollback(self, tmp_path):
        improver = CodeImprover(
            improvements_path=tmp_path / "improvements.json",
            history_path=tmp_path / "history.jsonl",
        )

        from src.introspection.feedback_generator import Feedback

        fb = Feedback(
            episode_id="test-003",
            verdict="failure",
            parameter_adjustments=[
                {
                    "target": "motion_planner",
                    "parameter": "speed_factor",
                    "direction": "decrease",
                    "reason": "test",
                    "suggested_factor": 0.8,
                }
            ],
        )

        improver.process_feedback(fb)
        val_after = improver.get_parameter("motion_planner", "speed_factor")

        success = improver.rollback_last("motion_planner", "speed_factor")
        assert success is True
        val_rolled = improver.get_parameter("motion_planner", "speed_factor")
        assert val_rolled != val_after  # should have changed back


# ---------------------------------------------------------------------------
# IntrospectionManager integration tests
# ---------------------------------------------------------------------------


class TestIntrospectionManager:
    def test_full_pipeline(self):
        manager = IntrospectionManager(replay_window_s=15.0)

        # Simulate a task
        manager.begin_task(
            "go_home",
            goal="return to home position",
            target_position=np.zeros(6),
        )

        # Push some arm data
        for i in range(30):
            frac = i / 29.0
            positions = np.zeros(7)
            positions[1] = 0.3 * (1 - frac)
            manager.replay_buffer.push_joint_state(positions, np.zeros(7), np.zeros(7))

        # End task triggers introspection
        report = manager.end_task()
        assert report is not None
        assert report.verdict in ("success", "partial", "failure", "unknown")
        assert report.narrative != ""
        assert report.episode is not None
        assert report.assessment is not None

    def test_manual_introspect(self):
        manager = IntrospectionManager(auto_introspect=False)

        for _ in range(10):
            manager.replay_buffer.push_joint_state(np.zeros(7), np.zeros(7), np.zeros(7))

        report = manager.introspect(lookback_seconds=5.0)
        assert report.episode is not None
        assert report.assessment is not None

    def test_learning_summary(self):
        manager = IntrospectionManager()
        summary = manager.learning_summary()
        assert "total_introspections" in summary
        assert "buffer_stats" in summary

    def test_quick_introspect(self):
        manager = IntrospectionManager()
        for _ in range(10):
            manager.replay_buffer.push_joint_state(np.zeros(7), np.zeros(7), np.zeros(7))
        narrative = manager.quick_introspect(lookback_seconds=5.0)
        assert isinstance(narrative, str)
