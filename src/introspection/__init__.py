"""
Introspection subsystem for th3cl4w.

Allows the arm to watch back what it did, understand whether it succeeded,
generate self-feedback, and propose code/parameter improvements.

Pipeline:
    ReplayBuffer → WorldModel → EpisodeAnalyzer → FeedbackGenerator → CodeImprover
    (orchestrated by IntrospectionManager)

Quick start:
    from src.introspection import IntrospectionManager
    from src.telemetry.collector import get_collector

    manager = IntrospectionManager()
    manager.start(get_collector())

    manager.begin_task("pick_and_place", goal="pick the red block")
    # ... arm executes task ...
    report = manager.end_task()

    print(report.narrative)  # detailed self-feedback
    print(report.verdict)    # "success", "partial", or "failure"
"""

from src.introspection.manager import IntrospectionManager, IntrospectionReport
from src.introspection.replay_buffer import ReplayBuffer, Episode, TaskContext
from src.introspection.world_model import WorldModel, TrajectoryReconstruction
from src.introspection.episode_analyzer import EpisodeAnalyzer, EpisodeAssessment, Verdict
from src.introspection.feedback_generator import FeedbackGenerator, Feedback
from src.introspection.code_improver import CodeImprover

__all__ = [
    "IntrospectionManager",
    "IntrospectionReport",
    "ReplayBuffer",
    "Episode",
    "TaskContext",
    "WorldModel",
    "TrajectoryReconstruction",
    "EpisodeAnalyzer",
    "EpisodeAssessment",
    "Verdict",
    "FeedbackGenerator",
    "Feedback",
    "CodeImprover",
]
