import logging

logger = logging.getLogger(__name__)

"""
Introspection subsystem for th3cl4w.

Allows the arm to watch back what it did, understand whether it succeeded,
generate self-feedback, and propose code/parameter improvements.

Pipeline:
    ReplayBuffer → WorldModel → EpisodeAnalyzer → FeedbackGenerator → CodeImprover
    (orchestrated by IntrospectionManager)

Quick start:
    from services.introspection import IntrospectionManager
    from services.telemetry.collector import get_collector

    manager = IntrospectionManager()
    manager.start(get_collector())

    manager.begin_task("pick_and_place", goal="pick the red block")
    # ... arm executes task ...
    report = manager.end_task()

    logger.info(report.narrative)  # detailed self-feedback
    logger.info(report.verdict)    # "success", "partial", or "failure"
"""

from .manager import IntrospectionManager, IntrospectionReport
from .replay_buffer import ReplayBuffer, Episode, TaskContext
from .world_model import WorldModel, TrajectoryReconstruction
from .episode_analyzer import EpisodeAnalyzer, EpisodeAssessment, Verdict
from .feedback_generator import FeedbackGenerator, Feedback
from .code_improver import CodeImprover

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
