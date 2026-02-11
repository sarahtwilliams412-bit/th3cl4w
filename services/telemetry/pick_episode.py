"""Pick Episode Recorder — structured logging of every pick attempt."""

from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("th3cl4w.telemetry.pick_episode")

EPISODES_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "pick_episodes.jsonl"


@dataclass
class PhaseRecord:
    name: str  # "open_gripper", "approach", "lower", "grip", "lift", "verify"
    start_time: float = 0.0
    end_time: float = 0.0
    joints_at_start: list[float] = field(default_factory=list)
    joints_at_end: list[float] = field(default_factory=list)
    gripper_at_start: float = 0.0
    gripper_at_end: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class PickEpisode:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: str = "physical"  # "physical" or "simulation"
    target: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    # Detection
    detection_method: str = ""  # "hsv", "llm", "manual"
    detected_position_px: tuple[int, int] = (0, 0)
    detected_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detection_confidence: float = 0.0
    detection_camera: int = 0

    # Planning
    planned_joints: list[float] = field(default_factory=list)
    planned_gripper_mm: float = 0.0
    approach_joints: list[float] = field(default_factory=list)

    # Execution phases
    phases: list[PhaseRecord] = field(default_factory=list)

    # Result
    success: bool = False
    failure_reason: str = ""
    grip_verified: bool = False
    gripped_object: str = ""

    # Telemetry stats
    total_commands_sent: int = 0
    peak_tracking_error_deg: float = 0.0


class PickEpisodeRecorder:
    """Records pick episodes to JSONL file."""

    def __init__(self, episodes_file: Optional[Path] = None):
        self._file = episodes_file or EPISODES_FILE
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._current: Optional[PickEpisode] = None
        self._current_phase: Optional[PhaseRecord] = None

    def start(self, mode: str = "physical", target: str = "") -> PickEpisode:
        """Start recording a new pick episode."""
        self._current = PickEpisode(
            mode=mode,
            target=target,
            start_time=time.time(),
        )
        logger.info("Pick episode started: %s (mode=%s, target=%s)",
                     self._current.episode_id, mode, target)
        return self._current

    @property
    def current(self) -> Optional[PickEpisode]:
        return self._current

    def record_detection(self, method: str, position_px: tuple, position_mm: tuple,
                         confidence: float = 0.0, camera: int = 0):
        if self._current:
            self._current.detection_method = method
            self._current.detected_position_px = position_px
            self._current.detected_position_mm = position_mm
            self._current.detection_confidence = confidence
            self._current.detection_camera = camera

    def record_plan(self, joints: list[float], gripper_mm: float = 32.5,
                    approach_joints: Optional[list[float]] = None):
        if self._current:
            self._current.planned_joints = list(joints)
            self._current.planned_gripper_mm = gripper_mm
            if approach_joints:
                self._current.approach_joints = list(approach_joints)

    def start_phase(self, name: str, joints: list[float] = None, gripper: float = 0.0):
        self._current_phase = PhaseRecord(
            name=name,
            start_time=time.time(),
            joints_at_start=list(joints) if joints else [],
            gripper_at_start=gripper,
        )

    def end_phase(self, success: bool = True, error: str = "",
                  joints: list[float] = None, gripper: float = 0.0):
        if self._current_phase and self._current:
            self._current_phase.end_time = time.time()
            self._current_phase.success = success
            self._current_phase.error = error
            self._current_phase.joints_at_end = list(joints) if joints else []
            self._current_phase.gripper_at_end = gripper
            self._current.phases.append(self._current_phase)
            self._current_phase = None

    def record_result(self, success: bool, failure_reason: str = "",
                      grip_verified: bool = False, gripped_object: str = ""):
        if self._current:
            self._current.success = success
            self._current.failure_reason = failure_reason
            self._current.grip_verified = grip_verified
            self._current.gripped_object = gripped_object

    def finish(self) -> Optional[PickEpisode]:
        """Finish and save the current episode."""
        if not self._current:
            return None
        self._current.end_time = time.time()
        episode = self._current
        self._save(episode)
        self._current = None
        self._current_phase = None
        logger.info("Pick episode finished: %s (success=%s, %.1fs)",
                     episode.episode_id, episode.success,
                     episode.end_time - episode.start_time)
        return episode

    def _save(self, episode: PickEpisode):
        """Append episode to JSONL file."""
        try:
            data = asdict(episode)
            # Convert tuples to lists for JSON
            data["detected_position_px"] = list(data["detected_position_px"])
            data["detected_position_mm"] = list(data["detected_position_mm"])
            with open(self._file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error("Failed to save episode: %s", e)

    def load_episodes(self, limit: int = 100) -> list[dict]:
        """Load recent episodes from file."""
        if not self._file.exists():
            return []
        episodes = []
        try:
            with open(self._file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to load episodes: %s", e)
        return episodes[-limit:]
