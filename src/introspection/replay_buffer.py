"""
Replay Buffer — rolling window of arm state for introspection.

Captures joint states, commands, camera frame metadata, and task context
over a configurable time window (default 15 seconds). Subscribes to the
telemetry collector so recording happens automatically.

The buffer stores lightweight snapshots rather than raw camera pixels;
frame references (timestamps + camera IDs) are stored so the vision
pipeline can re-fetch or replay from disk if needed.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("th3cl4w.introspection.replay_buffer")


# ---------------------------------------------------------------------------
# Snapshot data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JointSnapshot:
    """A single capture of joint state at one instant."""

    ts: float  # monotonic time
    wall_ts: float  # wall-clock epoch
    positions: np.ndarray  # (7,) rad — 6 arm + gripper
    velocities: np.ndarray  # (7,) rad/s
    torques: np.ndarray  # (7,) Nm
    gripper: float  # normalized 0-1


@dataclass(frozen=True, slots=True)
class CommandSnapshot:
    """A command that was sent to the arm."""

    ts: float
    wall_ts: float
    funcode: int
    joint_id: int | None
    target_value: float | None
    data: dict | None


@dataclass(frozen=True, slots=True)
class CameraFrameRef:
    """Reference to a camera frame (not the pixels themselves)."""

    ts: float
    wall_ts: float
    camera_id: str
    fps: float | None
    motion_score: float | None
    connected: bool


@dataclass
class TaskContext:
    """What task the arm was trying to accomplish."""

    task_name: str = ""
    task_params: dict = field(default_factory=dict)
    goal_description: str = ""
    target_position: np.ndarray | None = None  # (6,) degrees — desired EE goal
    target_gripper_mm: float | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    planned_trajectory_label: str = ""


@dataclass
class Episode:
    """A complete captured episode — a slice of the replay buffer.

    This is the unit of data that flows through the introspection pipeline:
    replay → world model → analyzer → feedback → improver.
    """

    episode_id: str = ""
    task: TaskContext = field(default_factory=TaskContext)
    joint_snapshots: list[JointSnapshot] = field(default_factory=list)
    commands: list[CommandSnapshot] = field(default_factory=list)
    camera_refs: list[CameraFrameRef] = field(default_factory=list)
    safety_events: list[dict] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0

    @property
    def duration_s(self) -> float:
        if not self.joint_snapshots:
            return 0.0
        return self.joint_snapshots[-1].ts - self.joint_snapshots[0].ts

    @property
    def n_states(self) -> int:
        return len(self.joint_snapshots)

    def positions_array(self) -> np.ndarray:
        """Return (N, 7) array of joint positions over time."""
        if not self.joint_snapshots:
            return np.empty((0, 7))
        return np.array([s.positions for s in self.joint_snapshots])

    def velocities_array(self) -> np.ndarray:
        if not self.joint_snapshots:
            return np.empty((0, 7))
        return np.array([s.velocities for s in self.joint_snapshots])

    def torques_array(self) -> np.ndarray:
        if not self.joint_snapshots:
            return np.empty((0, 7))
        return np.array([s.torques for s in self.joint_snapshots])

    def times_array(self) -> np.ndarray:
        if not self.joint_snapshots:
            return np.empty(0)
        return np.array([s.ts for s in self.joint_snapshots])

    def commanded_targets(self) -> list[tuple[float, int | None, float | None]]:
        """Return list of (ts, joint_id, target_value) from commands."""
        return [(c.ts, c.joint_id, c.target_value) for c in self.commands]


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Rolling time-window buffer of arm activity.

    Subscribes to the telemetry collector and continuously stores snapshots.
    Call ``capture_episode()`` to extract the last N seconds as an Episode.

    Thread-safe: internal lists are guarded by a lock.
    """

    def __init__(self, window_seconds: float = 15.0, max_snapshots: int = 10000) -> None:
        self.window_seconds = window_seconds
        self.max_snapshots = max_snapshots

        self._lock = threading.Lock()
        self._joint_buf: list[JointSnapshot] = []
        self._cmd_buf: list[CommandSnapshot] = []
        self._cam_buf: list[CameraFrameRef] = []
        self._safety_buf: list[dict] = []

        # Current task context (set by the orchestrator before a task runs)
        self._current_task: TaskContext = TaskContext()

        self._running = False
        self._episode_counter = 0

    # -- Task context management --

    def set_task_context(self, task: TaskContext) -> None:
        with self._lock:
            self._current_task = copy.deepcopy(task)

    def get_task_context(self) -> TaskContext:
        with self._lock:
            return copy.deepcopy(self._current_task)

    # -- Telemetry subscriber callback --

    def on_telemetry_event(self, event: dict) -> None:
        """Callback for TelemetryCollector.subscribe().

        Routes events to the appropriate internal buffer.
        """
        etype = event.get("event_type", "")
        ts = event.get("timestamp_ms", 0.0) / 1000.0  # convert to seconds
        wall_ts = event.get("wall_time_ms", 0.0) / 1000.0
        payload = event.get("payload", {})

        if etype == "dds_receive":
            self._ingest_feedback(ts, wall_ts, payload)
        elif etype == "dds_publish":
            self._ingest_command(ts, wall_ts, payload)
        elif etype == "cam_frame":
            self._ingest_camera(ts, wall_ts, payload)
        elif etype == "state_update":
            # Could be safety or smoother events
            if payload.get("system_event_type") in ("safety_violation", "estop", "collision"):
                self._ingest_safety_event(ts, wall_ts, payload)
        elif etype == "error":
            self._ingest_safety_event(ts, wall_ts, payload)

    def _ingest_feedback(self, ts: float, wall_ts: float, payload: dict) -> None:
        angles = payload.get("angles", {})
        if not angles:
            return
        positions = np.array([
            angles.get(f"angle{i}", 0.0) or 0.0 for i in range(7)
        ])
        # Velocities and torques come from the state, not always in DDS feedback.
        # Default to zeros if not present.
        velocities = np.zeros(7)
        torques = np.zeros(7)
        gripper = positions[6] if len(positions) > 6 else 0.0

        snap = JointSnapshot(
            ts=ts, wall_ts=wall_ts,
            positions=positions, velocities=velocities, torques=torques,
            gripper=gripper,
        )
        with self._lock:
            self._joint_buf.append(snap)
            self._prune_buffer(self._joint_buf)

    def _ingest_command(self, ts: float, wall_ts: float, payload: dict) -> None:
        snap = CommandSnapshot(
            ts=ts, wall_ts=wall_ts,
            funcode=payload.get("funcode", 0),
            joint_id=payload.get("joint_id"),
            target_value=payload.get("target_value"),
            data=payload.get("data"),
        )
        with self._lock:
            self._cmd_buf.append(snap)
            self._prune_buffer(self._cmd_buf)

    def _ingest_camera(self, ts: float, wall_ts: float, payload: dict) -> None:
        ref = CameraFrameRef(
            ts=ts, wall_ts=wall_ts,
            camera_id=payload.get("camera_id", ""),
            fps=payload.get("actual_fps"),
            motion_score=payload.get("motion_score"),
            connected=bool(payload.get("connected", False)),
        )
        with self._lock:
            self._cam_buf.append(ref)
            self._prune_buffer(self._cam_buf)

    def _ingest_safety_event(self, ts: float, wall_ts: float, payload: dict) -> None:
        event = {"ts": ts, "wall_ts": wall_ts, **payload}
        with self._lock:
            self._safety_buf.append(event)
            self._prune_buffer(self._safety_buf)

    def _prune_buffer(self, buf: list) -> None:
        """Remove entries older than the time window and enforce max size."""
        if not buf:
            return
        cutoff = buf[-1].ts if hasattr(buf[-1], "ts") else buf[-1].get("ts", 0.0)
        cutoff -= self.window_seconds

        # Binary-ish prune: drop from front
        while buf and self._get_ts(buf[0]) < cutoff:
            buf.pop(0)

        # Hard cap
        while len(buf) > self.max_snapshots:
            buf.pop(0)

    @staticmethod
    def _get_ts(item: Any) -> float:
        if hasattr(item, "ts"):
            return item.ts
        if isinstance(item, dict):
            return item.get("ts", 0.0)
        return 0.0

    # -- Episode capture --

    def capture_episode(self, lookback_seconds: float | None = None) -> Episode:
        """Snapshot the last N seconds of buffer into an Episode.

        Parameters
        ----------
        lookback_seconds : how far back to look. Defaults to the full window.
        """
        window = lookback_seconds if lookback_seconds is not None else self.window_seconds

        with self._lock:
            self._episode_counter += 1
            eid = f"ep-{self._episode_counter:05d}-{int(time.time())}"
            now = time.monotonic()
            cutoff = now - window

            joints = [s for s in self._joint_buf if s.ts >= cutoff]
            cmds = [s for s in self._cmd_buf if s.ts >= cutoff]
            cams = [s for s in self._cam_buf if s.ts >= cutoff]
            safety = [s for s in self._safety_buf if s.get("ts", 0) >= cutoff]
            task = copy.deepcopy(self._current_task)

        return Episode(
            episode_id=eid,
            task=task,
            joint_snapshots=joints,
            commands=cmds,
            camera_refs=cams,
            safety_events=safety,
            start_ts=cutoff,
            end_ts=now,
        )

    # -- Manual ingestion (for testing or non-telemetry sources) --

    def push_joint_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        torques: np.ndarray,
        gripper: float = 0.0,
    ) -> None:
        """Directly push a joint state (useful for testing or direct polling)."""
        now_mono = time.monotonic()
        now_wall = time.time()
        snap = JointSnapshot(
            ts=now_mono, wall_ts=now_wall,
            positions=np.asarray(positions, dtype=float),
            velocities=np.asarray(velocities, dtype=float),
            torques=np.asarray(torques, dtype=float),
            gripper=float(gripper),
        )
        with self._lock:
            self._joint_buf.append(snap)
            self._prune_buffer(self._joint_buf)

    def push_command(
        self, funcode: int, joint_id: int | None = None, target_value: float | None = None
    ) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()
        snap = CommandSnapshot(
            ts=now_mono, wall_ts=now_wall,
            funcode=funcode, joint_id=joint_id,
            target_value=target_value, data=None,
        )
        with self._lock:
            self._cmd_buf.append(snap)
            self._prune_buffer(self._cmd_buf)

    # -- Stats --

    def stats(self) -> dict:
        with self._lock:
            return {
                "joint_snapshots": len(self._joint_buf),
                "commands": len(self._cmd_buf),
                "camera_refs": len(self._cam_buf),
                "safety_events": len(self._safety_buf),
                "window_seconds": self.window_seconds,
                "episodes_captured": self._episode_counter,
            }

    def clear(self) -> None:
        with self._lock:
            self._joint_buf.clear()
            self._cmd_buf.clear()
            self._cam_buf.clear()
            self._safety_buf.clear()
