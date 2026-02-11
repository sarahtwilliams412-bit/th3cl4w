"""LeRobot v2 format writer for th3cl4w demonstration data.

Converts D1 arm teleoperation recordings to the standardized LeRobot v2
dataset format used by the NVIDIA PhysicalAI-Robotics-Kitchen-Sim-Demos
and the broader HuggingFace robotics ecosystem.

Format spec: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3

Output structure:
    lerobot/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   ├── episodes.jsonl
    │   └── stats.json
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── videos/
        └── chunk-000/
            ├── observation.images.cam0/
            │   └── episode_000000.mp4
            └── observation.images.cam1/
                └── episode_000000.mp4
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# D1 arm observation/action dimensions
D1_STATE_DIM = 7       # 6 joint angles (deg) + 1 gripper (mm)
D1_ACTION_DIM = 7      # 6 joint deltas (deg) + 1 gripper delta (mm)
D1_FPS = 10            # Default recording rate (matches arm state publish)
CHUNK_SIZE = 1000       # Episodes per chunk directory
D1_ROBOT_TYPE = "unitree_d1"

# Feature names for the state/action vectors
STATE_NAMES = ["j0_deg", "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "gripper_mm"]
ACTION_NAMES = ["dj0_deg", "dj1_deg", "dj2_deg", "dj3_deg", "dj4_deg", "dj5_deg", "dgripper_mm"]


@dataclass
class LeRobotFrame:
    """A single frame of a LeRobot episode."""

    observation_state: np.ndarray   # (7,) float32: joints + gripper
    action: np.ndarray              # (7,) float32: joint deltas + gripper delta
    cam0_jpeg: Optional[bytes] = None
    cam1_jpeg: Optional[bytes] = None
    timestamp: float = 0.0


@dataclass
class LeRobotEpisode:
    """A complete LeRobot episode ready for serialization."""

    task: str
    frames: List[LeRobotFrame] = field(default_factory=list)
    success: bool = False

    @property
    def length(self) -> int:
        return len(self.frames)


class LeRobotWriter:
    """Writes demonstration data in LeRobot v2 format.

    Usage:
        writer = LeRobotWriter(Path("data/lerobot_demos"))
        writer.add_episode(episode)
        writer.add_episode(episode2)
        writer.finalize()  # writes meta files and computes stats
    """

    def __init__(self, output_dir: Path, fps: int = D1_FPS):
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._episodes: List[LeRobotEpisode] = []
        self._tasks: Dict[str, int] = {}  # task_text -> task_index
        self._task_counter = 0

    def _get_task_index(self, task: str) -> int:
        """Get or create a task index for a language instruction."""
        if task not in self._tasks:
            self._tasks[task] = self._task_counter
            self._task_counter += 1
        return self._tasks[task]

    def add_episode(self, episode: LeRobotEpisode) -> int:
        """Add an episode to the dataset. Returns the episode index."""
        idx = len(self._episodes)
        self._get_task_index(episode.task)
        self._episodes.append(episode)
        logger.info(
            "Added episode %d: task='%s', %d frames",
            idx, episode.task, episode.length,
        )
        return idx

    def add_episode_from_demo_steps(
        self,
        steps: list,
        task: str,
        success: bool = False,
    ) -> int:
        """Convert th3cl4w DemoStep objects to a LeRobot episode.

        Args:
            steps: List of DemoStep dataclass instances from data_collector.py
            task: Language instruction for this episode
            success: Whether the demonstration was successful
        """
        frames = []
        for i, step in enumerate(steps):
            # Build state vector: 6 joints + gripper
            state = np.array(
                step.joints_before + [step.gripper_before],
                dtype=np.float32,
            )

            # Build action vector: joint deltas + gripper delta
            joint_deltas = [
                step.joints_after[j] - step.joints_before[j]
                for j in range(min(len(step.joints_before), 6))
            ]
            gripper_delta = step.gripper_after - step.gripper_before
            action = np.array(joint_deltas + [gripper_delta], dtype=np.float32)

            frames.append(LeRobotFrame(
                observation_state=state,
                action=action,
                timestamp=step.timestamp if hasattr(step, "timestamp") else i / self._fps,
            ))

        episode = LeRobotEpisode(task=task, frames=frames, success=success)
        return self.add_episode(episode)

    def _write_parquet(self, episode_idx: int, episode: LeRobotEpisode):
        """Write episode data as a parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback: write as JSON lines if pyarrow not available
            self._write_parquet_fallback(episode_idx, episode)
            return

        chunk_id = episode_idx // CHUNK_SIZE
        task_idx = self._get_task_index(episode.task)

        # Build columnar data
        rows = []
        for frame_idx, frame in enumerate(episode.frames):
            rows.append({
                "observation.state": frame.observation_state.tolist(),
                "action": frame.action.tolist(),
                "episode_index": episode_idx,
                "frame_index": frame_idx,
                "timestamp": frame.timestamp,
                "task_index": task_idx,
            })

        table = pa.Table.from_pylist(rows)
        parquet_dir = self._output_dir / "data" / f"chunk-{chunk_id:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_dir / f"episode_{episode_idx:06d}.parquet")

    def _write_parquet_fallback(self, episode_idx: int, episode: LeRobotEpisode):
        """Fallback: write episode as JSONL if pyarrow unavailable."""
        chunk_id = episode_idx // CHUNK_SIZE
        task_idx = self._get_task_index(episode.task)

        data_dir = self._output_dir / "data" / f"chunk-{chunk_id:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)

        path = data_dir / f"episode_{episode_idx:06d}.jsonl"
        with open(path, "w") as f:
            for frame_idx, frame in enumerate(episode.frames):
                row = {
                    "observation.state": frame.observation_state.tolist(),
                    "action": frame.action.tolist(),
                    "episode_index": episode_idx,
                    "frame_index": frame_idx,
                    "timestamp": frame.timestamp,
                    "task_index": task_idx,
                }
                f.write(json.dumps(row) + "\n")

    def _write_video(self, episode_idx: int, episode: LeRobotEpisode, cam_key: str):
        """Encode camera frames to MP4 video using ffmpeg."""
        chunk_id = episode_idx // CHUNK_SIZE
        cam_attr = "cam0_jpeg" if cam_key == "cam0" else "cam1_jpeg"

        frames_with_data = [
            f for f in episode.frames if getattr(f, cam_attr) is not None
        ]
        if not frames_with_data:
            return

        video_dir = (
            self._output_dir / "videos" / f"chunk-{chunk_id:03d}"
            / f"observation.images.{cam_key}"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"episode_{episode_idx:06d}.mp4"

        # Write frames to temp dir, then encode with ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames_with_data):
                jpeg_data = getattr(frame, cam_attr)
                frame_path = os.path.join(tmpdir, f"frame_{i:06d}.jpg")
                with open(frame_path, "wb") as fp:
                    fp.write(jpeg_data)

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(self._fps),
                        "-i", os.path.join(tmpdir, "frame_%06d.jpg"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "23",
                        str(video_path),
                    ],
                    capture_output=True,
                    check=True,
                )
                logger.debug("Wrote video: %s", video_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(
                    "ffmpeg encoding failed for %s (episode %d): %s. "
                    "Saving raw frames instead.",
                    cam_key, episode_idx, e,
                )
                # Fallback: save individual JPEGs
                raw_dir = video_dir / f"episode_{episode_idx:06d}_frames"
                raw_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(frames_with_data):
                    jpeg_data = getattr(frame, cam_attr)
                    (raw_dir / f"frame_{i:06d}.jpg").write_bytes(jpeg_data)

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute aggregated statistics across all episodes."""
        all_states = []
        all_actions = []

        for ep in self._episodes:
            for frame in ep.frames:
                all_states.append(frame.observation_state)
                all_actions.append(frame.action)

        if not all_states:
            return {}

        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)

        def _stats_for(arr: np.ndarray, names: List[str]) -> Dict:
            return {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "names": names,
            }

        return {
            "observation.state": _stats_for(states, STATE_NAMES),
            "action": _stats_for(actions, ACTION_NAMES),
        }

    def _write_meta(self):
        """Write all metadata files."""
        meta_dir = self._output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        total_frames = sum(ep.length for ep in self._episodes)
        total_chunks = (len(self._episodes) // CHUNK_SIZE) + 1

        # info.json
        info = {
            "codebase_version": "v2.0",
            "robot_type": D1_ROBOT_TYPE,
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": len(self._episodes) * 2,  # 2 cameras
            "total_chunks": total_chunks,
            "chunks_size": CHUNK_SIZE,
            "fps": self._fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [D1_STATE_DIM],
                    "names": STATE_NAMES,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [D1_ACTION_DIM],
                    "names": ACTION_NAMES,
                },
                "observation.images.cam0": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "video_info": {"codec": "h264", "fps": self._fps},
                },
                "observation.images.cam1": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "video_info": {"codec": "h264", "fps": self._fps},
                },
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # tasks.jsonl
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for task_text, task_idx in sorted(self._tasks.items(), key=lambda x: x[1]):
                f.write(json.dumps({
                    "task_index": task_idx,
                    "language_instruction": task_text,
                }) + "\n")

        # episodes.jsonl
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for i, ep in enumerate(self._episodes):
                f.write(json.dumps({
                    "episode_index": i,
                    "language_instruction": ep.task,
                    "episode_length": ep.length,
                    "success": ep.success,
                }) + "\n")

        # stats.json
        stats = self._compute_stats()
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def finalize(self):
        """Write all episodes to disk in LeRobot v2 format.

        Call this after adding all episodes via add_episode().
        """
        if not self._episodes:
            logger.warning("No episodes to write")
            return

        logger.info(
            "Finalizing LeRobot dataset: %d episodes, %d tasks, output=%s",
            len(self._episodes), len(self._tasks), self._output_dir,
        )

        # Write each episode's data and videos
        for idx, episode in enumerate(self._episodes):
            self._write_parquet(idx, episode)
            self._write_video(idx, episode, "cam0")
            self._write_video(idx, episode, "cam1")

        # Write metadata
        self._write_meta()

        logger.info(
            "LeRobot dataset written: %d episodes, %d total frames",
            len(self._episodes),
            sum(ep.length for ep in self._episodes),
        )


def convert_legacy_demos(demo_dir: Path, output_dir: Path) -> int:
    """Convert existing th3cl4w demonstrations to LeRobot format.

    Reads from data/demonstrations/<demo_id>/ directories that contain
    trajectory.jsonl and frames/ subdirectories.

    Returns the number of episodes converted.
    """
    writer = LeRobotWriter(output_dir)
    converted = 0

    if not demo_dir.exists():
        logger.warning("Demo directory does not exist: %s", demo_dir)
        return 0

    for demo_path in sorted(demo_dir.iterdir()):
        if not demo_path.is_dir():
            continue

        traj_path = demo_path / "trajectory.jsonl"
        meta_path = demo_path / "metadata.json"

        if not traj_path.exists():
            continue

        # Read metadata
        task = "unknown"
        success = False
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                task = meta.get("task", "unknown")
                success = meta.get("success", False)

        # Read trajectory
        frames = []
        with open(traj_path) as f:
            for line in f:
                step = json.loads(line)
                joints = step.get("joints_before", [0.0] * 6)
                gripper = step.get("gripper_before", 0.0)
                joints_after = step.get("joints_after", joints)
                gripper_after = step.get("gripper_after", gripper)

                state = np.array(joints[:6] + [gripper], dtype=np.float32)
                deltas = [joints_after[j] - joints[j] for j in range(min(len(joints), 6))]
                action = np.array(deltas + [gripper_after - gripper], dtype=np.float32)

                # Load camera frames if available
                step_num = step.get("step", len(frames))
                cam0_path = demo_path / "frames" / f"step_{step_num:03d}_cam0.jpg"
                cam1_path = demo_path / "frames" / f"step_{step_num:03d}_cam1.jpg"

                cam0_data = cam0_path.read_bytes() if cam0_path.exists() else None
                cam1_data = cam1_path.read_bytes() if cam1_path.exists() else None

                frames.append(LeRobotFrame(
                    observation_state=state,
                    action=action,
                    cam0_jpeg=cam0_data,
                    cam1_jpeg=cam1_data,
                    timestamp=step.get("timestamp", len(frames) / D1_FPS),
                ))

        if frames:
            episode = LeRobotEpisode(task=task, frames=frames, success=success)
            writer.add_episode(episode)
            converted += 1

    writer.finalize()
    logger.info("Converted %d legacy demonstrations to LeRobot format", converted)
    return converted
