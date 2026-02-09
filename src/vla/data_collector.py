"""Data Collector — record demonstrations for VLA fine-tuning.

Records (observation, action) pairs as the arm is teleoperated,
saving camera frames and joint trajectories for later training.

Usage:
    collector = DataCollector()
    collector.start("pick up the red bull can")
    # ... teleoperate the arm ...
    collector.record_step(joints_before, joints_after, action, cam0_jpg, cam1_jpg)
    # ... more steps ...
    demo_path = collector.stop(success=True)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "demonstrations"


@dataclass
class DemoStep:
    """One step of a demonstration."""
    step: int
    timestamp: float
    joints_before: List[float]
    joints_after: List[float]
    gripper_before: float
    gripper_after: float
    action: Dict[str, Any]
    task: str


@dataclass
class DemoMetadata:
    """Metadata for a recorded demonstration."""
    demo_id: str
    task: str
    start_time: float
    end_time: float = 0.0
    success: bool = False
    num_steps: int = 0
    total_duration_s: float = 0.0
    notes: str = ""


class DataCollector:
    """Records teleoperation demonstrations for VLA training."""

    ARM_API = "http://localhost:8080"
    CAM_API = "http://localhost:8081"

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or DEFAULT_DATA_DIR
        self._recording = False
        self._demo_id: Optional[str] = None
        self._demo_dir: Optional[Path] = None
        self._task: Optional[str] = None
        self._steps: List[DemoStep] = []
        self._start_time: float = 0.0
        self._step_counter: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def demo_id(self) -> Optional[str]:
        return self._demo_id

    @property
    def step_count(self) -> int:
        return self._step_counter

    def start(self, task: str, notes: str = "") -> str:
        """Start recording a new demonstration.

        Args:
            task: Natural language task description
            notes: Optional notes about this demonstration

        Returns:
            Demo ID (directory name)
        """
        if self._recording:
            raise RuntimeError("Already recording. Stop current demo first.")

        # Generate demo ID from timestamp
        self._demo_id = f"demo_{int(time.time())}_{task[:20].replace(' ', '_')}"
        self._demo_dir = self._data_dir / self._demo_id
        self._demo_dir.mkdir(parents=True, exist_ok=True)
        (self._demo_dir / "frames").mkdir(exist_ok=True)

        self._task = task
        self._steps = []
        self._step_counter = 0
        self._start_time = time.time()
        self._recording = True

        # Write initial metadata
        meta = DemoMetadata(
            demo_id=self._demo_id,
            task=task,
            start_time=self._start_time,
            notes=notes,
        )
        self._write_metadata(meta)

        logger.info("Demo recording started: %s (task: '%s')", self._demo_id, task)
        return self._demo_id

    def stop(self, success: bool = True, notes: str = "") -> Optional[str]:
        """Stop recording and finalize the demonstration.

        Args:
            success: Whether the task was completed successfully
            notes: Additional notes

        Returns:
            Path to the demonstration directory
        """
        if not self._recording:
            logger.warning("Not recording — nothing to stop")
            return None

        self._recording = False
        end_time = time.time()

        # Write final trajectory
        self._write_trajectory()

        # Update metadata
        meta = DemoMetadata(
            demo_id=self._demo_id,
            task=self._task,
            start_time=self._start_time,
            end_time=end_time,
            success=success,
            num_steps=self._step_counter,
            total_duration_s=end_time - self._start_time,
            notes=notes,
        )
        self._write_metadata(meta)

        demo_path = str(self._demo_dir)
        logger.info(
            "Demo recording stopped: %s (%d steps, %.1fs, success=%s)",
            self._demo_id, self._step_counter,
            end_time - self._start_time, success,
        )

        self._demo_id = None
        self._demo_dir = None
        self._task = None
        return demo_path

    def record_step(
        self,
        joints_before: List[float],
        joints_after: List[float],
        gripper_before: float,
        gripper_after: float,
        action: Dict[str, Any],
        cam0_jpeg: Optional[bytes] = None,
        cam1_jpeg: Optional[bytes] = None,
    ):
        """Record one step of the demonstration.

        Args:
            joints_before: Joint angles before the action
            joints_after: Joint angles after the action
            gripper_before: Gripper position before
            gripper_after: Gripper position after
            action: Action dict (e.g., {"type": "joint", "id": 0, "delta": 5.0})
            cam0_jpeg: Front camera frame (optional, will be captured if None)
            cam1_jpeg: Overhead camera frame (optional)
        """
        if not self._recording:
            logger.warning("Not recording — ignoring step")
            return

        step = DemoStep(
            step=self._step_counter,
            timestamp=time.time(),
            joints_before=joints_before,
            joints_after=joints_after,
            gripper_before=gripper_before,
            gripper_after=gripper_after,
            action=action,
            task=self._task,
        )
        self._steps.append(step)

        # Save camera frames
        if cam0_jpeg:
            frame_path = self._demo_dir / "frames" / f"step_{self._step_counter:03d}_cam0.jpg"
            frame_path.write_bytes(cam0_jpeg)
        if cam1_jpeg:
            frame_path = self._demo_dir / "frames" / f"step_{self._step_counter:03d}_cam1.jpg"
            frame_path.write_bytes(cam1_jpeg)

        self._step_counter += 1

    async def record_step_auto(self, action: Dict[str, Any]):
        """Record a step by automatically capturing cameras and arm state.

        Captures before/after state and camera frames automatically.
        Call this AFTER executing the action.
        """
        if not self._recording:
            return

        async with httpx.AsyncClient(timeout=5.0) as c:
            state_resp = await c.get(f"{self.ARM_API}/api/state")
            cam0_resp = await c.get(f"{self.CAM_API}/snap/0")
            cam1_resp = await c.get(f"{self.CAM_API}/snap/1")

        state = state_resp.json()
        self.record_step(
            joints_before=state["joints"],  # Approximate — ideally capture before too
            joints_after=state["joints"],
            gripper_before=state["gripper"],
            gripper_after=state["gripper"],
            action=action,
            cam0_jpeg=cam0_resp.content,
            cam1_jpeg=cam1_resp.content,
        )

    def _write_trajectory(self):
        """Write trajectory as JSONL file."""
        if not self._demo_dir or not self._steps:
            return

        traj_path = self._demo_dir / "trajectory.jsonl"
        with open(traj_path, "w") as f:
            for step in self._steps:
                line = {
                    "step": step.step,
                    "timestamp": step.timestamp,
                    "joints_before": step.joints_before,
                    "joints_after": step.joints_after,
                    "gripper_before": step.gripper_before,
                    "gripper_after": step.gripper_after,
                    "action": step.action,
                    "task": step.task,
                }
                f.write(json.dumps(line) + "\n")

    def _write_metadata(self, meta: DemoMetadata):
        """Write metadata JSON."""
        if not self._demo_dir:
            return

        meta_path = self._demo_dir / "metadata.json"
        data = {
            "demo_id": meta.demo_id,
            "task": meta.task,
            "start_time": meta.start_time,
            "end_time": meta.end_time,
            "success": meta.success,
            "num_steps": meta.num_steps,
            "total_duration_s": meta.total_duration_s,
            "notes": meta.notes,
        }
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

    def list_demos(self) -> List[Dict[str, Any]]:
        """List all recorded demonstrations."""
        demos = []
        if not self._data_dir.exists():
            return demos

        for demo_dir in sorted(self._data_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            meta_path = demo_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    demos.append(json.load(f))
            else:
                demos.append({
                    "demo_id": demo_dir.name,
                    "task": "unknown",
                    "num_steps": 0,
                })
        return demos

    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        return {
            "recording": self._recording,
            "demo_id": self._demo_id,
            "task": self._task,
            "step_count": self._step_counter,
            "elapsed_s": time.time() - self._start_time if self._recording else 0,
            "total_demos": len(self.list_demos()),
        }
