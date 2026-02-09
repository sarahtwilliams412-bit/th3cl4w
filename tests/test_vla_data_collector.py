"""Tests for VLA data collector."""

import json
import pytest
from pathlib import Path
from src.vla.data_collector import DataCollector


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path / "demos"


@pytest.fixture
def collector(tmp_data_dir):
    return DataCollector(data_dir=tmp_data_dir)


class TestDataCollector:
    def test_initial_state(self, collector):
        assert not collector.is_recording
        assert collector.demo_id is None
        assert collector.step_count == 0

    def test_start_recording(self, collector, tmp_data_dir):
        demo_id = collector.start("pick up the can")
        assert collector.is_recording
        assert collector.demo_id == demo_id
        assert (tmp_data_dir / demo_id).is_dir()
        assert (tmp_data_dir / demo_id / "frames").is_dir()
        assert (tmp_data_dir / demo_id / "metadata.json").exists()

    def test_double_start_raises(self, collector):
        collector.start("task 1")
        with pytest.raises(RuntimeError):
            collector.start("task 2")

    def test_stop_without_start(self, collector):
        result = collector.stop()
        assert result is None

    def test_record_step(self, collector, tmp_data_dir):
        collector.start("test task")
        collector.record_step(
            joints_before=[0.0] * 6,
            joints_after=[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper_before=0.0,
            gripper_after=0.0,
            action={"type": "joint", "id": 0, "delta": 5.0},
            cam0_jpeg=b"fake_cam0_jpeg",
            cam1_jpeg=b"fake_cam1_jpeg",
        )
        assert collector.step_count == 1

        # Check frames were saved
        demo_dir = tmp_data_dir / collector.demo_id
        assert (demo_dir / "frames" / "step_000_cam0.jpg").exists()
        assert (demo_dir / "frames" / "step_001_cam0.jpg").exists() is False

    def test_full_recording_cycle(self, collector, tmp_data_dir):
        demo_id = collector.start("pick up object", notes="test run")

        # Record a few steps
        for i in range(3):
            collector.record_step(
                joints_before=[float(i)] * 6,
                joints_after=[float(i + 1)] * 6,
                gripper_before=0.0,
                gripper_after=0.0,
                action={"type": "joint", "id": 0, "delta": 1.0},
            )

        demo_path = collector.stop(success=True, notes="completed")
        assert demo_path is not None
        assert not collector.is_recording

        # Check trajectory file
        traj_path = Path(demo_path) / "trajectory.jsonl"
        assert traj_path.exists()
        lines = traj_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify first line
        step = json.loads(lines[0])
        assert step["step"] == 0
        assert step["task"] == "pick up object"

        # Check metadata
        meta_path = Path(demo_path) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["success"] is True
        assert meta["num_steps"] == 3
        assert meta["task"] == "pick up object"
        assert meta["notes"] == "completed"

    def test_list_demos(self, collector, tmp_data_dir):
        assert collector.list_demos() == []

        collector.start("task 1")
        collector.stop(success=True)

        demos = collector.list_demos()
        assert len(demos) == 1
        assert demos[0]["task"] == "task 1"

    def test_status(self, collector):
        status = collector.get_status()
        assert status["recording"] is False
        assert status["total_demos"] == 0

        collector.start("test")
        status = collector.get_status()
        assert status["recording"] is True
        assert status["task"] == "test"

    def test_record_without_cameras(self, collector):
        """Should work even without camera frames."""
        collector.start("test")
        collector.record_step(
            joints_before=[0.0] * 6,
            joints_after=[5.0] * 6,
            gripper_before=0.0,
            gripper_after=50.0,
            action={"type": "gripper", "position_mm": 50.0},
        )
        assert collector.step_count == 1
        collector.stop()
