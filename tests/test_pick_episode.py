"""Tests for pick episode recorder."""
import json
import time
from pathlib import Path

import pytest

from src.telemetry.pick_episode import PhaseRecord, PickEpisode, PickEpisodeRecorder


@pytest.fixture
def tmp_jsonl(tmp_path):
    return tmp_path / "episodes.jsonl"


@pytest.fixture
def recorder(tmp_jsonl):
    return PickEpisodeRecorder(episodes_file=tmp_jsonl)


def test_pick_episode_defaults():
    ep = PickEpisode()
    assert ep.mode == "physical"
    assert ep.success is False
    assert ep.detected_position_px == (0, 0)
    assert ep.phases == []
    assert ep.episode_id  # non-empty UUID


def test_start_creates_episode(recorder):
    ep = recorder.start(mode="simulation", target="red_block")
    assert ep.mode == "simulation"
    assert ep.target == "red_block"
    assert ep.start_time > 0
    assert recorder.current is ep


def test_record_detection(recorder):
    recorder.start()
    recorder.record_detection("hsv", (320, 240), (100.0, 50.0, 25.0), confidence=0.95, camera=1)
    ep = recorder.current
    assert ep.detection_method == "hsv"
    assert ep.detected_position_px == (320, 240)
    assert ep.detected_position_mm == (100.0, 50.0, 25.0)
    assert ep.detection_confidence == 0.95
    assert ep.detection_camera == 1


def test_record_plan(recorder):
    recorder.start()
    joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    approach = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    recorder.record_plan(joints, gripper_mm=30.0, approach_joints=approach)
    ep = recorder.current
    assert ep.planned_joints == joints
    assert ep.planned_gripper_mm == 30.0
    assert ep.approach_joints == approach


def test_phase_recording(recorder):
    recorder.start()
    recorder.start_phase("approach", joints=[0.1, 0.2], gripper=32.5)
    time.sleep(0.01)
    recorder.end_phase(success=True, joints=[0.3, 0.4], gripper=32.5)
    ep = recorder.current
    assert len(ep.phases) == 1
    phase = ep.phases[0]
    assert phase.name == "approach"
    assert phase.end_time > phase.start_time
    assert phase.success is True
    assert phase.joints_at_start == [0.1, 0.2]
    assert phase.joints_at_end == [0.3, 0.4]


def test_finish_saves_to_jsonl(recorder, tmp_jsonl):
    recorder.start(target="cube")
    recorder.record_result(success=True, grip_verified=True, gripped_object="cube")
    ep = recorder.finish()
    assert ep is not None
    assert ep.end_time > ep.start_time
    assert recorder.current is None
    # Check file
    assert tmp_jsonl.exists()
    data = json.loads(tmp_jsonl.read_text().strip())
    assert data["target"] == "cube"
    assert data["success"] is True


def test_load_episodes(recorder, tmp_jsonl):
    recorder.start(target="a")
    recorder.record_result(success=True)
    recorder.finish()
    recorder.start(target="b")
    recorder.record_result(success=False, failure_reason="dropped")
    recorder.finish()
    episodes = recorder.load_episodes()
    assert len(episodes) == 2
    assert episodes[0]["target"] == "a"
    assert episodes[1]["target"] == "b"
    assert episodes[1]["failure_reason"] == "dropped"


def test_multiple_episodes_append(recorder, tmp_jsonl):
    for i in range(5):
        recorder.start(target=f"obj_{i}")
        recorder.finish()
    lines = [l for l in tmp_jsonl.read_text().strip().split("\n") if l]
    assert len(lines) == 5


def test_phase_timing():
    p = PhaseRecord(name="lift", start_time=100.0, end_time=100.5)
    assert p.end_time > p.start_time
    assert p.name == "lift"


def test_load_empty(recorder, tmp_jsonl):
    assert recorder.load_episodes() == []
