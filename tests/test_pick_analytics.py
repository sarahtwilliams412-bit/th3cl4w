"""Tests for PickAnalytics."""

import json
import tempfile
from pathlib import Path

import pytest

from src.telemetry.pick_analytics import PickAnalytics


def _write_episodes(tmp: Path, episodes: list[dict]):
    with open(tmp, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


@pytest.fixture
def empty_file(tmp_path):
    p = tmp_path / "episodes.jsonl"
    return p  # doesn't exist


@pytest.fixture
def single_episode(tmp_path):
    p = tmp_path / "episodes.jsonl"
    _write_episodes(
        p,
        [
            {
                "episode_id": "ep1",
                "mode": "physical",
                "target": "redbull",
                "start_time": 1000.0,
                "end_time": 1005.0,
                "success": True,
                "phases": [
                    {"name": "approach", "start_time": 1000.0, "end_time": 1002.0, "success": True},
                    {"name": "grip", "start_time": 1002.0, "end_time": 1003.5, "success": True},
                    {"name": "lift", "start_time": 1003.5, "end_time": 1005.0, "success": True},
                ],
            }
        ],
    )
    return p


@pytest.fixture
def multi_episodes(tmp_path):
    p = tmp_path / "episodes.jsonl"
    _write_episodes(
        p,
        [
            {
                "episode_id": "ep1",
                "mode": "physical",
                "target": "redbull",
                "start_time": 1000.0,
                "end_time": 1005.0,
                "success": True,
                "phases": [
                    {"name": "approach", "start_time": 1000.0, "end_time": 1002.0, "success": True},
                    {"name": "grip", "start_time": 1002.0, "end_time": 1003.0, "success": True},
                ],
            },
            {
                "episode_id": "ep2",
                "mode": "simulation",
                "target": "redbull",
                "start_time": 2000.0,
                "end_time": 2003.0,
                "success": False,
                "failure_reason": "stall",
                "phases": [
                    {"name": "approach", "start_time": 2000.0, "end_time": 2001.5, "success": True},
                    {"name": "grip", "start_time": 2001.5, "end_time": 2003.0, "success": False},
                ],
            },
            {
                "episode_id": "ep3",
                "mode": "simulation",
                "target": "blue",
                "start_time": 3000.0,
                "end_time": 3004.0,
                "success": True,
                "phases": [
                    {"name": "approach", "start_time": 3000.0, "end_time": 3001.0, "success": True},
                    {"name": "grip", "start_time": 3001.0, "end_time": 3002.0, "success": True},
                    {"name": "lift", "start_time": 3002.0, "end_time": 3004.0, "success": True},
                ],
            },
        ],
    )
    return p


class TestEmptyData:
    def test_summary_empty_missing_file(self, empty_file):
        a = PickAnalytics(empty_file)
        s = a.summary()
        assert s["total_episodes"] == 0
        assert s["success_rate"] == 0.0

    def test_phase_breakdown_empty(self, empty_file):
        a = PickAnalytics(empty_file)
        assert a.phase_breakdown() == []

    def test_by_target_empty(self, empty_file):
        a = PickAnalytics(empty_file)
        assert a.by_target() == {}

    def test_by_mode_empty(self, empty_file):
        a = PickAnalytics(empty_file)
        assert a.by_mode() == {}


class TestSingleEpisode:
    def test_summary_single(self, single_episode):
        a = PickAnalytics(single_episode)
        s = a.summary()
        assert s["total_episodes"] == 1
        assert s["successes"] == 1
        assert s["failures"] == 0
        assert s["success_rate"] == 1.0
        assert s["avg_duration_s"] == 5.0

    def test_phase_breakdown_single(self, single_episode):
        a = PickAnalytics(single_episode)
        phases = a.phase_breakdown()
        names = [p["phase"] for p in phases]
        assert "approach" in names
        assert "grip" in names
        assert "lift" in names
        approach = next(p for p in phases if p["phase"] == "approach")
        assert approach["count"] == 1
        assert approach["avg_time_s"] == 2.0


class TestMultipleEpisodes:
    def test_summary_multi(self, multi_episodes):
        a = PickAnalytics(multi_episodes)
        s = a.summary()
        assert s["total_episodes"] == 3
        assert s["successes"] == 2
        assert s["failures"] == 1
        assert abs(s["success_rate"] - 0.6667) < 0.01

    def test_by_target(self, multi_episodes):
        a = PickAnalytics(multi_episodes)
        targets = a.by_target()
        assert "redbull" in targets
        assert "blue" in targets
        assert targets["redbull"]["total"] == 2
        assert targets["redbull"]["successes"] == 1
        assert targets["blue"]["success_rate"] == 1.0

    def test_by_mode(self, multi_episodes):
        a = PickAnalytics(multi_episodes)
        modes = a.by_mode()
        assert "physical" in modes
        assert "simulation" in modes
        assert modes["physical"]["total"] == 1
        assert modes["physical"]["success_rate"] == 1.0
        assert modes["simulation"]["total"] == 2

    def test_phase_breakdown_multi(self, multi_episodes):
        a = PickAnalytics(multi_episodes)
        phases = a.phase_breakdown()
        approach = next(p for p in phases if p["phase"] == "approach")
        assert approach["count"] == 3
        grip = next(p for p in phases if p["phase"] == "grip")
        assert grip["count"] == 3
        # grip had 1 failure
        assert grip["success_rate"] < 1.0

    def test_reload(self, multi_episodes):
        a = PickAnalytics(multi_episodes)
        assert len(a.episodes) == 3
        # Append another
        with open(multi_episodes, "a") as f:
            f.write(
                json.dumps(
                    {
                        "episode_id": "ep4",
                        "success": True,
                        "start_time": 4000,
                        "end_time": 4001,
                        "phases": [],
                        "mode": "physical",
                        "target": "green",
                    }
                )
                + "\n"
            )
        a.reload()
        assert len(a.episodes) == 4
