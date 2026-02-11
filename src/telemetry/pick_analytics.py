"""Pick Analytics — aggregate statistics from pick episode data."""

from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("th3cl4w.telemetry.pick_analytics")

EPISODES_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "pick_episodes.jsonl"


class PickAnalytics:
    """Reads pick_episodes.jsonl and computes aggregate analytics."""

    def __init__(self, episodes_file: Optional[Path] = None):
        self._file = episodes_file or EPISODES_FILE
        self._episodes: list[dict] = []
        self.reload()

    def reload(self):
        """Load episodes from disk."""
        self._episodes = []
        if not self._file.exists():
            return
        try:
            with open(self._file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._episodes.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("Failed to load episodes: %s", e)

    @property
    def episodes(self) -> list[dict]:
        return self._episodes

    def summary(self) -> dict:
        """Overall pick statistics."""
        eps = self._episodes
        if not eps:
            return {
                "total_episodes": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration_s": 0.0,
                "total_duration_s": 0.0,
            }

        successes = sum(1 for e in eps if e.get("success"))
        total = len(eps)
        durations = []
        for e in eps:
            start = e.get("start_time", 0)
            end = e.get("end_time", 0)
            if start and end and end > start:
                durations.append(end - start)

        return {
            "total_episodes": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 4) if total else 0.0,
            "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "total_duration_s": round(sum(durations), 2),
        }

    def phase_breakdown(self) -> list[dict]:
        """Per-phase timing breakdown across all episodes."""
        phase_stats: dict[str, dict] = defaultdict(
            lambda: {
                "count": 0,
                "total_time_s": 0.0,
                "successes": 0,
                "failures": 0,
            }
        )

        for ep in self._episodes:
            for phase in ep.get("phases", []):
                name = phase.get("name", "unknown")
                stats = phase_stats[name]
                stats["count"] += 1
                start = phase.get("start_time", 0)
                end = phase.get("end_time", 0)
                if start and end and end > start:
                    stats["total_time_s"] += end - start
                if phase.get("success", True):
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1

        result = []
        for name, stats in sorted(phase_stats.items()):
            avg = stats["total_time_s"] / stats["count"] if stats["count"] else 0.0
            result.append(
                {
                    "phase": name,
                    "count": stats["count"],
                    "avg_time_s": round(avg, 3),
                    "total_time_s": round(stats["total_time_s"], 3),
                    "success_rate": (
                        round(stats["successes"] / stats["count"], 4) if stats["count"] else 0.0
                    ),
                }
            )
        return result

    def by_target(self) -> dict[str, dict]:
        """Per-target success rate."""
        target_stats: dict[str, dict] = defaultdict(
            lambda: {
                "total": 0,
                "successes": 0,
                "durations": [],
            }
        )

        for ep in self._episodes:
            target = ep.get("target", "unknown") or "unknown"
            stats = target_stats[target]
            stats["total"] += 1
            if ep.get("success"):
                stats["successes"] += 1
            start = ep.get("start_time", 0)
            end = ep.get("end_time", 0)
            if start and end and end > start:
                stats["durations"].append(end - start)

        result = {}
        for target, stats in sorted(target_stats.items()):
            total = stats["total"]
            result[target] = {
                "total": total,
                "successes": stats["successes"],
                "failures": total - stats["successes"],
                "success_rate": round(stats["successes"] / total, 4) if total else 0.0,
                "avg_duration_s": (
                    round(sum(stats["durations"]) / len(stats["durations"]), 2)
                    if stats["durations"]
                    else 0.0
                ),
            }
        return result

    def by_mode(self) -> dict[str, dict]:
        """Sim vs physical comparison."""
        mode_stats: dict[str, dict] = defaultdict(
            lambda: {
                "total": 0,
                "successes": 0,
                "durations": [],
            }
        )

        for ep in self._episodes:
            mode = ep.get("mode", "unknown") or "unknown"
            stats = mode_stats[mode]
            stats["total"] += 1
            if ep.get("success"):
                stats["successes"] += 1
            start = ep.get("start_time", 0)
            end = ep.get("end_time", 0)
            if start and end and end > start:
                stats["durations"].append(end - start)

        result = {}
        for mode, stats in sorted(mode_stats.items()):
            total = stats["total"]
            result[mode] = {
                "total": total,
                "successes": stats["successes"],
                "failures": total - stats["successes"],
                "success_rate": round(stats["successes"] / total, 4) if total else 0.0,
                "avg_duration_s": (
                    round(sum(stats["durations"]) / len(stats["durations"]), 2)
                    if stats["durations"]
                    else 0.0
                ),
            }
        return result
