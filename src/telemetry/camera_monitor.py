"""Camera health monitoring for th3cl4w."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

_FPS_WINDOW_S = 10.0
_STALL_THRESHOLD_S = 2.0


class CameraHealthMonitor:
    """Per-camera health tracker with FPS, drop, and motion metrics."""

    def __init__(self, camera_id: str, target_fps: float = 30.0) -> None:
        self.camera_id = camera_id
        self.target_fps = target_fps

        self._connected = False
        self._resolution: tuple[int, int] | None = None
        self._drop_count = 0
        self._motion_score = 0.0
        self._last_frame: np.ndarray | None = None

        # monotonic timestamps of received frames (within window)
        self._frame_times: deque[float] = deque()
        self._last_frame_time: float | None = None

    # -- ingest ----------------------------------------------------------

    def on_frame(self, resolution: tuple[int, int], connected: bool = True) -> None:
        now = time.monotonic()
        self._connected = connected
        self._resolution = resolution
        self._last_frame_time = now
        self._frame_times.append(now)
        self._prune_frame_times(now)

    def on_drop(self) -> None:
        self._drop_count += 1

    def compute_motion(self, frame_gray: np.ndarray) -> float:
        """Compute motion score 0-1 via absolute frame diff."""
        if self._last_frame is not None and self._last_frame.shape == frame_gray.shape:
            diff = np.abs(frame_gray.astype(np.int16) - self._last_frame.astype(np.int16))
            score = float(np.mean(diff) / 255.0)
            self._motion_score = min(max(score, 0.0), 1.0)
        else:
            self._motion_score = 0.0
        self._last_frame = frame_gray.copy()
        return self._motion_score

    # -- properties ------------------------------------------------------

    @property
    def actual_fps(self) -> float:
        if not self._frame_times:
            return 0.0
        now = time.monotonic()
        self._prune_frame_times(now)
        count = len(self._frame_times)
        if count < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (count - 1) / span if span > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        last_age_ms: float | None = None
        if self._last_frame_time is not None:
            last_age_ms = (now - self._last_frame_time) * 1000

        stalled = (
            last_age_ms is not None and last_age_ms > _STALL_THRESHOLD_S * 1000
        ) or self._last_frame_time is None

        return {
            "camera_id": self.camera_id,
            "connected": self._connected,
            "actual_fps": self.actual_fps,
            "target_fps": self.target_fps,
            "drop_count": self._drop_count,
            "last_frame_age_ms": last_age_ms,
            "stalled": stalled,
            "resolution": self._resolution,
            "motion_score": self._motion_score,
        }

    # -- internal --------------------------------------------------------

    def _prune_frame_times(self, now: float) -> None:
        cutoff = now - _FPS_WINDOW_S
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
