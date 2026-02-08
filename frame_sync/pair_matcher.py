"""
Timestamp-based frame pair matching.

Matches top-down and profile ASCII frames by timestamp proximity,
dropping expired frames to prevent unbounded queue growth.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

GRID_SIZE = 128
MAX_TIMESTAMP_SKEW_MS = 16  # Maximum allowed skew between paired frames
EXPIRY_MS = 50  # Drop frames older than this


@dataclass
class FramePair:
    """Synchronized pair of ASCII frames from top-down and profile cameras."""

    timestamp_ms: int
    top_down: np.ndarray  # uint8 [128, 128] — ASCII codepoints
    profile: np.ndarray  # uint8 [128, 128] — ASCII codepoints


@dataclass
class _TimestampedFrame:
    """Internal frame with arrival time for expiry tracking."""

    timestamp_ms: int
    grid: np.ndarray  # uint8 [128, 128]
    arrival_time: float  # time.monotonic()


class PairMatcher:
    """Match top-down and profile frames by timestamp proximity.

    When a frame arrives, checks the other camera's queue for a frame
    within MAX_TIMESTAMP_SKEW_MS. If found, emits a FramePair. If not,
    enqueues the frame with a 50ms expiry.

    Parameters
    ----------
    on_pair : callable
        Callback invoked with each matched FramePair.
    max_skew_ms : int
        Maximum timestamp difference for a valid pair.
    expiry_ms : int
        Drop queued frames older than this.
    """

    def __init__(
        self,
        on_pair: Callable[[FramePair], None],
        max_skew_ms: int = MAX_TIMESTAMP_SKEW_MS,
        expiry_ms: int = EXPIRY_MS,
    ):
        self.on_pair = on_pair
        self.max_skew_ms = max_skew_ms
        self.expiry_ms = expiry_ms

        self._top_queue: deque[_TimestampedFrame] = deque(maxlen=64)
        self._prof_queue: deque[_TimestampedFrame] = deque(maxlen=64)

        # Stats
        self._pairs_emitted = 0
        self._frames_dropped = 0

    def feed(self, camera_id: str, timestamp_ms: int, grid: np.ndarray) -> None:
        """Feed a new frame. Triggers pair matching.

        Parameters
        ----------
        camera_id : str
            'T' for top-down, 'P' for profile.
        timestamp_ms : int
            Frame timestamp in milliseconds.
        grid : np.ndarray
            uint8 [128, 128] array of ASCII codepoints.
        """
        now = time.monotonic()
        frame = _TimestampedFrame(
            timestamp_ms=timestamp_ms,
            grid=grid,
            arrival_time=now,
        )

        if camera_id == "T":
            self._try_match(frame, self._prof_queue, is_top=True, now=now)
        elif camera_id == "P":
            self._try_match(frame, self._top_queue, is_top=False, now=now)
        else:
            logger.warning("Unknown camera ID: %r", camera_id)

    def _try_match(
        self,
        incoming: _TimestampedFrame,
        other_queue: deque[_TimestampedFrame],
        is_top: bool,
        now: float,
    ) -> None:
        """Try to match incoming frame against the other camera's queue."""
        # Purge expired frames from both queues
        self._purge_expired(self._top_queue, now)
        self._purge_expired(self._prof_queue, now)

        # Search other queue for a matching timestamp
        best_idx: Optional[int] = None
        best_skew = self.max_skew_ms + 1

        for idx, candidate in enumerate(other_queue):
            skew = abs(incoming.timestamp_ms - candidate.timestamp_ms)
            if skew <= self.max_skew_ms and skew < best_skew:
                best_skew = skew
                best_idx = idx

        if best_idx is not None:
            # Found a match — remove from queue and emit pair
            matched = other_queue[best_idx]
            del other_queue[best_idx]

            if is_top:
                pair = FramePair(
                    timestamp_ms=(incoming.timestamp_ms + matched.timestamp_ms) // 2,
                    top_down=incoming.grid,
                    profile=matched.grid,
                )
            else:
                pair = FramePair(
                    timestamp_ms=(incoming.timestamp_ms + matched.timestamp_ms) // 2,
                    top_down=matched.grid,
                    profile=incoming.grid,
                )

            self._pairs_emitted += 1
            self.on_pair(pair)
        else:
            # No match — enqueue for future matching
            own_queue = self._top_queue if is_top else self._prof_queue
            own_queue.append(incoming)

    def _purge_expired(self, queue: deque[_TimestampedFrame], now: float) -> None:
        """Remove frames older than expiry_ms from the front of the queue."""
        expiry_threshold = now - (self.expiry_ms / 1000.0)
        while queue and queue[0].arrival_time < expiry_threshold:
            queue.popleft()
            self._frames_dropped += 1

    @property
    def stats(self) -> dict:
        """Return matching statistics."""
        return {
            "pairs_emitted": self._pairs_emitted,
            "frames_dropped": self._frames_dropped,
            "top_queue_size": len(self._top_queue),
            "prof_queue_size": len(self._prof_queue),
        }
