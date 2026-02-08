"""
Collision/Stall Detector for D1 Arm.

Monitors commanded vs actual joint positions and detects stalls —
when a joint can't reach its target (e.g., blocked by an obstacle).
Thread-safe, designed to be called from the WebSocket state loop.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("th3cl4w.collision_detector")


@dataclass
class StallEvent:
    """Info about a detected stall/collision."""

    joint_id: int
    commanded_deg: float
    actual_deg: float
    error_deg: float
    last_good_position: float
    timestamp: float = field(default_factory=time.time)


# Type for the stall callback
StallCallback = Callable[[StallEvent], None]


class CollisionDetector:
    """
    Detects arm stalls by comparing commanded vs actual joint positions.

    A stall is detected when |commanded - actual| > position_error_deg
    persists for longer than stall_duration_s on any joint.

    After firing a stall callback, the detector enters a cooldown period
    to avoid repeated triggers.
    """

    def __init__(
        self,
        num_joints: int = 6,
        position_error_deg: float = 3.0,
        stall_duration_s: float = 0.5,
        cooldown_s: float = 5.0,
    ):
        self._num_joints = num_joints
        self._position_error_deg = position_error_deg
        self._stall_duration_s = stall_duration_s
        self._cooldown_s = cooldown_s

        self._lock = threading.Lock()

        # When each joint first entered error state (None = not in error)
        self._error_start: List[Optional[float]] = [None] * num_joints
        # Last known good positions (before error)
        self._last_good: List[float] = [0.0] * num_joints
        # Last time a stall was fired per joint (for cooldown)
        # Initialize to -infinity so first stall can always fire
        self._last_stall_time: List[float] = [float("-inf")] * num_joints

        self._callbacks: List[StallCallback] = []
        self._enabled = True
        self._recent_stalls: List[StallEvent] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val

    @property
    def last_good_positions(self) -> List[float]:
        with self._lock:
            return list(self._last_good)

    @property
    def recent_stalls(self) -> List[StallEvent]:
        with self._lock:
            return list(self._recent_stalls)

    def on_stall(self, callback: StallCallback) -> None:
        """Register a callback for stall events."""
        self._callbacks.append(callback)

    def update(
        self,
        commanded: List[float],
        actual: List[float],
        now: Optional[float] = None,
    ) -> Optional[StallEvent]:
        """
        Feed commanded and actual joint positions. Call every update cycle.

        Returns a StallEvent if a new stall was just detected, else None.
        """
        if not self._enabled:
            return None

        if now is None:
            now = time.time()

        n = min(len(commanded), len(actual), self._num_joints)

        with self._lock:
            for j in range(n):
                error = abs(commanded[j] - actual[j])

                if error > self._position_error_deg:
                    # Joint is in error state
                    if self._error_start[j] is None:
                        # Just entered error — record when
                        self._error_start[j] = now

                    duration = now - self._error_start[j]
                    if duration >= self._stall_duration_s:
                        # Check cooldown
                        if (now - self._last_stall_time[j]) >= self._cooldown_s:
                            # STALL DETECTED
                            event = StallEvent(
                                joint_id=j,
                                commanded_deg=commanded[j],
                                actual_deg=actual[j],
                                error_deg=error,
                                last_good_position=self._last_good[j],
                                timestamp=now,
                            )
                            self._last_stall_time[j] = now
                            self._recent_stalls.append(event)
                            # Keep only last 50
                            if len(self._recent_stalls) > 50:
                                self._recent_stalls = self._recent_stalls[-50:]

                            # Fire callbacks outside lock
                            self._fire_callbacks(event)
                            return event
                else:
                    # Joint is OK — update last known good and clear error
                    self._last_good[j] = actual[j]
                    self._error_start[j] = None

        return None

    def _fire_callbacks(self, event: StallEvent):
        """Fire all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Stall callback error: %s", e)

    def reset(self):
        """Reset all tracking state."""
        with self._lock:
            self._error_start = [None] * self._num_joints
            self._last_stall_time = [float("-inf")] * self._num_joints
