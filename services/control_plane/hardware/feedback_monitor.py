"""
Feedback Quality Monitor for D1 DDS Connection

Tracks feedback samples, filters zero-readings, and provides reliable state
that downstream consumers (command smoother, web API) can trust.

The core problem: DDS feedback intermittently returns all-zero joint angles
even when the arm is moving. This module maintains a rolling buffer of recent
samples and returns the most recent NON-ZERO reading as the "reliable" state.
"""

import logging
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NUM_JOINTS = 7  # 6 arm + 1 gripper


@dataclass
class FeedbackSample:
    """A single feedback sample with metadata."""

    joint_angles: Dict[str, float]  # angle0..angle6
    timestamp: float  # monotonic time
    wall_time: float  # wall clock time
    is_zero: bool  # True if ALL joints are zero
    seq: int = 0


@dataclass
class FeedbackHealth:
    """Aggregate feedback health metrics."""

    total_samples: int = 0
    zero_samples: int = 0
    zero_rate: float = 0.0  # percentage of zero reads in recent window
    samples_per_second: float = 0.0
    last_good_age_s: float = float("inf")  # seconds since last non-zero reading
    last_any_age_s: float = float("inf")  # seconds since any reading
    is_healthy: bool = False
    degraded_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "zero_samples": self.zero_samples,
            "zero_rate_pct": round(self.zero_rate * 100, 1),
            "samples_per_second": round(self.samples_per_second, 1),
            "last_good_age_s": (
                round(self.last_good_age_s, 3) if self.last_good_age_s != float("inf") else None
            ),
            "last_any_age_s": (
                round(self.last_any_age_s, 3) if self.last_any_age_s != float("inf") else None
            ),
            "is_healthy": self.is_healthy,
            "degraded_reason": self.degraded_reason,
        }


def _is_zero_reading(angles: Dict[str, float]) -> bool:
    """Check if a joint angle reading is all zeros (or effectively zero)."""
    for i in range(NUM_JOINTS):
        key = f"angle{i}"
        val = angles.get(key, 0.0)
        if isinstance(val, (int, float)) and abs(val) > 0.001:
            return False
    return True


def _has_nan_or_inf(angles: Dict[str, float]) -> bool:
    """Check if any joint angle is NaN or Inf."""
    for i in range(NUM_JOINTS):
        key = f"angle{i}"
        val = angles.get(key, 0.0)
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return True
    return False


class FeedbackMonitor:
    """
    Monitors DDS feedback quality and provides filtered, reliable state.

    Maintains a rolling window of recent feedback samples and filters out
    zero-readings to provide the most recent trustworthy joint state.

    Thread-safe: all public methods acquire the internal lock.
    """

    # Thresholds
    MAX_GOOD_AGE_S = 2.0  # Max age for "good" feedback before declaring stale
    ZERO_RATE_WARN = 0.3  # Warn if >30% of recent samples are zero
    ZERO_RATE_CRITICAL = 0.7  # Critical if >70% zero
    MIN_RATE_HZ = 2.0  # Minimum acceptable feedback rate

    def __init__(self, window_size: int = 100):
        self._lock = threading.Lock()
        self._samples: deque[FeedbackSample] = deque(maxlen=window_size)
        self._last_good_sample: Optional[FeedbackSample] = None
        self._last_any_sample: Optional[FeedbackSample] = None
        self._total_count = 0
        self._zero_count = 0
        self._window_size = window_size
        # Callback for degradation alerts
        self._on_degraded = None

    def set_degradation_callback(self, callback):
        """Set a callback(health: FeedbackHealth) called when quality degrades."""
        self._on_degraded = callback

    def record_sample(self, angles: Dict[str, float], seq: int = 0) -> bool:
        """Record a new feedback sample.

        Returns True if the sample is "good" (non-zero, no NaN/Inf).
        Returns False if the sample was rejected or is zero.
        """
        now_mono = time.monotonic()
        now_wall = time.time()

        # Reject NaN/Inf readings entirely
        if _has_nan_or_inf(angles):
            logger.warning("Rejecting feedback with NaN/Inf values: seq=%d", seq)
            return False

        is_zero = _is_zero_reading(angles)
        sample = FeedbackSample(
            joint_angles=dict(angles),
            timestamp=now_mono,
            wall_time=now_wall,
            is_zero=is_zero,
            seq=seq,
        )

        with self._lock:
            self._samples.append(sample)
            self._total_count += 1
            self._last_any_sample = sample

            if is_zero:
                self._zero_count += 1
            else:
                self._last_good_sample = sample

        # Check for degradation (outside lock to avoid callback deadlocks)
        if self._total_count % 10 == 0:  # Check every 10 samples
            health = self.get_health()
            if not health.is_healthy and self._on_degraded:
                try:
                    self._on_degraded(health)
                except Exception:
                    pass

        return not is_zero

    def get_reliable_state(self) -> Optional[Dict[str, float]]:
        """Return the most recent NON-ZERO joint angle reading.

        Returns None if no good readings exist or the last good reading
        is too old (>MAX_GOOD_AGE_S).
        """
        with self._lock:
            if self._last_good_sample is None:
                return None
            age = time.monotonic() - self._last_good_sample.timestamp
            if age > self.MAX_GOOD_AGE_S:
                return None
            return dict(self._last_good_sample.joint_angles)

    def get_reliable_angles(self) -> Optional[np.ndarray]:
        """Return reliable joint angles as a (7,) numpy array, or None."""
        state = self.get_reliable_state()
        if state is None:
            return None
        return np.array(
            [state.get(f"angle{i}", 0.0) for i in range(NUM_JOINTS)],
            dtype=np.float64,
        )

    def get_reliable_gripper(self) -> Optional[float]:
        """Return reliable gripper position (joint 6), or None."""
        state = self.get_reliable_state()
        if state is None:
            return None
        return float(state.get("angle6", 0.0))

    def get_health(self) -> FeedbackHealth:
        """Compute current feedback health metrics."""
        now = time.monotonic()
        with self._lock:
            samples = list(self._samples)
            last_good = self._last_good_sample
            last_any = self._last_any_sample
            total = self._total_count
            zeros = self._zero_count

        if not samples:
            return FeedbackHealth(
                is_healthy=False,
                degraded_reason="no_samples",
            )

        # Compute rate from recent samples
        recent = [s for s in samples if (now - s.timestamp) < 10.0]
        if len(recent) >= 2:
            time_span = recent[-1].timestamp - recent[0].timestamp
            rate = (len(recent) - 1) / time_span if time_span > 0 else 0.0
        else:
            rate = 0.0

        # Zero rate in recent window
        recent_zeros = sum(1 for s in recent if s.is_zero)
        zero_rate = recent_zeros / len(recent) if recent else 0.0

        last_good_age = (now - last_good.timestamp) if last_good else float("inf")
        last_any_age = (now - last_any.timestamp) if last_any else float("inf")

        # Determine health
        reason = None
        healthy = True

        if last_any_age > self.MAX_GOOD_AGE_S:
            healthy = False
            reason = "no_recent_feedback"
        elif last_good_age > self.MAX_GOOD_AGE_S:
            healthy = False
            reason = "no_recent_good_feedback"
        elif zero_rate > self.ZERO_RATE_CRITICAL:
            healthy = False
            reason = f"critical_zero_rate_{zero_rate:.0%}"
        elif zero_rate > self.ZERO_RATE_WARN:
            healthy = True  # Still usable but degraded
            reason = f"high_zero_rate_{zero_rate:.0%}"
        elif rate < self.MIN_RATE_HZ and len(recent) > 2:
            healthy = False
            reason = f"low_rate_{rate:.1f}Hz"

        return FeedbackHealth(
            total_samples=total,
            zero_samples=zeros,
            zero_rate=zero_rate,
            samples_per_second=rate,
            last_good_age_s=last_good_age,
            last_any_age_s=last_any_age,
            is_healthy=healthy,
            degraded_reason=reason,
        )

    def get_recent_samples(self, n: int = 20) -> List[dict]:
        """Return the last N samples as dicts (for diagnostics API)."""
        with self._lock:
            samples = list(self._samples)[-n:]
        now = time.monotonic()
        return [
            {
                "seq": s.seq,
                "age_s": round(now - s.timestamp, 3),
                "is_zero": s.is_zero,
                "angles": [s.joint_angles.get(f"angle{i}", 0.0) for i in range(NUM_JOINTS)],
            }
            for s in samples
        ]

    def is_feedback_fresh(self, max_age_s: float = 0.5) -> bool:
        """Check if we have recent GOOD (non-zero) feedback.

        This is stricter than the DDS connection's is_feedback_fresh()
        because it requires non-zero readings, not just any feedback.
        """
        with self._lock:
            if self._last_good_sample is None:
                return False
            return (time.monotonic() - self._last_good_sample.timestamp) < max_age_s
