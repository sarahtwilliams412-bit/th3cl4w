"""
Centralized Gemini API rate limiter.

Shared across all modules (location tracker, object labeler, visual servo, etc.)
to prevent 429 errors from concurrent usage.

Usage:
    from shared.utils.gemini_limiter import gemini_limiter

    if gemini_limiter.acquire():
        # safe to call Gemini
        try:
            result = call_gemini(...)
            gemini_limiter.record_success()
        except RateLimitError:
            gemini_limiter.record_429()
    else:
        # rate-limited, skip this call
        pass
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("th3cl4w.gemini_limiter")


class GeminiRateLimiter:
    """Process-wide Gemini rate limiter with exponential backoff.

    - Enforces a minimum interval between calls (default 5s)
    - Exponential backoff on 429 errors (60s -> 120s -> ... -> 600s max)
    - Pauses after 5 consecutive 429s for 10 minutes
    - Can be paused/resumed from UI
    - Thread-safe
    """

    def __init__(
        self,
        min_interval_s: float = 5.0,
        backoff_initial_s: float = 60.0,
        backoff_max_s: float = 600.0,
        backoff_multiplier: float = 2.0,
        pause_after_consecutive: int = 5,
        pause_duration_s: float = 600.0,
    ):
        self._lock = threading.Lock()
        self._min_interval = min_interval_s
        self._backoff_initial = backoff_initial_s
        self._backoff_max = backoff_max_s
        self._backoff_multiplier = backoff_multiplier
        self._pause_after = pause_after_consecutive
        self._pause_duration = pause_duration_s

        # State
        self._last_call_time: float = 0.0
        self._backoff_s: float = 0.0
        self._blocked_until: float = 0.0
        self._consecutive_429s: int = 0
        self._total_calls: int = 0
        self._total_429s: int = 0
        self._paused: bool = False  # manual pause from UI

    def acquire(self) -> bool:
        """Try to acquire permission to call Gemini."""
        with self._lock:
            now = time.time()

            if self._paused:
                return False

            if now < self._blocked_until:
                return False

            if now - self._last_call_time < self._min_interval:
                return False

            self._last_call_time = now
            self._total_calls += 1
            return True

    def record_success(self):
        """Record a successful Gemini call. Resets backoff."""
        with self._lock:
            self._backoff_s = 0.0
            self._blocked_until = 0.0
            self._consecutive_429s = 0

    def record_429(self):
        """Record a 429 error. Applies exponential backoff."""
        with self._lock:
            self._consecutive_429s += 1
            self._total_429s += 1

            if self._consecutive_429s >= self._pause_after:
                self._backoff_s = self._pause_duration
                logger.warning(
                    "Gemini: %d consecutive 429s — pausing for %ds",
                    self._consecutive_429s,
                    int(self._pause_duration),
                )
            elif self._backoff_s == 0:
                self._backoff_s = self._backoff_initial
            else:
                self._backoff_s = min(
                    self._backoff_s * self._backoff_multiplier, self._backoff_max
                )

            self._blocked_until = time.time() + self._backoff_s
            logger.info(
                "Gemini 429 backoff: %.0fs (consecutive: %d, total: %d)",
                self._backoff_s,
                self._consecutive_429s,
                self._total_429s,
            )

    def pause(self):
        """Manually pause all Gemini calls (from UI)."""
        with self._lock:
            self._paused = True
            logger.info("Gemini rate limiter: PAUSED by user")

    def resume(self):
        """Resume Gemini calls after manual pause."""
        with self._lock:
            self._paused = False
            self._blocked_until = 0.0
            self._backoff_s = 0.0
            self._consecutive_429s = 0
            logger.info("Gemini rate limiter: RESUMED by user")

    @property
    def is_limited(self) -> bool:
        """Check if currently rate-limited (without consuming a slot)."""
        with self._lock:
            if self._paused:
                return True
            return time.time() < self._blocked_until

    @property
    def status(self) -> dict:
        """Current limiter state for diagnostics/UI."""
        with self._lock:
            now = time.time()
            remaining = max(0, self._blocked_until - now)
            return {
                "paused": self._paused,
                "rate_limited": now < self._blocked_until,
                "backoff_remaining_s": round(remaining, 1),
                "consecutive_429s": self._consecutive_429s,
                "total_calls": self._total_calls,
                "total_429s": self._total_429s,
                "min_interval_s": self._min_interval,
            }


# Singleton instance — import this everywhere
gemini_limiter = GeminiRateLimiter()
