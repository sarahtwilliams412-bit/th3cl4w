"""Tests for the centralized Gemini rate limiter."""

import time
from unittest.mock import patch

import pytest

from src.utils.gemini_limiter import GeminiRateLimiter


class TestAcquire:
    def test_first_call_allowed(self):
        limiter = GeminiRateLimiter(min_interval_s=5.0)
        assert limiter.acquire() is True

    def test_second_call_within_interval_blocked(self):
        limiter = GeminiRateLimiter(min_interval_s=5.0)
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_call_after_interval_allowed(self):
        limiter = GeminiRateLimiter(min_interval_s=0.1)
        assert limiter.acquire() is True
        time.sleep(0.15)
        assert limiter.acquire() is True


class TestBackoff:
    def test_record_429_triggers_backoff(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0, backoff_initial_s=1.0)
        limiter.acquire()
        limiter.record_429()
        # Should be blocked now
        assert limiter.acquire() is False
        assert limiter.status["consecutive_429s"] == 1
        assert limiter.status["backoff_remaining_s"] > 0

    def test_record_success_resets_backoff(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0, backoff_initial_s=1.0)
        limiter.acquire()
        limiter.record_429()
        limiter.record_success()
        assert limiter.status["consecutive_429s"] == 0
        assert limiter.status["backoff_remaining_s"] == 0

    def test_exponential_backoff(self):
        limiter = GeminiRateLimiter(
            min_interval_s=0.0,
            backoff_initial_s=10.0,
            backoff_multiplier=2.0,
            backoff_max_s=100.0,
            pause_after_consecutive=99,  # don't trigger extended pause
        )
        limiter.acquire()
        limiter.record_429()  # 10s
        s1 = limiter.status["backoff_remaining_s"]
        limiter.record_429()  # 20s
        s2 = limiter.status["backoff_remaining_s"]
        assert s2 > s1

    def test_consecutive_429s_trigger_extended_pause(self):
        limiter = GeminiRateLimiter(
            min_interval_s=0.0,
            backoff_initial_s=1.0,
            pause_after_consecutive=3,
            pause_duration_s=100.0,
        )
        limiter.acquire()
        for _ in range(3):
            limiter.record_429()
        assert limiter.status["consecutive_429s"] == 3
        assert limiter.status["backoff_remaining_s"] > 50  # ~100s


class TestPauseResume:
    def test_pause_blocks_acquire(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0)
        limiter.pause()
        assert limiter.acquire() is False
        assert limiter.status["paused"] is True

    def test_resume_allows_acquire(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0)
        limiter.pause()
        limiter.resume()
        assert limiter.acquire() is True
        assert limiter.status["paused"] is False

    def test_resume_clears_backoff(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0, backoff_initial_s=60.0)
        limiter.acquire()
        limiter.record_429()
        limiter.resume()
        assert limiter.status["backoff_remaining_s"] == 0
        assert limiter.status["consecutive_429s"] == 0


class TestStatus:
    def test_status_keys(self):
        limiter = GeminiRateLimiter()
        s = limiter.status
        expected_keys = {
            "paused", "rate_limited", "backoff_remaining_s",
            "consecutive_429s", "total_calls", "total_429s", "min_interval_s",
        }
        assert set(s.keys()) == expected_keys

    def test_status_initial_values(self):
        limiter = GeminiRateLimiter(min_interval_s=5.0)
        s = limiter.status
        assert s["paused"] is False
        assert s["rate_limited"] is False
        assert s["total_calls"] == 0
        assert s["total_429s"] == 0
        assert s["min_interval_s"] == 5.0

    def test_is_limited_property(self):
        limiter = GeminiRateLimiter(min_interval_s=0.0, backoff_initial_s=10.0)
        assert limiter.is_limited is False
        limiter.acquire()
        limiter.record_429()
        assert limiter.is_limited is True
