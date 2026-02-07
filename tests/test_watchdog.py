"""Tests for the watchdog timer."""

import threading
import time

import pytest

from src.safety.watchdog import Watchdog


class TestWatchdog:
    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="positive"):
            Watchdog(timeout=0, callback=lambda: None)

        with pytest.raises(ValueError, match="positive"):
            Watchdog(timeout=-1, callback=lambda: None)

    def test_triggers_on_timeout(self):
        triggered = threading.Event()
        wd = Watchdog(timeout=0.05, callback=triggered.set)
        wd.start()
        try:
            assert triggered.wait(timeout=1.0), "Watchdog should have triggered"
            assert wd.is_triggered
        finally:
            wd.stop()

    def test_does_not_trigger_when_fed(self):
        triggered = threading.Event()
        wd = Watchdog(timeout=0.1, callback=triggered.set)
        wd.start()
        try:
            for _ in range(10):
                wd.feed()
                time.sleep(0.02)
            assert not triggered.is_set(), "Watchdog should not have triggered"
            assert not wd.is_triggered
        finally:
            wd.stop()

    def test_feed_resets_trigger(self):
        triggered = threading.Event()
        wd = Watchdog(timeout=0.05, callback=triggered.set)
        wd.start()
        try:
            assert triggered.wait(timeout=1.0)
            assert wd.is_triggered

            # Feed should reset
            wd.feed()
            assert not wd.is_triggered
        finally:
            wd.stop()

    def test_stop_is_idempotent(self):
        wd = Watchdog(timeout=1.0, callback=lambda: None)
        wd.start()
        wd.stop()
        wd.stop()  # should not raise

    def test_start_is_idempotent(self):
        wd = Watchdog(timeout=1.0, callback=lambda: None)
        wd.start()
        wd.start()  # should not raise
        wd.stop()

    def test_callback_exception_does_not_crash(self):
        def bad_callback():
            raise RuntimeError("oops")

        wd = Watchdog(timeout=0.05, callback=bad_callback)
        wd.start()
        try:
            time.sleep(0.2)  # let it trigger
            assert wd.is_triggered
        finally:
            wd.stop()
