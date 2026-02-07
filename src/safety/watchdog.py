"""
Watchdog timer for D1 arm safety.

Monitors the time since the last command was sent and triggers an
emergency idle command if the timeout is exceeded.  This prevents
the arm from continuing to execute a stale command if the control
program hangs or crashes.
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Watchdog:
    """Watchdog timer that calls a callback if not fed within a timeout.

    Typical usage with D1Connection::

        def emergency_stop():
            conn.send_command(D1Command(mode=0))

        wd = Watchdog(timeout=0.05, callback=emergency_stop)  # 50ms
        wd.start()

        # In your control loop:
        conn.send_command(cmd)
        wd.feed()

        # When done:
        wd.stop()
    """

    def __init__(
        self,
        timeout: float,
        callback: Callable[[], None],
        name: str = "D1Watchdog",
    ):
        if timeout <= 0:
            raise ValueError(f"Watchdog timeout must be positive, got {timeout}")
        self.timeout = timeout
        self._callback = callback
        self._name = name
        self._last_feed: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._triggered = False

    def start(self):
        """Start the watchdog timer."""
        if self._running:
            return
        self._running = True
        self._triggered = False
        self._last_feed = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("Watchdog started (timeout=%.3fs)", self.timeout)

    def stop(self):
        """Stop the watchdog timer."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("Watchdog stopped")

    def feed(self):
        """Reset the watchdog timer. Call this after each successful command."""
        self._last_feed = time.monotonic()
        self._triggered = False

    @property
    def is_triggered(self) -> bool:
        """True if the watchdog fired since the last feed."""
        return self._triggered

    def _run(self):
        """Background thread that checks the timer."""
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._last_feed
            if elapsed >= self.timeout and not self._triggered:
                self._triggered = True
                logger.warning(
                    "Watchdog triggered — no command for %.3fs (limit %.3fs)",
                    elapsed,
                    self.timeout,
                )
                try:
                    self._callback()
                except Exception as e:
                    logger.error("Watchdog callback failed: %s", e)
            # Check at 4x the timeout frequency for responsiveness
            self._stop_event.wait(timeout=self.timeout / 4.0)
