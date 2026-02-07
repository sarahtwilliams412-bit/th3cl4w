"""
Real-time control loop for the D1 arm.

Runs at a fixed frequency (default 500Hz to match D1 hardware),
reading state, computing control, enforcing safety limits, and
sending commands.
"""

import logging
import threading
import time
from typing import Callable, Optional

from src.control.controller import Controller
from src.interface.d1_connection import D1Command, D1Connection, D1State
from src.safety.limits import D1SafetyLimits, clamp_command
from src.safety.watchdog import Watchdog

logger = logging.getLogger(__name__)


class ControlLoop:
    """Fixed-frequency control loop for the D1 arm.

    Integrates connection, controller, safety limits, and watchdog into
    a single run loop::

        conn = D1Connection()
        ctrl = JointPositionController()
        loop = ControlLoop(conn, ctrl)

        ctrl.set_target(target_positions)
        loop.start()
        # ... later ...
        loop.stop()
    """

    DEFAULT_FREQUENCY = 500.0  # Hz — matches D1 hardware

    def __init__(
        self,
        connection: D1Connection,
        controller: Controller,
        frequency: float = DEFAULT_FREQUENCY,
        safety_limits: Optional[D1SafetyLimits] = None,
        watchdog_timeout: float = 0.1,
        on_state: Optional[Callable[[D1State], None]] = None,
    ):
        """
        Args:
            connection: An already-connected D1Connection.
            controller: Controller that computes commands from state.
            frequency: Loop frequency in Hz.
            safety_limits: Safety limits applied to every command. Uses
                defaults if None.
            watchdog_timeout: Watchdog timeout in seconds. Set to 0 to
                disable the watchdog.
            on_state: Optional callback invoked with each new state reading.
        """
        self.connection = connection
        self.controller = controller
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.safety_limits = safety_limits or D1SafetyLimits()
        self.on_state = on_state

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._watchdog: Optional[Watchdog] = None
        self._watchdog_timeout = watchdog_timeout

        # Diagnostics
        self._cycle_count: int = 0
        self._last_state: Optional[D1State] = None
        self._max_jitter: float = 0.0

    def start(self) -> None:
        """Start the control loop in a background thread."""
        if self._running:
            logger.warning("Control loop already running")
            return

        if not self.connection.is_connected:
            raise RuntimeError("Connection is not established")

        self._running = True
        self._cycle_count = 0
        self._max_jitter = 0.0
        self._stop_event.clear()

        # Start watchdog
        if self._watchdog_timeout > 0:
            self._watchdog = Watchdog(
                timeout=self._watchdog_timeout,
                callback=self._emergency_stop,
                name="ControlLoopWatchdog",
            )
            self._watchdog.start()

        self._thread = threading.Thread(
            target=self._run, name="D1ControlLoop", daemon=True
        )
        self._thread.start()
        logger.info("Control loop started at %.0f Hz", self.frequency)

    def stop(self) -> None:
        """Stop the control loop. Sends an idle command before exiting."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if self._watchdog:
            self._watchdog.stop()
            self._watchdog = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        # Send a final idle command
        self.connection.send_command(D1Command(mode=0))
        logger.info(
            "Control loop stopped after %d cycles (max jitter: %.3fms)",
            self._cycle_count,
            self._max_jitter * 1000,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_state(self) -> Optional[D1State]:
        return self._last_state

    def _run(self) -> None:
        """Main loop body — runs in a background thread."""
        next_time = time.monotonic()

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()

            # Track jitter
            jitter = abs(cycle_start - next_time)
            if jitter > self._max_jitter:
                self._max_jitter = jitter

            try:
                # 1. Read state
                state = self.connection.get_state()
                if state is not None:
                    self._last_state = state
                    if self.on_state:
                        self.on_state(state)

                    # 2. Compute command
                    cmd = self.controller.compute(state)

                    # 3. Apply safety limits
                    cmd = clamp_command(cmd, self.safety_limits)

                    # 4. Send command
                    if self.connection.send_command(cmd):
                        if self._watchdog:
                            self._watchdog.feed()

                self._cycle_count += 1

            except Exception as e:
                logger.error("Control loop error: %s", e)
                # Send idle on error to be safe
                try:
                    self.connection.send_command(D1Command(mode=0))
                except Exception:
                    pass

            # Sleep until next cycle
            next_time += self.dt
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
            else:
                # We're behind schedule — log if significantly late
                if -sleep_time > self.dt:
                    logger.warning(
                        "Control loop overrun: %.3fms behind",
                        -sleep_time * 1000,
                    )
                next_time = time.monotonic()

    def _emergency_stop(self) -> None:
        """Watchdog callback — sends idle command."""
        logger.warning("Emergency stop triggered by watchdog")
        try:
            self.connection.send_command(D1Command(mode=0))
        except Exception as e:
            logger.error("Emergency stop failed: %s", e)
