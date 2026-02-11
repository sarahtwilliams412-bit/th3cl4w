"""Sim Telemetry Bridge — polls SimulatedArm and emits telemetry events.

When the server runs in simulation mode the real DDS feedback loop doesn't
exist.  This bridge polls the SimulatedArm at 10 Hz and emits identical
telemetry events so that analysis tools, the telemetry watcher, and the UI
telemetry page all work identically in sim and physical mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SimTelemetryBridge"]


class SimTelemetryBridge:
    """Bridges SimulatedArm state into the telemetry system."""

    def __init__(self, sim_arm: Any, collector: Any | None = None) -> None:
        self._arm = sim_arm
        self._collector = collector
        self._task: asyncio.Task | None = None
        self._running = False
        self._event_count = 0
        self._seq = 0
        self._last_angles: list[float] | None = None

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._loop())
        logger.info("SimTelemetryBridge started")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("SimTelemetryBridge stopped")

    # -- main loop --------------------------------------------------------

    async def _loop(self) -> None:
        try:
            while self._running:
                self._emit_feedback()
                await asyncio.sleep(0.1)  # 10 Hz
        except asyncio.CancelledError:
            pass

    # -- emit -------------------------------------------------------------

    def _emit_feedback(self) -> None:
        """Poll arm state and emit telemetry events."""
        angles_arr = self._arm.get_joint_angles()
        if angles_arr is None:
            return

        angles_list = [float(a) for a in angles_arr]
        # Arm joints only (first 6), exclude gripper
        arm_angles = angles_list[:6] if len(angles_list) > 6 else angles_list

        angles_dict = {f"angle{i}": v for i, v in enumerate(arm_angles)}

        self._seq += 1
        seq = self._seq

        # --- funcode 1: joint angles ---
        self._emit_event(
            "dds_receive",
            {
                "seq": seq,
                "funcode": 1,
                "angles": angles_dict,
                "status": None,
                "sim": True,
            },
        )

        # --- funcode 3: status ---
        with self._arm._lock:
            power = self._arm._powered
            enable = self._arm._enabled
            error = self._arm._error

        status_dict = {
            "power_status": int(power),
            "enable_status": int(enable),
            "error_status": error,
            "recv_status": 0,
            "exec_status": 0,
        }

        self._emit_event(
            "dds_receive",
            {
                "seq": seq,
                "funcode": 3,
                "angles": None,
                "status": status_dict,
                "sim": True,
            },
        )

        # --- state_update on angle change ---
        if self._last_angles is not None:
            changed = any(abs(a - b) > 0.05 for a, b in zip(arm_angles, self._last_angles))
        else:
            changed = True  # first reading

        if changed:
            self._emit_event(
                "state_update",
                {
                    "angles": angles_dict,
                    "sim": True,
                },
            )

        self._last_angles = list(arm_angles)

    def _emit_event(self, event_type_str: str, payload: dict) -> None:
        self._event_count += 1
        if self._collector is None:
            return
        try:
            from src.telemetry.collector import EventType

            et = EventType(event_type_str)
            self._collector.emit(source="sim_bridge", event_type=et, payload=payload)
        except Exception:
            logger.debug("SimTelemetryBridge: failed to emit %s", event_type_str, exc_info=True)

    # -- stats ------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "event_count": self._event_count,
            "seq": self._seq,
        }
