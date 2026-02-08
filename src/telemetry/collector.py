"""Central telemetry event bus for th3cl4w."""

from __future__ import annotations

import enum
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class EventType(enum.Enum):
    CMD_SENT = "cmd_sent"
    DDS_PUBLISH = "dds_publish"
    DDS_RECEIVE = "dds_receive"
    CMD_ACK = "cmd_ack"
    CMD_EXEC = "cmd_exec"
    STATE_UPDATE = "state_update"
    WS_SEND = "ws_send"
    WS_RECEIVE = "ws_receive"
    CAM_FRAME = "cam_frame"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    timestamp_ms: float
    wall_time_ms: float
    source: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None


_RATE_WINDOW_S = 10.0
_MAX_CORRELATIONS = 500
_DEFAULT_BUFFER_SIZE = 1000


class TelemetryCollector:
    """Thread-safe telemetry collector with ring buffer, rate tracking, and correlation."""

    def __init__(self, maxlen: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._lock = threading.Lock()
        self._events: deque[TelemetryEvent] = deque(maxlen=maxlen)
        self._enabled = False
        # rate tracking: event_type -> deque of monotonic timestamps
        self._rate_timestamps: dict[EventType, deque[float]] = {
            et: deque() for et in EventType
        }
        # correlation map: correlation_id -> list of events (bounded)
        self._correlations: dict[str, list[TelemetryEvent]] = {}
        self._correlation_order: deque[str] = deque()

    # -- enable / disable ------------------------------------------------

    def enable(self) -> None:
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    # -- emit ------------------------------------------------------------

    @staticmethod
    def new_correlation_id() -> str:
        return uuid.uuid4().hex[:12]

    def emit(
        self,
        source: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> TelemetryEvent | None:
        with self._lock:
            if not self._enabled:
                return None

            now_mono = time.monotonic_ns() / 1_000_000
            now_wall = time.time() * 1000

            event = TelemetryEvent(
                timestamp_ms=now_mono,
                wall_time_ms=now_wall,
                source=source,
                event_type=event_type,
                payload=payload or {},
                correlation_id=correlation_id,
            )

            self._events.append(event)

            # rate tracking
            ts_deque = self._rate_timestamps[event_type]
            ts_deque.append(now_mono)
            # prune old entries
            cutoff = now_mono - _RATE_WINDOW_S * 1000
            while ts_deque and ts_deque[0] < cutoff:
                ts_deque.popleft()

            # correlation tracking
            if correlation_id is not None:
                if correlation_id not in self._correlations:
                    # evict oldest if at capacity
                    if len(self._correlations) >= _MAX_CORRELATIONS:
                        oldest = self._correlation_order.popleft()
                        self._correlations.pop(oldest, None)
                    self._correlations[correlation_id] = []
                    self._correlation_order.append(correlation_id)
                self._correlations[correlation_id].append(event)

            return event

    # -- queries ---------------------------------------------------------

    def get_events(
        self,
        limit: int = 100,
        event_type: EventType | None = None,
        source: str | None = None,
    ) -> list[TelemetryEvent]:
        with self._lock:
            result: list[TelemetryEvent] = []
            for ev in reversed(self._events):
                if event_type is not None and ev.event_type != event_type:
                    continue
                if source is not None and ev.source != source:
                    continue
                result.append(ev)
                if len(result) >= limit:
                    break
            result.reverse()
            return result

    def get_pipeline(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return events for a correlation_id with inter-event latencies."""
        with self._lock:
            events = self._correlations.get(correlation_id, [])
            out: list[dict[str, Any]] = []
            for i, ev in enumerate(events):
                entry: dict[str, Any] = {
                    "event": ev,
                    "latency_ms": (
                        ev.timestamp_ms - events[i - 1].timestamp_ms if i > 0 else 0.0
                    ),
                }
                out.append(entry)
            return out

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            now_mono = time.monotonic_ns() / 1_000_000
            cutoff = now_mono - _RATE_WINDOW_S * 1000

            rates: dict[str, float] = {}
            for et, ts_deque in self._rate_timestamps.items():
                # prune
                while ts_deque and ts_deque[0] < cutoff:
                    ts_deque.popleft()
                rates[et.value] = len(ts_deque) / _RATE_WINDOW_S

            # staleness: time since last event
            staleness_ms: float | None = None
            if self._events:
                staleness_ms = now_mono - self._events[-1].timestamp_ms

            # avg latencies per correlation
            latencies: list[float] = []
            for evts in self._correlations.values():
                if len(evts) >= 2:
                    latencies.append(evts[-1].timestamp_ms - evts[0].timestamp_ms)
            avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

            return {
                "total_events": len(self._events),
                "rates_per_sec": rates,
                "staleness_ms": staleness_ms,
                "avg_pipeline_latency_ms": avg_latency_ms,
                "active_correlations": len(self._correlations),
            }


# -- singleton -----------------------------------------------------------

_collector: TelemetryCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> TelemetryCollector:
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TelemetryCollector()
    return _collector
