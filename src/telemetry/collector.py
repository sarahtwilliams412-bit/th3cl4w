"""Central telemetry event bus for th3cl4w — SQLite-backed."""

from __future__ import annotations

import collections
import enum
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# EventType enum (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Legacy TelemetryEvent dataclass (kept for backward compat)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    timestamp_ms: float
    wall_time_ms: float
    source: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS dds_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    ts_mono     REAL NOT NULL,
    seq         INTEGER,
    funcode     INTEGER NOT NULL,
    joint_id    INTEGER,
    target_value REAL,
    data_json   TEXT,
    correlation_id TEXT,
    raw_len     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dds_commands_ts ON dds_commands(ts);
CREATE INDEX IF NOT EXISTS idx_dds_commands_cid ON dds_commands(correlation_id);
CREATE INDEX IF NOT EXISTS idx_dds_commands_funcode ON dds_commands(funcode);

CREATE TABLE IF NOT EXISTS dds_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    ts_mono     REAL NOT NULL,
    seq         INTEGER,
    funcode     INTEGER NOT NULL,
    angle0      REAL, angle1 REAL, angle2 REAL, angle3 REAL,
    angle4      REAL, angle5 REAL, angle6 REAL,
    power_status  INTEGER,
    enable_status INTEGER,
    error_status  INTEGER,
    recv_status   INTEGER,
    exec_status   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dds_feedback_ts ON dds_feedback(ts);
CREATE INDEX IF NOT EXISTS idx_dds_feedback_funcode ON dds_feedback(funcode);

CREATE TABLE IF NOT EXISTS smoother_state (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    joint_id    INTEGER NOT NULL,
    target      REAL NOT NULL,
    current     REAL NOT NULL,
    sent        REAL NOT NULL,
    step_size   REAL,
    dirty       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_smoother_ts ON smoother_state(ts);
CREATE INDEX IF NOT EXISTS idx_smoother_joint ON smoother_state(joint_id, ts);

CREATE TABLE IF NOT EXISTS web_requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    endpoint    TEXT NOT NULL,
    method      TEXT NOT NULL DEFAULT 'POST',
    params_json TEXT,
    response_ms REAL,
    status_code INTEGER,
    correlation_id TEXT,
    ok          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_web_requests_ts ON web_requests(ts);
CREATE INDEX IF NOT EXISTS idx_web_requests_endpoint ON web_requests(endpoint, ts);

CREATE TABLE IF NOT EXISTS camera_health (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    camera_id   TEXT NOT NULL,
    actual_fps  REAL,
    target_fps  REAL,
    drop_count  INTEGER,
    motion_score REAL,
    connected   INTEGER,
    resolution_w INTEGER,
    resolution_h INTEGER,
    stalled     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_camera_health_ts ON camera_health(ts);
CREATE INDEX IF NOT EXISTS idx_camera_health_cam ON camera_health(camera_id, ts);

CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    event_type  TEXT NOT NULL,
    source      TEXT NOT NULL,
    detail      TEXT,
    data_json   TEXT,
    correlation_id TEXT,
    level       TEXT DEFAULT 'info'
);
CREATE INDEX IF NOT EXISTS idx_system_events_ts ON system_events(ts);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type, ts);
"""

# ---------------------------------------------------------------------------
# Internal queue record types
# ---------------------------------------------------------------------------

_TABLE_DDS_COMMANDS = "dds_commands"
_TABLE_DDS_FEEDBACK = "dds_feedback"
_TABLE_SMOOTHER = "smoother_state"
_TABLE_WEB = "web_requests"
_TABLE_CAMERA = "camera_health"
_TABLE_SYSTEM = "system_events"

_INSERT_SQL: dict[str, str] = {
    _TABLE_DDS_COMMANDS: (
        "INSERT INTO dds_commands (ts, ts_mono, seq, funcode, joint_id, target_value, data_json, correlation_id, raw_len) "
        "VALUES (?,?,?,?,?,?,?,?,?)"
    ),
    _TABLE_DDS_FEEDBACK: (
        "INSERT INTO dds_feedback (ts, ts_mono, seq, funcode, angle0, angle1, angle2, angle3, angle4, angle5, angle6, "
        "power_status, enable_status, error_status, recv_status, exec_status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    ),
    _TABLE_SMOOTHER: (
        "INSERT INTO smoother_state (ts, joint_id, target, current, sent, step_size, dirty) "
        "VALUES (?,?,?,?,?,?,?)"
    ),
    _TABLE_WEB: (
        "INSERT INTO web_requests (ts, endpoint, method, params_json, response_ms, status_code, correlation_id, ok) "
        "VALUES (?,?,?,?,?,?,?,?)"
    ),
    _TABLE_CAMERA: (
        "INSERT INTO camera_health (ts, camera_id, actual_fps, target_fps, drop_count, motion_score, connected, resolution_w, resolution_h, stalled) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)"
    ),
    _TABLE_SYSTEM: (
        "INSERT INTO system_events (ts, event_type, source, detail, data_json, correlation_id, level) "
        "VALUES (?,?,?,?,?,?,?)"
    ),
}

_RATE_WINDOW_S = 10.0

# Sentinel to signal writer thread to stop
_STOP = object()


class TelemetryCollector:
    """Thread-safe telemetry collector with SQLite backend and async batch writes."""

    def __init__(
        self,
        db_path: str = "data/telemetry.db",
        batch_size: int = 50,
        flush_interval_s: float = 0.1,
        maxlen: int = 1000,
    ) -> None:
        self._db_path = db_path
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._enabled = False
        self._queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()

        # Legacy rate tracking (for get_stats compat)
        self._rate_lock = threading.Lock()
        self._rate_timestamps: dict[EventType, list[float]] = {
            et: [] for et in EventType
        }

        # Subscribers for real-time streaming (list of callables)
        self._subscribers: list[Any] = []
        self._sub_lock = threading.Lock()

        # Legacy in-memory event buffer (populated by emit() for backward compat)
        self._events: collections.deque[TelemetryEvent] = collections.deque(maxlen=maxlen)
        self._events_lock = threading.Lock()

        # Init DB
        self._init_db()

    # -- DB init ---------------------------------------------------------

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.close()

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # -- Enable / disable ------------------------------------------------

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- Lifecycle -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="telemetry-writer"
            )
            self._writer_thread.start()

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        self._queue.put(_STOP)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

    # -- Background writer -----------------------------------------------

    def _writer_loop(self) -> None:
        conn = self._make_conn()
        try:
            while True:
                batch: list[tuple[str, tuple]] = []
                # Block waiting for first item
                try:
                    item = self._queue.get(timeout=self._flush_interval_s)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                batch.append(item)
                # Drain more
                while len(batch) < self._batch_size:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _STOP:
                        self._flush_batch(conn, batch)
                        return
                    batch.append(item)
                self._flush_batch(conn, batch)
        finally:
            # Drain remaining
            remaining: list[tuple[str, tuple]] = []
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    continue
                remaining.append(item)
            if remaining:
                self._flush_batch(conn, remaining)
            conn.close()

    def _flush_batch(self, conn: sqlite3.Connection, batch: list[tuple[str, tuple]]) -> None:
        if not batch:
            return
        # Group by table
        by_table: dict[str, list[tuple]] = {}
        for table, params in batch:
            by_table.setdefault(table, []).append(params)
        with conn:
            for table, rows in by_table.items():
                sql = _INSERT_SQL[table]
                conn.executemany(sql, rows)

    # -- Enqueue helpers -------------------------------------------------

    def _enqueue(self, table: str, params: tuple) -> None:
        self._queue.put((table, params))

    # -- Typed log methods -----------------------------------------------

    @staticmethod
    def new_correlation_id() -> str:
        return uuid.uuid4().hex[:12]

    def log_dds_command(
        self,
        seq: int | None = None,
        funcode: int = 0,
        joint_id: int | None = None,
        target_value: float | None = None,
        data: dict | None = None,
        correlation_id: str | None = None,
        raw_len: int = 0,
        _from_emit: bool = False,
    ) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        data_json = json.dumps(data) if data is not None else None
        self._enqueue(_TABLE_DDS_COMMANDS, (
            ts, ts_mono, seq, funcode, joint_id, target_value, data_json, correlation_id, raw_len
        ))
        if not _from_emit:
            self._notify_subscribers(TelemetryEvent(
                timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
                source="dds", event_type=EventType.DDS_PUBLISH,
                payload={"seq": seq, "funcode": funcode, "joint_id": joint_id, "target_value": target_value, "data": data, "raw_len": raw_len},
                correlation_id=correlation_id,
            ))

    def log_dds_feedback(
        self,
        seq: int | None = None,
        funcode: int = 0,
        angles: dict | None = None,
        status: dict | None = None,
        _from_emit: bool = False,
    ) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        a = angles or {}
        s = status or {}
        self._enqueue(_TABLE_DDS_FEEDBACK, (
            ts, ts_mono, seq, funcode,
            a.get("angle0"), a.get("angle1"), a.get("angle2"), a.get("angle3"),
            a.get("angle4"), a.get("angle5"), a.get("angle6"),
            s.get("power_status"), s.get("enable_status"), s.get("error_status"),
            s.get("recv_status"), s.get("exec_status"),
        ))
        if not _from_emit:
            self._notify_subscribers(TelemetryEvent(
                timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
                source="dds", event_type=EventType.DDS_RECEIVE,
                payload={"seq": seq, "funcode": funcode, "angles": angles, "status": status},
            ))

    def log_smoother_state(self, states: list[dict]) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        for s in states:
            self._enqueue(_TABLE_SMOOTHER, (
                ts, s["joint_id"], s["target"], s["current"], s["sent"],
                s.get("step_size"), s.get("dirty", 1),
            ))
        self._notify_subscribers(TelemetryEvent(
            timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
            source="smoother", event_type=EventType.STATE_UPDATE,
            payload={"joints": states},
        ))

    def log_web_request(
        self,
        endpoint: str = "",
        method: str = "POST",
        params: dict | None = None,
        response_ms: float = 0.0,
        status_code: int = 200,
        ok: bool = True,
        correlation_id: str | None = None,
        _from_emit: bool = False,
    ) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        params_json = json.dumps(params) if params is not None else None
        self._enqueue(_TABLE_WEB, (
            ts, endpoint, method, params_json, response_ms, status_code, correlation_id, 1 if ok else 0
        ))
        if not _from_emit:
            self._notify_subscribers(TelemetryEvent(
                timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
                source="web", event_type=EventType.CMD_SENT,
                payload={"endpoint": endpoint, "method": method, "response_ms": response_ms, "status_code": status_code, "ok": ok},
                correlation_id=correlation_id,
            ))

    def log_camera_health(
        self,
        camera_id: str = "",
        stats: dict | None = None,
    ) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        s = stats or {}
        self._enqueue(_TABLE_CAMERA, (
            ts, camera_id,
            s.get("actual_fps"), s.get("target_fps"), s.get("drop_count"),
            s.get("motion_score"), s.get("connected"), s.get("resolution_w"),
            s.get("resolution_h"), s.get("stalled"),
        ))
        self._notify_subscribers(TelemetryEvent(
            timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
            source="camera", event_type=EventType.CAM_FRAME,
            payload={"camera_id": camera_id, **s},
        ))

    def log_system_event(
        self,
        event_type: str = "",
        source: str = "",
        detail: str = "",
        data: dict | None = None,
        correlation_id: str | None = None,
        level: str = "info",
        _from_emit: bool = False,
    ) -> None:
        ts = time.time()
        ts_mono = time.monotonic()
        data_json = json.dumps(data) if data is not None else None
        self._enqueue(_TABLE_SYSTEM, (
            ts, event_type, source, detail, data_json, correlation_id, level
        ))
        if not _from_emit:
            evt_type = EventType.ERROR if level == "error" else EventType.STATE_UPDATE
            self._notify_subscribers(TelemetryEvent(
                timestamp_ms=ts_mono * 1000, wall_time_ms=ts * 1000,
                source=source, event_type=evt_type,
                payload={"system_event_type": event_type, "detail": detail, "data": data, "level": level},
                correlation_id=correlation_id,
            ))

    # -- Subscribers for real-time streaming ------------------------------

    def subscribe(self, callback: Any) -> None:
        """Register a callback(event_dict) for real-time streaming."""
        with self._sub_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Any) -> None:
        """Remove a subscriber callback."""
        with self._sub_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify_subscribers(self, event: TelemetryEvent) -> None:
        """Notify all subscribers with a dict version of the event."""
        with self._sub_lock:
            subs = list(self._subscribers)
        if not subs:
            return
        event_dict = {
            "timestamp_ms": event.timestamp_ms,
            "wall_time_ms": event.wall_time_ms,
            "source": event.source,
            "event_type": event.event_type.value,
            "payload": event.payload,
            "correlation_id": event.correlation_id,
        }
        for cb in subs:
            try:
                cb(event_dict)
            except Exception:
                pass

    # -- Backward-compatible emit ----------------------------------------

    def emit(
        self,
        source: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> TelemetryEvent | None:
        if not self._enabled:
            return None

        now_mono = time.monotonic_ns() / 1_000_000
        now_wall = time.time() * 1000
        p = payload or {}

        event = TelemetryEvent(
            timestamp_ms=now_mono,
            wall_time_ms=now_wall,
            source=source,
            event_type=event_type,
            payload=p,
            correlation_id=correlation_id,
        )

        # Store in legacy buffer
        with self._events_lock:
            self._events.append(event)

        # Rate tracking
        with self._rate_lock:
            self._rate_timestamps[event_type].append(now_mono)

        # Notify subscribers
        self._notify_subscribers(event)

        # Route to typed methods (with _from_emit=True to avoid double notification)
        if event_type == EventType.DDS_PUBLISH:
            self.log_dds_command(
                seq=p.get("seq"),
                funcode=p.get("funcode", 0),
                joint_id=p.get("joint_id"),
                target_value=p.get("target_value"),
                data=p.get("data"),
                correlation_id=correlation_id,
                raw_len=p.get("raw_len", 0),
                _from_emit=True,
            )
        elif event_type == EventType.DDS_RECEIVE:
            self.log_dds_feedback(
                seq=p.get("seq"),
                funcode=p.get("funcode", 0),
                angles=p.get("angles"),
                status=p.get("status"),
                _from_emit=True,
            )
        elif event_type == EventType.CMD_SENT:
            self.log_web_request(
                endpoint=p.get("endpoint", ""),
                method=p.get("method", "POST"),
                params=p.get("params"),
                response_ms=p.get("response_ms", 0.0),
                status_code=p.get("status_code", 200),
                ok=p.get("ok", True),
                correlation_id=correlation_id,
                _from_emit=True,
            )
        elif event_type == EventType.CAM_FRAME:
            self.log_camera_health(
                camera_id=p.get("camera_id", ""),
                stats=p,
            )
        elif event_type == EventType.WS_SEND:
            pass  # Skip — too noisy
        else:
            # ERROR, STATE_UPDATE, CMD_ACK, CMD_EXEC, WS_RECEIVE
            self.log_system_event(
                event_type=event_type.value,
                source=source,
                detail=p.get("detail", ""),
                data=p if p else None,
                correlation_id=correlation_id,
                level="error" if event_type == EventType.ERROR else "info",
                _from_emit=True,
            )

        return event

    # -- Legacy query methods (backward compat) --------------------------

    def get_events(
        self,
        limit: int = 100,
        event_type: EventType | None = None,
        source: str | None = None,
    ) -> list[TelemetryEvent]:
        """Return events from the in-memory buffer (populated by emit())."""
        with self._events_lock:
            events = list(self._events)
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if source is not None:
            events = [e for e in events if e.source == source]
        return events[-limit:]

    def get_pipeline(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return events matching a correlation ID from the in-memory buffer."""
        with self._events_lock:
            matched = [e for e in self._events if e.correlation_id == correlation_id]
        matched.sort(key=lambda e: e.timestamp_ms)
        result = []
        for i, ev in enumerate(matched):
            result.append({
                "event": {
                    "ts": ev.timestamp_ms,
                    "type": ev.event_type.value,
                    "source": ev.source,
                },
                "latency_ms": (ev.timestamp_ms - matched[i - 1].timestamp_ms) if i > 0 else 0.0,
            })
        return result

    def get_stats(self) -> dict[str, Any]:
        now_mono = time.monotonic_ns() / 1_000_000
        cutoff = now_mono - _RATE_WINDOW_S * 1000

        rates: dict[str, float] = {}
        with self._rate_lock:
            for et, ts_list in self._rate_timestamps.items():
                # prune
                self._rate_timestamps[et] = [t for t in ts_list if t >= cutoff]
                rates[et.value] = len(self._rate_timestamps[et]) / _RATE_WINDOW_S

        # Total from in-memory buffer (immediate, no flush needed)
        with self._events_lock:
            total = len(self._events)

        return {
            "total_events": total,
            "rates_per_sec": rates,
            "staleness_ms": None,
            "avg_pipeline_latency_ms": 0.0,
            "active_correlations": 0,
        }


# -- Singleton -----------------------------------------------------------

_collector: TelemetryCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> TelemetryCollector:
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TelemetryCollector()
                _collector.start()
    return _collector
