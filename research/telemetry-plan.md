# Telemetry System Plan — th3cl4w

## Overview

Replace the in-memory ring buffer telemetry (`src/telemetry/collector.py`) with a SQLite-backed persistent telemetry system. The current `TelemetryCollector` holds ~1000 events in a `deque` and loses everything on restart. The new system writes to `data/telemetry.db` asynchronously with batch inserts, while preserving the existing `emit()` API so all current call sites (DDS connection, server, camera) continue working.

## File Layout

```
src/telemetry/
├── __init__.py              # re-export (update imports)
├── collector.py             # REPLACE — new TelemetryCollector with SQLite backend
├── camera_monitor.py        # KEEP as-is
├── query.py                 # NEW — TelemetryQuery class
├── schema.py                # NEW — schema definitions + migration
data/
├── telemetry.db             # created at runtime
tools/
├── query_telemetry.py       # NEW — CLI query tool
```

## SQLite Schema

```sql
-- schema.py will execute these on first run

CREATE TABLE IF NOT EXISTS dds_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,           -- time.time() epoch seconds
    ts_mono     REAL NOT NULL,           -- time.monotonic() for interval math
    seq         INTEGER,                 -- DDS sequence number
    funcode     INTEGER NOT NULL,        -- 1=set_joint, 2=set_all, 5=enable, 6=power, 7=reset
    joint_id    INTEGER,                 -- NULL for multi-joint/system commands
    target_value REAL,                   -- angle_deg or gripper_mm
    data_json   TEXT,                    -- full data dict as JSON (for set_all_joints etc)
    correlation_id TEXT,
    raw_len     INTEGER                  -- bytes published
);
CREATE INDEX IF NOT EXISTS idx_dds_commands_ts ON dds_commands(ts);
CREATE INDEX IF NOT EXISTS idx_dds_commands_cid ON dds_commands(correlation_id);
CREATE INDEX IF NOT EXISTS idx_dds_commands_funcode ON dds_commands(funcode);

CREATE TABLE IF NOT EXISTS dds_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    ts_mono     REAL NOT NULL,
    seq         INTEGER,
    funcode     INTEGER NOT NULL,        -- 1=angles, 3=status
    angle0      REAL, angle1 REAL, angle2 REAL, angle3 REAL,
    angle4      REAL, angle5 REAL, angle6 REAL,  -- angle6 = gripper
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
    joint_id    INTEGER NOT NULL,        -- 0-5 for joints, 6 for gripper
    target      REAL NOT NULL,
    current     REAL NOT NULL,
    sent        REAL NOT NULL,           -- what was actually sent to arm
    step_size   REAL,                    -- the interpolation step applied
    dirty       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_smoother_ts ON smoother_state(ts);
CREATE INDEX IF NOT EXISTS idx_smoother_joint ON smoother_state(joint_id, ts);

CREATE TABLE IF NOT EXISTS web_requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    endpoint    TEXT NOT NULL,            -- e.g. "/api/command/set-joint"
    method      TEXT NOT NULL DEFAULT 'POST',
    params_json TEXT,                     -- request parameters as JSON
    response_ms REAL,                    -- response time in ms
    status_code INTEGER,
    correlation_id TEXT,
    ok          INTEGER                  -- 1=success, 0=failure
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
    event_type  TEXT NOT NULL,           -- 'power_on','power_off','enable','disable','error','estop','connect','disconnect'
    source      TEXT NOT NULL,           -- 'web','dds','smoother','camera','system'
    detail      TEXT,                    -- human-readable detail
    data_json   TEXT,                    -- structured data as JSON
    correlation_id TEXT,
    level       TEXT DEFAULT 'info'      -- 'info','warning','error'
);
CREATE INDEX IF NOT EXISTS idx_system_events_ts ON system_events(ts);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type, ts);
```

## Class APIs

### `TelemetryCollector` (replaces `src/telemetry/collector.py`)

Keeps the existing `emit()` API signature for backward compat, but writes to SQLite via a background thread with batch inserts.

```python
class TelemetryCollector:
    def __init__(self, db_path: str = "data/telemetry.db", 
                 batch_size: int = 50, flush_interval_s: float = 1.0):
        """
        Args:
            db_path: Path to SQLite database file (created if missing).
            batch_size: Flush to disk after this many events queued.
            flush_interval_s: Max seconds between flushes even if batch not full.
        """

    def enable(self) -> None: ...
    def disable(self) -> None: ...
    @property
    def enabled(self) -> bool: ...
    
    @staticmethod
    def new_correlation_id() -> str: ...

    # -- Primary emit (backward-compat with existing call sites) --
    def emit(self, source: str, event_type: EventType, 
             payload: dict | None = None, 
             correlation_id: str | None = None) -> None:
        """
        Enqueue event for async write. Internally routes to the correct
        table based on event_type:
          - DDS_PUBLISH -> dds_commands
          - DDS_RECEIVE (funcode=1) -> dds_feedback (angles)
          - DDS_RECEIVE (funcode=3) -> dds_feedback (status)
          - CMD_SENT -> web_requests
          - CAM_FRAME -> camera_health
          - ERROR, STATE_UPDATE, CMD_ACK, CMD_EXEC -> system_events
        """

    # -- Typed insert methods (for new integration points) --
    def log_command(self, seq: int, funcode: int, joint_id: int | None,
                    target_value: float | None, data: dict | None,
                    correlation_id: str | None = None, raw_len: int = 0) -> None:
        """Insert into dds_commands."""

    def log_feedback(self, seq: int, funcode: int, angles: dict | None,
                     status: dict | None) -> None:
        """Insert into dds_feedback."""

    def log_smoother_tick(self, states: list[dict]) -> None:
        """Batch insert smoother state for all active joints.
        Each dict: {joint_id, target, current, sent, step_size, dirty}
        """

    def log_web_request(self, endpoint: str, method: str, params: dict | None,
                        response_ms: float, status_code: int, ok: bool,
                        correlation_id: str | None = None) -> None:
        """Insert into web_requests."""

    def log_camera_health(self, camera_id: str, stats: dict) -> None:
        """Insert into camera_health. stats from CameraHealthMonitor.stats"""

    def log_system_event(self, event_type: str, source: str, detail: str = "",
                         data: dict | None = None, correlation_id: str | None = None,
                         level: str = "info") -> None:
        """Insert into system_events."""

    # -- Lifecycle --
    def start(self) -> None:
        """Start the background writer thread. Called once at app startup."""

    def stop(self) -> None:
        """Flush remaining events and close the database."""

    # -- Legacy compat (used by server.py debug endpoints) --
    def get_events(self, limit: int = 100, event_type: EventType | None = None,
                   source: str | None = None) -> list[TelemetryEvent]: ...
    def get_pipeline(self, correlation_id: str) -> list[dict]: ...
    def get_stats(self) -> dict: ...
```

**Internal design:**
- A `queue.Queue` receives all events from any thread
- A single daemon `threading.Thread` drains the queue, batches by table, and does `executemany()` inserts
- WAL mode for concurrent reads during writes
- `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` for performance

### `TelemetryQuery` (`src/telemetry/query.py`)

Read-only query interface. All methods accept optional `time_range: tuple[float, float] | None` as `(start_epoch, end_epoch)`.

```python
class TelemetryQuery:
    def __init__(self, db_path: str = "data/telemetry.db"):
        """Opens a read-only connection."""

    # -- Command queries --
    def get_commands(self, time_range: tuple[float,float] | None = None,
                     funcode: int | None = None, limit: int = 1000) -> list[dict]:
        """Return dds_commands rows as dicts."""

    def get_command_rate(self, window_s: float = 10.0) -> dict:
        """Return {total_commands, rate_hz, by_funcode: {funcode: rate_hz}}
        for the last window_s seconds."""

    # -- Feedback / joint queries --
    def get_joint_history(self, joint: int, time_range: tuple[float,float] | None = None,
                          limit: int = 5000) -> list[dict]:
        """Return [{ts, angle}] for a specific joint from dds_feedback."""

    def get_all_joints_snapshot(self, time_range: tuple[float,float] | None = None,
                                 limit: int = 1000) -> list[dict]:
        """Return full angle snapshots from dds_feedback (funcode=1)."""

    def get_gripper_log(self, time_range: tuple[float,float] | None = None,
                        limit: int = 1000) -> list[dict]:
        """Return [{ts, position, source}] combining commands (funcode=1,joint=6) 
        and feedback (angle6)."""

    # -- Tracking error --
    def get_tracking_error(self, joint: int, time_range: tuple[float,float] | None = None,
                            delay_s: float = 0.2) -> dict:
        """Compare commanded vs actual position with a time delay.
        Returns {mean_error_deg, max_error_deg, samples, errors: [{ts, target, actual, error}]}"""

    # -- Smoother --
    def get_smoother_history(self, joint: int, time_range: tuple[float,float] | None = None,
                              limit: int = 2000) -> list[dict]:
        """Return [{ts, target, current, sent, step_size}] for a joint."""

    # -- Web --
    def get_web_requests(self, time_range: tuple[float,float] | None = None,
                          endpoint: str | None = None, limit: int = 500) -> list[dict]:
        """Return web request log."""

    def get_web_latency_stats(self, time_range: tuple[float,float] | None = None) -> dict:
        """Return {mean_ms, p50_ms, p95_ms, p99_ms, by_endpoint: {...}}"""

    # -- Camera --
    def get_camera_health(self, camera_id: str | None = None,
                           time_range: tuple[float,float] | None = None,
                           limit: int = 500) -> list[dict]:
        """Return camera health snapshots."""

    # -- System events --
    def get_events(self, event_type: str | None = None,
                   source: str | None = None,
                   level: str | None = None,
                   time_range: tuple[float,float] | None = None,
                   limit: int = 500) -> list[dict]:
        """Return system events."""

    # -- Utilities --
    def get_session_summary(self) -> dict:
        """Return {first_event_ts, last_event_ts, duration_s, 
        total_commands, total_feedback, total_events, 
        command_rate_hz, feedback_rate_hz, error_count}"""

    def get_db_stats(self) -> dict:
        """Return {table_name: row_count} and db file size."""
```

### `tools/query_telemetry.py` — CLI Tool

```
usage: query_telemetry.py [-h] [--db PATH] COMMAND [options]

commands:
  summary                          Session summary stats
  commands [--last N] [--funcode F] Recent commands
  joints [--joint J] [--last Ns]   Joint angle history
  gripper [--last Ns]              Gripper position log
  tracking [--joint J]             Tracking error analysis
  smoother [--joint J] [--last Ns] Smoother interpolation history
  requests [--endpoint E]          Web request log
  latency                          Web request latency stats
  cameras [--cam ID]               Camera health log
  events [--type T] [--level L]    System events
  rate                             Current command/feedback rates
  tail [--follow]                  Live tail of system events (polls every 1s)
  export [--format csv|json]       Export all tables
  dbstats                          Database size and row counts
```

---

## Execution Tasks

### Task 1: Collector & Storage (`src/telemetry/schema.py`, `src/telemetry/collector.py`)

**What to build:**
1. `src/telemetry/schema.py` — All CREATE TABLE/INDEX statements above. A function `init_db(db_path) -> sqlite3.Connection` that creates the database with WAL mode and all tables/indexes.
2. Rewrite `src/telemetry/collector.py` — New `TelemetryCollector` class per the API above. Key implementation details:
   - Use `queue.Queue` (thread-safe, unbounded) as the write buffer
   - Background `threading.Thread` (daemon=True) runs `_writer_loop()`:
     - `queue.get(timeout=flush_interval_s)` to batch events
     - Drain up to `batch_size` items, group by table
     - `executemany()` for each table's batch
     - Wrap in a single transaction
   - The `emit()` method routes events to typed insert methods based on `EventType`:
     - `DDS_PUBLISH` → `log_command()` (extract seq, funcode, data from payload)
     - `DDS_RECEIVE` → `log_feedback()` (extract angles or status from payload)
     - `CMD_SENT` → `log_web_request()` (partial — endpoint/params; response_ms filled later)
     - `CAM_FRAME` → `log_camera_health()`
     - `ERROR` → `log_system_event(event_type="error", ...)`
     - `STATE_UPDATE`, `CMD_ACK`, `CMD_EXEC` → `log_system_event()`
     - `WS_SEND` → skip (too noisy at 10Hz × N clients, not worth storing)
   - Keep the legacy `get_events()`, `get_pipeline()`, `get_stats()` methods working by querying SQLite instead of the in-memory deque. The debug API endpoints in `server.py` depend on these.
   - `EventType` enum: keep existing values, no changes needed
   - `data/` directory: create if not exists in `init_db()`
3. Update `src/telemetry/__init__.py` — Same exports, just ensure `get_collector()` calls `start()` automatically on first access.

**Backward compat requirements:**
- All existing `tc.emit("dds", EventType.DDS_PUBLISH, {...}, correlation_id)` calls in `d1_dds_connection.py` must work unchanged
- All existing `tc.emit("web", ...)` calls in `server.py` must work unchanged
- `get_events()`, `get_pipeline()`, `get_stats()` must return compatible data structures
- `TelemetryEvent` dataclass must still exist (returned by legacy methods)

### Task 2: Query & Analysis (`src/telemetry/query.py`, `tools/query_telemetry.py`)

**What to build:**
1. `src/telemetry/query.py` — `TelemetryQuery` class per the API above. Implementation notes:
   - Open connection as `sqlite3.connect(db_path, uri=True)` with `?mode=ro` for read-only
   - `row_factory = sqlite3.Row` for dict-like access
   - All time_range filters use `WHERE ts BETWEEN ? AND ?`
   - `get_tracking_error()`: JOIN dds_commands with dds_feedback where feedback.ts is closest to command.ts + delay_s. Use a subquery or window function.
   - `get_gripper_log()`: UNION of `dds_commands WHERE funcode=1 AND joint_id=6` and `dds_feedback WHERE funcode=1` (select angle6)
   - `get_web_latency_stats()`: Use SQLite percentile via sorted subquery (no numpy needed)
   - `get_command_rate()`: COUNT + GROUP BY funcode WHERE ts > now - window_s
2. `tools/query_telemetry.py` — CLI tool using `argparse` with subcommands. Each subcommand instantiates `TelemetryQuery` and formats output.
   - Default output: human-readable table (use simple column alignment, no external deps)
   - `--format json` option for machine-readable output
   - `tail --follow`: loop with `time.sleep(1)`, query events with `ts > last_seen_ts`
   - `export`: write CSV or JSON files per table to `data/export/`
   - Resolve db_path relative to project root (default `data/telemetry.db`)

### Task 3: Integration & Wiring

**What to wire up — changes to existing files:**

1. **`web/server.py`**:
   - In `lifespan()`: call `get_collector().start()` at startup, `get_collector().stop()` at shutdown
   - Add FastAPI middleware to log every `/api/` request to `log_web_request()` with timing:
     ```python
     @app.middleware("http")
     async def telemetry_middleware(request, call_next):
         if request.url.path.startswith("/api/"):
             t0 = time.monotonic()
             response = await call_next(request)
             elapsed_ms = (time.monotonic() - t0) * 1000
             tc.log_web_request(endpoint=request.url.path, method=request.method,
                                params=..., response_ms=elapsed_ms,
                                status_code=response.status_code, ok=response.status_code < 400)
             return response
         return await call_next(request)
     ```
   - Remove the per-endpoint `_telem_cmd_sent()` calls (middleware handles it)
   - Add new debug endpoint `GET /api/debug/query?table=...&limit=...` that proxies to `TelemetryQuery`

2. **`web/command_smoother.py`**:
   - Import telemetry collector
   - In `_tick()`, after computing interpolation, call `log_smoother_tick()` with current state for all dirty joints
   - Only log every Nth tick (e.g., every 5th = 2Hz) to avoid flooding (100ms ticks × 6 joints = too much)
   - Add a `_tick_counter % 5 == 0` guard

3. **`web/camera_server.py`**:
   - Every 5 seconds, snapshot each camera's `_health.stats` and call `log_camera_health()`
   - Add a simple timer in the capture loop or a separate periodic thread

4. **`src/interface/d1_dds_connection.py`**:
   - In `send_command()`: replace the `tc.emit("dds", EventType.DDS_PUBLISH, ...)` with `tc.log_command(seq, funcode, joint_id, target_value, data, correlation_id, raw_len)` for richer structured data
   - In `_process_feedback()`: replace `tc.emit("dds", EventType.DDS_RECEIVE, ...)` with `tc.log_feedback(seq, funcode, angles_dict, status_dict)` for structured storage
   - Keep `tc.emit()` calls for ERROR and STATE_UPDATE events (they go to system_events via emit routing)
   - In `connect()`/`disconnect()`: add `tc.log_system_event("connect"/"disconnect", "dds", ...)`

5. **Data retention** (add to collector):
   - On startup, delete rows older than 7 days: `DELETE FROM <table> WHERE ts < ?`
   - Configurable via `TELEMETRY_RETENTION_DAYS` env var (default 7)
