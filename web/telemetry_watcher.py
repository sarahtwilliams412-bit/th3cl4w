"""Telemetry Watcher — monitors arm telemetry with user feedback.

Separate FastAPI app on port 8085. Connects to the existing telemetry
SQLite database, accepts binary good/bad feedback from the user, and
correlates telemetry data with feedback windows so you can see what
the arm was doing when things went wrong vs. when things were fine.

Endpoints:
    POST /api/feedback              — submit binary feedback (good/bad)
    GET  /api/feedback/history      — all feedback entries
    GET  /api/feedback/sessions     — feedback grouped into sessions
    GET  /api/snapshot/{id}         — telemetry snapshot for a feedback entry
    GET  /api/analysis              — comparative analysis (good vs bad periods)
    GET  /api/status                — service health
    WS   /ws/watcher                — real-time feedback + analysis updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("th3cl4w.telemetry_watcher")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEMETRY_DB = os.environ.get("TELEMETRY_DB", "data/telemetry.db")
WATCHER_DB = os.environ.get("WATCHER_DB", "data/telemetry_watcher.db")
SNAPSHOT_WINDOW_S = float(os.environ.get("SNAPSHOT_WINDOW_S", "5.0"))
PORT = int(os.environ.get("WATCHER_PORT", "8085"))

# ---------------------------------------------------------------------------
# Watcher database (feedback + snapshots)
# ---------------------------------------------------------------------------

_WATCHER_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    rating      TEXT NOT NULL CHECK(rating IN ('good', 'bad')),
    note        TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback(ts);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id     INTEGER NOT NULL REFERENCES feedback(id),
    window_start    REAL NOT NULL,
    window_end      REAL NOT NULL,
    dds_commands    TEXT,
    dds_feedback    TEXT,
    smoother_state  TEXT,
    system_events   TEXT,
    web_requests    TEXT,
    camera_health   TEXT,
    summary_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_fid ON snapshots(feedback_id);
"""


def _init_watcher_db(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_WATCHER_SCHEMA)
    conn.close()


def _watcher_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _telemetry_conn(path: str) -> sqlite3.Connection:
    """Read-only connection to the main telemetry database."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry database not found: {path}")
    uri = f"file:{p.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


# ---------------------------------------------------------------------------
# Snapshot capture — grabs telemetry around a feedback timestamp
# ---------------------------------------------------------------------------


def _capture_snapshot(telemetry_db: str, ts: float, window_s: float) -> dict:
    """Pull all telemetry data from [ts - window_s, ts] from the telemetry DB."""
    t_start = ts - window_s
    t_end = ts

    try:
        conn = _telemetry_conn(telemetry_db)
    except FileNotFoundError:
        return {"error": "telemetry database not found", "window_start": t_start, "window_end": t_end}

    def _fetch(sql: str, params: tuple = ()) -> list[dict]:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    try:
        dds_cmds = _fetch(
            "SELECT ts, seq, funcode, joint_id, target_value, data_json, correlation_id "
            "FROM dds_commands WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )

        dds_fb = _fetch(
            "SELECT ts, seq, funcode, angle0, angle1, angle2, angle3, angle4, angle5, angle6, "
            "power_status, enable_status, error_status "
            "FROM dds_feedback WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )

        smoother = _fetch(
            "SELECT ts, joint_id, target, current, sent, step_size, dirty "
            "FROM smoother_state WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )

        sys_events = _fetch(
            "SELECT ts, event_type, source, detail, data_json, level "
            "FROM system_events WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )

        web_reqs = _fetch(
            "SELECT ts, endpoint, method, response_ms, status_code, ok "
            "FROM web_requests WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )

        cam_health = _fetch(
            "SELECT ts, camera_id, actual_fps, target_fps, drop_count, motion_score, stalled "
            "FROM camera_health WHERE ts BETWEEN ? AND ? ORDER BY ts",
            (t_start, t_end),
        )
    finally:
        conn.close()

    # Build summary metrics for this window
    error_events = [e for e in sys_events if e.get("level") == "error"]
    error_count = len(error_events)

    # Tracking error: for each smoother row, |target - current|
    tracking_errors = []
    for s in smoother:
        if s["target"] is not None and s["current"] is not None:
            tracking_errors.append(abs(s["target"] - s["current"]))
    avg_tracking_error = sum(tracking_errors) / len(tracking_errors) if tracking_errors else 0.0
    max_tracking_error = max(tracking_errors) if tracking_errors else 0.0

    # Feedback rate
    fb_count = len(dds_fb)
    fb_rate = fb_count / window_s if window_s > 0 else 0.0

    # Command rate
    cmd_count = len(dds_cmds)
    cmd_rate = cmd_count / window_s if window_s > 0 else 0.0

    # DDS error statuses
    dds_errors = [f for f in dds_fb if f.get("error_status") and f["error_status"] != 0]

    # Web request failures
    web_failures = [r for r in web_reqs if not r.get("ok")]

    # Camera stalls
    cam_stalls = [c for c in cam_health if c.get("stalled")]

    summary = {
        "window_start": t_start,
        "window_end": t_end,
        "window_duration_s": window_s,
        "dds_command_count": cmd_count,
        "dds_command_rate_hz": round(cmd_rate, 2),
        "dds_feedback_count": fb_count,
        "dds_feedback_rate_hz": round(fb_rate, 2),
        "avg_tracking_error": round(avg_tracking_error, 6),
        "max_tracking_error": round(max_tracking_error, 6),
        "error_event_count": error_count,
        "dds_error_count": len(dds_errors),
        "web_failure_count": len(web_failures),
        "camera_stall_count": len(cam_stalls),
        "smoother_sample_count": len(smoother),
        "system_event_count": len(sys_events),
    }

    return {
        "window_start": t_start,
        "window_end": t_end,
        "dds_commands": dds_cmds,
        "dds_feedback": dds_fb,
        "smoother_state": smoother,
        "system_events": sys_events,
        "web_requests": web_reqs,
        "camera_health": cam_health,
        "summary": summary,
    }


def _save_snapshot(watcher_db: str, feedback_id: int, snapshot: dict) -> int:
    """Persist snapshot to the watcher database. Returns snapshot id."""
    conn = _watcher_conn(watcher_db)
    try:
        cur = conn.execute(
            "INSERT INTO snapshots "
            "(feedback_id, window_start, window_end, dds_commands, dds_feedback, "
            "smoother_state, system_events, web_requests, camera_health, summary_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                feedback_id,
                snapshot["window_start"],
                snapshot["window_end"],
                json.dumps(snapshot.get("dds_commands", [])),
                json.dumps(snapshot.get("dds_feedback", [])),
                json.dumps(snapshot.get("smoother_state", [])),
                json.dumps(snapshot.get("system_events", [])),
                json.dumps(snapshot.get("web_requests", [])),
                json.dumps(snapshot.get("camera_health", [])),
                json.dumps(snapshot.get("summary", {})),
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analysis — compare good vs bad telemetry
# ---------------------------------------------------------------------------


def _compute_analysis(watcher_db: str) -> dict:
    """Aggregate and compare metrics across good and bad feedback windows."""
    conn = _watcher_conn(watcher_db)
    try:
        rows = conn.execute(
            "SELECT f.rating, s.summary_json "
            "FROM feedback f JOIN snapshots s ON s.feedback_id = f.id "
            "ORDER BY f.ts"
        ).fetchall()
    finally:
        conn.close()

    good_summaries: list[dict] = []
    bad_summaries: list[dict] = []

    for row in rows:
        try:
            summary = json.loads(row["summary_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if row["rating"] == "good":
            good_summaries.append(summary)
        else:
            bad_summaries.append(summary)

    def _avg(items: list[dict], key: str) -> float:
        vals = [s[key] for s in items if key in s and s[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _aggregate(items: list[dict]) -> dict:
        return {
            "count": len(items),
            "avg_tracking_error": _avg(items, "avg_tracking_error"),
            "avg_max_tracking_error": _avg(items, "max_tracking_error"),
            "avg_dds_command_rate_hz": _avg(items, "dds_command_rate_hz"),
            "avg_dds_feedback_rate_hz": _avg(items, "dds_feedback_rate_hz"),
            "avg_error_event_count": _avg(items, "error_event_count"),
            "avg_dds_error_count": _avg(items, "dds_error_count"),
            "avg_web_failure_count": _avg(items, "web_failure_count"),
            "avg_camera_stall_count": _avg(items, "camera_stall_count"),
        }

    good_agg = _aggregate(good_summaries)
    bad_agg = _aggregate(bad_summaries)

    # Identify likely problem indicators — metrics that are notably worse in bad windows
    indicators: list[dict] = []
    compare_keys = [
        ("avg_tracking_error", "higher", "Tracking error (avg)"),
        ("avg_max_tracking_error", "higher", "Tracking error (max)"),
        ("avg_error_event_count", "higher", "Error events"),
        ("avg_dds_error_count", "higher", "DDS errors"),
        ("avg_web_failure_count", "higher", "Web request failures"),
        ("avg_camera_stall_count", "higher", "Camera stalls"),
        ("avg_dds_feedback_rate_hz", "lower", "DDS feedback rate"),
        ("avg_dds_command_rate_hz", "lower", "DDS command rate"),
    ]

    for key, direction, label in compare_keys:
        good_val = good_agg.get(key, 0)
        bad_val = bad_agg.get(key, 0)
        if good_agg["count"] == 0 or bad_agg["count"] == 0:
            continue

        if direction == "higher" and bad_val > good_val and good_val >= 0:
            ratio = bad_val / good_val if good_val > 0 else float("inf")
            if ratio > 1.5 or (good_val == 0 and bad_val > 0):
                indicators.append({
                    "metric": label,
                    "good_avg": good_val,
                    "bad_avg": bad_val,
                    "ratio": round(ratio, 2) if ratio != float("inf") else "inf",
                    "direction": "higher_when_bad",
                    "severity": "high" if ratio > 3 else "medium",
                })
        elif direction == "lower" and bad_val < good_val and good_val > 0:
            ratio = good_val / bad_val if bad_val > 0 else float("inf")
            if ratio > 1.5:
                indicators.append({
                    "metric": label,
                    "good_avg": good_val,
                    "bad_avg": bad_val,
                    "ratio": round(ratio, 2) if ratio != float("inf") else "inf",
                    "direction": "lower_when_bad",
                    "severity": "high" if ratio > 3 else "medium",
                })

    # Sort by severity
    severity_order = {"high": 0, "medium": 1}
    indicators.sort(key=lambda x: severity_order.get(x["severity"], 2))

    return {
        "total_feedback": good_agg["count"] + bad_agg["count"],
        "good": good_agg,
        "bad": bad_agg,
        "problem_indicators": indicators,
    }


# ---------------------------------------------------------------------------
# WebSocket clients
# ---------------------------------------------------------------------------

ws_clients: set[WebSocket] = set()


async def _broadcast(data: dict) -> None:
    """Send data to all connected WebSocket clients."""
    if not ws_clients:
        return
    text = json.dumps(data)
    dead: set[WebSocket] = set()
    for ws in ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_watcher_db(WATCHER_DB)
    logger.info("Telemetry Watcher started — watching %s, storing in %s", TELEMETRY_DB, WATCHER_DB)
    yield
    logger.info("Telemetry Watcher shutting down")


app = FastAPI(title="th3cl4w Telemetry Watcher", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/feedback")
async def submit_feedback(rating: str = "bad", note: str = ""):
    """Submit binary feedback: 'good' or 'bad'."""
    if rating not in ("good", "bad"):
        return JSONResponse({"error": "rating must be 'good' or 'bad'"}, status_code=400)

    ts = time.time()

    # Record feedback
    conn = _watcher_conn(WATCHER_DB)
    try:
        cur = conn.execute(
            "INSERT INTO feedback (ts, rating, note) VALUES (?, ?, ?)",
            (ts, rating, note),
        )
        conn.commit()
        feedback_id = cur.lastrowid
    finally:
        conn.close()

    # Capture telemetry snapshot for the window leading up to this feedback
    snapshot = _capture_snapshot(TELEMETRY_DB, ts, SNAPSHOT_WINDOW_S)
    snapshot_id = _save_snapshot(WATCHER_DB, feedback_id, snapshot)

    result = {
        "feedback_id": feedback_id,
        "snapshot_id": snapshot_id,
        "ts": ts,
        "rating": rating,
        "note": note,
        "summary": snapshot.get("summary", {}),
    }

    # Broadcast to websocket clients
    await _broadcast({"type": "feedback", "data": result})

    logger.info("Feedback recorded: %s (id=%d, snapshot=%d)", rating, feedback_id, snapshot_id)
    return JSONResponse(result)


@app.get("/api/feedback/history")
async def feedback_history(limit: int = 100, rating: str | None = None):
    """Get feedback history with summaries."""
    conn = _watcher_conn(WATCHER_DB)
    try:
        if rating and rating in ("good", "bad"):
            rows = conn.execute(
                "SELECT f.id, f.ts, f.rating, f.note, s.summary_json "
                "FROM feedback f LEFT JOIN snapshots s ON s.feedback_id = f.id "
                "WHERE f.rating = ? "
                "ORDER BY f.ts DESC LIMIT ?",
                (rating, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT f.id, f.ts, f.rating, f.note, s.summary_json "
                "FROM feedback f LEFT JOIN snapshots s ON s.feedback_id = f.id "
                "ORDER BY f.ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    result = []
    for row in rows:
        entry = {
            "id": row["id"],
            "ts": row["ts"],
            "rating": row["rating"],
            "note": row["note"],
            "summary": None,
        }
        if row["summary_json"]:
            try:
                entry["summary"] = json.loads(row["summary_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(entry)

    return JSONResponse(result)


@app.get("/api/feedback/sessions")
async def feedback_sessions():
    """Group feedback into sessions (gaps > 60s separate sessions)."""
    conn = _watcher_conn(WATCHER_DB)
    try:
        rows = conn.execute(
            "SELECT f.id, f.ts, f.rating, f.note, s.summary_json "
            "FROM feedback f LEFT JOIN snapshots s ON s.feedback_id = f.id "
            "ORDER BY f.ts"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return JSONResponse([])

    sessions: list[dict] = []
    current_session: dict = {"start_ts": rows[0]["ts"], "entries": [], "good": 0, "bad": 0}

    for row in rows:
        entry = {
            "id": row["id"],
            "ts": row["ts"],
            "rating": row["rating"],
            "note": row["note"],
        }
        if row["summary_json"]:
            try:
                entry["summary"] = json.loads(row["summary_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # New session if gap > 60s
        if current_session["entries"] and row["ts"] - current_session["entries"][-1]["ts"] > 60:
            current_session["end_ts"] = current_session["entries"][-1]["ts"]
            sessions.append(current_session)
            current_session = {"start_ts": row["ts"], "entries": [], "good": 0, "bad": 0}

        current_session["entries"].append(entry)
        current_session[row["rating"]] += 1

    # Close final session
    if current_session["entries"]:
        current_session["end_ts"] = current_session["entries"][-1]["ts"]
        sessions.append(current_session)

    return JSONResponse(sessions)


@app.get("/api/snapshot/{feedback_id}")
async def get_snapshot(feedback_id: int):
    """Get the full telemetry snapshot for a feedback entry."""
    conn = _watcher_conn(WATCHER_DB)
    try:
        row = conn.execute(
            "SELECT * FROM snapshots WHERE feedback_id = ?", (feedback_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return JSONResponse({"error": "snapshot not found"}, status_code=404)

    result = {
        "id": row["id"],
        "feedback_id": row["feedback_id"],
        "window_start": row["window_start"],
        "window_end": row["window_end"],
    }

    for col in ("dds_commands", "dds_feedback", "smoother_state", "system_events", "web_requests", "camera_health", "summary_json"):
        try:
            result[col] = json.loads(row[col]) if row[col] else []
        except (json.JSONDecodeError, TypeError):
            result[col] = []

    return JSONResponse(result)


@app.get("/api/analysis")
async def get_analysis():
    """Compare telemetry metrics across good vs bad feedback periods."""
    analysis = _compute_analysis(WATCHER_DB)
    return JSONResponse(analysis)


@app.get("/api/status")
async def get_status():
    """Service health and counts."""
    conn = _watcher_conn(WATCHER_DB)
    try:
        fb_count = conn.execute("SELECT COUNT(*) AS cnt FROM feedback").fetchone()["cnt"]
        good_count = conn.execute("SELECT COUNT(*) AS cnt FROM feedback WHERE rating='good'").fetchone()["cnt"]
        bad_count = conn.execute("SELECT COUNT(*) AS cnt FROM feedback WHERE rating='bad'").fetchone()["cnt"]
        snap_count = conn.execute("SELECT COUNT(*) AS cnt FROM snapshots").fetchone()["cnt"]
    finally:
        conn.close()

    telemetry_available = Path(TELEMETRY_DB).exists()

    return JSONResponse({
        "status": "ok",
        "telemetry_db": TELEMETRY_DB,
        "telemetry_available": telemetry_available,
        "watcher_db": WATCHER_DB,
        "total_feedback": fb_count,
        "good_count": good_count,
        "bad_count": bad_count,
        "snapshot_count": snap_count,
        "snapshot_window_s": SNAPSHOT_WINDOW_S,
        "ws_clients": len(ws_clients),
    })


@app.websocket("/ws/watcher")
async def websocket_watcher(ws: WebSocket):
    """Real-time updates for feedback and analysis."""
    await ws.accept()
    ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(ws_clients))
    try:
        while True:
            # Keep alive — clients can also send ping
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(ws_clients))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="th3cl4w Telemetry Watcher")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--telemetry-db", default=TELEMETRY_DB)
    parser.add_argument("--watcher-db", default=WATCHER_DB)
    parser.add_argument("--window", type=float, default=SNAPSHOT_WINDOW_S,
                        help="Seconds of telemetry to capture before each feedback event")
    args = parser.parse_args()

    TELEMETRY_DB = args.telemetry_db
    WATCHER_DB = args.watcher_db
    SNAPSHOT_WINDOW_S = args.window

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
