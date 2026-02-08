"""Read-only query interface for th3cl4w telemetry database."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


def _parse_time(t: float | str | None) -> float | None:
    """Convert ISO timestamp string or epoch float to epoch float."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    # ISO format
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return float(t)


def _time_range(
    start: float | str | None, end: float | str | None
) -> tuple[float | None, float | None]:
    return _parse_time(start), _parse_time(end)


def _add_time_filter(
    query: str, params: list, start: float | None, end: float | None, col: str = "ts"
) -> str:
    if start is not None:
        query += f" AND {col} >= ?"
        params.append(start)
    if end is not None:
        query += f" AND {col} <= ?"
        params.append(end)
    return query


class TelemetryQuery:
    """Read-only query interface for the telemetry SQLite database."""

    def __init__(self, db_path: str = "data/telemetry.db") -> None:
        p = Path(db_path)
        if not p.exists():
            raise FileNotFoundError(f"Telemetry database not found: {db_path}")
        uri = f"file:{p.resolve()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TelemetryQuery:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _fetchall(self, sql: str, params: tuple | list = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    # -- Joint history --
    def get_joint_history(
        self,
        joint: int,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 5000,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        col = f"angle{joint}"
        sql = f"SELECT ts, {col} AS angle FROM dds_feedback WHERE funcode = 1"
        params: list = []
        sql = _add_time_filter(sql, params, s, e)
        sql += f" ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Tracking error --
    def get_tracking_error(
        self,
        joint: int,
        start: float | str | None = None,
        end: float | str | None = None,
        delay_s: float = 0.2,
    ) -> dict:
        s, e = _time_range(start, end)
        # Get commands for this joint
        cmd_sql = "SELECT ts, target_value FROM dds_commands WHERE joint_id = ? AND target_value IS NOT NULL"
        cmd_params: list = [joint]
        cmd_sql = _add_time_filter(cmd_sql, cmd_params, s, e)
        cmd_sql += " ORDER BY ts"
        commands = self._fetchall(cmd_sql, cmd_params)

        col = f"angle{joint}"
        errors: list[dict] = []
        for cmd in commands:
            fb_ts = cmd["ts"] + delay_s
            fb = self._fetchall(
                f"SELECT ts, {col} AS angle FROM dds_feedback WHERE funcode = 1 "
                f"AND ts BETWEEN ? AND ? ORDER BY ABS(ts - ?) LIMIT 1",
                (fb_ts - 1.0, fb_ts + 1.0, fb_ts),
            )
            if fb and fb[0]["angle"] is not None:
                err = abs(cmd["target_value"] - fb[0]["angle"])
                errors.append(
                    {
                        "ts": cmd["ts"],
                        "target": cmd["target_value"],
                        "actual": fb[0]["angle"],
                        "error": err,
                    }
                )
        if not errors:
            return {"mean_error_deg": 0.0, "max_error_deg": 0.0, "samples": 0, "errors": []}
        errs = [e["error"] for e in errors]
        return {
            "mean_error_deg": sum(errs) / len(errs),
            "max_error_deg": max(errs),
            "samples": len(errors),
            "errors": errors,
        }

    # -- Command rate --
    def get_command_rate(self, window_seconds: float = 10.0) -> dict:
        cutoff = time.time() - window_seconds
        rows = self._fetchall(
            "SELECT funcode, COUNT(*) AS cnt FROM dds_commands WHERE ts >= ? GROUP BY funcode",
            (cutoff,),
        )
        total = sum(r["cnt"] for r in rows)
        by_funcode = {r["funcode"]: r["cnt"] / window_seconds for r in rows}
        return {
            "total_commands": total,
            "rate_hz": total / window_seconds,
            "by_funcode": by_funcode,
        }

    # -- Feedback rate --
    def get_feedback_rate(self, window_seconds: float = 10.0) -> dict:
        cutoff = time.time() - window_seconds
        rows = self._fetchall(
            "SELECT COUNT(*) AS cnt FROM dds_feedback WHERE ts >= ?",
            (cutoff,),
        )
        total = rows[0]["cnt"] if rows else 0
        return {"total_feedback": total, "rate_hz": total / window_seconds}

    # -- Gripper log --
    def get_gripper_log(
        self,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        # Commands targeting gripper (joint_id=6)
        cmd_sql = "SELECT ts, target_value AS position, 'command' AS source FROM dds_commands WHERE joint_id = 6"
        cmd_params: list = []
        cmd_sql = _add_time_filter(cmd_sql, cmd_params, s, e)
        # Feedback angle6
        fb_sql = "SELECT ts, angle6 AS position, 'feedback' AS source FROM dds_feedback WHERE funcode = 1 AND angle6 IS NOT NULL"
        fb_params: list = []
        fb_sql = _add_time_filter(fb_sql, fb_params, s, e)
        sql = f"{cmd_sql} UNION ALL {fb_sql} ORDER BY ts DESC LIMIT ?"
        params = cmd_params + fb_params + [limit]
        return self._fetchall(sql, params)

    # -- System events --
    def get_system_events(
        self,
        event_type: str | None = None,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        sql = "SELECT * FROM system_events WHERE 1=1"
        params: list = []
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql = _add_time_filter(sql, params, s, e)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Camera health --
    def get_camera_health(
        self,
        camera_id: str | None = None,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        sql = "SELECT * FROM camera_health WHERE 1=1"
        params: list = []
        if camera_id:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        sql = _add_time_filter(sql, params, s, e)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Web request latency --
    def get_web_request_latency(
        self,
        endpoint: str | None = None,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        sql = "SELECT * FROM web_requests WHERE 1=1"
        params: list = []
        if endpoint:
            sql += " AND endpoint = ?"
            params.append(endpoint)
        sql = _add_time_filter(sql, params, s, e)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Smoother state --
    def get_smoother_state(
        self,
        joint: int | None = None,
        start: float | str | None = None,
        end: float | str | None = None,
        limit: int = 2000,
    ) -> list[dict]:
        s, e = _time_range(start, end)
        sql = "SELECT * FROM smoother_state WHERE 1=1"
        params: list = []
        if joint is not None:
            sql += " AND joint_id = ?"
            params.append(joint)
        sql = _add_time_filter(sql, params, s, e)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Latest feedback --
    def get_latest_feedback(self) -> dict | None:
        rows = self._fetchall(
            "SELECT * FROM dds_feedback WHERE funcode = 1 ORDER BY ts DESC LIMIT 1"
        )
        return rows[0] if rows else None

    # -- Command count --
    def get_command_count(
        self,
        start: float | str | None = None,
        end: float | str | None = None,
    ) -> int:
        s, e = _time_range(start, end)
        sql = "SELECT COUNT(*) AS cnt FROM dds_commands WHERE 1=1"
        params: list = []
        sql = _add_time_filter(sql, params, s, e)
        rows = self._fetchall(sql, params)
        return rows[0]["cnt"] if rows else 0

    # -- Tail --
    def tail(self, limit: int = 50, event_types: list[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM system_events WHERE 1=1"
        params: list = []
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            sql += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._fetchall(sql, params)

    # -- Summary --
    def summary(
        self,
        start: float | str | None = None,
        end: float | str | None = None,
    ) -> dict:
        s, e = _time_range(start, end)

        def _count(table: str) -> int:
            sql = f"SELECT COUNT(*) AS cnt FROM {table} WHERE 1=1"
            p: list = []
            sql = _add_time_filter(sql, p, s, e)
            rows = self._fetchall(sql, p)
            return rows[0]["cnt"] if rows else 0

        def _ts_range(table: str) -> tuple[float | None, float | None]:
            sql = f"SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM {table} WHERE 1=1"
            p: list = []
            sql = _add_time_filter(sql, p, s, e)
            rows = self._fetchall(sql, p)
            if rows:
                return rows[0]["first_ts"], rows[0]["last_ts"]
            return None, None

        total_cmds = _count("dds_commands")
        total_fb = _count("dds_feedback")
        total_events = _count("system_events")
        error_count = 0
        esql = "SELECT COUNT(*) AS cnt FROM system_events WHERE level = 'error'"
        ep: list = []
        esql = _add_time_filter(esql, ep, s, e)
        erows = self._fetchall(esql, ep)
        if erows:
            error_count = erows[0]["cnt"]

        # Find overall time range across tables
        first_ts, last_ts = None, None
        for tbl in ("dds_commands", "dds_feedback", "system_events"):
            ft, lt = _ts_range(tbl)
            if ft is not None:
                first_ts = min(first_ts, ft) if first_ts is not None else ft
            if lt is not None:
                last_ts = max(last_ts, lt) if last_ts is not None else lt

        duration_s = (last_ts - first_ts) if (first_ts and last_ts) else 0.0
        cmd_rate = total_cmds / duration_s if duration_s > 0 else 0.0
        fb_rate = total_fb / duration_s if duration_s > 0 else 0.0

        return {
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
            "duration_s": duration_s,
            "total_commands": total_cmds,
            "total_feedback": total_fb,
            "total_events": total_events,
            "command_rate_hz": round(cmd_rate, 2),
            "feedback_rate_hz": round(fb_rate, 2),
            "error_count": error_count,
        }

    def get_db_stats(self) -> dict:
        tables = [
            "dds_commands",
            "dds_feedback",
            "smoother_state",
            "web_requests",
            "camera_health",
            "system_events",
        ]
        stats: dict[str, int] = {}
        for t in tables:
            try:
                rows = self._fetchall(f"SELECT COUNT(*) AS cnt FROM {t}")
                stats[t] = rows[0]["cnt"]
            except sqlite3.OperationalError:
                stats[t] = 0
        return stats
