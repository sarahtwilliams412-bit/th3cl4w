"""Tests for TelemetryQuery and CLI tool."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.telemetry.query import TelemetryQuery

# Schema from the plan
SCHEMA = """
CREATE TABLE IF NOT EXISTS dds_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, ts_mono REAL NOT NULL,
    seq INTEGER, funcode INTEGER NOT NULL, joint_id INTEGER, target_value REAL,
    data_json TEXT, correlation_id TEXT, raw_len INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dds_commands_ts ON dds_commands(ts);
CREATE INDEX IF NOT EXISTS idx_dds_commands_funcode ON dds_commands(funcode);

CREATE TABLE IF NOT EXISTS dds_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, ts_mono REAL NOT NULL,
    seq INTEGER, funcode INTEGER NOT NULL,
    angle0 REAL, angle1 REAL, angle2 REAL, angle3 REAL,
    angle4 REAL, angle5 REAL, angle6 REAL,
    power_status INTEGER, enable_status INTEGER, error_status INTEGER,
    recv_status INTEGER, exec_status INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dds_feedback_ts ON dds_feedback(ts);

CREATE TABLE IF NOT EXISTS smoother_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    joint_id INTEGER NOT NULL, target REAL NOT NULL, current REAL NOT NULL,
    sent REAL NOT NULL, step_size REAL, dirty INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_smoother_joint ON smoother_state(joint_id, ts);

CREATE TABLE IF NOT EXISTS web_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    endpoint TEXT NOT NULL, method TEXT NOT NULL DEFAULT 'POST',
    params_json TEXT, response_ms REAL, status_code INTEGER,
    correlation_id TEXT, ok INTEGER
);
CREATE INDEX IF NOT EXISTS idx_web_requests_ts ON web_requests(ts);
CREATE INDEX IF NOT EXISTS idx_web_requests_endpoint ON web_requests(endpoint, ts);

CREATE TABLE IF NOT EXISTS camera_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    camera_id TEXT NOT NULL, actual_fps REAL, target_fps REAL,
    drop_count INTEGER, motion_score REAL, connected INTEGER,
    resolution_w INTEGER, resolution_h INTEGER, stalled INTEGER
);
CREATE INDEX IF NOT EXISTS idx_camera_health_ts ON camera_health(ts);
CREATE INDEX IF NOT EXISTS idx_camera_health_cam ON camera_health(camera_id, ts);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    event_type TEXT NOT NULL, source TEXT NOT NULL,
    detail TEXT, data_json TEXT, correlation_id TEXT,
    level TEXT DEFAULT 'info'
);
CREATE INDEX IF NOT EXISTS idx_system_events_ts ON system_events(ts);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type, ts);
"""

NOW = time.time()


def _create_test_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    # Insert test data
    for i in range(10):
        t = NOW - 5.0 + i * 0.5
        conn.execute(
            "INSERT INTO dds_commands (ts, ts_mono, seq, funcode, joint_id, target_value) VALUES (?,?,?,?,?,?)",
            (t, t, i, 1, 0, 10.0 + i),
        )
        conn.execute(
            "INSERT INTO dds_feedback (ts, ts_mono, seq, funcode, angle0, angle1, angle2, angle3, angle4, angle5, angle6) "
            "VALUES (?,?,?,1,?,?,?,?,?,?,?)",
            (t + 0.2, t + 0.2, i, 10.0 + i + 0.1, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0 + i),
        )

    # Gripper commands
    conn.execute(
        "INSERT INTO dds_commands (ts, ts_mono, seq, funcode, joint_id, target_value) VALUES (?,?,?,1,6,50.0)",
        (NOW, NOW, 100),
    )

    # Smoother state
    for i in range(5):
        conn.execute(
            "INSERT INTO smoother_state (ts, joint_id, target, current, sent, step_size, dirty) VALUES (?,?,?,?,?,?,?)",
            (NOW - 2 + i * 0.5, 0, 15.0, 14.0 + i * 0.2, 14.0 + i * 0.2, 0.2, 1),
        )

    # Web requests
    for i in range(5):
        conn.execute(
            "INSERT INTO web_requests (ts, endpoint, method, response_ms, status_code, ok) VALUES (?,?,?,?,?,?)",
            (NOW - 2 + i * 0.4, "/api/command/set-joint", "POST", 5.0 + i, 200, 1),
        )

    # Camera health
    conn.execute(
        "INSERT INTO camera_health (ts, camera_id, actual_fps, target_fps, drop_count, connected, stalled) VALUES (?,?,?,?,?,?,?)",
        (NOW, "cam0", 29.5, 30.0, 2, 1, 0),
    )

    # System events
    for i, (etype, level) in enumerate(
        [
            ("power_on", "info"),
            ("enable", "info"),
            ("error", "error"),
            ("connect", "info"),
            ("disconnect", "warning"),
        ]
    ):
        conn.execute(
            "INSERT INTO system_events (ts, event_type, source, detail, level) VALUES (?,?,?,?,?)",
            (NOW - 4 + i, etype, "dds", f"Test event {i}", level),
        )

    conn.commit()
    conn.close()


class TestTelemetryQuery(unittest.TestCase):
    db_path: str
    q: TelemetryQuery

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls._tmpdir.name, "test.db")
        _create_test_db(cls.db_path)
        cls.q = TelemetryQuery(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.q.close()
        cls._tmpdir.cleanup()

    def test_get_joint_history(self) -> None:
        rows = self.q.get_joint_history(0)
        self.assertGreater(len(rows), 0)
        self.assertIn("ts", rows[0])
        self.assertIn("angle", rows[0])

    def test_get_joint_history_with_time_range(self) -> None:
        rows = self.q.get_joint_history(0, start=NOW - 3, end=NOW)
        for r in rows:
            self.assertGreaterEqual(r["ts"], NOW - 3)
            self.assertLessEqual(r["ts"], NOW)

    def test_get_tracking_error(self) -> None:
        result = self.q.get_tracking_error(0)
        self.assertIn("mean_error_deg", result)
        self.assertIn("max_error_deg", result)
        self.assertIn("samples", result)
        self.assertGreater(result["samples"], 0)
        # Our test data has ~0.1 degree offset
        self.assertAlmostEqual(result["mean_error_deg"], 0.1, places=0)

    def test_get_command_rate(self) -> None:
        result = self.q.get_command_rate(window_seconds=60.0)
        self.assertIn("total_commands", result)
        self.assertIn("rate_hz", result)
        self.assertGreater(result["total_commands"], 0)

    def test_get_feedback_rate(self) -> None:
        result = self.q.get_feedback_rate(window_seconds=60.0)
        self.assertGreater(result["total_feedback"], 0)

    def test_get_gripper_log(self) -> None:
        rows = self.q.get_gripper_log()
        self.assertGreater(len(rows), 0)
        sources = {r["source"] for r in rows}
        self.assertTrue(sources & {"command", "feedback"})

    def test_get_system_events(self) -> None:
        rows = self.q.get_system_events()
        self.assertEqual(len(rows), 5)

    def test_get_system_events_filtered(self) -> None:
        rows = self.q.get_system_events(event_type="error")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["level"], "error")

    def test_get_camera_health(self) -> None:
        rows = self.q.get_camera_health(camera_id="cam0")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["actual_fps"], 29.5)

    def test_get_web_request_latency(self) -> None:
        rows = self.q.get_web_request_latency()
        self.assertEqual(len(rows), 5)

    def test_get_web_request_latency_filtered(self) -> None:
        rows = self.q.get_web_request_latency(endpoint="/api/command/set-joint")
        self.assertEqual(len(rows), 5)

    def test_get_smoother_state(self) -> None:
        rows = self.q.get_smoother_state(joint=0)
        self.assertEqual(len(rows), 5)

    def test_get_latest_feedback(self) -> None:
        fb = self.q.get_latest_feedback()
        self.assertIsNotNone(fb)
        self.assertIn("angle0", fb)

    def test_get_command_count(self) -> None:
        count = self.q.get_command_count()
        self.assertEqual(count, 11)  # 10 joint + 1 gripper

    def test_get_command_count_time_range(self) -> None:
        count = self.q.get_command_count(start=NOW - 2, end=NOW + 1)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 11)

    def test_tail(self) -> None:
        rows = self.q.tail(limit=3)
        self.assertEqual(len(rows), 3)

    def test_tail_with_event_types(self) -> None:
        rows = self.q.tail(limit=50, event_types=["error"])
        self.assertEqual(len(rows), 1)

    def test_summary(self) -> None:
        s = self.q.summary()
        self.assertGreater(s["total_commands"], 0)
        self.assertGreater(s["total_feedback"], 0)
        self.assertGreater(s["duration_s"], 0)
        self.assertEqual(s["error_count"], 1)

    def test_get_db_stats(self) -> None:
        stats = self.q.get_db_stats()
        self.assertEqual(stats["dds_commands"], 11)
        self.assertEqual(stats["dds_feedback"], 10)
        self.assertEqual(stats["system_events"], 5)

    def test_iso_timestamp_parsing(self) -> None:
        rows = self.q.get_system_events(start="2020-01-01T00:00:00", end="2099-01-01T00:00:00")
        self.assertEqual(len(rows), 5)

    def test_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            TelemetryQuery("/nonexistent/path.db")

    def test_context_manager(self) -> None:
        with TelemetryQuery(self.db_path) as q:
            self.assertIsNotNone(q.get_db_stats())


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls._tmpdir.name, "test.db")
        _create_test_db(cls.db_path)
        cls.cli = str(_project_root / "tools" / "query_telemetry.py")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, self.cli, "--db", self.db_path, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_summary(self) -> None:
        r = self._run("summary")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Session Summary", r.stdout)
        self.assertIn("Commands:", r.stdout)

    def test_tail(self) -> None:
        r = self._run("tail", "-n", "3")
        self.assertEqual(r.returncode, 0)
        self.assertIn("error", r.stdout.lower())

    def test_joints(self) -> None:
        r = self._run("joints", "-j", "0", "-n", "5")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Angle", r.stdout)

    def test_events(self) -> None:
        r = self._run("events", "-t", "error")
        self.assertEqual(r.returncode, 0)

    def test_export_json(self) -> None:
        r = self._run("export")
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertIn("summary", data)

    def test_rate(self) -> None:
        r = self._run("rate", "-w", "600")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Hz", r.stdout)

    def test_cameras(self) -> None:
        r = self._run("cameras")
        self.assertEqual(r.returncode, 0)

    def test_latency(self) -> None:
        r = self._run("latency")
        self.assertEqual(r.returncode, 0)

    def test_smoother(self) -> None:
        r = self._run("smoother", "-j", "0")
        self.assertEqual(r.returncode, 0)

    def test_gripper(self) -> None:
        r = self._run("gripper")
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
