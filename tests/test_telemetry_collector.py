"""Tests for the SQLite-backed TelemetryCollector."""

import os
import sqlite3
import tempfile
import threading
import time
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from telemetry.collector import TelemetryCollector, EventType, TelemetryEvent


class TestTelemetryCollectorSchema(unittest.TestCase):
    """Test that schema is created correctly on init."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        self.tc = TelemetryCollector(db_path=self.db_path)

    def tearDown(self):
        self.tc.close()

    def test_tables_exist(self):
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()
        expected = {"dds_commands", "dds_feedback", "smoother_state", "web_requests", "camera_health", "system_events"}
        self.assertEqual(expected, tables)

    def test_indexes_exist(self):
        conn = sqlite3.connect(self.db_path)
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()}
        conn.close()
        self.assertIn("idx_dds_commands_ts", indexes)
        self.assertIn("idx_dds_feedback_ts", indexes)
        self.assertIn("idx_smoother_ts", indexes)
        self.assertIn("idx_web_requests_ts", indexes)
        self.assertIn("idx_camera_health_ts", indexes)
        self.assertIn("idx_system_events_ts", indexes)

    def test_wal_mode(self):
        conn = sqlite3.connect(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal")

    def test_data_dir_auto_created(self):
        nested = os.path.join(self.tmp, "sub", "dir", "test.db")
        tc2 = TelemetryCollector(db_path=nested)
        self.assertTrue(os.path.exists(os.path.dirname(nested)))
        tc2.close()


class TestLogMethods(unittest.TestCase):
    """Test each typed log method writes to the correct table."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        self.tc = TelemetryCollector(db_path=self.db_path, flush_interval_s=0.05)
        self.tc.start()

    def tearDown(self):
        self.tc.close()

    def _wait_flush(self):
        time.sleep(0.2)

    def _count(self, table):
        conn = sqlite3.connect(self.db_path)
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return n

    def test_log_dds_command(self):
        self.tc.log_dds_command(seq=1, funcode=1, joint_id=0, target_value=45.0, data={"test": 1}, correlation_id="abc", raw_len=32)
        self._wait_flush()
        self.assertEqual(self._count("dds_commands"), 1)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM dds_commands").fetchone()
        conn.close()
        self.assertEqual(row["funcode"], 1)
        self.assertEqual(row["joint_id"], 0)
        self.assertAlmostEqual(row["target_value"], 45.0)
        self.assertEqual(row["correlation_id"], "abc")

    def test_log_dds_feedback(self):
        self.tc.log_dds_feedback(seq=5, funcode=1, angles={"angle0": 10.0, "angle1": 20.0})
        self._wait_flush()
        self.assertEqual(self._count("dds_feedback"), 1)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM dds_feedback").fetchone()
        conn.close()
        self.assertAlmostEqual(row["angle0"], 10.0)
        self.assertAlmostEqual(row["angle1"], 20.0)

    def test_log_smoother_state(self):
        self.tc.log_smoother_state([
            {"joint_id": 0, "target": 10.0, "current": 8.0, "sent": 9.0, "step_size": 1.0, "dirty": 1},
            {"joint_id": 1, "target": 20.0, "current": 18.0, "sent": 19.0},
        ])
        self._wait_flush()
        self.assertEqual(self._count("smoother_state"), 2)

    def test_log_web_request(self):
        self.tc.log_web_request(endpoint="/api/test", method="GET", params={"a": 1}, response_ms=12.5, status_code=200, ok=True)
        self._wait_flush()
        self.assertEqual(self._count("web_requests"), 1)

    def test_log_camera_health(self):
        self.tc.log_camera_health(camera_id="cam0", stats={"actual_fps": 29.5, "connected": 1})
        self._wait_flush()
        self.assertEqual(self._count("camera_health"), 1)

    def test_log_system_event(self):
        self.tc.log_system_event(event_type="power_on", source="dds", detail="Arm powered on", level="info")
        self._wait_flush()
        self.assertEqual(self._count("system_events"), 1)


class TestEmitRouting(unittest.TestCase):
    """Test that emit() routes to the correct table."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        self.tc = TelemetryCollector(db_path=self.db_path, flush_interval_s=0.05)
        self.tc.start()
        self.tc.enable()

    def tearDown(self):
        self.tc.close()

    def _wait_flush(self):
        time.sleep(0.2)

    def _count(self, table):
        conn = sqlite3.connect(self.db_path)
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return n

    def test_emit_dds_publish(self):
        self.tc.emit("dds", EventType.DDS_PUBLISH, {"funcode": 1, "seq": 1, "joint_id": 0, "target_value": 30.0})
        self._wait_flush()
        self.assertEqual(self._count("dds_commands"), 1)

    def test_emit_dds_receive(self):
        self.tc.emit("dds", EventType.DDS_RECEIVE, {"funcode": 1, "angles": {"angle0": 5.0}})
        self._wait_flush()
        self.assertEqual(self._count("dds_feedback"), 1)

    def test_emit_cmd_sent(self):
        self.tc.emit("web", EventType.CMD_SENT, {"endpoint": "/api/cmd", "status_code": 200})
        self._wait_flush()
        self.assertEqual(self._count("web_requests"), 1)

    def test_emit_cam_frame(self):
        self.tc.emit("camera", EventType.CAM_FRAME, {"camera_id": "cam0", "actual_fps": 30})
        self._wait_flush()
        self.assertEqual(self._count("camera_health"), 1)

    def test_emit_error(self):
        self.tc.emit("dds", EventType.ERROR, {"detail": "Connection lost"})
        self._wait_flush()
        self.assertEqual(self._count("system_events"), 1)

    def test_emit_state_update(self):
        self.tc.emit("system", EventType.STATE_UPDATE, {"detail": "enabled"})
        self._wait_flush()
        self.assertEqual(self._count("system_events"), 1)

    def test_emit_ws_send_skipped(self):
        self.tc.emit("web", EventType.WS_SEND, {"data": "test"})
        self._wait_flush()
        # WS_SEND is skipped
        self.assertEqual(self._count("system_events"), 0)

    def test_emit_disabled_returns_none(self):
        self.tc.disable()
        result = self.tc.emit("dds", EventType.DDS_PUBLISH, {"funcode": 1})
        self.assertIsNone(result)

    def test_emit_returns_event(self):
        result = self.tc.emit("dds", EventType.DDS_PUBLISH, {"funcode": 1})
        self.assertIsInstance(result, TelemetryEvent)


class TestBatchFlushing(unittest.TestCase):
    """Test batch flushing behavior."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")

    def test_flush_on_close(self):
        tc = TelemetryCollector(db_path=self.db_path, flush_interval_s=10.0, batch_size=1000)
        tc.start()
        for i in range(25):
            tc.log_system_event(event_type="test", source="test", detail=f"event {i}")
        # Don't wait — close should flush
        tc.close()
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
        conn.close()
        self.assertEqual(n, 25)

    def test_batch_size_trigger(self):
        tc = TelemetryCollector(db_path=self.db_path, flush_interval_s=10.0, batch_size=10)
        tc.start()
        for i in range(15):
            tc.log_system_event(event_type="test", source="test", detail=f"event {i}")
        time.sleep(0.3)  # Give writer time to process batch
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(n, 10)
        tc.close()


class TestConcurrentWrites(unittest.TestCase):
    """Test concurrent writes from multiple threads."""

    def test_concurrent_writes(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        tc = TelemetryCollector(db_path=db_path, flush_interval_s=0.05)
        tc.start()
        tc.enable()

        n_threads = 5
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def writer(tid):
            barrier.wait()
            for i in range(n_per_thread):
                tc.log_system_event(event_type="test", source=f"thread-{tid}", detail=f"event-{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        tc.close()

        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
        conn.close()
        self.assertEqual(n, n_threads * n_per_thread)


class TestLegacyCompat(unittest.TestCase):
    """Test legacy get_events, get_pipeline, get_stats."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        self.tc = TelemetryCollector(db_path=self.db_path, flush_interval_s=0.05)
        self.tc.start()
        self.tc.enable()

    def tearDown(self):
        self.tc.close()

    def test_get_stats(self):
        self.tc.emit("dds", EventType.DDS_PUBLISH, {"funcode": 1})
        time.sleep(0.2)
        stats = self.tc.get_stats()
        self.assertIn("total_events", stats)
        self.assertIn("rates_per_sec", stats)
        self.assertGreaterEqual(stats["total_events"], 1)

    def test_get_events_from_system(self):
        self.tc.emit("dds", EventType.ERROR, {"detail": "test error"})
        time.sleep(0.2)
        events = self.tc.get_events(event_type=EventType.ERROR)
        self.assertGreaterEqual(len(events), 1)
        self.assertIsInstance(events[0], TelemetryEvent)

    def test_get_pipeline(self):
        cid = TelemetryCollector.new_correlation_id()
        self.tc.emit("web", EventType.CMD_SENT, {"endpoint": "/api/test"}, correlation_id=cid)
        self.tc.emit("dds", EventType.DDS_PUBLISH, {"funcode": 1}, correlation_id=cid)
        time.sleep(0.2)
        pipeline = self.tc.get_pipeline(cid)
        self.assertEqual(len(pipeline), 2)

    def test_new_correlation_id(self):
        cid = TelemetryCollector.new_correlation_id()
        self.assertIsInstance(cid, str)
        self.assertEqual(len(cid), 12)


if __name__ == "__main__":
    unittest.main()
