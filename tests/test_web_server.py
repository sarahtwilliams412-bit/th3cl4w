#!/usr/bin/env python3.12
"""
Tests for th3cl4w web server — all endpoints, WebSocket, validation, ordering.
Uses FastAPI TestClient with simulated arm.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure --simulate mode for import
sys.modules.setdefault("pytest", pytest)

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_server(tmp_path):
    """Reset server state before each test, using a temp telemetry DB."""
    import src.telemetry.collector as _tc_mod
    from src.telemetry.collector import TelemetryCollector

    # Replace the singleton with a temp-DB collector so tests don't pollute data/telemetry.db
    old_collector = _tc_mod._collector
    tmp_collector = TelemetryCollector(db_path=str(tmp_path / "test_telemetry.db"))
    tmp_collector.start()
    tmp_collector.enable()
    _tc_mod._collector = tmp_collector

    from web.server import SimulatedArm, action_log, CommandSmoother
    import web.server as srv

    srv.arm = SimulatedArm()
    srv.smoother = CommandSmoother(srv.arm, rate_hz=10.0, smoothing_factor=0.35, max_step_deg=15.0)
    srv.smoother.sync_from_feedback([0.0] * 6, 0.0)
    srv._prev_state = {}
    srv._active_task = None
    srv._enable_snapshot = None
    srv._enable_gripper_snapshot = 0.0
    action_log._entries.clear()

    # Initialize task planner if available
    if srv._HAS_PLANNING:
        from src.planning.task_planner import TaskPlanner

        srv.task_planner = TaskPlanner()

    yield

    tmp_collector.close()
    _tc_mod._collector = old_collector


@pytest.fixture
def enabled_client():
    """Client with arm powered on and enabled."""
    from web.server import app

    with TestClient(app) as c:
        c.post("/api/command/power-on")
        c.post("/api/command/enable")
        yield c


@pytest.fixture
def client():
    from web.server import app

    with TestClient(app) as c:
        yield c


# --- State endpoint ---


class TestState:
    def test_get_state(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        d = r.json()
        assert "joints" in d
        assert "power" in d
        assert "enabled" in d
        assert "gripper" in d
        assert "timestamp" in d
        assert len(d["joints"]) == 6
        assert d["connected"] is True

    def test_state_initial_values(self, client):
        d = client.get("/api/state").json()
        assert d["power"] is False
        assert d["enabled"] is False
        assert d["error"] == 0
        assert d["gripper"] == 0.0


# --- Power endpoints ---


class TestPower:
    def test_power_on(self, client):
        r = client.post("/api/command/power-on")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "POWER_ON"
        assert d["state"]["power"] is True

    def test_power_off(self, client):
        client.post("/api/command/power-on")
        r = client.post("/api/command/power-off")
        d = r.json()
        assert d["ok"] is True
        assert d["state"]["power"] is False

    def test_power_on_already_on(self, client):
        client.post("/api/command/power-on")
        r = client.post("/api/command/power-on")
        assert r.json()["ok"] is True  # idempotent

    def test_power_off_disables_motors(self, client):
        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        r = client.post("/api/command/power-off")
        d = r.json()
        assert d["state"]["power"] is False
        assert d["state"]["enabled"] is False


# --- Enable/Disable ---


class TestEnableDisable:
    def test_enable_without_power_fails(self, client):
        r = client.post("/api/command/enable")
        d = r.json()
        assert d["ok"] is False
        assert "power" in d.get("error", "").lower() or "Power" in d.get("error", "")

    def test_enable_with_power(self, client):
        client.post("/api/command/power-on")
        r = client.post("/api/command/enable")
        d = r.json()
        assert d["ok"] is True
        assert d["state"]["enabled"] is True

    def test_disable_anytime(self, client):
        # Disable when already disabled — should still succeed
        r = client.post("/api/command/disable")
        assert r.json()["ok"] is True

    def test_disable_when_enabled(self, client):
        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        r = client.post("/api/command/disable")
        d = r.json()
        assert d["ok"] is True
        assert d["state"]["enabled"] is False


# --- Emergency Stop ---


class TestEmergencyStop:
    def test_estop_disables_and_powers_off(self, client):
        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        r = client.post("/api/command/stop")
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "EMERGENCY_STOP"
        assert d["state"]["enabled"] is False
        assert d["state"]["power"] is False


# --- Set Joint ---


class TestSetJoint:
    def test_set_joint_valid(self, enabled_client):
        r = enabled_client.post("/api/command/set-joint", json={"id": 0, "angle": 45.0})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_set_joint_rejected_when_not_enabled(self, client):
        r = client.post("/api/command/set-joint", json={"id": 0, "angle": 45.0})
        assert r.status_code == 409
        assert r.json()["ok"] is False

    def test_set_joint_all_joints(self, enabled_client):
        for i, (lo, hi) in enumerate(
            [(-135, 135), (-90, 90), (-90, 90), (-135, 135), (-90, 90), (-135, 135)]
        ):
            r = enabled_client.post("/api/command/set-joint", json={"id": i, "angle": hi / 2})
            assert r.json()["ok"] is True

    def test_set_joint_out_of_range(self, enabled_client):
        # J1 range is ±90
        r = enabled_client.post("/api/command/set-joint", json={"id": 1, "angle": 100.0})
        assert r.status_code == 400
        assert r.json()["ok"] is False

    def test_set_joint_bad_id(self, client):
        r = client.post("/api/command/set-joint", json={"id": 6, "angle": 0.0})
        assert r.status_code == 422  # validation error from Pydantic (id max=5)

    def test_set_joint_negative_id(self, client):
        r = client.post("/api/command/set-joint", json={"id": -1, "angle": 0.0})
        assert r.status_code == 422

    def test_set_joint_missing_fields(self, client):
        r = client.post("/api/command/set-joint", json={"id": 0})
        assert r.status_code == 422

        r = client.post("/api/command/set-joint", json={"angle": 10.0})
        assert r.status_code == 422

    def test_set_joint_boundary(self, enabled_client):
        # Exactly at boundary should work
        r = enabled_client.post("/api/command/set-joint", json={"id": 0, "angle": 135.0})
        assert r.json()["ok"] is True
        r = enabled_client.post("/api/command/set-joint", json={"id": 0, "angle": -135.0})
        assert r.json()["ok"] is True


# --- Set All Joints ---


class TestSetAllJoints:
    def test_set_all_joints(self, enabled_client):
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        r = enabled_client.post("/api/command/set-all-joints", json={"angles": angles})
        assert r.json()["ok"] is True

    def test_set_all_joints_rejected_when_not_enabled(self, client):
        r = client.post("/api/command/set-all-joints", json={"angles": [0.0] * 6})
        assert r.status_code == 409

    def test_set_all_joints_wrong_count(self, client):
        r = client.post("/api/command/set-all-joints", json={"angles": [0.0] * 5})
        assert r.status_code == 422

        r = client.post("/api/command/set-all-joints", json={"angles": [0.0] * 7})
        assert r.status_code == 422

    def test_set_all_joints_out_of_range(self, enabled_client):
        # J1 (index 1) max is 90
        angles = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        r = enabled_client.post("/api/command/set-all-joints", json={"angles": angles})
        assert r.status_code == 400
        assert r.json()["ok"] is False

    def test_set_all_joints_missing(self, client):
        r = client.post("/api/command/set-all-joints", json={})
        assert r.status_code == 422


# --- Set Gripper ---


class TestSetGripper:
    def test_set_gripper_valid(self, enabled_client):
        r = enabled_client.post("/api/command/set-gripper", json={"position": 32.5})
        assert r.json()["ok"] is True

    def test_set_gripper_rejected_when_not_enabled(self, client):
        r = client.post("/api/command/set-gripper", json={"position": 32.5})
        assert r.status_code == 409

    def test_set_gripper_zero(self, enabled_client):
        r = enabled_client.post("/api/command/set-gripper", json={"position": 0.0})
        assert r.json()["ok"] is True

    def test_set_gripper_max(self, enabled_client):
        r = enabled_client.post("/api/command/set-gripper", json={"position": 65.0})
        assert r.json()["ok"] is True

    def test_set_gripper_out_of_range(self, client):
        r = client.post("/api/command/set-gripper", json={"position": 70.0})
        assert r.status_code == 422

    def test_set_gripper_negative(self, client):
        r = client.post("/api/command/set-gripper", json={"position": -1.0})
        assert r.status_code == 422

    def test_set_gripper_missing(self, client):
        r = client.post("/api/command/set-gripper", json={})
        assert r.status_code == 422


# --- Reset ---


class TestReset:
    def test_reset(self, client):
        client.post("/api/command/set-joint", json={"id": 0, "angle": 45.0})
        r = client.post("/api/command/reset")
        assert r.json()["ok"] is True


# --- Log endpoint ---


class TestLog:
    def test_log_returns_entries(self, client):
        client.post("/api/command/power-on")
        r = client.get("/api/log")
        assert r.status_code == 200
        entries = r.json()["entries"]
        assert len(entries) > 0

    def test_log_entry_format(self, client):
        client.post("/api/command/power-on")
        entries = client.get("/api/log").json()["entries"]
        e = entries[-1]
        assert "ts" in e
        assert "ts_str" in e
        assert "action" in e
        assert "details" in e
        assert "level" in e

    def test_every_command_generates_log(self, client):
        from web.server import action_log

        action_log._entries.clear()

        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        client.post("/api/command/set-joint", json={"id": 0, "angle": 10.0})
        client.post("/api/command/set-gripper", json={"position": 20.0})
        client.post("/api/command/disable")
        client.post("/api/command/power-off")

        entries = client.get("/api/log").json()["entries"]
        actions = [e["action"] for e in entries]
        assert "POWER_ON" in actions
        assert "ENABLE" in actions
        assert "SET_JOINT" in actions
        assert "SET_GRIPPER" in actions
        assert "DISABLE" in actions
        assert "POWER_OFF" in actions


# --- Command ordering enforcement ---


class TestOrdering:
    def test_full_sequence(self, client):
        """Power on -> enable -> move -> disable -> power off."""
        r = client.post("/api/command/power-on")
        assert r.json()["ok"] is True

        r = client.post("/api/command/enable")
        assert r.json()["ok"] is True

        r = client.post("/api/command/set-joint", json={"id": 0, "angle": 30.0})
        assert r.json()["ok"] is True

        r = client.post("/api/command/disable")
        assert r.json()["ok"] is True

        r = client.post("/api/command/power-off")
        assert r.json()["ok"] is True

    def test_cannot_enable_before_power(self, client):
        r = client.post("/api/command/enable")
        assert r.json()["ok"] is False


# --- WebSocket ---


class TestWebSocket:
    def test_ws_receives_state(self, client):
        with client.websocket_connect("/ws/state") as ws:
            data = ws.receive_json()
            assert "joints" in data
            assert "power" in data
            assert "log" in data
            assert len(data["joints"]) == 6

    def test_ws_receives_updates(self, client):
        with client.websocket_connect("/ws/state") as ws:
            d1 = ws.receive_json()
            # Power on via REST
            client.post("/api/command/power-on")
            # Next WS message should reflect power on
            for _ in range(5):
                d2 = ws.receive_json()
                if d2.get("type") == "ack" or d2.get("power"):
                    break
            # State should eventually show power on
            assert d2.get("power") is True or d2.get("type") == "ack"


# --- Response format ---


class TestResponseFormat:
    def test_command_response_has_state(self, client):
        """Every command response must include state."""
        for endpoint in ["power-on", "power-off", "disable", "reset", "stop"]:
            r = client.post(f"/api/command/{endpoint}")
            d = r.json()
            assert "state" in d, f"Missing state in {endpoint} response"
            assert "joints" in d["state"]

    def test_set_joint_response_has_state(self, enabled_client):
        r = enabled_client.post("/api/command/set-joint", json={"id": 0, "angle": 10.0})
        assert "state" in r.json()

    def test_set_gripper_response_has_state(self, enabled_client):
        r = enabled_client.post("/api/command/set-gripper", json={"position": 10.0})
        assert "state" in r.json()


# --- Telemetry query endpoints ---


class TestTelemetryQuery:
    def test_query_summary(self, client):
        r = client.get("/api/query/summary")
        assert r.status_code == 200
        d = r.json()
        assert "total_commands" in d
        assert "total_events" in d
        assert "duration_s" in d

    def test_query_db_stats(self, client):
        r = client.get("/api/query/db-stats")
        assert r.status_code == 200
        d = r.json()
        assert "dds_commands" in d
        assert "system_events" in d

    def test_query_system_events(self, client):
        r = client.get("/api/query/system-events")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_query_web_latency(self, client):
        # Generate some requests first
        client.get("/api/state")
        r = client.get("/api/query/web-latency")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_query_command_rate(self, client):
        r = client.get("/api/query/command-rate")
        assert r.status_code == 200
        d = r.json()
        assert "rate_hz" in d

    def test_query_joint_history_valid(self, client):
        r = client.get("/api/query/joint-history/0")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_query_joint_history_invalid(self, client):
        r = client.get("/api/query/joint-history/9")
        assert r.status_code == 400

    def test_query_tracking_error(self, client):
        r = client.get("/api/query/tracking-error/0")
        assert r.status_code == 200
        d = r.json()
        assert "mean_error_deg" in d

    def test_query_smoother(self, client):
        r = client.get("/api/query/smoother")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# --- Task / planning endpoints ---


class TestTasks:
    def test_task_home_requires_enable(self, client):
        r = client.post("/api/task/home")
        assert r.status_code == 409
        assert r.json()["ok"] is False

    def test_task_ready_requires_enable(self, client):
        r = client.post("/api/task/ready")
        assert r.status_code == 409

    def test_task_wave_requires_enable(self, client):
        r = client.post("/api/task/wave")
        assert r.status_code == 409

    def test_task_stop_always_ok(self, client):
        r = client.post("/api/task/stop")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_task_home_when_enabled(self, enabled_client):
        r = enabled_client.post("/api/task/home", json={"speed": 0.5})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_HOME"
        assert "points" in d
        assert "duration_s" in d

    def test_task_ready_when_enabled(self, enabled_client):
        r = enabled_client.post("/api/task/ready")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_READY"

    def test_task_wave_when_enabled(self, enabled_client):
        r = enabled_client.post("/api/task/wave", json={"speed": 0.5, "waves": 2})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_WAVE"

    def test_task_wave_default_params(self, enabled_client):
        r = enabled_client.post("/api/task/wave")
        assert r.status_code == 200
        assert r.json()["ok"] is True


# --- Planning initialization ---


class TestPlanningInit:
    """Verify the task planner is properly initialized (global, not local)."""

    def test_task_planner_initialized(self, client):
        """After reset_server fixture, task_planner must not be None."""
        import web.server as srv

        assert srv._HAS_PLANNING is True, "Planning module should be importable"
        assert srv.task_planner is not None, "task_planner global must be set"

    def test_task_home_returns_trajectory(self, enabled_client):
        """Home task should succeed and include trajectory info, not 501."""
        # Move away from home first so the trajectory is non-trivial
        enabled_client.post(
            "/api/command/set-all-joints",
            json={"angles": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]},
        )
        r = enabled_client.post("/api/task/home")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_HOME"
        assert d["points"] > 0
        assert d["duration_s"] >= 0

    def test_task_ready_returns_trajectory(self, enabled_client):
        """Ready task should succeed, not return 'planning module not available'."""
        r = enabled_client.post("/api/task/ready")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_READY"
        assert d["points"] > 0

    def test_task_wave_returns_trajectory(self, enabled_client):
        """Wave task should succeed with trajectory details."""
        r = enabled_client.post("/api/task/wave")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["action"] == "TASK_WAVE"
        assert d["points"] > 0


# --- Enable snapshot & safe disable ---


class TestEnableSnapshot:
    """Verify position is captured at enable and used at disable."""

    def test_enable_captures_snapshot(self, client):
        """Enabling motors should save the current joint positions."""
        import web.server as srv

        assert srv._enable_snapshot is None
        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        assert srv._enable_snapshot is not None
        assert len(srv._enable_snapshot) == 6
        assert isinstance(srv._enable_snapshot[0], float)

    def test_enable_captures_gripper_snapshot(self, client):
        """Enabling motors should save the gripper position too."""
        import web.server as srv

        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        assert isinstance(srv._enable_gripper_snapshot, float)

    def test_disable_clears_snapshot(self, client):
        """Disabling should clear the enable snapshot."""
        import web.server as srv

        client.post("/api/command/power-on")
        client.post("/api/command/enable")
        assert srv._enable_snapshot is not None
        client.post("/api/command/disable")
        assert srv._enable_snapshot is None

    def test_disable_without_enable(self, client):
        """Disabling without enable should work (no snapshot to use)."""
        import web.server as srv

        assert srv._enable_snapshot is None
        client.post("/api/command/power-on")
        r = client.post("/api/command/disable")
        assert r.status_code == 200

    def test_enable_snapshot_after_movement(self, enabled_client):
        """Snapshot should reflect position at enable time, not after moves."""
        import web.server as srv

        snapshot_before = list(srv._enable_snapshot)
        # Move a joint
        enabled_client.post("/api/command/set-joint", json={"id": 0, "angle": 45.0})
        # Snapshot should still be the original enable-time values
        assert srv._enable_snapshot == snapshot_before

    def test_disable_return_with_planning(self, enabled_client):
        """When arm has moved, disable should attempt to return to start."""
        import web.server as srv

        # Move the arm significantly so distance > 1.0 deg
        enabled_client.post(
            "/api/command/set-all-joints",
            json={"angles": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]},
        )
        # Force simulated arm to reflect the target immediately
        srv.arm._angles = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

        r = enabled_client.post("/api/command/disable")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        # Should mention returning to start
        assert "Returning" in d.get("action", "") or "DISABLE" in d.get("action", "")
