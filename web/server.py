#!/usr/bin/env python3.12
"""
th3cl4w — Web Control Panel for Unitree D1 Arm
FastAPI backend with WebSocket state streaming and REST command API.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from src.telemetry import get_collector, EventType

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False

try:
    from src.telemetry.query import TelemetryQuery

    _HAS_QUERY = True
except ImportError:
    _HAS_QUERY = False

import numpy as np

try:
    from src.planning.task_planner import TaskPlanner, TaskStatus
    from src.planning.motion_planner import Waypoint

    _HAS_PLANNING = True
except ImportError:
    _HAS_PLANNING = False

try:
    from src.vision.calibration import StereoCalibrator
    from src.vision.workspace_mapper import WorkspaceMapper
    from src.planning.collision_preview import CollisionPreview

    _HAS_BIFOCAL = True
except ImportError:
    _HAS_BIFOCAL = False

_web_dir = str(Path(__file__).resolve().parent)
if _web_dir not in sys.path:
    sys.path.insert(0, _web_dir)

from command_smoother import CommandSmoother

# ---------------------------------------------------------------------------
# CLI args (parsed early so lifespan can access them)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w D1 Arm Web Control Panel")
parser.add_argument("--simulate", action="store_true", help="Run with simulated arm state")
parser.add_argument("--interface", default="eno1", help="Network interface for DDS (default: eno1)")
parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")

# Only parse if running as main (not under test)
if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args(["--simulate"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("th3cl4w.web")

# ---------------------------------------------------------------------------
# D1 Joint specs
# ---------------------------------------------------------------------------

JOINT_LIMITS_DEG = {
    0: (-135.0, 135.0),  # J0 base yaw
    1: (-90.0, 90.0),  # J1 shoulder pitch
    2: (-90.0, 90.0),  # J2 elbow pitch
    3: (-135.0, 135.0),  # J3 wrist roll
    4: (-90.0, 90.0),  # J4 wrist pitch
    5: (-135.0, 135.0),  # J5 wrist roll
}

GRIPPER_RANGE = (0.0, 65.0)  # mm

# ---------------------------------------------------------------------------
# Structured action log
# ---------------------------------------------------------------------------


class ActionLog:
    """Thread-safe circular log of action entries."""

    def __init__(self, maxlen: int = 200):
        self._entries: deque = deque(maxlen=maxlen)

    def add(self, action: str, details: str, level: str = "info"):
        ts = time.time()
        d = time.localtime(ts)
        ms = int((ts % 1) * 1000)
        ts_str = f"{d.tm_hour:02d}:{d.tm_min:02d}:{d.tm_sec:02d}.{ms:03d}"
        entry = {"ts": ts, "ts_str": ts_str, "action": action, "details": details, "level": level}
        self._entries.append(entry)
        logger.log(
            {"info": logging.INFO, "error": logging.ERROR, "warning": logging.WARNING}.get(
                level, logging.INFO
            ),
            "%s | %s | %s",
            ts_str,
            action,
            details,
        )

    def last(self, n: int = 50) -> List[Dict]:
        items = list(self._entries)
        return items[-n:]


action_log = ActionLog()

# ---------------------------------------------------------------------------
# Simulated arm for --simulate mode
# ---------------------------------------------------------------------------


class SimulatedArm:
    """Fake arm that holds state in memory with proper power→enable ordering."""

    def __init__(self):
        self._angles = [0.0] * 6
        self._target_angles = [0.0] * 6
        self._gripper = 0.0
        self._target_gripper = 0.0
        self._powered = False
        self._enabled = False
        self._error = 0
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_joint_angles(self):
        import numpy as np

        for i in range(6):
            diff = self._target_angles[i] - self._angles[i]
            self._angles[i] += diff * 0.15
        self._gripper += (self._target_gripper - self._gripper) * 0.15
        return np.array(self._angles)

    def get_gripper_position(self) -> float:
        return round(self._gripper, 2)

    def get_status(self):
        return {
            "power_status": 1 if self._powered else 0,
            "enable_status": 1 if self._enabled else 0,
            "error_status": self._error,
        }

    def power_on(self, **kwargs):
        self._powered = True
        return True

    def power_off(self, **kwargs):
        self._enabled = False
        self._powered = False
        return True

    def enable_motors(self, **kwargs):
        if not self._powered:
            return False
        self._enabled = True
        return True

    def disable_motors(self, **kwargs):
        self._enabled = False
        return True

    def reset_to_zero(self, **kwargs):
        self._target_angles = [0.0] * 6
        self._target_gripper = 0.0
        return True

    def set_joint(self, joint_id: int, angle_deg: float, delay_ms: int = 0, **kwargs):
        lo, hi = JOINT_LIMITS_DEG.get(joint_id, (-135, 135))
        if not (0 <= joint_id <= 5):
            return False
        self._target_angles[joint_id] = max(lo, min(hi, angle_deg))
        return True

    def set_all_joints(self, angles_deg: list, mode: int = 0, **kwargs):
        if len(angles_deg) != 6:
            return False
        for i, a in enumerate(angles_deg):
            lo, hi = JOINT_LIMITS_DEG.get(i, (-135, 135))
            self._target_angles[i] = max(lo, min(hi, a))
        return True

    def set_gripper(self, position_mm: float, **kwargs):
        self._target_gripper = max(GRIPPER_RANGE[0], min(GRIPPER_RANGE[1], position_mm))
        return True

    def disconnect(self):
        self._connected = False


# ---------------------------------------------------------------------------
# Global arm reference
# ---------------------------------------------------------------------------

arm: Any = None
smoother: Optional[CommandSmoother] = None
task_planner: Any = None  # TaskPlanner instance, initialized in lifespan
workspace_mapper: Any = None  # WorkspaceMapper for bifocal vision
collision_preview: Any = None  # CollisionPreview for path checking

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arm, smoother
    # Start telemetry collector
    if _HAS_TELEMETRY:
        tc = get_collector()
        tc.start()
        tc.enable()
        tc.log_system_event("startup", "system", "th3cl4w starting up")

    if args.simulate:
        arm = SimulatedArm()
        action_log.add("SYSTEM", "Simulated arm initialized", "info")
        logger.info("Running in SIMULATION mode")
    else:
        try:
            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.interface.d1_dds_connection import D1DDSConnection

            arm = D1DDSConnection(collector=tc if _HAS_TELEMETRY else None)
            if arm.connect(interface_name=args.interface):
                action_log.add("SYSTEM", f"DDS connected on {args.interface}", "info")
                if _HAS_TELEMETRY:
                    get_collector().log_system_event(
                        "connect", "dds", f"Connected on {args.interface}"
                    )
            else:
                action_log.add("SYSTEM", f"DDS connection FAILED on {args.interface}", "error")
                if _HAS_TELEMETRY:
                    get_collector().log_system_event(
                        "connect_failed", "dds", f"Failed on {args.interface}", level="error"
                    )
        except Exception as e:
            action_log.add("SYSTEM", f"DDS init error: {e}", "error")
            logger.exception("Failed to initialize DDS connection")
            arm = None

    # Start the command smoother for smooth motion
    if arm is not None:
        tc_ref = get_collector() if _HAS_TELEMETRY else None
        smoother = CommandSmoother(
            arm, rate_hz=10.0, smoothing_factor=0.35, max_step_deg=15.0, collector=tc_ref
        )
        await smoother.start()
        action_log.add(
            "SYSTEM", f"Command smoother started (10Hz, α=0.35, synced={smoother.synced})", "info"
        )

    # Initialize task planner for pre-built motion sequences
    if _HAS_PLANNING:
        task_planner = TaskPlanner()
        action_log.add("SYSTEM", "Task planner initialized", "info")

    # Initialize bifocal workspace mapper and collision preview
    global workspace_mapper, collision_preview
    if _HAS_BIFOCAL:
        calibrator = StereoCalibrator()
        # Try to load existing calibration
        calib_path = Path(__file__).resolve().parent.parent / "calibration" / "stereo.npz"
        if calib_path.exists():
            try:
                calibrator.load(calib_path)
                action_log.add("SYSTEM", "Stereo calibration loaded", "info")
            except Exception as e:
                action_log.add("SYSTEM", f"Calibration load failed: {e}", "warning")
        workspace_mapper = WorkspaceMapper(calibrator)
        collision_preview = CollisionPreview()
        action_log.add(
            "SYSTEM", "Bifocal workspace mapper initialized (disabled by default)", "info"
        )

    yield

    if smoother is not None:
        await smoother.stop()
    if arm is not None:
        try:
            arm.disconnect()
        except Exception:
            pass
    # Stop telemetry collector
    if _HAS_TELEMETRY:
        get_collector().log_system_event("shutdown", "system", "th3cl4w shutting down")
        get_collector().stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="th3cl4w", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def telemetry_middleware(request, call_next):
    """Log all /api/ requests to telemetry."""
    if _HAS_TELEMETRY and request.url.path.startswith("/api/"):
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        tc = get_collector()
        if tc.enabled:
            tc.log_web_request(
                endpoint=request.url.path,
                method=request.method,
                params=None,
                response_ms=elapsed_ms,
                status_code=response.status_code,
                ok=response.status_code < 400,
            )
        return response
    return await call_next(request)


# ---------------------------------------------------------------------------
# Connected WebSocket clients for command acks
# ---------------------------------------------------------------------------
ws_clients: list[WebSocket] = []

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetJointRequest(BaseModel):
    id: int = Field(ge=0, le=5)
    angle: float


class SetAllJointsRequest(BaseModel):
    angles: List[float] = Field(min_length=6, max_length=6)


class SetGripperRequest(BaseModel):
    position: float = Field(ge=0.0, le=65.0)


class RawCommandRequest(BaseModel):
    payload: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_prev_state: Dict[str, Any] = {}


def get_arm_state() -> Dict[str, Any]:
    global _prev_state
    if arm is None:
        return {
            "connected": False,
            "joints": [0.0] * 6,
            "gripper": 0.0,
            "power": False,
            "enabled": False,
            "error": 0,
            "timestamp": time.time(),
        }

    angles_raw = arm.get_joint_angles()
    angles = [round(float(a), 2) for a in angles_raw] if angles_raw is not None else [0.0] * 6
    # Ensure exactly 6 joints
    angles = angles[:6] if len(angles) >= 6 else angles + [0.0] * (6 - len(angles))

    gripper = 0.0
    if hasattr(arm, "get_gripper_position"):
        gripper = round(float(arm.get_gripper_position()), 2)

    status = arm.get_status() or {}

    state = {
        "connected": arm.is_connected,
        "joints": angles,
        "gripper": gripper,
        "power": bool(status.get("power_status", 0)),
        "enabled": bool(status.get("enable_status", 0)),
        "error": status.get("error_status", 0),
        "timestamp": time.time(),
    }

    # SAFETY: Sync smoother from arm feedback if not yet synced
    if smoother and not smoother.synced:
        smoother.sync_from_feedback(angles, gripper)

    # NOTE: smoother arm_enabled is ONLY changed by explicit command handlers
    # (enable, disable, power-off, e-stop, reset). DO NOT sync from feedback here
    # — the arm takes time to process enable commands, so feedback would race.

    # Log state transitions
    if _prev_state:
        if _prev_state.get("power") != state["power"]:
            action_log.add("STATE", f"Power: {'ON' if state['power'] else 'OFF'}", "warning")
        if _prev_state.get("enabled") != state["enabled"]:
            action_log.add(
                "STATE", f"Motors: {'ENABLED' if state['enabled'] else 'DISABLED'}", "warning"
            )
        if _prev_state.get("error") != state["error"] and state["error"]:
            action_log.add("STATE", f"Error changed: {state['error']}", "error")
    _prev_state = state.copy()

    return state


def cmd_response(
    success: bool, action: str, extra: str = "", correlation_id: str | None = None
) -> JSONResponse:
    state = get_arm_state()
    level = "info" if success else "error"
    detail = f"{'OK' if success else 'FAILED'}"
    if extra:
        detail += f" — {extra}"
    action_log.add(action, detail, level)
    resp_data: Dict[str, Any] = {"ok": success, "action": action, "state": state}
    if correlation_id is not None:
        resp_data["correlation_id"] = correlation_id
    return JSONResponse(resp_data)


def _telem_cmd_sent(endpoint: str, params: Dict[str, Any], correlation_id: str) -> None:
    """Emit CMD_SENT if telemetry is available and enabled."""
    if _HAS_TELEMETRY:
        tc = get_collector()
        if tc.enabled:
            tc.emit(
                "web", EventType.CMD_SENT, {"endpoint": endpoint, "params": params}, correlation_id
            )


def _new_cid() -> str | None:
    """Generate a correlation_id if telemetry is available."""
    if _HAS_TELEMETRY:
        return get_collector().new_correlation_id()
    return None


async def broadcast_ack(action: str, success: bool):
    """Send command acknowledgment to all connected WS clients."""
    msg = {"type": "ack", "action": action, "ok": success, "timestamp": time.time()}
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/cameras")
async def api_cameras():
    """Proxy camera status from the camera server."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:8081/status")
            return resp.json()
    except Exception:
        return {
            "error": "Camera server unavailable",
            "0": {"connected": False},
            "1": {"connected": False},
        }


@app.get("/api/state")
async def api_state():
    return get_arm_state()


@app.get("/api/log")
async def api_log():
    return {"entries": action_log.last(50)}


@app.post("/api/command/enable")
async def cmd_enable():
    cid = _new_cid()
    _telem_cmd_sent("enable", {}, cid)
    if arm is None:
        return cmd_response(False, "ENABLE", "No arm connected", cid)
    state = arm.get_status() or {}
    if not state.get("power_status"):
        action_log.add("ENABLE", "REJECTED — power is off, power on first", "error")
        resp_data: Dict[str, Any] = {
            "ok": False,
            "action": "ENABLE",
            "error": "Power must be on before enabling",
            "state": get_arm_state(),
        }
        if cid:
            resp_data["correlation_id"] = cid
        return JSONResponse(resp_data)
    ok = arm.enable_motors(_correlation_id=cid)
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "enable",
            "web",
            f"Motors enable: {'OK' if ok else 'FAILED'}",
            correlation_id=cid,
            level="info" if ok else "error",
        )
    if ok and smoother:
        smoother.set_arm_enabled(True)
    resp = cmd_response(ok, "ENABLE", correlation_id=cid)
    await broadcast_ack("ENABLE", ok)
    return resp


@app.post("/api/command/disable")
async def cmd_disable():
    cid = _new_cid()
    _telem_cmd_sent("disable", {}, cid)
    ok = arm.disable_motors(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "disable", "web", f"Motors disable: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "DISABLE", correlation_id=cid)
    await broadcast_ack("DISABLE", ok)
    return resp


@app.post("/api/command/power-on")
async def cmd_power_on():
    cid = _new_cid()
    _telem_cmd_sent("power-on", {}, cid)
    ok = arm.power_on(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "power_on", "web", f"Power on: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    resp = cmd_response(ok, "POWER_ON", correlation_id=cid)
    await broadcast_ack("POWER_ON", ok)
    return resp


@app.post("/api/command/power-off")
async def cmd_power_off():
    cid = _new_cid()
    _telem_cmd_sent("power-off", {}, cid)
    ok = arm.power_off(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "power_off", "web", f"Power off: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "POWER_OFF", correlation_id=cid)
    await broadcast_ack("POWER_OFF", ok)
    return resp


@app.post("/api/command/reset")
async def cmd_reset():
    cid = _new_cid()
    _telem_cmd_sent("reset", {}, cid)
    ok = arm.reset_to_zero(_correlation_id=cid) if arm else False
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "RESET", correlation_id=cid)
    await broadcast_ack("RESET", ok)
    return resp


@app.post("/api/command/set-joint")
async def cmd_set_joint(req: SetJointRequest):
    cid = _new_cid()
    _telem_cmd_sent("set-joint", {"id": req.id, "angle": req.angle}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_JOINT", f"Request: J{req.id} -> {req.angle}°", "info")
    lo, hi = JOINT_LIMITS_DEG.get(req.id, (-135, 135))
    if not (lo <= req.angle <= hi):
        action_log.add(
            "SET_JOINT", f"REJECTED — J{req.id} angle {req.angle}° outside [{lo}, {hi}]", "error"
        )
        resp_data: Dict[str, Any] = {
            "ok": False,
            "action": "SET_JOINT",
            "error": f"Angle {req.angle} out of range [{lo}, {hi}]",
            "state": get_arm_state(),
        }
        if cid:
            resp_data["correlation_id"] = cid
        return JSONResponse(resp_data, status_code=400)
    if smoother and smoother.running:
        smoother.set_joint_target(req.id, req.angle)
        ok = True
    else:
        ok = arm.set_joint(req.id, req.angle, _correlation_id=cid) if arm else False
    resp = cmd_response(ok, "SET_JOINT", f"J{req.id} = {req.angle}°", cid)
    await broadcast_ack("SET_JOINT", ok)
    return resp


@app.post("/api/command/set-all-joints")
async def cmd_set_all_joints(req: SetAllJointsRequest):
    cid = _new_cid()
    _telem_cmd_sent("set-all-joints", {"angles": req.angles}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_ALL_JOINTS", f"Request: {[round(a,1) for a in req.angles]}", "info")
    for i, a in enumerate(req.angles):
        lo, hi = JOINT_LIMITS_DEG.get(i, (-135, 135))
        if not (lo <= a <= hi):
            action_log.add(
                "SET_ALL_JOINTS", f"REJECTED — J{i} angle {a}° outside [{lo}, {hi}]", "error"
            )
            resp_data2: Dict[str, Any] = {
                "ok": False,
                "action": "SET_ALL_JOINTS",
                "error": f"J{i} angle {a} out of range [{lo}, {hi}]",
                "state": get_arm_state(),
            }
            if cid:
                resp_data2["correlation_id"] = cid
            return JSONResponse(resp_data2, status_code=400)
    if smoother and smoother.running:
        smoother.set_all_joints_target(req.angles)
        ok = True
    else:
        ok = arm.set_all_joints(req.angles, _correlation_id=cid) if arm else False
    resp = cmd_response(ok, "SET_ALL_JOINTS", correlation_id=cid)
    await broadcast_ack("SET_ALL_JOINTS", ok)
    return resp


@app.post("/api/command/set-gripper")
async def cmd_set_gripper(req: SetGripperRequest):
    cid = _new_cid()
    _telem_cmd_sent("set-gripper", {"position": req.position}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_GRIPPER", f"Request: {req.position} mm", "info")
    if smoother and smoother.running:
        smoother.set_gripper_target(req.position)
        ok = True
    else:
        ok = False
        if arm and hasattr(arm, "set_gripper"):
            ok = arm.set_gripper(req.position, _correlation_id=cid)
    resp = cmd_response(ok, "SET_GRIPPER", f"{req.position} mm", cid)
    await broadcast_ack("SET_GRIPPER", ok)
    return resp


@app.post("/api/command/stop")
async def cmd_stop():
    """Emergency stop: disable motors AND power off."""
    cid = _new_cid()
    _telem_cmd_sent("stop", {}, cid)
    action_log.add("EMERGENCY_STOP", "⚠ TRIGGERED", "error")
    if smoother:
        smoother.emergency_stop()
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "estop", "web", "Emergency stop triggered", correlation_id=cid, level="error"
        )
    ok1 = arm.disable_motors(_correlation_id=cid) if arm else False
    ok2 = arm.power_off(_correlation_id=cid) if arm else False
    ok = ok1 and ok2
    resp = cmd_response(
        ok,
        "EMERGENCY_STOP",
        f"disable={'OK' if ok1 else 'FAIL'} power_off={'OK' if ok2 else 'FAIL'}",
        cid,
    )
    await broadcast_ack("EMERGENCY_STOP", ok)
    return resp


# ---------------------------------------------------------------------------
# WebSocket — stream state at 10Hz
# ---------------------------------------------------------------------------


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    action_log.add("WS", "Client connected", "info")
    try:
        while True:
            state = get_arm_state()
            state["log"] = action_log.last(30)
            await ws.send_json(state)
            if _HAS_TELEMETRY:
                tc = get_collector()
                if tc.enabled:
                    tc.emit(
                        "web",
                        EventType.WS_SEND,
                        {
                            "client_count": len(ws_clients),
                            "state_timestamp": state.get("timestamp"),
                        },
                    )
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        action_log.add("WS", "Client disconnected", "info")
    except Exception:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Debug / telemetry endpoints
# ---------------------------------------------------------------------------


@app.get("/api/debug/telemetry")
async def debug_telemetry(
    limit: int = 100, event_type: str | None = None, source: str | None = None
):
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    tc = get_collector()
    et = None
    if event_type:
        try:
            et = EventType(event_type)
        except ValueError:
            return JSONResponse({"error": f"unknown event_type: {event_type}"}, status_code=400)
    events = tc.get_events(limit=limit, event_type=et, source=source)
    return [
        {
            "timestamp_ms": e.timestamp_ms,
            "wall_time_ms": e.wall_time_ms,
            "source": e.source,
            "event_type": e.event_type.value,
            "payload": e.payload,
            "correlation_id": e.correlation_id,
        }
        for e in events
    ]


@app.get("/api/debug/stats")
async def debug_stats():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    return get_collector().get_stats()


@app.post("/api/debug/enable")
async def debug_enable():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    get_collector().enable()
    return {"enabled": True}


@app.post("/api/debug/disable")
async def debug_disable():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    get_collector().disable()
    return {"enabled": False}


@app.get("/api/debug/pipeline/{correlation_id}")
async def debug_pipeline(correlation_id: str):
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    pipeline = get_collector().get_pipeline(correlation_id)
    return pipeline


# ---------------------------------------------------------------------------
# WebSocket — stream telemetry events in real-time
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Stream telemetry events to clients in real-time."""
    await ws.accept()
    if not _HAS_TELEMETRY:
        await ws.close(code=1011, reason="Telemetry not available")
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    loop = asyncio.get_event_loop()

    def on_event(event_dict):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event_dict)
        except Exception:
            pass

    tc = get_collector()
    tc.subscribe(on_event)
    try:
        while True:
            event_dict = await queue.get()
            await ws.send_json(event_dict)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        tc.unsubscribe(on_event)


# ---------------------------------------------------------------------------
# Telemetry query endpoints — expose TelemetryQuery (SQLite read-only) data
# ---------------------------------------------------------------------------


def _get_query() -> "TelemetryQuery | None":
    """Open a read-only TelemetryQuery against the active database."""
    if not _HAS_QUERY:
        return None
    db_path = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"
    if not db_path.exists():
        return None
    try:
        return TelemetryQuery(str(db_path))
    except Exception:
        return None


@app.get("/api/query/summary")
async def query_summary():
    """Session summary: total commands, feedback, events, rates, errors."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.summary()
    finally:
        q.close()


@app.get("/api/query/db-stats")
async def query_db_stats():
    """Row counts per telemetry table."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_db_stats()
    finally:
        q.close()


@app.get("/api/query/joint-history/{joint}")
async def query_joint_history(joint: int, limit: int = 500):
    """Feedback angle history for a single joint."""
    if not (0 <= joint <= 6):
        return JSONResponse({"error": "joint must be 0-6"}, status_code=400)
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_joint_history(joint, limit=limit)
    finally:
        q.close()


@app.get("/api/query/tracking-error/{joint}")
async def query_tracking_error(joint: int):
    """Command vs feedback tracking error for a joint."""
    if not (0 <= joint <= 5):
        return JSONResponse({"error": "joint must be 0-5"}, status_code=400)
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_tracking_error(joint)
    finally:
        q.close()


@app.get("/api/query/command-rate")
async def query_command_rate(window: float = 10.0):
    """DDS command rate over a time window."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_command_rate(window)
    finally:
        q.close()


@app.get("/api/query/web-latency")
async def query_web_latency(endpoint: str | None = None, limit: int = 100):
    """Web request latency history."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_web_request_latency(endpoint=endpoint, limit=limit)
    finally:
        q.close()


@app.get("/api/query/system-events")
async def query_system_events(event_type: str | None = None, limit: int = 100):
    """System event log."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_system_events(event_type=event_type, limit=limit)
    finally:
        q.close()


@app.get("/api/query/smoother")
async def query_smoother_state(joint: int | None = None, limit: int = 500):
    """Smoother state (target vs current vs sent per joint)."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_smoother_state(joint=joint, limit=limit)
    finally:
        q.close()


# ---------------------------------------------------------------------------
# Planning endpoints — execute pre-built tasks through the smoother
# ---------------------------------------------------------------------------

_active_task: Optional[asyncio.Task] = None


async def _execute_trajectory(trajectory, label: str) -> None:
    """Feed trajectory points into the smoother at their scheduled times.

    Runs as a background asyncio task so the HTTP endpoint returns immediately.
    """
    if smoother is None or not smoother.arm_enabled:
        action_log.add("TASK", f"{label} — arm not enabled, aborting", "error")
        return

    action_log.add(
        "TASK",
        f"Executing {label} ({len(trajectory.points)} points, {trajectory.duration:.1f}s)",
        "info",
    )
    t0 = time.monotonic()

    for pt in trajectory.points:
        if smoother is None or not smoother.arm_enabled:
            action_log.add("TASK", f"{label} — aborted (arm disabled)", "error")
            return

        angles = [float(a) for a in pt.positions[:6]]
        smoother.set_all_joints_target(angles)
        if pt.gripper_mm is not None:
            smoother.set_gripper_target(pt.gripper_mm)

        # Wait until this point's scheduled time
        elapsed = time.monotonic() - t0
        wait = pt.time - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

    action_log.add("TASK", f"{label} complete ({time.monotonic() - t0:.1f}s)", "info")


class TaskRequest(BaseModel):
    speed: float = Field(default=0.6, ge=0.1, le=1.0)


@app.post("/api/task/home")
async def task_home(req: TaskRequest = TaskRequest()):
    """Plan and execute a smooth return to home position."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    result = task_planner.go_home(current, speed_factor=req.speed, gripper_mm=gripper)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(result.trajectory, "Go Home"))
    action_log.add(
        "TASK",
        f"Home planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_HOME",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


@app.post("/api/task/ready")
async def task_ready(req: TaskRequest = TaskRequest()):
    """Plan and execute move to ready/neutral position."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    result = task_planner.go_ready(current, speed_factor=req.speed, gripper_mm=gripper)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(result.trajectory, "Go Ready"))
    action_log.add(
        "TASK",
        f"Ready planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_READY",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


class WaveRequest(BaseModel):
    speed: float = Field(default=0.8, ge=0.1, le=1.0)
    waves: int = Field(default=3, ge=1, le=10)


@app.post("/api/task/wave")
async def task_wave(req: WaveRequest = WaveRequest()):
    """Plan and execute a wave gesture."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    result = task_planner.wave(current, n_waves=req.waves, speed_factor=req.speed)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(
        _execute_trajectory(result.trajectory, f"Wave ({req.waves}x)")
    )
    action_log.add(
        "TASK",
        f"Wave planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_WAVE",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


@app.post("/api/task/stop")
async def task_stop():
    """Cancel any running task (does NOT e-stop the arm)."""
    global _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
        action_log.add("TASK", "Task cancelled by user", "warning")
        return {"ok": True, "action": "TASK_STOP"}
    return {"ok": True, "action": "TASK_STOP", "detail": "No task running"}


def _get_current_joints():
    """Get current joint angles as numpy array for the planner."""
    if arm is None:
        return np.zeros(6)
    angles_raw = arm.get_joint_angles()
    if angles_raw is not None and len(angles_raw) >= 6:
        return np.array([float(a) for a in angles_raw[:6]])
    return np.zeros(6)


def armState_gripper() -> float:
    """Get current gripper position in mm."""
    if arm is None:
        return 0.0
    if hasattr(arm, "get_gripper_position"):
        return float(arm.get_gripper_position())
    return 0.0


# ---------------------------------------------------------------------------
# Bifocal Workspace Mapping endpoints — planning/measurement only
# ---------------------------------------------------------------------------


@app.post("/api/bifocal/toggle")
async def bifocal_toggle():
    """Toggle the bifocal workspace mapper on/off."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    enabled = workspace_mapper.toggle()
    action_log.add("BIFOCAL", f"Workspace mapper {'enabled' if enabled else 'disabled'}", "info")
    return {"ok": True, "enabled": enabled}


@app.get("/api/bifocal/status")
async def bifocal_status():
    """Get bifocal workspace mapper status."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return {"available": False}
    status = workspace_mapper.get_status()
    status["available"] = True
    return status


@app.post("/api/bifocal/update")
async def bifocal_update():
    """Trigger a workspace map update from current camera frames."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    if not workspace_mapper.enabled:
        return JSONResponse({"ok": False, "error": "Mapper not enabled"}, status_code=409)

    # Grab snapshots from both cameras
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        import cv2

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        result = workspace_mapper.update_from_frames(left, right)
        return {"ok": True, **result}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/bifocal/workspace")
async def bifocal_workspace(max_points: int = 500):
    """Get occupied voxel positions for 3D visualization."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return {"points": [], "summary": {}}
    points = workspace_mapper.get_occupied_points(max_points)
    summary = workspace_mapper.get_occupancy_summary()
    return {"points": points, "summary": summary}


@app.post("/api/bifocal/clear")
async def bifocal_clear():
    """Clear the workspace map."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    workspace_mapper.clear()
    action_log.add("BIFOCAL", "Workspace map cleared", "info")
    return {"ok": True}


class CalibScaleRequest(BaseModel):
    square_size_mm: float = Field(default=25.0, gt=0)


@app.post("/api/bifocal/calibrate-scale")
async def bifocal_calibrate_scale(req: CalibScaleRequest = CalibScaleRequest()):
    """Calibrate real-world scale using a checkerboard pattern."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Failed to decode frames"}, status_code=502)

        result = workspace_mapper.calibrate_scale_from_checkerboard(left, right, req.square_size_mm)
        if result["ok"]:
            action_log.add("BIFOCAL", f"Scale calibrated: factor={result['scale_factor']}", "info")
        else:
            action_log.add(
                "BIFOCAL", f"Scale calibration failed: {result.get('error', '?')}", "warning"
            )
        return result

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class TapeMeasureRequest(BaseModel):
    known_length_mm: float = Field(gt=0)
    x1: int
    y1: int
    x2: int
    y2: int


@app.post("/api/bifocal/calibrate-tape")
async def bifocal_calibrate_tape(req: TapeMeasureRequest):
    """Calibrate scale using two points on a tape measure."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Failed to decode frames"}, status_code=502)

        result = workspace_mapper.calibrate_scale_from_tape_measure(
            left, right, req.known_length_mm, (req.x1, req.y1), (req.x2, req.y2)
        )
        if result["ok"]:
            action_log.add("BIFOCAL", f"Tape calibration: factor={result['scale_factor']}", "info")
        return result

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/bifocal/preview")
async def bifocal_preview():
    """Preview the current arm pose against the workspace map for collisions."""
    if not _HAS_BIFOCAL or workspace_mapper is None or collision_preview is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    current = _get_current_joints()
    result = collision_preview.preview_single_pose(current, workspace_mapper)
    arm_points = collision_preview.get_arm_envelope(current)

    return {
        "ok": True,
        "clear": result.clear,
        "summary": result.summary,
        "hits": [
            {
                "link": h.link_index,
                "point": h.link_point_mm,
                "severity": h.severity,
            }
            for h in result.hits
        ],
        "arm_points_mm": arm_points,
        "checked": result.checked_points,
        "elapsed_ms": result.elapsed_ms,
    }


@app.post("/api/bifocal/preview-target")
async def bifocal_preview_target(req: SetAllJointsRequest):
    """Preview a target pose against the workspace map for collisions."""
    if not _HAS_BIFOCAL or workspace_mapper is None or collision_preview is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    target = np.array(req.angles[:6])
    result = collision_preview.preview_single_pose(target, workspace_mapper)
    arm_points = collision_preview.get_arm_envelope(target)

    return {
        "ok": True,
        "clear": result.clear,
        "summary": result.summary,
        "hits": [
            {
                "link": h.link_index,
                "point": h.link_point_mm,
                "severity": h.severity,
            }
            for h in result.hits
        ],
        "arm_points_mm": arm_points,
        "checked": result.checked_points,
        "elapsed_ms": result.elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Telemetry viewer page route
# ---------------------------------------------------------------------------


@app.get("/telemetry")
async def telemetry_page():
    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "static" / "telemetry.html")


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Starting th3cl4w web panel on %s:%d (simulate=%s)", args.host, args.port, args.simulate
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
