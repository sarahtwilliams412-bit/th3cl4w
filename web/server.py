# th3cl4w V1 Server — Stable Base
#!/usr/bin/env python3.12
"""
th3cl4w — Web Control Panel for Unitree D1 Arm.

FastAPI backend with WebSocket state streaming and REST command API.
Supports both real DDS hardware and --simulate mode.
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

# Ensure project root is in sys.path so src.* imports work regardless of CWD
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
    from src.vision.workspace_mapper import WorkspaceMapper
    from src.planning.collision_preview import CollisionPreview

    _HAS_BIFOCAL = True
except ImportError:
    _HAS_BIFOCAL = False

try:
    from src.vision.arm_tracker import DualCameraArmTracker
    from src.vision.grasp_planner import VisualGraspPlanner
    from src.planning.pick_executor import PickExecutor, PickPhase

    _HAS_VISUAL_PICK = True
except ImportError:
    _HAS_VISUAL_PICK = False

try:
    from src.vision.scene_analyzer import SceneAnalyzer
    from src.planning.vision_task_planner import VisionTaskPlanner

    _HAS_VISION_PLANNING = True
except ImportError:
    _HAS_VISION_PLANNING = False

try:
    from src.vision.claw_position import ClawPositionPredictor

    _HAS_CLAW_PREDICT = True
except ImportError:
    _HAS_CLAW_PREDICT = False

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
arm_tracker: Any = None  # DualCameraArmTracker for visual object tracking
grasp_planner: Any = None  # VisualGraspPlanner for grasp pose computation
pick_executor: Any = None  # PickExecutor for autonomous pick operations
vision_task_planner: Any = None  # VisionTaskPlanner for camera-guided planning
scene_analyzer: Any = None  # SceneAnalyzer for scene understanding
claw_predictor: Any = None  # ClawPositionPredictor for visual claw tracking

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arm, smoother, task_planner
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
    global task_planner
    if _HAS_PLANNING:
        task_planner = TaskPlanner()
        action_log.add("SYSTEM", "Task planner initialized", "info")

    # Initialize bifocal workspace mapper and collision preview
    global workspace_mapper, collision_preview
    if _HAS_BIFOCAL:
        workspace_mapper = WorkspaceMapper()
        collision_preview = CollisionPreview()
        action_log.add(
            "SYSTEM", "Bifocal workspace mapper initialized (disabled by default)", "info"
        )

    # Initialize visual pick system (arm tracker + grasp planner + pick executor)
    global arm_tracker, grasp_planner, pick_executor
    if _HAS_VISUAL_PICK:
        arm_tracker = DualCameraArmTracker()
        grasp_planner = VisualGraspPlanner()
        pick_executor = PickExecutor(task_planner=task_planner if _HAS_PLANNING else None)
        action_log.add("SYSTEM", "Visual pick system initialized (tracker + grasp planner)", "info")

    # Initialize vision task planner for camera-guided planning
    global vision_task_planner, scene_analyzer
    if _HAS_VISION_PLANNING and _HAS_PLANNING:
        scene_analyzer = SceneAnalyzer()
        vision_task_planner = VisionTaskPlanner(task_planner=task_planner)
        action_log.add("SYSTEM", "Vision task planner initialized", "info")

    # Initialize claw position predictor
    global claw_predictor
    if _HAS_CLAW_PREDICT:
        claw_predictor = ClawPositionPredictor()
        action_log.add(
            "SYSTEM", "Claw position predictor initialized (disabled by default)", "info"
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
_cached_joint_angles: list = [0.0] * 6
_cached_gripper: float = 0.0


def get_arm_state() -> Dict[str, Any]:
    global _prev_state, _cached_joint_angles, _cached_gripper
    if arm is None:
        return {
            "connected": False,
            "joints": list(_cached_joint_angles),
            "gripper": _cached_gripper,
            "power": False,
            "enabled": False,
            "error": 0,
            "timestamp": time.time(),
        }

    angles_raw = arm.get_joint_angles()
    if angles_raw is not None:
        angles = [round(float(a), 2) for a in angles_raw]
    else:
        # Use cached values instead of zeros to prevent viz jumping
        angles = list(_cached_joint_angles)
    # Ensure exactly 6 joints
    angles = angles[:6] if len(angles) >= 6 else angles + [0.0] * (6 - len(angles))
    # Cache last known good values to avoid zero-snap on transient None reads
    _cached_joint_angles = list(angles)

    gripper = 0.0
    if hasattr(arm, "get_gripper_position"):
        gripper = round(float(arm.get_gripper_position()), 2)
    _cached_gripper = gripper

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
    """Return current arm state (joints, gripper, power, enabled, error)."""
    return get_arm_state()


@app.get("/api/log")
async def api_log():
    """Return recent action log entries."""
    return {"entries": action_log.last(50)}


@app.post("/api/command/enable")
async def cmd_enable():
    """Enable motors (requires power to be on first)."""
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
    """Disable motors."""
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
    """Power on the arm."""
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
    """Power off the arm (also disables motors)."""
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
    """Reset arm to zero position."""
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
    """Set a single joint to a target angle (degrees)."""
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
    """Set all 6 joints to target angles simultaneously."""
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

            # Include claw prediction data if predictor is enabled
            if claw_predictor is not None and claw_predictor.enabled:
                last_pred = claw_predictor.get_last_prediction()
                if last_pred is not None:
                    state["claw_prediction"] = last_pred.to_dict()
                else:
                    state["claw_prediction"] = {"enabled": True, "detected": False}
            elif claw_predictor is not None:
                state["claw_prediction"] = {"enabled": False}
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
    except Exception as e:
        action_log.add("WS", f"Client error: {e}", "error")
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
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("Telemetry WS error: %s", e)
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
# Visual Pick endpoints — dual-camera object detection + autonomous grasp
# ---------------------------------------------------------------------------


class VisualPickRequest(BaseModel):
    target: str = Field(default="redbull", description="Object to pick: redbull, red, blue, all")
    speed: float = Field(default=0.5, ge=0.1, le=1.0)
    execute: bool = Field(default=False, description="If True, execute immediately after planning")


class VisualPickFromPositionRequest(BaseModel):
    x_mm: float = Field(description="X position in arm-base frame (mm)")
    y_mm: float = Field(description="Y position in arm-base frame (mm)")
    z_mm: float = Field(description="Z position in arm-base frame (mm)")
    label: str = Field(default="redbull")
    speed: float = Field(default=0.5, ge=0.1, le=1.0)
    execute: bool = Field(default=False)


@app.get("/api/pick/status")
async def pick_status():
    """Get visual pick system status."""
    if not _HAS_VISUAL_PICK or pick_executor is None:
        return {"available": False, "error": "Visual pick module not available"}
    status = pick_executor.get_status()
    status["available"] = True
    status["calibrated"] = arm_tracker.calibrator.is_calibrated if arm_tracker else False
    return status


@app.post("/api/pick/detect")
async def pick_detect(req: VisualPickRequest = VisualPickRequest()):
    """Detect objects using dual cameras without planning a grasp.

    Returns detected objects with their 3D positions.
    """
    if not _HAS_VISUAL_PICK or arm_tracker is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )

    # Grab camera snapshots
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

        result = arm_tracker.track(left, right, target_label=req.target, annotate=False)

        objects_data = []
        for obj in result.objects:
            objects_data.append({
                "label": obj.label,
                "position_mm": [round(float(x), 1) for x in obj.position_mm],
                "position_cam_mm": [round(float(x), 1) for x in obj.position_cam_mm],
                "size_mm": list(obj.size_mm),
                "depth_mm": round(obj.depth_mm, 1),
                "confidence": round(obj.confidence, 3),
                "bbox_left": list(obj.bbox_left),
                "centroid_left": list(obj.centroid_left),
            })

        action_log.add(
            "VISION",
            f"Detected {len(result.objects)} '{req.target}' object(s) in {result.elapsed_ms:.0f}ms",
            "info",
        )

        return {
            "ok": True,
            "objects": objects_data,
            "count": len(result.objects),
            "elapsed_ms": result.elapsed_ms,
            "status": result.status,
            "message": result.message,
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/pick/plan")
async def pick_plan(req: VisualPickRequest = VisualPickRequest()):
    """Detect object and plan a full pick trajectory.

    Uses dual cameras to find the target, plans grasp approach, and optionally
    executes the trajectory through the command smoother.
    """
    global _active_task
    if not _HAS_VISUAL_PICK or pick_executor is None or arm_tracker is None or grasp_planner is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )
    if req.execute and not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled for execution"}, status_code=409)

    # Grab camera snapshots
    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        current = _get_current_joints()
        gripper = armState_gripper()

        result = pick_executor.plan_pick(
            left_frame=left,
            right_frame=right,
            arm_tracker=arm_tracker,
            grasp_planner=grasp_planner,
            current_angles_deg=current,
            current_gripper_mm=gripper,
            target_label=req.target,
            workspace_mapper=workspace_mapper,
            collision_preview=collision_preview,
        )

        response_data = {
            "ok": result.success,
            "phase": result.phase.value,
            "message": result.message,
            "elapsed_ms": result.elapsed_ms,
            "detected_count": len(result.detected_objects),
        }

        if result.success and result.trajectory is not None:
            response_data["trajectory"] = {
                "points": len(result.trajectory.points),
                "duration_s": round(result.trajectory.duration, 1),
            }
            response_data["grasp"] = {
                "approach_angles": result.approach_angles_deg,
                "grasp_angles": result.grasp_angles_deg,
                "retreat_angles": result.retreat_angles_deg,
                "gripper_open_mm": result.gripper_open_mm,
                "gripper_close_mm": result.gripper_close_mm,
            }

            if result.target_object is not None:
                response_data["target"] = {
                    "label": result.target_object.label,
                    "position_mm": [round(float(x), 1) for x in result.target_object.position_mm],
                    "depth_mm": round(result.target_object.depth_mm, 1),
                    "confidence": round(result.target_object.confidence, 3),
                }

            action_log.add(
                "PICK",
                f"Pick planned for '{req.target}': {len(result.trajectory.points)} pts, "
                f"{result.trajectory.duration:.1f}s",
                "info",
            )

            # Execute if requested
            if req.execute and smoother and smoother.arm_enabled:
                if _active_task and not _active_task.done():
                    _active_task.cancel()
                _active_task = asyncio.create_task(
                    _execute_trajectory(result.trajectory, f"Visual Pick ({req.target})")
                )
                response_data["executing"] = True
                action_log.add("PICK", f"Executing pick trajectory for '{req.target}'", "info")
        else:
            action_log.add(
                "PICK", f"Pick plan failed for '{req.target}': {result.message}", "warning"
            )

        return response_data

    except Exception as e:
        logger.exception("Pick plan error")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/pick/from-position")
async def pick_from_position(req: VisualPickFromPositionRequest):
    """Plan a pick from a known 3D position (skip camera detection).

    Useful when the object position is already known.
    """
    global _active_task
    if not _HAS_VISUAL_PICK or pick_executor is None or grasp_planner is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )
    if req.execute and not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled for execution"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    obj_pos = np.array([req.x_mm, req.y_mm, req.z_mm])

    result = pick_executor.plan_pick_from_position(
        object_position_mm=obj_pos,
        grasp_planner=grasp_planner,
        current_angles_deg=current,
        current_gripper_mm=gripper,
        object_label=req.label,
    )

    response_data = {
        "ok": result.success,
        "phase": result.phase.value,
        "message": result.message,
        "elapsed_ms": result.elapsed_ms,
    }

    if result.success and result.trajectory is not None:
        response_data["trajectory"] = {
            "points": len(result.trajectory.points),
            "duration_s": round(result.trajectory.duration, 1),
        }
        response_data["grasp"] = {
            "approach_angles": result.approach_angles_deg,
            "grasp_angles": result.grasp_angles_deg,
            "retreat_angles": result.retreat_angles_deg,
            "gripper_open_mm": result.gripper_open_mm,
            "gripper_close_mm": result.gripper_close_mm,
        }
        response_data["target_position_mm"] = [req.x_mm, req.y_mm, req.z_mm]

        action_log.add(
            "PICK",
            f"Pick from position [{req.x_mm:.0f},{req.y_mm:.0f},{req.z_mm:.0f}]: "
            f"{len(result.trajectory.points)} pts",
            "info",
        )

        if req.execute and smoother and smoother.arm_enabled:
            if _active_task and not _active_task.done():
                _active_task.cancel()
            _active_task = asyncio.create_task(
                _execute_trajectory(result.trajectory, f"Pick from position ({req.label})")
            )
            response_data["executing"] = True

    return response_data


@app.post("/api/pick/calibrate-camera-arm")
async def pick_calibrate_camera_arm():
    """Calibrate camera-to-arm transform using current arm pose and visual detection.

    Place a distinctive object (e.g. red marker) at the end-effector,
    then this endpoint detects it in both cameras and computes the
    camera-to-arm translation offset.
    """
    if not _HAS_VISUAL_PICK or arm_tracker is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )

    import httpx, cv2

    try:
        # Get camera frames
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Camera frames failed"}, status_code=502)

        # Detect red marker at EE
        result = arm_tracker.track(left, right, target_label="red", annotate=False)
        if not result.objects:
            return JSONResponse(
                {"ok": False, "error": "No red marker detected at end-effector"},
                status_code=400,
            )

        # Get current EE position from FK
        current = _get_current_joints()
        from src.kinematics.kinematics import D1Kinematics

        kin = D1Kinematics()
        q7 = np.zeros(7)
        q7[:6] = np.deg2rad(current)
        ee_pose = kin.forward_kinematics(q7)
        ee_pos_mm = ee_pose[:3, 3] * 1000  # meters to mm

        # Use the closest detection to EE as the calibration point
        cam_pos = result.objects[0].position_cam_mm
        arm_tracker.calibrate_cam_to_arm_from_known_point(cam_pos, ee_pos_mm)

        action_log.add(
            "CALIBRATION",
            f"Camera-to-arm calibrated: cam={[round(x,1) for x in cam_pos]}, "
            f"arm={[round(x,1) for x in ee_pos_mm]}",
            "info",
        )

        return {
            "ok": True,
            "camera_point_mm": [round(float(x), 1) for x in cam_pos],
            "arm_point_mm": [round(float(x), 1) for x in ee_pos_mm],
            "message": "Camera-to-arm transform updated",
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Vision Task Planning endpoints — look at camera, build a plan
# ---------------------------------------------------------------------------


class VisionPlanRequest(BaseModel):
    instruction: str = Field(
        ..., min_length=1, max_length=500, description="What to do, e.g. 'pick up the red object'"
    )
    camera: int = Field(default=0, ge=0, le=1, description="Camera index to use (0 or 1)")
    execute: bool = Field(default=False, description="Execute the plan immediately if True")


@app.post("/api/vision/plan")
async def vision_plan(req: VisionPlanRequest):
    """Analyze camera feed and build a task plan from an instruction."""
    if not _HAS_VISION_PLANNING or vision_task_planner is None or scene_analyzer is None:
        return JSONResponse(
            {"ok": False, "error": "Vision planning module not available"}, status_code=501
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"http://localhost:8081/snap/{req.camera}")
        if resp.status_code != 200:
            return JSONResponse(
                {"ok": False, "error": f"Camera {req.camera} snapshot failed"}, status_code=502
            )

        import cv2

        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frame"}, status_code=502
            )
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Camera error: {e}"}, status_code=502)

    import time as _time

    scene = scene_analyzer.analyze(frame, timestamp=_time.time())
    action_log.add(
        "VISION",
        f"Scene analyzed: {scene.object_count} objects from camera {req.camera}",
        "info",
    )

    current = _get_current_joints()
    plan = vision_task_planner.plan(req.instruction, scene, current)

    action_log.add(
        "VISION",
        f"Plan: {plan.action.value} -> {'OK' if plan.success else 'FAILED'}: "
        + (plan.error or (plan.task_result.message if plan.task_result else "")),
        "info" if plan.success else "warning",
    )

    global _active_task
    if req.execute and plan.success and plan.trajectory:
        if not (smoother and smoother.arm_enabled):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Arm not enabled — plan created but not executed",
                    "plan": plan.to_dict(),
                    "scene": scene.to_dict(),
                },
                status_code=409,
            )
        if _active_task and not _active_task.done():
            _active_task.cancel()
        _active_task = asyncio.create_task(
            _execute_trajectory(plan.trajectory, f"Vision: {plan.action.value}")
        )
        action_log.add("VISION", f"Executing vision plan: {plan.action.value}", "info")

    return {
        "ok": plan.success,
        "plan": plan.to_dict(),
        "scene": scene.to_dict(),
        "executed": req.execute and plan.success,
    }


@app.post("/api/vision/analyze")
async def vision_analyze(camera: int = 0):
    """Analyze the current camera view and return a scene description."""
    if not _HAS_VISION_PLANNING or scene_analyzer is None:
        return JSONResponse(
            {"ok": False, "error": "Vision planning module not available"}, status_code=501
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"http://localhost:8081/snap/{camera}")
        if resp.status_code != 200:
            return JSONResponse(
                {"ok": False, "error": f"Camera {camera} snapshot failed"}, status_code=502
            )

        import cv2

        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frame"}, status_code=502
            )
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Camera error: {e}"}, status_code=502)

    import time as _time

    scene = scene_analyzer.analyze(frame, timestamp=_time.time())
    action_log.add(
        "VISION",
        f"Scene analysis: {scene.object_count} objects from camera {camera}",
        "info",
    )

    return {"ok": True, "scene": scene.to_dict()}


# ---------------------------------------------------------------------------
# Claw Position Prediction endpoints — visual tracking of end-effector
# ---------------------------------------------------------------------------


@app.post("/api/claw-predict/toggle")
async def claw_predict_toggle():
    """Toggle the claw position predictor on/off."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    enabled = claw_predictor.toggle()
    action_log.add("CLAW_PREDICT", f"Predictor {'enabled' if enabled else 'disabled'}", "info")
    return {"ok": True, "enabled": enabled}


@app.get("/api/claw-predict/status")
async def claw_predict_status():
    """Get claw position predictor status and last prediction."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return {"available": False}
    status = claw_predictor.get_status()
    status["available"] = True
    return status


@app.post("/api/claw-predict/update")
async def claw_predict_update():
    """Trigger a claw position prediction from current camera frames."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    if not claw_predictor.enabled:
        return JSONResponse({"ok": False, "error": "Predictor not enabled"}, status_code=409)

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        import cv2

        cam0 = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        cam1 = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if cam0 is None or cam1 is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        # Sync scale factor from workspace mapper if available
        if workspace_mapper is not None and workspace_mapper.scale_calibrated:
            claw_predictor.set_scale_factor(workspace_mapper._scale_factor)

        # Compute FK position for comparison
        fk_pos = _get_fk_position_mm()

        result = claw_predictor.predict(cam0, cam1, fk_position_mm=fk_pos)
        return {"ok": True, "prediction": result.to_dict()}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class ClawHSVRequest(BaseModel):
    h_lower: int = Field(ge=0, le=180)
    s_lower: int = Field(ge=0, le=255)
    v_lower: int = Field(ge=0, le=255)
    h_upper: int = Field(ge=0, le=180)
    s_upper: int = Field(ge=0, le=255)
    v_upper: int = Field(ge=0, le=255)


@app.post("/api/claw-predict/set-hsv")
async def claw_predict_set_hsv(req: ClawHSVRequest):
    """Set custom HSV color range for claw detection."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.set_hsv_range(
        (req.h_lower, req.s_lower, req.v_lower),
        (req.h_upper, req.s_upper, req.v_upper),
    )
    action_log.add(
        "CLAW_PREDICT",
        f"HSV range set: [{req.h_lower},{req.s_lower},{req.v_lower}]-[{req.h_upper},{req.s_upper},{req.v_upper}]",
        "info",
    )
    return {"ok": True}


class ClawROIRequest(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


@app.post("/api/claw-predict/set-roi")
async def claw_predict_set_roi(req: ClawROIRequest):
    """Set a region of interest for claw detection."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.set_detection_roi(req.x, req.y, req.w, req.h)
    action_log.add("CLAW_PREDICT", f"ROI set: ({req.x},{req.y}) {req.w}x{req.h}", "info")
    return {"ok": True}


@app.post("/api/claw-predict/clear-roi")
async def claw_predict_clear_roi():
    """Clear the detection ROI."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.clear_detection_roi()
    action_log.add("CLAW_PREDICT", "ROI cleared", "info")
    return {"ok": True}


def _get_fk_position_mm() -> Optional[list[float]]:
    """Compute the current end-effector position from forward kinematics."""
    try:
        from src.kinematics import D1Kinematics

        kin = D1Kinematics()
        joints_6 = _get_current_joints()
        # FK expects 7 joints (6 arm + gripper); gripper angle is 0
        joints_7 = np.append(joints_6, 0.0)
        joint_radians = np.deg2rad(joints_7)
        T = kin.forward_kinematics(joint_radians)
        # Position in meters from FK, convert to mm
        pos_mm = T[:3, 3] * 1000.0
        return list(float(v) for v in pos_mm)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Viz Calibration endpoints
# ---------------------------------------------------------------------------

try:
    from src.vision.viz_calibrator import (
        run_calibration as _run_viz_calibration,
        load_calibration as _load_viz_calibration,
        OUTPUT_PATH as _VIZ_CALIB_PATH,
    )
    _HAS_VIZ_CALIB = True
except ImportError:
    _HAS_VIZ_CALIB = False

_viz_calib_running = False


@app.get("/api/viz/calibration")
async def get_viz_calibration():
    """Return saved visualization calibration data."""
    if not _HAS_VIZ_CALIB:
        return JSONResponse({"ok": False, "error": "Viz calibrator not available"}, status_code=501)
    data = _load_viz_calibration()
    if data is None:
        return JSONResponse({"ok": False, "error": "No calibration data found"}, status_code=404)
    return {"ok": True, **data}


@app.post("/api/viz/calibrate")
async def run_viz_calibration():
    """Run camera-based visualization calibration (moves the arm!)."""
    global _viz_calib_running
    if not _HAS_VIZ_CALIB:
        return JSONResponse({"ok": False, "error": "Viz calibrator not available"}, status_code=501)
    if _viz_calib_running:
        return JSONResponse({"ok": False, "error": "Calibration already in progress"}, status_code=409)
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    _viz_calib_running = True
    try:
        result = await _run_viz_calibration()
        action_log.add(
            "VIZ_CALIB",
            f"{'OK' if result.success else 'FAILED'} — {result.n_observations} obs, residual={result.residual}",
            "info" if result.success else "error",
        )
        import math
        def _sanitize(v):
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                return None
            return v
        return {
            "ok": result.success,
            "links_mm": {k: _sanitize(v) for k, v in result.links_mm.items()},
            "joint_viz_offsets": [_sanitize(v) for v in result.joint_viz_offsets],
            "residual": _sanitize(result.residual),
            "n_observations": result.n_observations,
            "message": result.message,
        }
    except Exception as e:
        action_log.add("VIZ_CALIB", f"Error: {e}", "error")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        _viz_calib_running = False


# ---------------------------------------------------------------------------
# Telemetry viewer page route
# ---------------------------------------------------------------------------


@app.get("/telemetry")
async def telemetry_page():
    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "static" / "telemetry.html")


# ---------------------------------------------------------------------------
# Static files — versioned UIs all pointing to the same server
# /v1/ → V1 stable base, /v2/ → V2 Cartesian controls, / → V1 (default)
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

v1_dir = static_dir / "v1"
v2_dir = static_dir / "v2"
if v1_dir.is_dir():
    app.mount("/v1", StaticFiles(directory=str(v1_dir), html=True), name="static-v1")
if v2_dir.is_dir():
    app.mount("/v2", StaticFiles(directory=str(v2_dir), html=True), name="static-v2")
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Starting th3cl4w web panel on %s:%d (simulate=%s)", args.host, args.port, args.simulate
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
