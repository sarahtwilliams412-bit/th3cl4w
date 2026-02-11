"""
Control Plane Server — Manual arm control service.

This is a standalone FastAPI app for direct control of the Unitree D1 arm.
It provides:
  - Joint control (sliders, set-all-joints, home positions)
  - Gripper control
  - Power on/off, enable/disable
  - Emergency stop
  - Real-time state broadcasting via WebSocket
  - State broadcasting to Redis message bus

Port: 8090 (configurable via CONTROL_PLANE_PORT env var)

Usage:
    python -m services.control_plane.server
    python -m services.control_plane.server --simulate
    python -m services.control_plane.server --interface eno1
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.arm_model.joint_service import (
    ALL_JOINTS,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
    MAX_STEP_DEG,
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    all_joints_dict,
    joint_dict,
    joint_limits_deg_array,
    HOME_POSITION,
    READY_POSITION,
)
from shared.safety.safety_monitor import SafetyMonitor
from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.control_plane")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Control Plane — Manual Arm Control")
parser.add_argument("--simulate", action="store_true", help="Run with simulated arm")
parser.add_argument("--interface", default="eno1", help="Network interface for DDS")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args(["--simulate"])

setup_logging(server_name="control_plane", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Joint limits
# ---------------------------------------------------------------------------

_UNIFIED_JOINT_LIMITS_DEG = joint_limits_deg_array()
JOINT_LIMITS_DEG = {
    i: (float(_UNIFIED_JOINT_LIMITS_DEG[i, 0]), float(_UNIFIED_JOINT_LIMITS_DEG[i, 1]))
    for i in range(6)
}
GRIPPER_RANGE = (GRIPPER_MIN_MM, GRIPPER_MAX_MM)

# ---------------------------------------------------------------------------
# Action log
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
# Global state
# ---------------------------------------------------------------------------

arm: Any = None
smoother: Any = None
safety_monitor: Optional[SafetyMonitor] = None
_sim_mode: bool = False

# Message bus publisher (optional, connects to Redis)
_publisher: Any = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arm, smoother, safety_monitor, _sim_mode, _publisher

    # Initialize arm connection
    if args.simulate or ServiceConfig.SIMULATE:
        from services.control_plane.hardware.simulated_arm import SimulatedArm
        arm = SimulatedArm()
        arm.start_feedback_loop(rate_hz=10.0)
        _sim_mode = True
        action_log.add("SYSTEM", "Simulated arm initialized (SIM mode)")
    else:
        try:
            os.environ.setdefault("CYCLONEDDS_URI", f"<General><NetworkInterfaceAddress>{args.interface}</NetworkInterfaceAddress></General>")
            from services.control_plane.hardware.d1_dds_connection import D1DDSConnection
            arm = D1DDSConnection()
            arm.start_feedback_loop(rate_hz=200.0)
            _sim_mode = False
            action_log.add("SYSTEM", f"DDS arm connection on {args.interface}")
        except Exception as e:
            logger.error("DDS connection failed: %s — falling back to simulation", e)
            from services.control_plane.hardware.simulated_arm import SimulatedArm
            arm = SimulatedArm()
            arm.start_feedback_loop(rate_hz=10.0)
            _sim_mode = True
            action_log.add("SYSTEM", f"DDS failed ({e}), using simulated arm", "warning")

    # Initialize safety monitor
    safety_monitor = SafetyMonitor()

    # Initialize command smoother
    try:
        from services.control_plane.control.command_smoother import CommandSmoother
        smoother = CommandSmoother(arm, safety_monitor)
    except ImportError:
        logger.warning("CommandSmoother not available")

    # Connect to message bus (non-blocking)
    try:
        from shared.bus.publisher import EventPublisher
        _publisher = EventPublisher()
        await _publisher.connect()
    except Exception as e:
        logger.info("Message bus not available: %s (operating standalone)", e)

    action_log.add("SYSTEM", "Control Plane ready")

    yield

    # Shutdown
    action_log.add("SYSTEM", "Control Plane shutting down")
    if _publisher:
        await _publisher.close()
    if arm and hasattr(arm, "stop_feedback_loop"):
        arm.stop_feedback_loop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Control Plane",
    description="Manual arm control service for the Unitree D1",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (control panel UI)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "control_plane",
        "sim_mode": _sim_mode,
        "arm_connected": arm is not None,
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class JointMoveRequest(BaseModel):
    joint_id: int = Field(ge=0, le=5, description="Joint ID (0-5)")
    angle_deg: float = Field(description="Target angle in degrees")


class AllJointsMoveRequest(BaseModel):
    angles_deg: List[float] = Field(min_length=6, max_length=6, description="6 joint angles in degrees")


class GripperRequest(BaseModel):
    position_mm: float = Field(ge=0.0, le=65.0, description="Gripper opening in mm")


# ---------------------------------------------------------------------------
# State API
# ---------------------------------------------------------------------------


@app.get("/api/state")
async def get_state():
    """Get current arm state."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)

    state = arm.get_state()
    if state is None:
        return JSONResponse({"error": "No state available"}, status_code=503)

    joints_deg = [float(np.degrees(state.joint_positions[i])) for i in range(NUM_ARM_JOINTS)]
    gripper_mm = float(state.gripper_position * GRIPPER_MAX_MM)

    return {
        "joints": joints_deg,
        "gripper_mm": gripper_mm,
        "gripper_normalized": float(state.gripper_position),
        "velocities": [float(v) for v in state.joint_velocities],
        "torques": [float(t) for t in state.joint_torques],
        "timestamp": state.timestamp,
        "sim_mode": _sim_mode,
        "connected": True,
        "powered": getattr(arm, "powered", True),
        "enabled": getattr(arm, "enabled", True),
        "estop": safety_monitor.estop_active if safety_monitor else False,
    }


@app.get("/api/joints")
async def get_joints():
    """Get joint configuration and current angles."""
    state = arm.get_state() if arm else None
    joints_deg = (
        [float(np.degrees(state.joint_positions[i])) for i in range(NUM_ARM_JOINTS)]
        if state
        else [0.0] * NUM_ARM_JOINTS
    )

    configs = all_joints_dict()
    for i, cfg in enumerate(configs):
        if i < len(joints_deg):
            cfg["current_deg"] = joints_deg[i]
    return {"joints": configs}


@app.get("/api/joint/{joint_id}")
async def get_joint(joint_id: int):
    """Get single joint info."""
    if joint_id < 0 or joint_id > 6:
        return JSONResponse({"error": f"Invalid joint ID: {joint_id}"}, status_code=400)
    return joint_dict(joint_id)


# ---------------------------------------------------------------------------
# Joint control API
# ---------------------------------------------------------------------------


@app.post("/api/command/set-joint")
async def set_joint(req: JointMoveRequest):
    """Move a single joint to a target angle."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)

    jid = req.joint_id
    angle = req.angle_deg

    # Clamp to limits
    lo, hi = JOINT_LIMITS_DEG.get(jid, (-135.0, 135.0))
    clamped = max(lo, min(hi, angle))
    if clamped != angle:
        action_log.add("CLAMP", f"J{jid}: {angle:.1f} -> {clamped:.1f} (limits [{lo}, {hi}])", "warning")

    if smoother:
        smoother.set_target_joint(jid, clamped)
    else:
        # Direct command fallback
        state = arm.get_state()
        if state:
            positions = state.joint_positions.copy()
            positions[jid] = np.radians(clamped)
            from shared.arm_model.d1_state import D1Command
            cmd = D1Command(mode=1, joint_positions=positions)
            arm.send_command(cmd)

    action_log.add("JOINT", f"J{jid} -> {clamped:.1f}°")
    return {"ok": True, "joint_id": jid, "target_deg": clamped}


@app.post("/api/command/set-all-joints")
async def set_all_joints(req: AllJointsMoveRequest):
    """Move all joints simultaneously."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)

    clamped = []
    for i, angle in enumerate(req.angles_deg):
        lo, hi = JOINT_LIMITS_DEG.get(i, (-135.0, 135.0))
        clamped.append(max(lo, min(hi, angle)))

    if smoother:
        for i, angle in enumerate(clamped):
            smoother.set_target_joint(i, angle)
    else:
        positions = np.radians(clamped + [0.0])  # pad for gripper
        from shared.arm_model.d1_state import D1Command
        cmd = D1Command(mode=1, joint_positions=positions)
        arm.send_command(cmd)

    action_log.add("JOINTS", f"All -> {[f'{a:.1f}' for a in clamped]}")
    return {"ok": True, "target_deg": clamped}


@app.post("/api/command/set-gripper")
async def set_gripper(req: GripperRequest):
    """Set gripper opening."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)

    mm = max(GRIPPER_RANGE[0], min(GRIPPER_RANGE[1], req.position_mm))
    normalized = mm / GRIPPER_MAX_MM

    if smoother:
        smoother.set_target_gripper(mm)
    else:
        from shared.arm_model.d1_state import D1Command
        cmd = D1Command(mode=1, gripper_position=normalized)
        arm.send_command(cmd)

    action_log.add("GRIPPER", f"-> {mm:.1f}mm")
    return {"ok": True, "gripper_mm": mm}


# ---------------------------------------------------------------------------
# Power / Enable / E-Stop
# ---------------------------------------------------------------------------


@app.post("/api/command/power-on")
async def power_on():
    """Power on the arm."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)
    if hasattr(arm, "power_on"):
        arm.power_on()
    action_log.add("POWER", "Power ON")
    return {"ok": True}


@app.post("/api/command/power-off")
async def power_off():
    """Power off the arm."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)
    if hasattr(arm, "power_off"):
        arm.power_off()
    action_log.add("POWER", "Power OFF")
    return {"ok": True}


@app.post("/api/command/enable")
async def enable():
    """Enable arm for motion."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)
    if hasattr(arm, "enable"):
        arm.enable()
    action_log.add("ENABLE", "Arm enabled")
    return {"ok": True}


@app.post("/api/command/disable")
async def disable():
    """Disable arm motion."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)
    if hasattr(arm, "disable"):
        arm.disable()
    action_log.add("DISABLE", "Arm disabled")
    return {"ok": True}


@app.post("/api/command/estop")
async def estop():
    """Trigger emergency stop."""
    if safety_monitor:
        safety_monitor.trigger_estop("manual")
    action_log.add("E-STOP", "Emergency stop TRIGGERED", "error")
    return {"ok": True, "estop": True}


@app.post("/api/command/estop-reset")
async def estop_reset():
    """Reset emergency stop."""
    if safety_monitor:
        safety_monitor.reset_estop()
    action_log.add("E-STOP", "Emergency stop RESET")
    return {"ok": True, "estop": False}


@app.post("/api/command/home")
async def go_home():
    """Move arm to home position (all zeros)."""
    if arm is None:
        return JSONResponse({"error": "Arm not initialized"}, status_code=503)

    home = HOME_POSITION.tolist()
    if smoother:
        for i, angle in enumerate(home):
            smoother.set_target_joint(i, angle)
    else:
        positions = np.radians(home + [0.0])
        from shared.arm_model.d1_state import D1Command
        cmd = D1Command(mode=1, joint_positions=positions)
        arm.send_command(cmd)

    action_log.add("HOME", "Moving to home position")
    return {"ok": True, "target_deg": home}


# ---------------------------------------------------------------------------
# Safety status
# ---------------------------------------------------------------------------


@app.get("/api/safety")
async def safety_status():
    """Get safety system status."""
    return {
        "estop_active": safety_monitor.estop_active if safety_monitor else False,
        "sim_mode": _sim_mode,
        "joint_limits": JOINT_LIMITS_DEG,
        "gripper_range": GRIPPER_RANGE,
    }


# ---------------------------------------------------------------------------
# Simulation toggle
# ---------------------------------------------------------------------------


@app.get("/api/sim_status")
async def sim_status():
    """Get simulation mode status."""
    return {"sim_mode": _sim_mode}


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------


@app.get("/api/actions")
async def get_actions(n: int = 50):
    """Get recent action log entries."""
    return {"actions": action_log.last(n)}


# ---------------------------------------------------------------------------
# WebSocket state stream
# ---------------------------------------------------------------------------


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    """Real-time arm state WebSocket (10Hz)."""
    await websocket.accept()
    try:
        while True:
            if arm is None:
                await asyncio.sleep(0.5)
                continue

            state = arm.get_state()
            if state is None:
                await asyncio.sleep(0.1)
                continue

            joints_deg = [float(np.degrees(state.joint_positions[i])) for i in range(NUM_ARM_JOINTS)]
            gripper_mm = float(state.gripper_position * GRIPPER_MAX_MM)

            payload = {
                "type": "state",
                "joints": joints_deg,
                "gripper_mm": gripper_mm,
                "velocities": [float(v) for v in state.joint_velocities],
                "torques": [float(t) for t in state.joint_torques],
                "timestamp": state.timestamp,
                "sim_mode": _sim_mode,
                "connected": True,
                "powered": getattr(arm, "powered", True),
                "enabled": getattr(arm, "enabled", True),
                "estop": safety_monitor.estop_active if safety_monitor else False,
            }

            await websocket.send_json(payload)

            # Also publish to message bus if connected
            if _publisher and _publisher.is_connected:
                from shared.messages.arm_state import ArmStateMessage
                msg = ArmStateMessage(
                    joint_angles_deg=joints_deg,
                    joint_velocities=[float(v) for v in state.joint_velocities],
                    joint_torques=[float(t) for t in state.joint_torques],
                    gripper_mm=gripper_mm,
                    powered=getattr(arm, "powered", True),
                    enabled=getattr(arm, "enabled", True),
                    estop=safety_monitor.estop_active if safety_monitor else False,
                    sim_mode=_sim_mode,
                    connected=True,
                    timestamp=state.timestamp,
                )
                await _publisher.publish("arm.state", msg)

            await asyncio.sleep(0.1)  # 10Hz

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.CONTROL_PLANE_PORT
    logger.info("Starting Control Plane on port %d (sim=%s)", port, _sim_mode or args.simulate)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
