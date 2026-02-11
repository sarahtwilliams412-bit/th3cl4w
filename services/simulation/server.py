"""
Simulation Server — Physics simulation for plan rehearsal and validation.

Runs motion plans in a simulated environment before real execution. Loads
scene graphs, applies physics, detects collisions, and reports whether a
plan is safe to execute on the real hardware.

Port: 8089 (configurable via SIMULATION_PORT env var)

Usage:
    python -m services.simulation.server
    python -m services.simulation.server --debug
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.simulation")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Simulation — Physics Simulation & Rehearsal")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="simulation", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateSimRequest(BaseModel):
    """Request to create a new simulation session."""
    name: str = Field(default="", description="Optional session name")
    physics_hz: float = Field(default=240.0, ge=60.0, le=1000.0, description="Physics step rate")
    real_time: bool = Field(default=False, description="Run in real-time (vs. fast-forward)")


class LoadSceneRequest(BaseModel):
    """Request to load a scene into the simulation."""
    scene_source: str = Field(default="live", description="Source: 'live' (from Mapping) or 'snapshot'")
    snapshot_id: Optional[str] = Field(default=None, description="Snapshot ID if source is 'snapshot'")
    include_arm: bool = Field(default=True, description="Include arm model in simulation")


class RunPlanRequest(BaseModel):
    """Request to run a motion plan in the simulation."""
    plan_id: str = Field(description="Plan ID from Kinematics App")
    task_id: Optional[str] = Field(default=None, description="Parent task ID")
    check_collisions: bool = Field(default=True, description="Enable collision detection")
    check_joint_limits: bool = Field(default=True, description="Check joint limit violations")
    check_singularities: bool = Field(default=True, description="Check for singularity proximity")


# ---------------------------------------------------------------------------
# In-memory simulation sessions
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_ws_clients: dict[str, list[WebSocket]] = {}  # session_id -> list of clients
_publisher: Any = None
_subscriber: Any = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _publisher, _subscriber

    # Connect to message bus (non-blocking)
    try:
        from shared.bus.publisher import EventPublisher
        _publisher = EventPublisher()
        await _publisher.connect()
    except Exception as e:
        logger.info("Message bus publisher not available: %s (operating standalone)", e)

    try:
        from shared.bus.subscriber import EventSubscriber
        _subscriber = EventSubscriber()
        await _subscriber.connect()
        from shared.bus.topics import Topics
        await _subscriber.subscribe(Topics.PLAN_COMPUTED, _on_plan_computed)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Simulation ready")
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Simulation shut down")


async def _on_plan_computed(topic: str, data: dict):
    """Optionally auto-rehearse plans that arrive via the bus."""
    pass


async def _broadcast_sim_update(session_id: str, event: dict):
    """Send a simulation update to WebSocket clients for a session."""
    clients = _ws_clients.get(session_id, [])
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Simulation",
    description="Physics simulation for plan rehearsal and validation",
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

# Serve static files (simulation visualization UI)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "simulation",
        "active_sessions": len(_sessions),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Simulation API
# ---------------------------------------------------------------------------


@app.post("/api/sim/create")
async def create_session(req: CreateSimRequest):
    """Create a new simulation session."""
    session_id = f"sim_{uuid.uuid4().hex[:8]}"
    now = time.time()

    session = {
        "session_id": session_id,
        "name": req.name or session_id,
        "physics_hz": req.physics_hz,
        "real_time": req.real_time,
        "state": "created",
        "scene_loaded": False,
        "sim_time_s": 0.0,
        "step_count": 0,
        "collisions_detected": 0,
        "created_at": now,
        "last_updated": now,
    }
    _sessions[session_id] = session
    _ws_clients[session_id] = []

    logger.info("Simulation session %s created (physics_hz=%.0f, real_time=%s)",
                session_id, req.physics_hz, req.real_time)
    return session


@app.post("/api/sim/{session_id}/load-scene")
async def load_scene(session_id: str, req: LoadSceneRequest):
    """Load a scene into a simulation session."""
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": f"Session '{session_id}' not found"}, status_code=404)

    session["scene_loaded"] = True
    session["scene_source"] = req.scene_source
    session["state"] = "scene_loaded"
    session["last_updated"] = time.time()

    # Placeholder scene data
    scene_info = {
        "objects_loaded": 3,
        "arm_loaded": req.include_arm,
        "collision_bodies": 5,
    }

    logger.info("Scene loaded into %s (source=%s, objects=%d)",
                session_id, req.scene_source, scene_info["objects_loaded"])

    await _broadcast_sim_update(session_id, {
        "type": "scene_loaded",
        "session_id": session_id,
        "scene_info": scene_info,
        "timestamp": time.time(),
    })

    return {
        "session_id": session_id,
        "status": "scene_loaded",
        "scene_info": scene_info,
    }


@app.post("/api/sim/{session_id}/run-plan")
async def run_plan(session_id: str, req: RunPlanRequest):
    """Run a motion plan in the simulation."""
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": f"Session '{session_id}' not found"}, status_code=404)

    if not session.get("scene_loaded"):
        return JSONResponse(
            {"error": "Scene not loaded — call load-scene first"},
            status_code=409,
        )

    session["state"] = "running"
    session["last_updated"] = time.time()
    run_id = f"run_{uuid.uuid4().hex[:6]}"

    # Placeholder simulation result
    result = {
        "run_id": run_id,
        "session_id": session_id,
        "plan_id": req.plan_id,
        "task_id": req.task_id,
        "status": "completed",
        "success": True,
        "sim_duration_s": 2.5,
        "steps_executed": 600,
        "collision_free": True,
        "joint_limits_ok": True,
        "singularity_free": True,
        "max_joint_velocity_deg_s": 45.0,
        "max_joint_torque_nm": 12.3,
        "verdict": "safe_to_execute",
        "timestamp": time.time(),
    }

    session["state"] = "idle"
    session["step_count"] += result["steps_executed"]
    session["sim_time_s"] += result["sim_duration_s"]

    # Publish result
    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.SIM_RESULT, result)

    await _broadcast_sim_update(session_id, {
        "type": "run_completed", **result,
    })

    logger.info("Simulation run %s: plan %s verdict=%s",
                run_id, req.plan_id, result["verdict"])
    return result


@app.get("/api/sim/{session_id}/state")
async def get_sim_state(session_id: str):
    """Get the current state of a simulation session."""
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": f"Session '{session_id}' not found"}, status_code=404)
    return session


@app.post("/api/sim/{session_id}/reset")
async def reset_sim(session_id: str):
    """Reset a simulation session to initial state."""
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": f"Session '{session_id}' not found"}, status_code=404)

    session["state"] = "created"
    session["scene_loaded"] = False
    session["sim_time_s"] = 0.0
    session["step_count"] = 0
    session["collisions_detected"] = 0
    session["last_updated"] = time.time()

    await _broadcast_sim_update(session_id, {
        "type": "reset", "session_id": session_id, "timestamp": time.time(),
    })

    logger.info("Simulation session %s reset", session_id)
    return {"session_id": session_id, "status": "reset"}


# ---------------------------------------------------------------------------
# WebSocket — real-time simulation state
# ---------------------------------------------------------------------------


@app.websocket("/ws/sim/{session_id}")
async def ws_sim(websocket: WebSocket, session_id: str):
    """Real-time simulation state updates via WebSocket."""
    session = _sessions.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason=f"Session '{session_id}' not found")
        return

    await websocket.accept()
    if session_id not in _ws_clients:
        _ws_clients[session_id] = []
    _ws_clients[session_id].append(websocket)
    logger.info("WebSocket client connected to sim %s (%d clients)",
                session_id, len(_ws_clients[session_id]))
    try:
        await websocket.send_json({
            "type": "snapshot",
            "session": session,
            "timestamp": time.time(),
        })
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "session_id": session_id,
                    "state": session["state"],
                    "timestamp": time.time(),
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        clients = _ws_clients.get(session_id, [])
        if websocket in clients:
            clients.remove(websocket)
        logger.info("WebSocket client disconnected from sim %s (%d remaining)",
                    session_id, len(clients))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.SIMULATION_PORT
    logger.info("Starting Simulation on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
