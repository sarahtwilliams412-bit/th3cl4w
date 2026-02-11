"""
Local Model Server — Near-field workspace model from the arm-mounted camera.

Maintains a high-resolution model of the immediate workspace area using the
arm-mounted camera. Provides workspace mapping, surface detection, and
real-time updates as the arm moves.

Port: 8083 (configurable via LOCAL_MODEL_PORT env var)

Usage:
    python -m services.local_model.server
    python -m services.local_model.server --debug
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

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.local_model")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Local Model — Near-field Workspace Model")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="local_model", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# In-memory local model state
# ---------------------------------------------------------------------------

_local_state: dict[str, Any] = {
    "arm_camera_active": False,
    "surface_detected": False,
    "surface_height_mm": 0.0,
    "near_objects": [],
    "workspace_bounds_mm": {
        "min": [-200.0, -200.0, 0.0],
        "max": [200.0, 200.0, 150.0],
    },
    "last_scan_time": 0.0,
    "scan_count": 0,
}

_workspace_map: dict[str, Any] = {
    "grid_resolution_mm": 10.0,
    "grid_size": [40, 40],
    "occupancy": [],  # Placeholder — would be a 2D grid
    "surface_normals": [],
    "last_updated": 0.0,
}

_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None


async def _broadcast_local_update(event: dict):
    """Send a local model update to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


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
        await _subscriber.subscribe(Topics.ARM_STATE, _on_arm_state)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Local Model ready")
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Local Model shut down")


async def _on_arm_state(topic: str, data: dict):
    """Handle arm state updates — triggers local model refresh when arm moves."""
    # In a real implementation, this would trigger a re-scan from the arm camera
    pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Local Model",
    description="Near-field workspace model from arm-mounted camera",
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

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "local_model",
        "arm_camera_active": _local_state["arm_camera_active"],
        "scan_count": _local_state["scan_count"],
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Local model API
# ---------------------------------------------------------------------------


@app.get("/api/local/state")
async def get_local_state():
    """Get the current local model state."""
    return {
        **_local_state,
        "timestamp": time.time(),
    }


@app.get("/api/local/workspace")
async def get_workspace():
    """Get the workspace occupancy map."""
    return {
        **_workspace_map,
        "timestamp": time.time(),
    }


@app.post("/api/local/scan")
async def trigger_arm_scan():
    """Trigger a scan using the arm-mounted camera."""
    scan_id = str(uuid.uuid4())[:8]
    _local_state["scan_count"] += 1
    _local_state["last_scan_time"] = time.time()

    logger.info("Arm camera scan triggered (scan_id=%s, total_scans=%d)",
                scan_id, _local_state["scan_count"])

    # Publish scan event
    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.LOCAL_UPDATED, {
            "event": "scan_triggered",
            "scan_id": scan_id,
            "timestamp": time.time(),
        })

    await _broadcast_local_update({
        "type": "scan_triggered",
        "scan_id": scan_id,
        "timestamp": time.time(),
    })

    return {
        "scan_id": scan_id,
        "status": "initiated",
        "message": "Arm camera scan initiated — results will arrive via local model updates",
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time local model updates
# ---------------------------------------------------------------------------


@app.websocket("/ws/local")
async def ws_local(websocket: WebSocket):
    """Real-time local model updates via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        # Send initial snapshot
        await websocket.send_json({
            "type": "snapshot",
            "state": _local_state,
            "workspace": _workspace_map,
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
                    "timestamp": time.time(),
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.LOCAL_MODEL_PORT
    logger.info("Starting Local Model on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
