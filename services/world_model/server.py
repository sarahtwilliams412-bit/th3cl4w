"""
World Model Server — Maintains a unified 3D model of all tracked objects.

Aggregates detections from Object ID, fuses with depth/position data, and
provides a single source of truth for what objects exist, where they are,
and whether they are reachable by the arm.

Port: 8082 (configurable via WORLD_MODEL_PORT env var)

Usage:
    python -m services.world_model.server
    python -m services.world_model.server --debug
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

logger = logging.getLogger("th3cl4w.world_model")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w World Model — 3D Object Tracker")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="world_model", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# In-memory world state
# ---------------------------------------------------------------------------

_objects: dict[str, dict] = {}
_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None

# Seed a few placeholder objects so the API is demonstrable
_SEED_OBJECTS = {
    "obj_001": {
        "id": "obj_001",
        "label": "red ball",
        "position_mm": [200.0, 100.0, 50.0],
        "dimensions_mm": [40.0, 40.0, 40.0],
        "confidence": 0.92,
        "category": "target",
        "reach_status": "reachable",
        "distance_mm": 245.0,
        "observation_count": 15,
        "stable": True,
        "first_seen": 0.0,
        "last_seen": 0.0,
    },
    "obj_002": {
        "id": "obj_002",
        "label": "blue cup",
        "position_mm": [350.0, -50.0, 30.0],
        "dimensions_mm": [70.0, 70.0, 100.0],
        "confidence": 0.87,
        "category": "target",
        "reach_status": "reachable",
        "distance_mm": 355.0,
        "observation_count": 8,
        "stable": True,
        "first_seen": 0.0,
        "last_seen": 0.0,
    },
    "obj_003": {
        "id": "obj_003",
        "label": "cardboard box",
        "position_mm": [800.0, 200.0, 100.0],
        "dimensions_mm": [300.0, 200.0, 150.0],
        "confidence": 0.78,
        "category": "obstacle",
        "reach_status": "out_of_range",
        "distance_mm": 825.0,
        "observation_count": 3,
        "stable": True,
        "first_seen": 0.0,
        "last_seen": 0.0,
    },
}


async def _broadcast_world_update(event: dict):
    """Send a world update to all connected WebSocket clients."""
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
    global _publisher, _subscriber, _objects

    now = time.time()
    for obj in _SEED_OBJECTS.values():
        obj["first_seen"] = now
        obj["last_seen"] = now
    _objects = dict(_SEED_OBJECTS)

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
        # Subscribe to object detection events to build world model
        from shared.bus.topics import Topics
        await _subscriber.subscribe(Topics.OBJECTS_DETECTED, _on_objects_detected)
        await _subscriber.subscribe(Topics.OBJECTS_UPDATED, _on_objects_updated)
        await _subscriber.subscribe(Topics.OBJECTS_REMOVED, _on_objects_removed)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("World Model ready — tracking %d objects", len(_objects))
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("World Model shut down")


async def _on_objects_detected(topic: str, data: dict):
    """Handle new object detections from Object ID service."""
    objects = data.get("objects", [data]) if isinstance(data, dict) else [data]
    for obj in objects:
        obj_id = obj.get("id", str(uuid.uuid4())[:8])
        _objects[obj_id] = obj
    await _broadcast_world_update({"type": "objects_detected", "objects": objects})


async def _on_objects_updated(topic: str, data: dict):
    """Handle object position/state updates."""
    obj_id = data.get("id")
    if obj_id and obj_id in _objects:
        _objects[obj_id].update(data)
    await _broadcast_world_update({"type": "objects_updated", "data": data})


async def _on_objects_removed(topic: str, data: dict):
    """Handle object removal."""
    obj_id = data.get("id")
    if obj_id and obj_id in _objects:
        del _objects[obj_id]
    await _broadcast_world_update({"type": "objects_removed", "id": obj_id})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w World Model",
    description="Unified 3D object tracker — single source of truth for the workspace",
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
        "service": "world_model",
        "tracked_objects": len(_objects),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Object API
# ---------------------------------------------------------------------------


@app.get("/api/objects")
async def list_objects():
    """List all tracked objects in the world model."""
    return {
        "objects": list(_objects.values()),
        "count": len(_objects),
        "timestamp": time.time(),
    }


@app.get("/api/objects/reachable")
async def list_reachable_objects():
    """List objects classified as reachable by the arm."""
    reachable = [
        obj for obj in _objects.values()
        if obj.get("reach_status") == "reachable"
    ]
    return {
        "objects": reachable,
        "count": len(reachable),
        "timestamp": time.time(),
    }


@app.get("/api/object/{object_id}")
async def get_object(object_id: str):
    """Get detailed info for a single tracked object."""
    obj = _objects.get(object_id)
    if obj is None:
        return JSONResponse(
            {"error": f"Object '{object_id}' not found"},
            status_code=404,
        )
    return obj


@app.post("/api/scan")
async def trigger_scan():
    """Trigger a full world scan — requests all cameras to capture and re-detect."""
    scan_id = str(uuid.uuid4())[:8]
    logger.info("Full world scan triggered (scan_id=%s)", scan_id)

    # Publish scan request to message bus
    if _publisher and _publisher.is_connected:
        await _publisher.publish("world.scan_requested", {
            "scan_id": scan_id,
            "timestamp": time.time(),
        })

    return {
        "scan_id": scan_id,
        "status": "initiated",
        "message": "Full scan requested — results will arrive via world updates",
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time world updates
# ---------------------------------------------------------------------------


@app.websocket("/ws/world")
async def ws_world(websocket: WebSocket):
    """Real-time world model updates via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        # Send initial snapshot
        await websocket.send_json({
            "type": "snapshot",
            "objects": list(_objects.values()),
            "timestamp": time.time(),
        })
        # Keep connection alive and relay updates
        while True:
            # Listen for client messages (e.g., filter requests)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Client can send ping or filter commands
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "tracked_objects": len(_objects),
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
    port = args.port or ServiceConfig.WORLD_MODEL_PORT
    logger.info("Starting World Model on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
