"""Location Server — persistent spatial awareness for the D1 arm.

Separate FastAPI app on port 8082. Continuously monitors cameras,
detects objects, and maintains a live world model of everything
within the arm's reach.

Endpoints:
    GET  /api/location/objects       — all tracked objects
    GET  /api/location/reachable     — only reachable objects
    GET  /api/location/object/{id}   — single object detail
    POST /api/location/scan          — trigger immediate full scan
    GET  /api/location/status        — service health
    WS   /ws/location                — real-time object updates
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.location.world_model import LocationWorldModel
from src.location.tracker import ObjectTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("th3cl4w.location_server")

# --- Global state ---
world_model = LocationWorldModel()
tracker = ObjectTracker(world_model)
ws_clients: set[WebSocket] = set()
_start_time = time.time()


def _on_objects_updated(changed):
    """Broadcast updates to WebSocket clients."""
    if not ws_clients:
        return
    data = json.dumps({
        "type": "update",
        "objects": [o.to_dict() for o in changed],
        "timestamp": time.time(),
    })
    # Schedule broadcast in the event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast(data))
    except RuntimeError:
        pass


async def _broadcast(data: str):
    """Send data to all connected WebSocket clients."""
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


world_model.register_callback(_on_objects_updated)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start tracker on startup, stop on shutdown."""
    logger.info("Location server starting...")
    await tracker.start()
    yield
    logger.info("Location server shutting down...")
    await tracker.stop()


app = FastAPI(title="th3cl4w Location Server", lifespan=lifespan)


# --- REST endpoints ---

@app.get("/api/location/objects")
async def get_objects():
    """All tracked objects with positions, confidence, reachability."""
    objects = world_model.get_all_objects()
    return {
        "ok": True,
        "count": len(objects),
        "objects": [o.to_dict() for o in objects],
    }


@app.get("/api/location/reachable")
async def get_reachable():
    """Only objects within arm reach."""
    objects = world_model.get_reachable()
    return {
        "ok": True,
        "count": len(objects),
        "objects": [o.to_dict() for o in objects],
    }


@app.get("/api/location/object/{obj_id}")
async def get_object(obj_id: str):
    """Single object detail."""
    obj = world_model.get_object(obj_id)
    if obj is None:
        return JSONResponse(
            {"ok": False, "error": f"Object '{obj_id}' not found"},
            status_code=404,
        )
    return {"ok": True, "object": obj.to_dict()}


@app.post("/api/location/scan")
async def trigger_scan():
    """Trigger an immediate full scan of all cameras."""
    await tracker.trigger_scan()
    objects = world_model.get_all_objects()
    return {
        "ok": True,
        "message": "Full scan completed",
        "count": len(objects),
        "objects": [o.to_dict() for o in objects],
    }


@app.get("/api/location/status")
async def get_status():
    """Service health, last scan time, camera status."""
    return {
        "ok": True,
        "uptime_s": round(time.time() - _start_time, 1),
        "scan_count": world_model.scan_count,
        "last_scan_time": world_model.last_scan_time,
        "object_count": world_model.object_count,
        "cameras": tracker.camera_status,
        "ws_clients": len(ws_clients),
    }


# --- WebSocket ---

@app.websocket("/ws/location")
async def ws_location(ws: WebSocket):
    """WebSocket stream of object updates in real-time."""
    await ws.accept()
    ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(ws_clients))

    try:
        # Send current state immediately
        objects = world_model.get_all_objects()
        await ws.send_text(json.dumps({
            "type": "snapshot",
            "objects": [o.to_dict() for o in objects],
            "timestamp": time.time(),
        }))

        # Keep alive — wait for client disconnect
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        ws_clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remain)", len(ws_clients))


# --- Main ---

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LOCATION_PORT", "8082"))
    logger.info("Starting location server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
