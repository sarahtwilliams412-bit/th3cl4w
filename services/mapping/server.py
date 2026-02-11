"""
Mapping Server — 3D scene graph, collision map, and point cloud management.

Builds and maintains a 3D representation of the workspace: scene graph with
objects and arm skeleton, voxel-based collision map, and fused point cloud
from multiple depth sources.

Port: 8086 (configurable via MAPPING_PORT env var)

Usage:
    python -m services.mapping.server
    python -m services.mapping.server --debug
"""

import argparse
import asyncio
import json
import logging
import sys
import time
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

logger = logging.getLogger("th3cl4w.mapping")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Mapping — 3D Scene & Collision Map")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="mapping", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# In-memory scene state
# ---------------------------------------------------------------------------

_scene_graph: dict[str, Any] = {
    "arm": {
        "joint_positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.05, 0.2],
            [0.1, 0.05, 0.28],
            [0.18, 0.08, 0.3],
            [0.23, 0.09, 0.31],
            [0.25, 0.1, 0.3],
        ],
        "link_radii": [0.04, 0.035, 0.03, 0.025, 0.02, 0.015],
    },
    "objects": [
        {
            "id": "obj_001",
            "label": "red ball",
            "position": [0.2, 0.1, 0.05],
            "dimensions": [0.04, 0.04, 0.04],
            "color": [1.0, 0.2, 0.2],
            "category": "target",
        },
        {
            "id": "obj_002",
            "label": "blue cup",
            "position": [0.35, -0.05, 0.03],
            "dimensions": [0.07, 0.07, 0.1],
            "color": [0.2, 0.3, 1.0],
            "category": "target",
        },
    ],
    "bounds_min": [-0.5, -0.5, 0.0],
    "bounds_max": [0.5, 0.5, 0.5],
    "timestamp": 0.0,
}

_collision_map: dict[str, Any] = {
    "voxel_size_m": 0.01,
    "grid_dimensions": [100, 100, 50],
    "occupied_voxels": 42,
    "total_voxels": 500000,
    "arm_volume_voxels": 28,
    "object_volume_voxels": 14,
    "last_updated": 0.0,
}

_point_cloud: dict[str, Any] = {
    "num_points": 0,
    "sources": [],
    "bounds_min": [-0.5, -0.5, 0.0],
    "bounds_max": [0.5, 0.5, 0.5],
    "last_updated": 0.0,
}

_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None


async def _broadcast_scene_update(event: dict):
    """Send a scene update to all connected WebSocket clients."""
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

    _scene_graph["timestamp"] = time.time()
    _collision_map["last_updated"] = time.time()
    _point_cloud["last_updated"] = time.time()

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
        await _subscriber.subscribe(Topics.OBJECTS_UPDATED, _on_objects_updated)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Mapping ready — scene has %d objects", len(_scene_graph["objects"]))
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Mapping shut down")


async def _on_arm_state(topic: str, data: dict):
    """Handle arm state updates to refresh arm skeleton in scene graph."""
    pass


async def _on_objects_updated(topic: str, data: dict):
    """Handle object updates to refresh scene graph."""
    pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Mapping",
    description="3D scene graph, collision map, and point cloud management",
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

# Serve static files (3D visualization UI)
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
        "service": "mapping",
        "scene_objects": len(_scene_graph["objects"]),
        "collision_voxels": _collision_map["occupied_voxels"],
        "point_cloud_size": _point_cloud["num_points"],
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Scene API
# ---------------------------------------------------------------------------


@app.get("/api/scene")
async def get_scene():
    """Get the current scene graph."""
    return {
        **_scene_graph,
        "timestamp": time.time(),
    }


@app.get("/api/collision")
async def get_collision():
    """Get the collision map summary.

    The full voxel grid is too large to send over HTTP — this returns
    metadata and statistics. Use the WebSocket for streaming updates.
    """
    return {
        **_collision_map,
        "timestamp": time.time(),
    }


@app.get("/api/pointcloud")
async def get_pointcloud():
    """Get point cloud metadata.

    Returns metadata about the fused point cloud. The actual point data
    is streamed via WebSocket or fetched as binary via a separate endpoint.
    """
    return {
        **_point_cloud,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time scene updates
# ---------------------------------------------------------------------------


@app.websocket("/ws/scene")
async def ws_scene(websocket: WebSocket):
    """Real-time scene graph updates via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        # Send initial snapshot
        await websocket.send_json({
            "type": "snapshot",
            "scene": _scene_graph,
            "collision": _collision_map,
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
                    "scene_objects": len(_scene_graph["objects"]),
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
    port = args.port or ServiceConfig.MAPPING_PORT
    logger.info("Starting Mapping on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
