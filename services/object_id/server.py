"""
Object ID Server — Object detection, identification, and classification.

Runs ML-based object detection on camera frames, identifies objects using
an ontology of known types, and publishes detection results to the message bus
for consumption by the World Model and other services.

Port: 8084 (configurable via OBJECT_ID_PORT env var)

Usage:
    python -m services.object_id.server
    python -m services.object_id.server --debug
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from collections import deque
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

logger = logging.getLogger("th3cl4w.object_id")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Object ID — Detection & Identification")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="object_id", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    """Request to run detection on a camera frame."""
    camera_id: int = Field(default=0, description="Camera ID to capture from")
    frame_base64: Optional[str] = Field(default=None, description="Base64-encoded image (if not using live camera)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class IdentifyRequest(BaseModel):
    """Request to identify a specific detected object in more detail."""
    crop_base64: Optional[str] = Field(default=None, description="Cropped image of the object")
    context: str = Field(default="", description="Additional context for identification")


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_recent_detections: deque = deque(maxlen=500)
_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None
_detection_count: int = 0

# Known object ontology
_ONTOLOGY = {
    "categories": ["target", "obstacle", "boundary", "debris", "unknown"],
    "known_types": [
        {"type": "ball", "shapes": ["spherical"], "typical_size_mm": [30, 80]},
        {"type": "cup", "shapes": ["cylindrical"], "typical_size_mm": [60, 120]},
        {"type": "can", "shapes": ["cylindrical"], "typical_size_mm": [50, 130]},
        {"type": "bottle", "shapes": ["cylindrical"], "typical_size_mm": [60, 250]},
        {"type": "box", "shapes": ["rectangular"], "typical_size_mm": [50, 300]},
        {"type": "pen", "shapes": ["elongated"], "typical_size_mm": [10, 180]},
        {"type": "phone", "shapes": ["rectangular", "flat"], "typical_size_mm": [60, 160]},
        {"type": "tool", "shapes": ["elongated", "irregular"], "typical_size_mm": [100, 400]},
        {"type": "fruit", "shapes": ["spherical", "ovoid"], "typical_size_mm": [40, 120]},
    ],
    "color_vocabulary": [
        "red", "blue", "green", "yellow", "orange", "purple",
        "white", "black", "silver", "brown", "pink",
    ],
    "material_vocabulary": [
        "plastic", "metal", "glass", "wood", "rubber", "fabric", "paper", "ceramic",
    ],
}


async def _broadcast_detection(event: dict):
    """Send a detection event to all connected WebSocket clients."""
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
        # Subscribe to camera frames for automatic detection
        from shared.bus.topics import Topics
        await _subscriber.subscribe(Topics.CAMERA_FRAME, _on_camera_frame)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Object ID ready — ontology loaded with %d known types",
                len(_ONTOLOGY["known_types"]))
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Object ID shut down")


async def _on_camera_frame(topic: str, data: dict):
    """Handle incoming camera frames for automatic detection."""
    # In a real implementation, this would run the detection pipeline
    logger.debug("Received camera frame on %s", topic)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Object ID",
    description="Object detection, identification, and classification service",
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
        "service": "object_id",
        "total_detections": _detection_count,
        "recent_detections": len(_recent_detections),
        "known_types": len(_ONTOLOGY["known_types"]),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Detection API
# ---------------------------------------------------------------------------


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Run object detection on a camera frame.

    If frame_base64 is provided, detects objects in that image.
    Otherwise, captures a frame from the specified camera and detects.
    """
    global _detection_count
    detection_id = str(uuid.uuid4())[:8]
    _detection_count += 1
    now = time.time()

    # Placeholder detection results
    detections = [
        {
            "detection_id": f"{detection_id}_0",
            "label": "red ball",
            "confidence": 0.94,
            "bbox": [120, 85, 180, 145],
            "position_mm": [200.0, 100.0, 50.0],
            "dimensions_mm": [40.0, 40.0, 40.0],
            "category": "target",
            "color": "red",
            "shape": "spherical",
            "camera_id": req.camera_id,
            "timestamp": now,
        },
        {
            "detection_id": f"{detection_id}_1",
            "label": "blue cup",
            "confidence": 0.87,
            "bbox": [300, 200, 390, 320],
            "position_mm": [350.0, -50.0, 30.0],
            "dimensions_mm": [70.0, 70.0, 100.0],
            "category": "target",
            "color": "blue",
            "shape": "cylindrical",
            "camera_id": req.camera_id,
            "timestamp": now,
        },
    ]

    # Filter by confidence threshold
    detections = [d for d in detections if d["confidence"] >= req.confidence_threshold]

    # Store in recent detections
    for d in detections:
        _recent_detections.append(d)

    # Publish to message bus
    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.OBJECTS_DETECTED, {
            "detection_id": detection_id,
            "objects": detections,
            "camera_id": req.camera_id,
            "timestamp": now,
        })

    # Broadcast to WebSocket clients
    await _broadcast_detection({
        "type": "detection",
        "detection_id": detection_id,
        "objects": detections,
        "timestamp": now,
    })

    logger.info("Detection %s: found %d objects (camera %d)",
                detection_id, len(detections), req.camera_id)

    return {
        "detection_id": detection_id,
        "objects": detections,
        "count": len(detections),
        "camera_id": req.camera_id,
        "timestamp": now,
    }


@app.post("/api/identify/{object_id}")
async def identify_object(object_id: str, req: IdentifyRequest):
    """Identify a specific object in more detail using ML models."""
    now = time.time()

    # Placeholder identification result
    identification = {
        "object_id": object_id,
        "label": "red ball",
        "detailed_label": "small red rubber ball",
        "confidence": 0.96,
        "characteristics": {
            "color": "red",
            "material": "rubber",
            "shape": "spherical",
            "texture": "smooth",
            "estimated_weight_g": 50.0,
        },
        "graspability": {
            "graspable": True,
            "grip_type": "spherical",
            "estimated_grip_force_n": 2.5,
        },
        "context": req.context,
        "timestamp": now,
    }

    logger.info("Identified object %s as '%s' (confidence=%.2f)",
                object_id, identification["detailed_label"], identification["confidence"])

    return identification


@app.get("/api/detections")
async def get_recent_detections(limit: int = 50):
    """Get recent detection results."""
    items = list(_recent_detections)
    return {
        "detections": items[-limit:],
        "count": len(items[-limit:]),
        "total_all_time": _detection_count,
        "timestamp": time.time(),
    }


@app.get("/api/ontology")
async def get_ontology():
    """Get the known object type ontology."""
    return _ONTOLOGY


# ---------------------------------------------------------------------------
# WebSocket — real-time detection stream
# ---------------------------------------------------------------------------


@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    """Real-time detection results via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        await websocket.send_json({
            "type": "connected",
            "total_detections": _detection_count,
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
                    "total_detections": _detection_count,
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
    port = args.port or ServiceConfig.OBJECT_ID_PORT
    logger.info("Starting Object ID on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
