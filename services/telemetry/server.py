"""
Telemetry Server — Event logging, pick episode tracking, and analytics.

Collects telemetry events from all services, tracks pick episodes end-to-end,
and provides analytics summaries for monitoring system performance.

Port: 8091 (configurable via TELEMETRY_PORT env var)

Usage:
    python -m services.telemetry.server
    python -m services.telemetry.server --debug
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
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.telemetry")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Telemetry — Event Logging & Analytics")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="telemetry", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# In-memory telemetry stores
# ---------------------------------------------------------------------------

_events: deque = deque(maxlen=10000)
_episodes: list[dict] = []
_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None
_event_count: int = 0

# Seed some example episodes
_SEED_EPISODES = [
    {
        "episode_id": "ep_001",
        "task_id": "task_abc12345",
        "target_object": "red ball",
        "status": "completed",
        "success": True,
        "phases": {
            "detection_s": 0.45,
            "planning_s": 0.32,
            "simulation_s": 1.20,
            "execution_s": 3.80,
            "total_s": 5.77,
        },
        "collisions": 0,
        "retries": 0,
        "started_at": 0.0,
        "completed_at": 0.0,
    },
    {
        "episode_id": "ep_002",
        "task_id": "task_def67890",
        "target_object": "blue cup",
        "status": "completed",
        "success": True,
        "phases": {
            "detection_s": 0.52,
            "planning_s": 0.41,
            "simulation_s": 1.35,
            "execution_s": 4.10,
            "total_s": 6.38,
        },
        "collisions": 0,
        "retries": 0,
        "started_at": 0.0,
        "completed_at": 0.0,
    },
    {
        "episode_id": "ep_003",
        "task_id": "task_ghi11223",
        "target_object": "small wrench",
        "status": "failed",
        "success": False,
        "phases": {
            "detection_s": 0.38,
            "planning_s": 0.28,
            "simulation_s": 1.10,
            "execution_s": 2.50,
            "total_s": 4.26,
        },
        "collisions": 0,
        "retries": 1,
        "error": "Grasp failed — object slipped",
        "started_at": 0.0,
        "completed_at": 0.0,
    },
]


async def _broadcast_telemetry(event: dict):
    """Send a telemetry event to all connected WebSocket clients."""
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
    global _publisher, _subscriber, _episodes

    now = time.time()
    for ep in _SEED_EPISODES:
        ep["started_at"] = now - ep["phases"]["total_s"]
        ep["completed_at"] = now
    _episodes = list(_SEED_EPISODES)

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
        # Subscribe to all major event topics for telemetry collection
        from shared.bus.topics import Topics
        for topic in [
            Topics.TASK_QUEUED, Topics.TASK_STARTED,
            Topics.TASK_COMPLETED, Topics.TASK_FAILED, Topics.TASK_CANCELLED,
            Topics.PLAN_REQUESTED, Topics.PLAN_COMPUTED, Topics.PLAN_EXECUTING,
            Topics.SIM_RESULT, Topics.ARM_ESTOP,
            Topics.OBJECTS_DETECTED,
        ]:
            await _subscriber.subscribe(topic, _on_telemetry_event)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Telemetry ready — %d seed episodes loaded", len(_episodes))
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Telemetry shut down")


async def _on_telemetry_event(topic: str, data: dict):
    """Handle any bus event and record it as telemetry."""
    global _event_count
    _event_count += 1
    event = {
        "event_id": _event_count,
        "topic": topic,
        "data": data,
        "timestamp": time.time(),
    }
    _events.append(event)
    await _broadcast_telemetry({"type": "event", **event})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Telemetry",
    description="Event logging, pick episode tracking, and analytics",
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

# Serve static files (telemetry dashboard UI)
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
        "service": "telemetry",
        "total_events": _event_count,
        "buffered_events": len(_events),
        "total_episodes": len(_episodes),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Telemetry API
# ---------------------------------------------------------------------------


@app.get("/api/events")
async def get_events(
    limit: int = Query(default=100, ge=1, le=1000),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    since: Optional[float] = Query(default=None, description="Events after this Unix timestamp"),
):
    """Query telemetry events."""
    items = list(_events)
    if topic:
        items = [e for e in items if e.get("topic") == topic]
    if since:
        items = [e for e in items if e.get("timestamp", 0) >= since]
    items = items[-limit:]
    return {
        "events": items,
        "count": len(items),
        "total_all_time": _event_count,
        "timestamp": time.time(),
    }


@app.get("/api/episodes")
async def get_episodes(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None, description="Filter by status"),
):
    """List pick episodes."""
    eps = list(_episodes)
    if status:
        eps = [e for e in eps if e.get("status") == status]
    eps.sort(key=lambda e: e.get("completed_at", 0), reverse=True)
    return {
        "episodes": eps[:limit],
        "count": len(eps[:limit]),
        "total": len(_episodes),
        "timestamp": time.time(),
    }


@app.get("/api/analytics")
async def get_analytics():
    """Get pick analytics summary."""
    total = len(_episodes)
    successful = sum(1 for e in _episodes if e.get("success"))
    failed = total - successful

    avg_duration = 0.0
    avg_phases: dict[str, float] = {}
    if total > 0:
        durations = [e["phases"]["total_s"] for e in _episodes if "phases" in e]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        phase_keys = ["detection_s", "planning_s", "simulation_s", "execution_s"]
        for key in phase_keys:
            vals = [e["phases"][key] for e in _episodes if "phases" in e and key in e["phases"]]
            avg_phases[key] = sum(vals) / len(vals) if vals else 0.0

    return {
        "summary": {
            "total_episodes": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration_s": round(avg_duration, 2),
            "avg_phases": {k: round(v, 3) for k, v in avg_phases.items()},
        },
        "total_telemetry_events": _event_count,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time telemetry stream
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """Real-time telemetry event stream via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        await websocket.send_json({
            "type": "connected",
            "total_events": _event_count,
            "total_episodes": len(_episodes),
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
                    "total_events": _event_count,
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
    port = args.port or ServiceConfig.TELEMETRY_PORT
    logger.info("Starting Telemetry on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
