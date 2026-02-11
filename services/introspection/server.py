"""
Introspection Server — Self-feedback, episode analysis, and code improvement.

Allows the arm to review what it did, assess success/failure, generate
self-feedback, and propose parameter or code improvements. Uses the
introspection pipeline: ReplayBuffer -> WorldModel -> EpisodeAnalyzer ->
FeedbackGenerator -> CodeImprover.

Port: 8092 (configurable via INTROSPECTION_PORT env var)

Usage:
    python -m services.introspection.server
    python -m services.introspection.server --debug
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

logger = logging.getLogger("th3cl4w.introspection")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Introspection — Self-Feedback & Analysis")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="introspection", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request to analyze a task episode."""
    task_id: str = Field(description="Task ID to analyze")
    include_trajectory: bool = Field(default=True, description="Include trajectory reconstruction")
    generate_feedback: bool = Field(default=True, description="Generate self-feedback")
    propose_improvements: bool = Field(default=False, description="Propose code/config changes")


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_analyses: list[dict] = []
_improvements: list[dict] = []
_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _publisher, _subscriber

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
        await _subscriber.subscribe(Topics.TASK_COMPLETED, _on_task_completed)
        await _subscriber.subscribe(Topics.TASK_FAILED, _on_task_failed)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Introspection ready")
    yield

    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Introspection shut down — analyzed %d episodes", len(_analyses))


async def _on_task_completed(topic: str, data: dict):
    """Auto-analyze completed tasks."""
    task_id = data.get("task_id", "")
    logger.info("Task %s completed — queued for introspection", task_id)


async def _on_task_failed(topic: str, data: dict):
    """Auto-analyze failed tasks (higher priority)."""
    task_id = data.get("task_id", "")
    logger.info("Task %s failed — queued for introspection (high priority)", task_id)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Introspection",
    description="Self-feedback, episode analysis, and code improvement",
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
        "service": "introspection",
        "total_analyses": len(_analyses),
        "total_improvements": len(_improvements),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Introspection API
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze_episode(req: AnalyzeRequest):
    """Analyze a task episode for feedback and improvements."""
    analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
    now = time.time()

    analysis = {
        "analysis_id": analysis_id,
        "task_id": req.task_id,
        "status": "pending",
        "trajectory_included": req.include_trajectory,
        "feedback": None,
        "verdict": None,
        "improvements": [],
        "started_at": now,
        "completed_at": None,
    }
    _analyses.append(analysis)

    logger.info("Analysis %s queued for task %s", analysis_id, req.task_id)
    return analysis


@app.get("/api/analyses")
async def list_analyses(limit: int = 20):
    """List past episode analyses."""
    items = _analyses[-limit:]
    items.reverse()
    return {
        "analyses": items,
        "count": len(items),
        "total": len(_analyses),
        "timestamp": time.time(),
    }


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get details of a specific analysis."""
    for a in _analyses:
        if a["analysis_id"] == analysis_id:
            return a
    return JSONResponse({"error": f"Analysis '{analysis_id}' not found"}, status_code=404)


@app.get("/api/improvements")
async def list_improvements(limit: int = 20):
    """List proposed code/config improvements."""
    items = _improvements[-limit:]
    items.reverse()
    return {
        "improvements": items,
        "count": len(items),
        "total": len(_improvements),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time introspection feed
# ---------------------------------------------------------------------------


@app.websocket("/ws/introspection")
async def ws_introspection(websocket: WebSocket):
    """Real-time introspection event feed."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        await websocket.send_json({
            "type": "connected",
            "total_analyses": len(_analyses),
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
                    "total_analyses": len(_analyses),
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
    port = args.port or ServiceConfig.INTROSPECTION_PORT
    logger.info("Starting Introspection on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
