"""
Tasker Server — Task queue, orchestration, and lifecycle management.

Accepts high-level tasks (pick, place, scan, text commands), orchestrates
their execution through planning, simulation rehearsal, and real execution,
and tracks task lifecycle from submission to completion.

Port: 8088 (configurable via TASKER_PORT env var)

Usage:
    python -m services.tasker.server
    python -m services.tasker.server --debug
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

logger = logging.getLogger("th3cl4w.tasker")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Tasker — Task Queue & Orchestration")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="tasker", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class TaskSubmitRequest(BaseModel):
    """Request to submit a new task."""
    task_type: str = Field(description="Task type: pick, place, scan, move_to, home, text_command")
    target: str = Field(default="", description="Target object label or position")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
    priority: int = Field(default=0, ge=0, le=10, description="Priority (0=normal, 10=urgent)")
    require_sim_rehearsal: bool = Field(default=True, description="Simulate before real execution")
    text_command: str = Field(default="", description="Natural language command (for text_command type)")


# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------

_tasks: dict[str, dict] = {}
_task_queue: deque = deque()
_ws_clients: list[WebSocket] = []
_publisher: Any = None
_subscriber: Any = None
_task_counter: int = 0


async def _broadcast_task_update(event: dict):
    """Send a task update to all connected WebSocket clients."""
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
        await _subscriber.subscribe(Topics.PLAN_COMPUTED, _on_plan_computed)
        await _subscriber.subscribe(Topics.SIM_RESULT, _on_sim_result)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Tasker ready")
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Tasker shut down")


async def _on_plan_computed(topic: str, data: dict):
    """Handle plan computation results from Kinematics App."""
    task_id = data.get("task_id")
    if task_id and task_id in _tasks:
        _tasks[task_id]["plan_id"] = data.get("plan_id")
        _tasks[task_id]["status"] = "planned"
        await _broadcast_task_update({
            "type": "task_updated", "task_id": task_id,
            "status": "planned", "timestamp": time.time(),
        })


async def _on_sim_result(topic: str, data: dict):
    """Handle simulation rehearsal results."""
    task_id = data.get("task_id")
    if task_id and task_id in _tasks:
        _tasks[task_id]["sim_result"] = data.get("result", "unknown")
        _tasks[task_id]["status"] = "rehearsed"
        await _broadcast_task_update({
            "type": "task_updated", "task_id": task_id,
            "status": "rehearsed", "timestamp": time.time(),
        })


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Tasker",
    description="Task queue, orchestration, and lifecycle management",
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
    active = sum(1 for t in _tasks.values() if t["status"] in ("queued", "planning", "rehearsing", "executing"))
    return {
        "status": "healthy",
        "service": "tasker",
        "total_tasks": len(_tasks),
        "active_tasks": active,
        "queue_depth": len(_task_queue),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Task API
# ---------------------------------------------------------------------------


@app.post("/api/task")
async def submit_task(req: TaskSubmitRequest):
    """Submit a new task to the queue."""
    global _task_counter
    _task_counter += 1
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    now = time.time()

    task = {
        "task_id": task_id,
        "task_number": _task_counter,
        "task_type": req.task_type,
        "target": req.target,
        "parameters": req.parameters,
        "priority": req.priority,
        "require_sim_rehearsal": req.require_sim_rehearsal,
        "text_command": req.text_command,
        "status": "queued",
        "plan_id": None,
        "sim_result": None,
        "error": None,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _tasks[task_id] = task
    _task_queue.append(task_id)

    # Publish to message bus
    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.TASK_QUEUED, {
            "task_id": task_id, "task_type": req.task_type,
            "target": req.target, "priority": req.priority,
        })

    await _broadcast_task_update({
        "type": "task_queued", "task": task, "timestamp": now,
    })

    logger.info("Task %s queued: type=%s target='%s' priority=%d",
                task_id, req.task_type, req.target, req.priority)
    return task


@app.get("/api/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 50):
    """List all tasks, optionally filtered by status."""
    tasks = list(_tasks.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    # Sort by creation time, newest first
    tasks.sort(key=lambda t: t["created_at"], reverse=True)
    return {
        "tasks": tasks[:limit],
        "count": len(tasks[:limit]),
        "total": len(_tasks),
        "timestamp": time.time(),
    }


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """Get the status and details of a specific task."""
    task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"error": f"Task '{task_id}' not found"}, status_code=404)
    return task


@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a task."""
    task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"error": f"Task '{task_id}' not found"}, status_code=404)

    if task["status"] in ("completed", "failed", "cancelled"):
        return JSONResponse(
            {"error": f"Task '{task_id}' is already {task['status']}"},
            status_code=409,
        )

    task["status"] = "cancelled"
    task["completed_at"] = time.time()

    # Remove from queue if still queued
    if task_id in _task_queue:
        _task_queue.remove(task_id)

    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.TASK_CANCELLED, {
            "task_id": task_id, "task_type": task["task_type"],
        })

    await _broadcast_task_update({
        "type": "task_cancelled", "task_id": task_id, "timestamp": time.time(),
    })

    logger.info("Task %s cancelled", task_id)
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/api/queue")
async def get_queue():
    """Get the current task queue status."""
    queued_tasks = [_tasks[tid] for tid in _task_queue if tid in _tasks]
    active_tasks = [
        t for t in _tasks.values()
        if t["status"] in ("planning", "rehearsing", "executing")
    ]
    return {
        "queue": queued_tasks,
        "queue_depth": len(queued_tasks),
        "active": active_tasks,
        "active_count": len(active_tasks),
        "total_tasks": len(_tasks),
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time task updates
# ---------------------------------------------------------------------------


@app.websocket("/ws/tasks")
async def ws_tasks(websocket: WebSocket):
    """Real-time task lifecycle updates via WebSocket."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        # Send initial snapshot
        active = [t for t in _tasks.values()
                  if t["status"] in ("queued", "planning", "rehearsing", "executing")]
        await websocket.send_json({
            "type": "snapshot",
            "active_tasks": active,
            "queue_depth": len(_task_queue),
            "total_tasks": len(_tasks),
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
                    "queue_depth": len(_task_queue),
                    "total_tasks": len(_tasks),
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
    port = args.port or ServiceConfig.TASKER_PORT
    logger.info("Starting Tasker on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
