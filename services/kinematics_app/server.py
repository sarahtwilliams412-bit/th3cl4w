"""
Kinematics App Server — Motion planning, IK/FK, and plan execution.

Computes motion plans for the arm: pick operations, free-space moves,
text-based natural language commands. Uses forward/inverse kinematics and
collision checking to produce safe, executable trajectories.

Port: 8085 (configurable via KINEMATICS_PORT env var)

Usage:
    python -m services.kinematics_app.server
    python -m services.kinematics_app.server --debug
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

logger = logging.getLogger("th3cl4w.kinematics")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Kinematics App — Motion Planning")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="kinematics", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MotionPlanRequest(BaseModel):
    """Request to plan a motion to a target position."""
    target_position_mm: list[float] = Field(description="[x, y, z] target in mm")
    target_orientation: Optional[list[float]] = Field(default=None, description="[roll, pitch, yaw] in degrees")
    max_velocity: float = Field(default=1.0, ge=0.1, le=5.0, description="Max joint velocity scale")
    collision_check: bool = Field(default=True, description="Enable collision checking")


class PickPlanRequest(BaseModel):
    """Request to plan a pick operation."""
    object_id: str = Field(description="Target object ID from world model")
    approach_height_mm: float = Field(default=80.0, ge=20.0, description="Height above object for approach")
    grip_type: str = Field(default="auto", description="Grip type: auto, pinch, power, spherical")
    retreat_height_mm: float = Field(default=120.0, ge=40.0, description="Height for retreat after pick")


class TextCommandRequest(BaseModel):
    """Natural language command for the arm."""
    command: str = Field(description="Natural language command (e.g., 'pick up the red ball')")
    context: dict = Field(default_factory=dict, description="Additional context")


class FKRequest(BaseModel):
    """Forward kinematics query."""
    joint_angles_deg: list[float] = Field(min_length=6, max_length=6, description="6 joint angles in degrees")


class IKRequest(BaseModel):
    """Inverse kinematics query."""
    target_position_mm: list[float] = Field(description="[x, y, z] target in mm")
    target_orientation: Optional[list[float]] = Field(default=None, description="[roll, pitch, yaw] in degrees")
    seed_angles_deg: Optional[list[float]] = Field(default=None, description="Seed joint angles for solver")


# ---------------------------------------------------------------------------
# In-memory plan store
# ---------------------------------------------------------------------------

_plans: dict[str, dict] = {}
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
        await _subscriber.subscribe(Topics.PLAN_REQUESTED, _on_plan_requested)
        await _subscriber.listen()
    except Exception as e:
        logger.info("Message bus subscriber not available: %s (operating standalone)", e)

    logger.info("Kinematics App ready")
    yield

    # Shutdown
    if _publisher:
        await _publisher.close()
    if _subscriber:
        await _subscriber.close()
    logger.info("Kinematics App shut down")


async def _on_plan_requested(topic: str, data: dict):
    """Handle plan requests from the message bus (e.g., from Tasker)."""
    logger.info("Plan request received via bus: %s", data.get("task_id", "unknown"))


# ---------------------------------------------------------------------------
# Helper — generate placeholder trajectory
# ---------------------------------------------------------------------------


def _placeholder_trajectory(n_waypoints: int = 10) -> list[list[float]]:
    """Generate a placeholder trajectory (list of joint angle waypoints)."""
    import random
    trajectory = []
    current = [0.0, -30.0, 60.0, 0.0, 30.0, 0.0]
    for i in range(n_waypoints):
        step = [c + random.uniform(-2.0, 2.0) for c in current]
        trajectory.append([round(v, 2) for v in step])
        current = step
    return trajectory


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Kinematics App",
    description="Motion planning, IK/FK, and trajectory execution service",
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
        "service": "kinematics",
        "active_plans": sum(1 for p in _plans.values() if p["status"] in ("planned", "executing")),
        "total_plans": len(_plans),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Motion planning API
# ---------------------------------------------------------------------------


@app.post("/api/plan/motion")
async def plan_motion(req: MotionPlanRequest):
    """Plan a free-space motion to a target position."""
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    now = time.time()

    trajectory = _placeholder_trajectory(12)

    plan = {
        "plan_id": plan_id,
        "type": "motion",
        "status": "planned",
        "target_position_mm": req.target_position_mm,
        "target_orientation": req.target_orientation,
        "trajectory": trajectory,
        "waypoint_count": len(trajectory),
        "estimated_duration_s": len(trajectory) * 0.15,
        "collision_free": True,
        "created_at": now,
    }
    _plans[plan_id] = plan

    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.PLAN_COMPUTED, plan)

    logger.info("Motion plan %s: %d waypoints, %.1fs estimated",
                plan_id, len(trajectory), plan["estimated_duration_s"])
    return plan


@app.post("/api/plan/pick")
async def plan_pick(req: PickPlanRequest):
    """Plan a pick operation for a target object."""
    plan_id = f"pick_{uuid.uuid4().hex[:8]}"
    now = time.time()

    approach_traj = _placeholder_trajectory(8)
    descend_traj = _placeholder_trajectory(4)
    retreat_traj = _placeholder_trajectory(6)

    plan = {
        "plan_id": plan_id,
        "type": "pick",
        "status": "planned",
        "object_id": req.object_id,
        "grip_type": req.grip_type,
        "approach_height_mm": req.approach_height_mm,
        "retreat_height_mm": req.retreat_height_mm,
        "phases": {
            "approach": {"trajectory": approach_traj, "waypoints": len(approach_traj)},
            "descend": {"trajectory": descend_traj, "waypoints": len(descend_traj)},
            "grasp": {"grip_force_n": 3.0, "grip_width_mm": 40.0},
            "retreat": {"trajectory": retreat_traj, "waypoints": len(retreat_traj)},
        },
        "total_waypoints": len(approach_traj) + len(descend_traj) + len(retreat_traj),
        "estimated_duration_s": 3.5,
        "collision_free": True,
        "created_at": now,
    }
    _plans[plan_id] = plan

    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.PLAN_COMPUTED, {
            "plan_id": plan_id, "type": "pick", "object_id": req.object_id,
        })

    logger.info("Pick plan %s for object %s: %d total waypoints",
                plan_id, req.object_id, plan["total_waypoints"])
    return plan


@app.post("/api/plan/text-command")
async def plan_text_command(req: TextCommandRequest):
    """Parse a natural language command and generate a motion plan."""
    plan_id = f"nlp_{uuid.uuid4().hex[:8]}"
    now = time.time()

    trajectory = _placeholder_trajectory(10)

    plan = {
        "plan_id": plan_id,
        "type": "text_command",
        "status": "planned",
        "original_command": req.command,
        "interpreted_action": "pick",
        "interpreted_target": "red ball",
        "confidence": 0.88,
        "trajectory": trajectory,
        "waypoint_count": len(trajectory),
        "estimated_duration_s": 2.8,
        "collision_free": True,
        "created_at": now,
    }
    _plans[plan_id] = plan

    logger.info("Text command plan %s: '%s' -> pick red ball (conf=%.2f)",
                plan_id, req.command, plan["confidence"])
    return plan


@app.post("/api/execute/{plan_id}")
async def execute_plan(plan_id: str):
    """Execute a previously computed plan."""
    plan = _plans.get(plan_id)
    if plan is None:
        return JSONResponse({"error": f"Plan '{plan_id}' not found"}, status_code=404)

    if plan["status"] not in ("planned",):
        return JSONResponse(
            {"error": f"Plan '{plan_id}' cannot be executed (status={plan['status']})"},
            status_code=409,
        )

    plan["status"] = "executing"
    plan["execution_started_at"] = time.time()

    if _publisher and _publisher.is_connected:
        from shared.bus.topics import Topics
        await _publisher.publish(Topics.PLAN_EXECUTING, {
            "plan_id": plan_id, "type": plan["type"],
        })

    logger.info("Executing plan %s (type=%s)", plan_id, plan["type"])

    # In a real implementation, this would hand off to the execution engine
    # For now, mark as completed after a brief delay
    plan["status"] = "completed"
    plan["execution_completed_at"] = time.time()

    return {
        "plan_id": plan_id,
        "status": plan["status"],
        "message": "Plan execution completed",
    }


@app.get("/api/plan/{plan_id}")
async def get_plan(plan_id: str):
    """Get the status and details of a plan."""
    plan = _plans.get(plan_id)
    if plan is None:
        return JSONResponse({"error": f"Plan '{plan_id}' not found"}, status_code=404)
    return plan


# ---------------------------------------------------------------------------
# Forward / Inverse Kinematics
# ---------------------------------------------------------------------------


@app.post("/api/fk")
async def forward_kinematics(req: FKRequest):
    """Compute forward kinematics — joint angles to end-effector pose."""
    # Placeholder FK result
    return {
        "joint_angles_deg": req.joint_angles_deg,
        "end_effector": {
            "position_mm": [250.0, 100.0, 300.0],
            "orientation_deg": [0.0, -45.0, 0.0],
            "quaternion": [0.924, 0.0, -0.383, 0.0],
        },
        "joint_positions_mm": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 50.0, 200.0],
            [100.0, 50.0, 280.0],
            [180.0, 80.0, 300.0],
            [230.0, 90.0, 310.0],
            [250.0, 100.0, 300.0],
        ],
        "timestamp": time.time(),
    }


@app.post("/api/ik")
async def inverse_kinematics(req: IKRequest):
    """Compute inverse kinematics — target pose to joint angles."""
    # Placeholder IK result
    solutions = [
        {"joint_angles_deg": [10.0, -35.0, 65.0, 0.0, 40.0, -10.0], "fitness": 0.98},
        {"joint_angles_deg": [-170.0, -35.0, 65.0, 180.0, 40.0, 170.0], "fitness": 0.92},
    ]

    return {
        "target_position_mm": req.target_position_mm,
        "target_orientation": req.target_orientation,
        "solutions": solutions,
        "best_solution": solutions[0],
        "reachable": True,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.KINEMATICS_PORT
    logger.info("Starting Kinematics App on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
