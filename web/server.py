"""
th3cl4w Web Control Server

FastAPI backend for controlling the Unitree D1 robotic arm.
Provides REST API + WebSocket for real-time state streaming.

Usage:
    python web/server.py                  # Connect to real hardware
    python web/server.py --simulate       # Simulated arm (no hardware)
    python web/server.py --simulate --port 8080
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.interface.d1_connection import D1Command, D1Connection, D1State, NUM_JOINTS

logger = logging.getLogger("th3cl4w.web")

# ---------------------------------------------------------------------------
# Simulated arm
# ---------------------------------------------------------------------------

class SimulatedD1:
    """Simulated D1 arm for development without hardware."""

    def __init__(self):
        self.joint_positions = np.zeros(NUM_JOINTS)
        self.joint_velocities = np.zeros(NUM_JOINTS)
        self.joint_torques = np.zeros(NUM_JOINTS)
        self.gripper_position = 0.0
        self.mode = 0
        self._target_positions = np.zeros(NUM_JOINTS)
        self._target_gripper = 0.0
        self._connected = True
        self._start_time = time.time()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def get_state(self) -> D1State:
        # Simulate smooth motion toward targets
        dt = 0.05
        speed = 2.0  # rad/s
        for i in range(NUM_JOINTS):
            diff = self._target_positions[i] - self.joint_positions[i]
            step = np.clip(diff, -speed * dt, speed * dt)
            self.joint_positions[i] += step
            self.joint_velocities[i] = step / dt if abs(step) > 1e-6 else 0.0
            self.joint_torques[i] = diff * 10.0  # fake spring torque

        gdiff = self._target_gripper - self.gripper_position
        self.gripper_position += np.clip(gdiff, -speed * dt, speed * dt)

        # Add tiny noise for realism
        noise = np.random.normal(0, 0.0005, NUM_JOINTS)

        return D1State(
            joint_positions=self.joint_positions.copy() + noise,
            joint_velocities=self.joint_velocities.copy(),
            joint_torques=self.joint_torques.copy(),
            gripper_position=float(self.gripper_position),
            timestamp=time.time() - self._start_time,
        )

    def send_command(self, cmd: D1Command) -> bool:
        self.mode = cmd.mode
        if cmd.joint_positions is not None:
            self._target_positions = np.array(cmd.joint_positions, dtype=np.float64)
        if cmd.gripper_position is not None:
            self._target_gripper = float(cmd.gripper_position)
        return True

    def stop(self):
        self.mode = 0
        self._target_positions = self.joint_positions.copy()
        self._target_gripper = self.gripper_position

    def home(self):
        self._target_positions = np.zeros(NUM_JOINTS)
        self._target_gripper = 0.0


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

arm = None  # D1Connection or SimulatedD1
simulate_mode = False
ws_clients: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class CommandRequest(BaseModel):
    mode: int = 0
    joint_positions: Optional[list[float]] = None
    joint_velocities: Optional[list[float]] = None
    joint_torques: Optional[list[float]] = None
    gripper_position: Optional[float] = None


class StateResponse(BaseModel):
    connected: bool
    mode: int
    joint_positions: list[float]
    joint_velocities: list[float]
    joint_torques: list[float]
    gripper_position: float
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_state_dict() -> dict:
    """Get current arm state as a dict."""
    if arm is None:
        return {"connected": False, "mode": 0, "joint_positions": [0]*7,
                "joint_velocities": [0]*7, "joint_torques": [0]*7,
                "gripper_position": 0.0, "timestamp": 0.0}

    state = arm.get_state() if simulate_mode else arm.get_state()
    if state is None:
        return {"connected": arm.is_connected, "mode": getattr(arm, 'mode', 0),
                "joint_positions": [0]*7, "joint_velocities": [0]*7,
                "joint_torques": [0]*7, "gripper_position": 0.0, "timestamp": 0.0}

    return {
        "connected": arm.is_connected,
        "mode": getattr(arm, 'mode', 0),
        "joint_positions": [round(float(x), 6) for x in state.joint_positions],
        "joint_velocities": [round(float(x), 6) for x in state.joint_velocities],
        "joint_torques": [round(float(x), 6) for x in state.joint_torques],
        "gripper_position": round(float(state.gripper_position), 6),
        "timestamp": round(float(state.timestamp), 4),
    }


async def broadcast_state():
    """Background task: stream state to all WebSocket clients at ~20Hz."""
    while True:
        if ws_clients:
            data = json.dumps(get_state_dict())
            dead = set()
            for ws in ws_clients:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.add(ws)
            ws_clients -= dead
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global arm
    if simulate_mode:
        arm = SimulatedD1()
        logger.info("Running in SIMULATE mode — no hardware")
    else:
        arm = D1Connection()
        if not arm.connect():
            logger.error("Failed to connect to D1 arm! Starting in disconnected state.")
    # Start broadcast task
    task = asyncio.create_task(broadcast_state())
    yield
    task.cancel()
    if arm:
        arm.disconnect()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="th3cl4w", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/state")
async def api_state():
    return get_state_dict()


@app.post("/api/command")
async def api_command(cmd: CommandRequest):
    if arm is None:
        return {"ok": False, "error": "Not connected"}

    d1cmd = D1Command(
        mode=cmd.mode,
        joint_positions=np.array(cmd.joint_positions) if cmd.joint_positions else None,
        joint_velocities=np.array(cmd.joint_velocities) if cmd.joint_velocities else None,
        joint_torques=np.array(cmd.joint_torques) if cmd.joint_torques else None,
        gripper_position=cmd.gripper_position,
    )

    if simulate_mode:
        ok = arm.send_command(d1cmd)
    else:
        ok = arm.send_command(d1cmd)

    return {"ok": ok}


@app.post("/api/stop")
async def api_stop():
    if arm is None:
        return {"ok": False, "error": "Not connected"}

    if simulate_mode:
        arm.stop()
        return {"ok": True}
    else:
        # Send idle command
        cmd = D1Command(mode=0)
        return {"ok": arm.send_command(cmd)}


@app.post("/api/home")
async def api_home():
    if arm is None:
        return {"ok": False, "error": "Not connected"}

    if simulate_mode:
        arm.home()
        arm.mode = 1
        return {"ok": True}
    else:
        cmd = D1Command(
            mode=1,
            joint_positions=np.zeros(NUM_JOINTS),
            gripper_position=0.0,
        )
        return {"ok": arm.send_command(cmd)}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        while True:
            # Keep connection alive; handle incoming messages (e.g. commands)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "command":
                    cmd = CommandRequest(**msg.get("data", {}))
                    d1cmd = D1Command(
                        mode=cmd.mode,
                        joint_positions=np.array(cmd.joint_positions) if cmd.joint_positions else None,
                        joint_velocities=np.array(cmd.joint_velocities) if cmd.joint_velocities else None,
                        joint_torques=np.array(cmd.joint_torques) if cmd.joint_torques else None,
                        gripper_position=cmd.gripper_position,
                    )
                    arm.send_command(d1cmd)
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global simulate_mode
    parser = argparse.ArgumentParser(description="th3cl4w Web Control Server")
    parser.add_argument("--simulate", action="store_true", help="Run without hardware")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7777, help="Bind port")
    args = parser.parse_args()

    simulate_mode = args.simulate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Starting th3cl4w web server on %s:%d (simulate=%s)", args.host, args.port, args.simulate)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
