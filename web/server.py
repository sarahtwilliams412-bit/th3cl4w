# th3cl4w V1 Server — Stable Base
#!/usr/bin/env python3.12
from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else
"""
th3cl4w — Web Control Panel for Unitree D1 Arm.

FastAPI backend with WebSocket state streaming and REST command API.
Supports both real DDS hardware and --simulate mode.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time

# Hint OpenCV to prefer GPU device for OpenCL acceleration (RX 580 eGPU)
os.environ.setdefault("OPENCV_OPENCL_DEVICE", ":GPU:0")
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path so src.* imports work regardless of CWD
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.telemetry import get_collector, EventType

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False

try:
    from src.telemetry.query import TelemetryQuery

    _HAS_QUERY = True
except ImportError:
    _HAS_QUERY = False

import numpy as np

try:
    from src.planning.task_planner import TaskPlanner, TaskStatus
    from src.planning.motion_planner import Waypoint

    _HAS_PLANNING = True
except ImportError:
    _HAS_PLANNING = False

try:
    from src.planning.text_command import parse_command, CommandType, ParsedCommand

    _HAS_TEXT_CMD = True
except ImportError:
    _HAS_TEXT_CMD = False

try:
    from src.vision.workspace_mapper import WorkspaceMapper
    from src.planning.collision_preview import CollisionPreview

    _HAS_BIFOCAL = True
except ImportError:
    _HAS_BIFOCAL = False

try:
    from src.vision.arm_tracker import DualCameraArmTracker
    from src.vision.grasp_planner import VisualGraspPlanner
    from src.planning.pick_executor import PickExecutor, PickPhase

    _HAS_VISUAL_PICK = True
except ImportError:
    _HAS_VISUAL_PICK = False

try:
    from src.vision.scene_analyzer import SceneAnalyzer
    from src.planning.vision_task_planner import VisionTaskPlanner

    _HAS_VISION_PLANNING = True
except ImportError:
    _HAS_VISION_PLANNING = False

try:
    from src.vla import VLAController, DataCollector, GeminiVLABackend

    _HAS_VLA = True
except ImportError:
    _HAS_VLA = False

try:
    from src.vision.claw_position import ClawPositionPredictor

    _HAS_CLAW_PREDICT = True
except ImportError:
    _HAS_CLAW_PREDICT = False

try:
    from src.safety.collision_detector import CollisionDetector, StallEvent
    from src.vision.collision_analyzer import CollisionAnalyzer

    _HAS_COLLISION = True
except ImportError:
    _HAS_COLLISION = False

try:
    from src.vision.pose_fusion import PoseFusion, CameraCalib, FusionSource
    from src.vision.fk_engine import fk_positions as fk_positions_fn
    from src.vision.arm_segmenter import ArmSegmenter
    from src.vision.joint_detector import JointDetector

    _HAS_POSE_FUSION = True
except ImportError:
    _HAS_POSE_FUSION = False

try:
    from src.vision.ascii_converter import (
        AsciiConverter,
        CHARSET_STANDARD,
        CHARSET_DETAILED,
        CHARSET_BLOCKS,
        CHARSET_MINIMAL,
    )

    _HAS_ASCII = True
except ImportError:
    _HAS_ASCII = False

try:
    from src.vision.gpu_preprocess import decode_jpeg_gpu, gpu_status as _gpu_status

    _HAS_GPU_PREPROCESS = True
except ImportError:
    _HAS_GPU_PREPROCESS = False

_web_dir = str(Path(__file__).resolve().parent)
if _web_dir not in sys.path:
    sys.path.insert(0, _web_dir)

from command_smoother import CommandSmoother

from src.safety.limits import (
    JOINT_LIMITS_DEG as _UNIFIED_JOINT_LIMITS_DEG,
    GRIPPER_MIN_MM,
    GRIPPER_MAX_MM,
    MAX_STEP_DEG,
)
from src.safety.safety_monitor import SafetyMonitor

# ---------------------------------------------------------------------------
# CLI args (parsed early so lifespan can access them)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w D1 Arm Web Control Panel")
parser.add_argument("--simulate", action="store_true", help="Run with simulated arm state")
parser.add_argument("--interface", default="eno1", help="Network interface for DDS (default: eno1)")
parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")

# Only parse if running as main (not under test)
if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args(["--simulate"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("th3cl4w.web")

# ---------------------------------------------------------------------------
# D1 Joint specs — imported from unified limits
# ---------------------------------------------------------------------------

JOINT_LIMITS_DEG = {
    i: (float(_UNIFIED_JOINT_LIMITS_DEG[i, 0]), float(_UNIFIED_JOINT_LIMITS_DEG[i, 1]))
    for i in range(6)
}

GRIPPER_RANGE = (GRIPPER_MIN_MM, GRIPPER_MAX_MM)

# ---------------------------------------------------------------------------
# Structured action log
# ---------------------------------------------------------------------------


class ActionLog:
    """Thread-safe circular log of action entries."""

    def __init__(self, maxlen: int = 200):
        self._entries: deque = deque(maxlen=maxlen)

    def add(self, action: str, details: str, level: str = "info"):
        ts = time.time()
        d = time.localtime(ts)
        ms = int((ts % 1) * 1000)
        ts_str = f"{d.tm_hour:02d}:{d.tm_min:02d}:{d.tm_sec:02d}.{ms:03d}"
        entry = {"ts": ts, "ts_str": ts_str, "action": action, "details": details, "level": level}
        self._entries.append(entry)
        logger.log(
            {"info": logging.INFO, "error": logging.ERROR, "warning": logging.WARNING}.get(
                level, logging.INFO
            ),
            "%s | %s | %s",
            ts_str,
            action,
            details,
        )

    def last(self, n: int = 50) -> List[Dict]:
        items = list(self._entries)
        return items[-n:]


action_log = ActionLog()

# ---------------------------------------------------------------------------
# Simulated arm — imported from src.interface.simulated_arm
# ---------------------------------------------------------------------------

from src.interface.simulated_arm import SimulatedArm

# Track whether we're in simulation mode (can be toggled at runtime)
_sim_mode: bool = False


# ---------------------------------------------------------------------------
# Global arm reference
# ---------------------------------------------------------------------------

arm: Any = None
smoother: Optional[CommandSmoother] = None
safety_monitor: Optional[SafetyMonitor] = None
task_planner: Any = None  # TaskPlanner instance, initialized in lifespan
workspace_mapper: Any = None  # WorkspaceMapper for bifocal vision
collision_preview: Any = None  # CollisionPreview for path checking
arm_tracker: Any = None  # DualCameraArmTracker for visual object tracking
grasp_planner: Any = None  # VisualGraspPlanner for grasp pose computation
pick_executor: Any = None  # PickExecutor for autonomous pick operations
vision_task_planner: Any = None  # VisionTaskPlanner for camera-guided planning
scene_analyzer: Any = None  # SceneAnalyzer for scene understanding
claw_predictor: Any = None  # ClawPositionPredictor for visual claw tracking
collision_detector: Any = None  # CollisionDetector for stall detection
collision_analyzer: Any = None  # CollisionAnalyzer for camera + vision analysis
collision_events: list = []  # Recent collision events for API
vla_controller: Any = None  # VLAController for vision-language-action
vla_data_collector: Any = None  # DataCollector for recording demonstrations
pose_fusion: Any = None  # PoseFusion engine
arm3d_segmenters: dict = {}  # Per-camera ArmSegmenter instances
arm3d_detector: Any = None  # JointDetector for arm3d pipeline
camera_models: dict = {}  # CameraModel instances keyed by camera_id

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global arm, smoother, task_planner

    # Initialize OpenCL GPU acceleration for OpenCV
    try:
        import cv2

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            # Log device info
            dev = cv2.ocl.Device.getDefault()
            if dev is not None and dev.available():
                logger.info(
                    "OpenCL GPU acceleration enabled: %s (%s)", dev.name(), dev.vendorName()
                )
            else:
                logger.info("OpenCL available but no device detected; falling back to CPU")
        else:
            logger.info("OpenCL not available; OpenCV will use CPU")
    except Exception as e:
        logger.warning("OpenCL init failed (%s); OpenCV will use CPU", e)

    # Start telemetry collector
    if _HAS_TELEMETRY:
        tc = get_collector()
        tc.start()
        tc.enable()
        tc.log_system_event("startup", "system", "th3cl4w starting up")

    if args.simulate:
        arm = SimulatedArm()
        arm.start_feedback_loop(rate_hz=10.0)
        _sim_mode = True
        action_log.add("SYSTEM", "Simulated arm initialized (SIM mode)", "info")
        logger.info("Running in SIMULATION mode")
    else:
        try:
            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.interface.d1_dds_connection import D1DDSConnection

            arm = D1DDSConnection(collector=tc if _HAS_TELEMETRY else None)
            if arm.connect(interface_name=args.interface):
                action_log.add("SYSTEM", f"DDS connected on {args.interface}", "info")
                if _HAS_TELEMETRY:
                    get_collector().log_system_event(
                        "connect", "dds", f"Connected on {args.interface}"
                    )
            else:
                action_log.add("SYSTEM", f"DDS connection FAILED on {args.interface}", "error")
                if _HAS_TELEMETRY:
                    get_collector().log_system_event(
                        "connect_failed", "dds", f"Failed on {args.interface}", level="error"
                    )
        except Exception as e:
            action_log.add("SYSTEM", f"DDS init error: {e}", "error")
            logger.exception("Failed to initialize DDS connection")
            arm = None

    # Instantiate the safety monitor — must exist before smoother
    global safety_monitor
    safety_monitor = SafetyMonitor()
    action_log.add("SYSTEM", "Safety monitor initialized", "info")

    # Start the command smoother for smooth motion
    if arm is not None:
        tc_ref = get_collector() if _HAS_TELEMETRY else None
        smoother = CommandSmoother(
            arm,
            rate_hz=10.0,
            smoothing_factor=0.35,
            max_step_deg=MAX_STEP_DEG,
            collector=tc_ref,
            safety_monitor=safety_monitor,
        )
        await smoother.start()
        action_log.add(
            "SYSTEM",
            f"Command smoother started (10Hz, α=0.35, max_step={MAX_STEP_DEG}°, synced={smoother.synced})",
            "info",
        )

    # Initialize task planner for pre-built motion sequences
    global task_planner
    if _HAS_PLANNING:
        task_planner = TaskPlanner()
        action_log.add("SYSTEM", "Task planner initialized", "info")

    # Initialize bifocal workspace mapper and collision preview
    global workspace_mapper, collision_preview
    if _HAS_BIFOCAL:
        workspace_mapper = WorkspaceMapper()
        collision_preview = CollisionPreview()
        action_log.add(
            "SYSTEM", "Bifocal workspace mapper initialized (disabled by default)", "info"
        )

    # Initialize visual pick system (arm tracker + grasp planner + pick executor)
    global arm_tracker, grasp_planner, pick_executor
    if _HAS_VISUAL_PICK:
        arm_tracker = DualCameraArmTracker()
        grasp_planner = VisualGraspPlanner()
        pick_executor = PickExecutor(task_planner=task_planner if _HAS_PLANNING else None)
        action_log.add("SYSTEM", "Visual pick system initialized (tracker + grasp planner)", "info")

    # Initialize vision task planner for camera-guided planning
    global vision_task_planner, scene_analyzer
    if _HAS_VISION_PLANNING and _HAS_PLANNING:
        scene_analyzer = SceneAnalyzer()
        vision_task_planner = VisionTaskPlanner(task_planner=task_planner)
        action_log.add("SYSTEM", "Vision task planner initialized", "info")

    # Initialize claw position predictor
    global claw_predictor
    if _HAS_CLAW_PREDICT:
        claw_predictor = ClawPositionPredictor()
        action_log.add(
            "SYSTEM", "Claw position predictor initialized (disabled by default)", "info"
        )

    # Initialize pose fusion pipeline
    global pose_fusion, arm3d_segmenters, arm3d_detector
    if _HAS_POSE_FUSION:
        pose_fusion = PoseFusion()
        arm3d_segmenters = {0: ArmSegmenter(), 1: ArmSegmenter()}
        arm3d_detector = JointDetector()
        action_log.add("SYSTEM", "Pose fusion pipeline initialized", "info")

    # Initialize VLA controller and data collector
    global vla_controller, vla_data_collector
    if _HAS_VLA:
        try:
            vla_controller = VLAController()
            vla_data_collector = DataCollector()
            action_log.add(
                "SYSTEM", "VLA controller initialized (Gemini backend, lazy-load)", "info"
            )
        except Exception as e:
            logger.warning("VLA init failed: %s (will work without VLA)", e)
            vla_controller = None
            vla_data_collector = None

    # Load calibrated camera models
    global camera_models
    try:
        from src.vision.camera_model import CameraModel

        for cam_id in [0, 1]:
            cm = CameraModel(cam_id)
            if cm.load():
                camera_models[cam_id] = cm
        if camera_models:
            action_log.add("SYSTEM", f"Camera models loaded: {list(camera_models.keys())}", "info")
    except Exception as e:
        logger.warning(f"Failed to load camera models: {e}")

    # Initialize collision detector + analyzer
    global collision_detector, collision_analyzer
    if _HAS_COLLISION:
        collision_detector = CollisionDetector()
        # TEMP: Disable collision detector — too aggressive for real arm
        # (3° threshold triggers on normal motion lag, backs off every command)
        collision_detector.enabled = False
        logger.info("Collision detector initialized but DISABLED (too aggressive for real arm)")
        collision_analyzer = CollisionAnalyzer()
        action_log.add(
            "SYSTEM",
            f"Collision detector initialized (vision={'yes' if collision_analyzer.vision_available else 'no'})",
            "info",
        )

    yield

    if smoother is not None:
        await smoother.stop()
    if arm is not None:
        try:
            arm.disconnect()
        except Exception:
            pass
    # Stop telemetry collector
    if _HAS_TELEMETRY:
        get_collector().log_system_event("shutdown", "system", "th3cl4w shutting down")
        get_collector().stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="th3cl4w", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def telemetry_middleware(request, call_next):
    """Log all /api/ requests to telemetry."""
    if _HAS_TELEMETRY and request.url.path.startswith("/api/"):
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        tc = get_collector()
        if tc.enabled:
            tc.log_web_request(
                endpoint=request.url.path,
                method=request.method,
                params=None,
                response_ms=elapsed_ms,
                status_code=response.status_code,
                ok=response.status_code < 400,
            )
        return response
    return await call_next(request)


# ---------------------------------------------------------------------------
# Connected WebSocket clients for command acks
# ---------------------------------------------------------------------------
ws_clients: list[WebSocket] = []

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SetJointRequest(BaseModel):
    id: int = Field(ge=0, le=5)
    angle: float


class SetAllJointsRequest(BaseModel):
    angles: List[float] = Field(min_length=6, max_length=6)


class SetGripperRequest(BaseModel):
    position: float = Field(ge=0.0, le=65.0)


class RawCommandRequest(BaseModel):
    payload: dict


class TextCommandRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_prev_state: Dict[str, Any] = {}
_cached_joint_angles: list = [0.0] * 6
_cached_gripper: float = 0.0


def get_arm_state() -> Dict[str, Any]:
    global _prev_state, _cached_joint_angles, _cached_gripper
    if arm is None:
        return {
            "connected": False,
            "joints": list(_cached_joint_angles),
            "gripper": _cached_gripper,
            "power": False,
            "enabled": False,
            "error": 0,
            "sim_mode": _sim_mode,
            "timestamp": time.time(),
        }

    # Prefer reliable (zero-filtered) joint angles if available
    if hasattr(arm, "get_reliable_joint_angles"):
        angles_raw = arm.get_reliable_joint_angles()
        if angles_raw is None:
            # Fall back to raw if reliable returns None (all stale)
            angles_raw = arm.get_joint_angles()
    else:
        angles_raw = arm.get_joint_angles()
    if angles_raw is not None:
        angles = [round(float(a), 2) for a in angles_raw]
    else:
        # Use cached values instead of zeros to prevent viz jumping
        angles = list(_cached_joint_angles)
    # Ensure exactly 6 joints
    angles = angles[:6] if len(angles) >= 6 else angles + [0.0] * (6 - len(angles))
    # Cache last known good values to avoid zero-snap on transient None reads
    _cached_joint_angles = list(angles)

    gripper = 0.0
    if hasattr(arm, "get_gripper_position"):
        gripper = round(float(arm.get_gripper_position()), 2)
    _cached_gripper = gripper

    status = arm.get_status() or {}

    state = {
        "connected": arm.is_connected,
        "joints": angles,
        "gripper": gripper,
        "power": bool(status.get("power_status", 0)),
        "enabled": bool(status.get("enable_status", 0)),
        "error": status.get("error_status", 0),
        "sim_mode": _sim_mode,
        "timestamp": time.time(),
    }

    # SAFETY: Sync smoother from arm feedback if not yet synced
    if smoother and not smoother.synced:
        smoother.sync_from_feedback(angles, gripper)
    # Keep feedback timestamp fresh for staleness gating
    elif smoother and smoother.synced:
        smoother._last_feedback_time = time.time()

    # Auto-sync enable state from DDS feedback on server restart:
    # If DDS reports enabled but smoother doesn't know, sync it.
    # This handles the case where server restarts while arm is already enabled.
    if smoother and not smoother._arm_enabled and state["enabled"] and state["power"]:
        smoother.set_arm_enabled(True)
        action_log.add("STATE", "Auto-synced enable state from DDS feedback", "warning")

    # Log state transitions + power-loss auto-recovery
    global _power_loss_recovery_task
    if _prev_state:
        if _prev_state.get("power") != state["power"]:
            action_log.add("STATE", f"Power: {'ON' if state['power'] else 'OFF'}", "warning")
            # Detect power loss transition (True -> False) and trigger auto-recovery
            if _prev_state.get("power") and not state["power"]:
                action_log.add("STATE", "Power loss detected — scheduling auto-recovery in 3s", "error")
                if _power_loss_recovery_task is None or _power_loss_recovery_task.done():
                    _power_loss_recovery_task = asyncio.create_task(_auto_recover_power())
        if _prev_state.get("enabled") != state["enabled"]:
            action_log.add(
                "STATE", f"Motors: {'ENABLED' if state['enabled'] else 'DISABLED'}", "warning"
            )
        if _prev_state.get("error") != state["error"] and state["error"]:
            action_log.add("STATE", f"Error changed: {state['error']}", "error")
    _prev_state = state.copy()

    return state


def cmd_response(
    success: bool, action: str, extra: str = "", correlation_id: str | None = None
) -> JSONResponse:
    state = get_arm_state()
    level = "info" if success else "error"
    detail = f"{'OK' if success else 'FAILED'}"
    if extra:
        detail += f" — {extra}"
    action_log.add(action, detail, level)
    resp_data: Dict[str, Any] = {"ok": success, "action": action, "state": state}
    if correlation_id is not None:
        resp_data["correlation_id"] = correlation_id
    return JSONResponse(resp_data)


def _telem_cmd_sent(endpoint: str, params: Dict[str, Any], correlation_id: str) -> None:
    """Emit CMD_SENT if telemetry is available and enabled."""
    if _HAS_TELEMETRY:
        tc = get_collector()
        if tc.enabled:
            tc.emit(
                "web", EventType.CMD_SENT, {"endpoint": endpoint, "params": params}, correlation_id
            )


def _new_cid() -> str | None:
    """Generate a correlation_id if telemetry is available."""
    if _HAS_TELEMETRY:
        return get_collector().new_correlation_id()
    return None


async def broadcast_ack(action: str, success: bool):
    """Send command acknowledgment to all connected WS clients."""
    msg = {"type": "ack", "action": action, "ok": success, "timestamp": time.time()}
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def _validate_env():
    """Check required env vars at startup and warn if missing."""
    if not os.environ.get("GEMINI_API_KEY"):
        logging.getLogger("th3cl4w").warning(
            "GEMINI_API_KEY not set — Gemini vision/LLM features will be unavailable. "
            "Add it to .env or set the environment variable."
        )

_validate_env()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/config/secrets-status")
async def api_secrets_status():
    """Return which secret keys are configured (True/False) without revealing values."""
    return {
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")
                               and os.environ.get("GEMINI_API_KEY") != "your-gemini-api-key-here"),
    }


## Camera config storage
CAMERA_CONFIG_PATH = Path(_project_root) / "data" / "camera_config.json"
VALID_PERSPECTIVES = ["overhead", "side", "front", "arm-mounted", "custom"]

def _load_camera_config() -> dict:
    """Load camera config from JSON file, return defaults if missing."""
    defaults = {
        str(i): {"label": f"Camera {i}", "perspective": "custom"}
        for i in range(3)
    }
    if CAMERA_CONFIG_PATH.exists():
        try:
            with open(CAMERA_CONFIG_PATH) as f:
                saved = json.load(f)
            # Merge with defaults so new cameras get defaults
            for k, v in defaults.items():
                if k not in saved:
                    saved[k] = v
            return saved
        except Exception:
            pass
    return defaults

def _save_camera_config(config: dict):
    """Save camera config to JSON file."""
    CAMERA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMERA_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


@app.get("/api/cameras")
async def api_cameras():
    """Proxy camera status from the camera server, enriched with labels/perspectives."""
    import httpx

    cam_config = _load_camera_config()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:8081/status")
            status = resp.json()
    except Exception:
        status = {
            "error": "Camera server unavailable",
            "0": {"connected": False},
            "1": {"connected": False},
        }

    # Enrich each camera entry with label and perspective
    for cam_id in list(status.keys()):
        if cam_id.isdigit() and isinstance(status[cam_id], dict):
            cfg = cam_config.get(cam_id, {"label": f"Camera {cam_id}", "perspective": "custom"})
            status[cam_id]["label"] = cfg.get("label", f"Camera {cam_id}")
            status[cam_id]["perspective"] = cfg.get("perspective", "custom")

    return status


@app.get("/api/cameras/config")
async def api_cameras_config_get():
    """Return camera label/perspective configuration."""
    return _load_camera_config()


class CameraConfigRequest(BaseModel):
    camera_id: int = Field(ge=0, le=2)
    label: str = Field(min_length=1, max_length=50)
    perspective: str


@app.post("/api/cameras/config")
async def api_cameras_config_set(req: CameraConfigRequest):
    """Set label and perspective for a camera."""
    if req.perspective not in VALID_PERSPECTIVES:
        return JSONResponse(
            {"ok": False, "error": f"Invalid perspective. Must be one of: {VALID_PERSPECTIVES}"},
            status_code=400,
        )
    config = _load_camera_config()
    config[str(req.camera_id)] = {"label": req.label, "perspective": req.perspective}
    _save_camera_config(config)
    action_log.add("CAMERA_CONFIG", f"Camera {req.camera_id}: label={req.label!r}, perspective={req.perspective}", "info")
    return {"ok": True, "config": config}


@app.get("/api/state")
async def api_state():
    """Return current arm state (joints, gripper, power, enabled, error)."""
    return get_arm_state()


@app.get("/api/sim/status")
async def api_sim_status():
    """Return whether the server is in simulation mode."""
    return {"sim_mode": _sim_mode}


@app.post("/api/sim/toggle")
async def api_sim_toggle():
    """Toggle between SIM and LIVE mode at runtime.

    When switching to SIM: creates a SimulatedArm, stops DDS.
    When switching to LIVE: attempts DDS connection, falls back to SIM on failure.
    """
    global arm, smoother, _sim_mode

    if _sim_mode:
        # Switch SIM → LIVE
        action_log.add("SYSTEM", "Switching from SIM to LIVE mode...", "warning")
        try:
            from src.interface.d1_dds_connection import D1DDSConnection

            tc = get_collector() if _HAS_TELEMETRY else None
            new_arm = D1DDSConnection(collector=tc)
            if new_arm.connect(interface_name=args.interface):
                # Tear down old sim arm
                if arm is not None:
                    arm.disconnect()
                if smoother is not None:
                    await smoother.stop()

                arm = new_arm
                _sim_mode = False

                # Restart smoother with new arm
                tc_ref = get_collector() if _HAS_TELEMETRY else None
                smoother = CommandSmoother(
                    arm,
                    rate_hz=10.0,
                    smoothing_factor=0.35,
                    max_step_deg=MAX_STEP_DEG,
                    collector=tc_ref,
                    safety_monitor=safety_monitor,
                )
                await smoother.start()

                action_log.add("SYSTEM", "Switched to LIVE mode (DDS connected)", "info")
                return {"ok": True, "sim_mode": False}
            else:
                new_arm.disconnect()
                action_log.add("SYSTEM", "Failed to connect DDS — staying in SIM mode", "error")
                return JSONResponse(
                    {"ok": False, "sim_mode": True, "error": "DDS connection failed"},
                    status_code=503,
                )
        except Exception as e:
            action_log.add("SYSTEM", f"Failed to switch to LIVE: {e}", "error")
            return JSONResponse({"ok": False, "sim_mode": True, "error": str(e)}, status_code=503)
    else:
        # Switch LIVE → SIM
        action_log.add("SYSTEM", "Switching from LIVE to SIM mode...", "warning")
        if smoother is not None:
            await smoother.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

        arm = SimulatedArm()
        arm.start_feedback_loop(rate_hz=10.0)
        _sim_mode = True

        tc_ref = get_collector() if _HAS_TELEMETRY else None
        smoother = CommandSmoother(
            arm,
            rate_hz=10.0,
            smoothing_factor=0.35,
            max_step_deg=MAX_STEP_DEG,
            collector=tc_ref,
            safety_monitor=safety_monitor,
        )
        await smoother.start()

        action_log.add("SYSTEM", "Switched to SIM mode", "info")
        return {"ok": True, "sim_mode": True}


@app.get("/api/diagnostics/feedback")
async def api_diagnostics_feedback():
    """Return DDS feedback health and recent samples for debugging."""
    if arm is None or not hasattr(arm, "get_feedback_health"):
        return {"available": False, "error": "No DDS connection or feedback monitor"}
    health = arm.get_feedback_health()
    recent = arm.feedback_monitor.get_recent_samples(20)
    return {"available": True, "health": health, "recent_samples": recent}


@app.get("/api/log")
async def api_log():
    """Return recent action log entries."""
    return {"entries": action_log.last(50)}


@app.post("/api/command/enable")
async def cmd_enable():
    """Enable motors (requires power to be on first)."""
    cid = _new_cid()
    _telem_cmd_sent("enable", {}, cid)
    if arm is None:
        return cmd_response(False, "ENABLE", "No arm connected", cid)
    state = arm.get_status() or {}
    if not state.get("power_status"):
        action_log.add("ENABLE", "REJECTED — power is off, power on first", "error")
        resp_data: Dict[str, Any] = {
            "ok": False,
            "action": "ENABLE",
            "error": "Power must be on before enabling",
            "state": get_arm_state(),
        }
        if cid:
            resp_data["correlation_id"] = cid
        return JSONResponse(resp_data)
    ok = arm.enable_motors(_correlation_id=cid)
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "enable",
            "web",
            f"Motors enable: {'OK' if ok else 'FAILED'}",
            correlation_id=cid,
            level="info" if ok else "error",
        )
    if ok and smoother:
        smoother.set_arm_enabled(True)
    resp = cmd_response(ok, "ENABLE", correlation_id=cid)
    await broadcast_ack("ENABLE", ok)
    return resp


@app.post("/api/command/disable")
async def cmd_disable():
    """Disable motors."""
    cid = _new_cid()
    _telem_cmd_sent("disable", {}, cid)
    ok = arm.disable_motors(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "disable", "web", f"Motors disable: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "DISABLE", correlation_id=cid)
    await broadcast_ack("DISABLE", ok)
    return resp


@app.post("/api/command/power-on")
async def cmd_power_on():
    """Power on the arm."""
    cid = _new_cid()
    _telem_cmd_sent("power-on", {}, cid)
    ok = arm.power_on(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "power_on", "web", f"Power on: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    resp = cmd_response(ok, "POWER_ON", correlation_id=cid)
    await broadcast_ack("POWER_ON", ok)
    return resp


@app.post("/api/command/power-off")
async def cmd_power_off():
    """Power off the arm (also disables motors)."""
    cid = _new_cid()
    _telem_cmd_sent("power-off", {}, cid)
    ok = arm.power_off(_correlation_id=cid) if arm else False
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "power_off", "web", f"Power off: {'OK' if ok else 'FAILED'}", correlation_id=cid
        )
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "POWER_OFF", correlation_id=cid)
    await broadcast_ack("POWER_OFF", ok)
    return resp


@app.post("/api/command/reset")
async def cmd_reset():
    """Reset arm to zero position."""
    cid = _new_cid()
    _telem_cmd_sent("reset", {}, cid)
    ok = arm.reset_to_zero(_correlation_id=cid) if arm else False
    if smoother:
        smoother.set_arm_enabled(False)
    resp = cmd_response(ok, "RESET", correlation_id=cid)
    await broadcast_ack("RESET", ok)
    return resp


@app.post("/api/command/reset-enable")
async def cmd_reset_enable():
    """Combined reset + enable: reset, wait 2s, then enable. Required after overcurrent."""
    cid = _new_cid()
    _telem_cmd_sent("reset-enable", {}, cid)
    if arm is None:
        return cmd_response(False, "RESET_ENABLE", "No arm connected", cid)
    action_log.add("RESET_ENABLE", "Starting reset+enable sequence", "info")
    ok_reset = arm.reset_to_zero(_correlation_id=cid)
    if smoother:
        smoother.set_arm_enabled(False)
    if not ok_reset:
        return cmd_response(False, "RESET_ENABLE", "Reset failed", cid)
    await asyncio.sleep(2.0)
    ok_enable = arm.enable_motors(_correlation_id=cid)
    if ok_enable and smoother:
        smoother.set_arm_enabled(True)
    action_log.add(
        "RESET_ENABLE",
        f"reset={'OK' if ok_reset else 'FAIL'} enable={'OK' if ok_enable else 'FAIL'}",
        "info" if ok_enable else "error",
    )
    resp = cmd_response(ok_enable, "RESET_ENABLE", f"reset=OK enable={'OK' if ok_enable else 'FAIL'}", cid)
    await broadcast_ack("RESET_ENABLE", ok_enable)
    return resp


# ---------------------------------------------------------------------------
# Power-loss auto-recovery state
# ---------------------------------------------------------------------------
_power_loss_recovery_task: Optional[asyncio.Task] = None


async def _auto_recover_power():
    """Wait 3s then attempt reset+enable after power loss."""
    await asyncio.sleep(3.0)
    action_log.add("AUTO_RECOVERY", "Attempting reset+enable after power loss", "warning")
    if arm is None:
        action_log.add("AUTO_RECOVERY", "No arm connected, aborting", "error")
        return
    ok_reset = arm.reset_to_zero()
    if smoother:
        smoother.set_arm_enabled(False)
    if not ok_reset:
        action_log.add("AUTO_RECOVERY", "Reset failed", "error")
        return
    await asyncio.sleep(2.0)
    ok_enable = arm.enable_motors()
    if ok_enable and smoother:
        smoother.set_arm_enabled(True)
    action_log.add(
        "AUTO_RECOVERY",
        f"Recovery {'succeeded' if ok_enable else 'failed'}: reset={'OK' if ok_reset else 'FAIL'} enable={'OK' if ok_enable else 'FAIL'}",
        "info" if ok_enable else "error",
    )


@app.post("/api/command/set-joint")
async def cmd_set_joint(req: SetJointRequest):
    """Set a single joint to a target angle (degrees)."""
    cid = _new_cid()
    _telem_cmd_sent("set-joint", {"id": req.id, "angle": req.angle}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_JOINT", f"Request: J{req.id} -> {req.angle}°", "info")
    lo, hi = JOINT_LIMITS_DEG.get(req.id, (-135, 135))
    if not (lo <= req.angle <= hi):
        action_log.add(
            "SET_JOINT", f"REJECTED — J{req.id} angle {req.angle}° outside [{lo}, {hi}]", "error"
        )
        resp_data: Dict[str, Any] = {
            "ok": False,
            "action": "SET_JOINT",
            "error": f"Angle {req.angle} out of range [{lo}, {hi}]",
            "state": get_arm_state(),
        }
        if cid:
            resp_data["correlation_id"] = cid
        return JSONResponse(resp_data, status_code=400)

    # Safe reach check: rough torque proxy using current targets
    current_joints = list(armState_joints())
    proposed = list(current_joints)
    proposed[req.id] = req.angle
    torque_proxy = abs(proposed[1]) + abs(proposed[2]) * 0.7
    if torque_proxy > 100:
        action_log.add(
            "SET_JOINT",
            f"REJECTED — torque proxy {torque_proxy:.1f} > 100 (J1={proposed[1]:.1f}° J2={proposed[2]:.1f}°)",
            "warning",
        )
        resp_data = {
            "ok": False,
            "action": "SET_JOINT",
            "error": "Pose may exceed torque limits",
            "torque_proxy": round(torque_proxy, 1),
            "state": get_arm_state(),
        }
        if cid:
            resp_data["correlation_id"] = cid
        return JSONResponse(resp_data, status_code=400)

    if smoother and smoother.running:
        smoother.set_joint_target(req.id, req.angle)
        ok = True
    else:
        ok = arm.set_joint(req.id, req.angle, _correlation_id=cid) if arm else False

    # Schedule stall detection check
    if ok:
        asyncio.create_task(_check_stall(req.id, req.angle, cid))

    resp = cmd_response(ok, "SET_JOINT", f"J{req.id} = {req.angle}°", cid)
    await broadcast_ack("SET_JOINT", ok)
    return resp


async def _check_stall(joint_id: int, target_angle: float, cid: str | None):
    """After 3s, check if joint reached within 5° of target. Log warning if stalled."""
    await asyncio.sleep(3.0)
    state = get_arm_state()
    actual = state["joints"][joint_id]
    if abs(actual - target_angle) > 5.0:
        msg = f"J{joint_id} stalled at {actual:.1f}° (target {target_angle:.1f}°)"
        action_log.add("STALL", msg, "warning")
        logger.warning(msg)


@app.post("/api/command/set-all-joints")
async def cmd_set_all_joints(req: SetAllJointsRequest):
    """Set all 6 joints to target angles simultaneously."""
    cid = _new_cid()
    _telem_cmd_sent("set-all-joints", {"angles": req.angles}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_ALL_JOINTS", f"Request: {[round(a,1) for a in req.angles]}", "info")
    for i, a in enumerate(req.angles):
        lo, hi = JOINT_LIMITS_DEG.get(i, (-135, 135))
        if not (lo <= a <= hi):
            action_log.add(
                "SET_ALL_JOINTS", f"REJECTED — J{i} angle {a}° outside [{lo}, {hi}]", "error"
            )
            resp_data2: Dict[str, Any] = {
                "ok": False,
                "action": "SET_ALL_JOINTS",
                "error": f"J{i} angle {a} out of range [{lo}, {hi}]",
                "state": get_arm_state(),
            }
            if cid:
                resp_data2["correlation_id"] = cid
            return JSONResponse(resp_data2, status_code=400)
    if smoother and smoother.running:
        smoother.set_all_joints_target(req.angles)
        ok = True
    else:
        ok = arm.set_all_joints(req.angles, _correlation_id=cid) if arm else False
    resp = cmd_response(ok, "SET_ALL_JOINTS", correlation_id=cid)
    await broadcast_ack("SET_ALL_JOINTS", ok)
    return resp


@app.post("/api/command/set-gripper")
async def cmd_set_gripper(req: SetGripperRequest):
    cid = _new_cid()
    _telem_cmd_sent("set-gripper", {"position": req.position}, cid)
    if not (smoother and smoother._arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)
    action_log.add("SET_GRIPPER", f"Request: {req.position} mm", "info")
    if smoother and smoother.running:
        smoother.set_gripper_target(req.position)
        ok = True
    else:
        ok = False
        if arm and hasattr(arm, "set_gripper"):
            ok = arm.set_gripper(req.position, _correlation_id=cid)
    resp = cmd_response(ok, "SET_GRIPPER", f"{req.position} mm", cid)
    await broadcast_ack("SET_GRIPPER", ok)
    return resp


# ── Gripper Contact Detection endpoints ──────────────────────────────
from src.control.contact_detector import GripperContactDetector, OBJECT_PROFILES

_contact_detector = GripperContactDetector()


@app.post("/api/gripper/adaptive-close")
async def gripper_adaptive_close(
    profile: Optional[str] = None,
    initial_mm: float = 15.0,
    object_min_mm: float = 25.0,
):
    """Close gripper with adaptive contact detection."""
    result = await _contact_detector.adaptive_grip(
        initial_mm=initial_mm,
        object_min_mm=object_min_mm,
        profile=profile,
    )
    return {
        "contacted": result.contacted,
        "final_mm": result.final_mm,
        "steps_taken": result.steps_taken,
        "grip_force_mm": result.grip_force_mm,
    }


@app.get("/api/gripper/contact-status")
async def gripper_contact_status():
    """Return the last contact detection result."""
    r = _contact_detector.last_result
    if r is None:
        return {"status": "no_detection_yet"}
    return {
        "contacted": r.contacted,
        "status": r.status.value,
        "final_mm": r.final_mm,
        "stable_mm": r.stable_mm,
        "time_s": round(r.time_s, 3),
    }


@app.post("/api/command/stop")
async def cmd_stop():
    """Emergency stop: disable motors AND power off."""
    cid = _new_cid()
    _telem_cmd_sent("stop", {}, cid)
    action_log.add("EMERGENCY_STOP", "⚠ TRIGGERED", "error")
    if safety_monitor:
        safety_monitor.trigger_estop("API emergency stop")
    if smoother:
        smoother.emergency_stop()
    if _HAS_TELEMETRY:
        get_collector().log_system_event(
            "estop", "web", "Emergency stop triggered", correlation_id=cid, level="error"
        )
    ok1 = arm.disable_motors(_correlation_id=cid) if arm else False
    ok2 = arm.power_off(_correlation_id=cid) if arm else False
    ok = ok1 and ok2
    resp = cmd_response(
        ok,
        "EMERGENCY_STOP",
        f"disable={'OK' if ok1 else 'FAIL'} power_off={'OK' if ok2 else 'FAIL'}",
        cid,
    )
    await broadcast_ack("EMERGENCY_STOP", ok)
    return resp


# ---------------------------------------------------------------------------
# Text command interface — natural language arm control
# ---------------------------------------------------------------------------


@app.post("/api/command/text")
async def cmd_text(req: TextCommandRequest):
    """Parse a natural language command and execute the corresponding arm action.

    Examples:
        "wave hello", "go home", "open gripper", "move joint 0 to 45",
        "reach forward", "point left", "close the claw", "stop"
    """
    if not _HAS_TEXT_CMD:
        return JSONResponse(
            {"ok": False, "error": "text command module not available"}, status_code=501
        )

    cid = _new_cid()
    parsed = parse_command(req.text)
    action_log.add(
        "TEXT_CMD",
        f"Input: {req.text!r} -> {parsed.command_type.value}: {parsed.description}",
        "info",
    )

    if parsed.command_type == CommandType.UNKNOWN:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Could not understand command: {req.text}",
                "parsed": {
                    "type": parsed.command_type.value,
                    "description": parsed.description,
                    "confidence": parsed.confidence,
                },
            },
            status_code=400,
        )

    # --- Stop ---
    if parsed.command_type == CommandType.STOP:
        return await cmd_stop()

    # --- Power commands ---
    if parsed.command_type == CommandType.POWER:
        if parsed.action == "power-on":
            return await cmd_power_on()
        elif parsed.action == "power-off":
            return await cmd_power_off()
        elif parsed.action == "enable":
            return await cmd_enable()
        elif parsed.action == "disable":
            return await cmd_disable()

    # --- Gripper ---
    if parsed.command_type == CommandType.SET_GRIPPER:
        if parsed.gripper_mm is not None:
            return await cmd_set_gripper(SetGripperRequest(position=parsed.gripper_mm))

    # --- Tasks (wave, home, ready, nod, shake) ---
    if parsed.command_type == CommandType.TASK:
        if parsed.action == "wave":
            return await task_wave(WaveRequest(speed=parsed.speed))
        elif parsed.action == "home":
            return await task_home(TaskRequest(speed=parsed.speed))
        elif parsed.action == "ready":
            return await task_ready(TaskRequest(speed=parsed.speed))
        elif parsed.action == "nod":
            return await _execute_nod(parsed.speed, cid)
        elif parsed.action == "shake":
            return await _execute_shake(parsed.speed, cid)

    # --- Set single joint ---
    if parsed.command_type == CommandType.SET_JOINT and parsed.joints:
        joint_id, angle = next(iter(parsed.joints.items()))
        return await cmd_set_joint(SetJointRequest(id=joint_id, angle=angle))

    # --- Set all joints (directional moves, named poses) ---
    if parsed.command_type == CommandType.SET_ALL_JOINTS and parsed.all_joints:
        if len(parsed.all_joints) == 6:
            # Use task planner for smooth trajectory if available
            if _HAS_PLANNING and smoother and smoother.arm_enabled:
                current = _get_current_joints()
                gripper = armState_gripper()
                traj = task_planner.planner.linear_joint_trajectory(
                    current,
                    np.array(parsed.all_joints),
                    parsed.speed,
                    gripper,
                    gripper,
                )
                global _active_task
                if _active_task and not _active_task.done():
                    _active_task.cancel()
                _active_task = asyncio.create_task(_execute_trajectory(traj, parsed.description))
                action_log.add(
                    "TEXT_CMD",
                    f"Executing: {parsed.description} ({len(traj.points)} pts, {traj.duration:.1f}s)",
                    "info",
                )
                return {
                    "ok": True,
                    "action": "TEXT_CMD",
                    "description": parsed.description,
                    "points": len(traj.points),
                    "duration_s": round(traj.duration, 1),
                    "parsed": {
                        "type": parsed.command_type.value,
                        "confidence": parsed.confidence,
                    },
                }
            else:
                return await cmd_set_all_joints(SetAllJointsRequest(angles=parsed.all_joints))

    return JSONResponse(
        {
            "ok": False,
            "error": f"Parsed but could not execute: {parsed.description}",
            "parsed": {
                "type": parsed.command_type.value,
                "description": parsed.description,
                "confidence": parsed.confidence,
            },
        },
        status_code=400,
    )


async def _execute_nod(speed: float, cid: str | None) -> dict:
    """Execute a nodding (yes) gesture by pitching the wrist up and down."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    waypoints = [Waypoint(current, gripper, speed)]

    nod_up = current.copy()
    nod_up[4] = max(current[4] - 20.0, -80.0)
    nod_down = current.copy()
    nod_down[4] = min(current[4] + 20.0, 80.0)

    for _ in range(2):
        waypoints.append(Waypoint(nod_up, gripper, speed))
        waypoints.append(Waypoint(nod_down, gripper, speed))
    waypoints.append(Waypoint(current, gripper, speed))

    traj = task_planner.planner.plan_waypoints(waypoints)
    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(traj, "Nod"))
    return {
        "ok": True,
        "action": "TEXT_CMD",
        "description": "Nod gesture",
        "points": len(traj.points),
        "duration_s": round(traj.duration, 1),
    }


async def _execute_shake(speed: float, cid: str | None) -> dict:
    """Execute a shaking (no) gesture by rotating the base yaw side to side."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    waypoints = [Waypoint(current, gripper, speed)]

    shake_left = current.copy()
    shake_left[0] = max(current[0] - 25.0, -135.0)
    shake_right = current.copy()
    shake_right[0] = min(current[0] + 25.0, 135.0)

    for _ in range(2):
        waypoints.append(Waypoint(shake_left, gripper, speed))
        waypoints.append(Waypoint(shake_right, gripper, speed))
    waypoints.append(Waypoint(current, gripper, speed))

    traj = task_planner.planner.plan_waypoints(waypoints)
    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(traj, "Shake"))
    return {
        "ok": True,
        "action": "TEXT_CMD",
        "description": "Shake gesture",
        "points": len(traj.points),
        "duration_s": round(traj.duration, 1),
    }


# ---------------------------------------------------------------------------
# WebSocket — stream state at 10Hz
# ---------------------------------------------------------------------------


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    action_log.add("WS", "Client connected", "info")
    try:
        while True:
            state = get_arm_state()
            state["log"] = action_log.last(30)

            # Include claw prediction data if predictor is enabled
            if claw_predictor is not None and claw_predictor.enabled:
                last_pred = claw_predictor.get_last_prediction()
                if last_pred is not None:
                    state["claw_prediction"] = last_pred.to_dict()
                else:
                    state["claw_prediction"] = {"enabled": True, "detected": False}
            elif claw_predictor is not None:
                state["claw_prediction"] = {"enabled": False}
            # Feed collision detector with commanded vs actual
            if _HAS_COLLISION and collision_detector and smoother and smoother.arm_enabled:
                commanded = []
                for j in range(6):
                    t = smoother._target[j]
                    c = smoother._current[j]
                    # Use target if set, else current smoother position
                    commanded.append(
                        t if t is not None else (c if c is not None else state["joints"][j])
                    )
                stall = collision_detector.update(commanded, state["joints"])
                if stall is not None:
                    asyncio.create_task(_handle_collision(stall, ws_clients[:]))

            await ws.send_json(state)
            if _HAS_TELEMETRY:
                tc = get_collector()
                if tc.enabled:
                    tc.emit(
                        "web",
                        EventType.WS_SEND,
                        {
                            "client_count": len(ws_clients),
                            "state_timestamp": state.get("timestamp"),
                        },
                    )
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        action_log.add("WS", "Client disconnected", "info")
    except Exception as e:
        action_log.add("WS", f"Client error: {e}", "error")
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Debug / telemetry endpoints
# ---------------------------------------------------------------------------


@app.get("/api/debug/telemetry")
async def debug_telemetry(
    limit: int = 100, event_type: str | None = None, source: str | None = None
):
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    tc = get_collector()
    et = None
    if event_type:
        try:
            et = EventType(event_type)
        except ValueError:
            return JSONResponse({"error": f"unknown event_type: {event_type}"}, status_code=400)
    events = tc.get_events(limit=limit, event_type=et, source=source)
    return [
        {
            "timestamp_ms": e.timestamp_ms,
            "wall_time_ms": e.wall_time_ms,
            "source": e.source,
            "event_type": e.event_type.value,
            "payload": e.payload,
            "correlation_id": e.correlation_id,
        }
        for e in events
    ]


@app.get("/api/debug/stats")
async def debug_stats():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    return get_collector().get_stats()


@app.post("/api/debug/enable")
async def debug_enable():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    get_collector().enable()
    return {"enabled": True}


@app.post("/api/debug/disable")
async def debug_disable():
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    get_collector().disable()
    return {"enabled": False}


@app.get("/api/debug/pipeline/{correlation_id}")
async def debug_pipeline(correlation_id: str):
    if not _HAS_TELEMETRY:
        return JSONResponse({"error": "telemetry not available"}, status_code=501)
    pipeline = get_collector().get_pipeline(correlation_id)
    return pipeline


# ---------------------------------------------------------------------------
# WebSocket — stream telemetry events in real-time
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Stream telemetry events to clients in real-time."""
    await ws.accept()
    if not _HAS_TELEMETRY:
        await ws.close(code=1011, reason="Telemetry not available")
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    loop = asyncio.get_event_loop()

    def on_event(event_dict):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event_dict)
        except Exception:
            pass

    tc = get_collector()
    tc.subscribe(on_event)
    try:
        while True:
            event_dict = await queue.get()
            await ws.send_json(event_dict)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("Telemetry WS error: %s", e)
    finally:
        tc.unsubscribe(on_event)


# ---------------------------------------------------------------------------
# Telemetry query endpoints — expose TelemetryQuery (SQLite read-only) data
# ---------------------------------------------------------------------------


def _get_query() -> "TelemetryQuery | None":
    """Open a read-only TelemetryQuery against the active database."""
    if not _HAS_QUERY:
        return None
    db_path = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"
    if not db_path.exists():
        return None
    try:
        return TelemetryQuery(str(db_path))
    except Exception:
        return None


@app.get("/api/query/summary")
async def query_summary():
    """Session summary: total commands, feedback, events, rates, errors."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.summary()
    finally:
        q.close()


@app.get("/api/query/db-stats")
async def query_db_stats():
    """Row counts per telemetry table."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_db_stats()
    finally:
        q.close()


@app.get("/api/query/joint-history/{joint}")
async def query_joint_history(joint: int, limit: int = 500):
    """Feedback angle history for a single joint."""
    if not (0 <= joint <= 6):
        return JSONResponse({"error": "joint must be 0-6"}, status_code=400)
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_joint_history(joint, limit=limit)
    finally:
        q.close()


@app.get("/api/query/tracking-error/{joint}")
async def query_tracking_error(joint: int):
    """Command vs feedback tracking error for a joint."""
    if not (0 <= joint <= 5):
        return JSONResponse({"error": "joint must be 0-5"}, status_code=400)
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_tracking_error(joint)
    finally:
        q.close()


@app.get("/api/query/command-rate")
async def query_command_rate(window: float = 10.0):
    """DDS command rate over a time window."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_command_rate(window)
    finally:
        q.close()


@app.get("/api/query/web-latency")
async def query_web_latency(endpoint: str | None = None, limit: int = 100):
    """Web request latency history."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_web_request_latency(endpoint=endpoint, limit=limit)
    finally:
        q.close()


@app.get("/api/query/system-events")
async def query_system_events(event_type: str | None = None, limit: int = 100):
    """System event log."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_system_events(event_type=event_type, limit=limit)
    finally:
        q.close()


@app.get("/api/query/smoother")
async def query_smoother_state(joint: int | None = None, limit: int = 500):
    """Smoother state (target vs current vs sent per joint)."""
    q = _get_query()
    if q is None:
        return JSONResponse({"error": "telemetry query not available"}, status_code=501)
    try:
        return q.get_smoother_state(joint=joint, limit=limit)
    finally:
        q.close()


# ---------------------------------------------------------------------------
# Planning endpoints — execute pre-built tasks through the smoother
# ---------------------------------------------------------------------------

_active_task: Optional[asyncio.Task] = None


async def _execute_trajectory(trajectory, label: str) -> None:
    """Feed trajectory points into the smoother at their scheduled times.

    Runs as a background asyncio task so the HTTP endpoint returns immediately.
    """
    if smoother is None or not smoother.arm_enabled:
        action_log.add("TASK", f"{label} — arm not enabled, aborting", "error")
        return

    action_log.add(
        "TASK",
        f"Executing {label} ({len(trajectory.points)} points, {trajectory.duration:.1f}s)",
        "info",
    )
    t0 = time.monotonic()

    for pt in trajectory.points:
        if smoother is None or not smoother.arm_enabled:
            action_log.add("TASK", f"{label} — aborted (arm disabled)", "error")
            return

        angles = [float(a) for a in pt.positions[:6]]
        smoother.set_all_joints_target(angles)
        if pt.gripper_mm is not None:
            smoother.set_gripper_target(pt.gripper_mm)

        # Wait until this point's scheduled time
        elapsed = time.monotonic() - t0
        wait = pt.time - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

    action_log.add("TASK", f"{label} complete ({time.monotonic() - t0:.1f}s)", "info")


class TaskRequest(BaseModel):
    speed: float = Field(default=0.6, ge=0.1, le=1.0)


@app.post("/api/task/home")
async def task_home(req: TaskRequest = TaskRequest()):
    """Plan and execute a smooth return to home position."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    result = task_planner.go_home(current, speed_factor=req.speed, gripper_mm=gripper)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(result.trajectory, "Go Home"))
    action_log.add(
        "TASK",
        f"Home planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_HOME",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


@app.post("/api/task/ready")
async def task_ready(req: TaskRequest = TaskRequest()):
    """Plan and execute move to ready/neutral position."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    gripper = armState_gripper()
    result = task_planner.go_ready(current, speed_factor=req.speed, gripper_mm=gripper)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(_execute_trajectory(result.trajectory, "Go Ready"))
    action_log.add(
        "TASK",
        f"Ready planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_READY",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


class WaveRequest(BaseModel):
    speed: float = Field(default=0.8, ge=0.1, le=1.0)
    waves: int = Field(default=3, ge=1, le=10)


@app.post("/api/task/wave")
async def task_wave(req: WaveRequest = WaveRequest()):
    """Plan and execute a wave gesture."""
    global _active_task
    if not _HAS_PLANNING:
        return JSONResponse(
            {"ok": False, "error": "planning module not available"}, status_code=501
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    current = _get_current_joints()
    result = task_planner.wave(current, n_waves=req.waves, speed_factor=req.speed)
    if result.status != TaskStatus.SUCCESS:
        return JSONResponse({"ok": False, "error": result.message}, status_code=500)

    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = asyncio.create_task(
        _execute_trajectory(result.trajectory, f"Wave ({req.waves}x)")
    )
    action_log.add(
        "TASK",
        f"Wave planned: {len(result.trajectory.points)} pts, {result.trajectory.duration:.1f}s",
        "info",
    )
    return {
        "ok": True,
        "action": "TASK_WAVE",
        "points": len(result.trajectory.points),
        "duration_s": round(result.trajectory.duration, 1),
    }


@app.post("/api/task/stop")
async def task_stop():
    """Cancel any running task (does NOT e-stop the arm)."""
    global _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
        action_log.add("TASK", "Task cancelled by user", "warning")
        return {"ok": True, "action": "TASK_STOP"}
    return {"ok": True, "action": "TASK_STOP", "detail": "No task running"}


def armState_joints() -> list:
    """Get current joint angles as a list of floats."""
    return list(_cached_joint_angles)


def _get_current_joints():
    """Get current joint angles as numpy array for the planner."""
    if arm is None:
        return np.zeros(6)
    angles_raw = arm.get_joint_angles()
    if angles_raw is not None and len(angles_raw) >= 6:
        return np.array([float(a) for a in angles_raw[:6]])
    return np.zeros(6)


def armState_gripper() -> float:
    """Get current gripper position in mm."""
    if arm is None:
        return 0.0
    if hasattr(arm, "get_gripper_position"):
        return float(arm.get_gripper_position())
    return 0.0


# ---------------------------------------------------------------------------
# Bifocal Workspace Mapping endpoints — planning/measurement only
# ---------------------------------------------------------------------------


@app.post("/api/bifocal/toggle")
async def bifocal_toggle():
    """Toggle the bifocal workspace mapper on/off."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    enabled = workspace_mapper.toggle()
    action_log.add("BIFOCAL", f"Workspace mapper {'enabled' if enabled else 'disabled'}", "info")
    return {"ok": True, "enabled": enabled}


@app.get("/api/bifocal/status")
async def bifocal_status():
    """Get bifocal workspace mapper status."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return {"available": False}
    status = workspace_mapper.get_status()
    status["available"] = True
    return status


@app.post("/api/bifocal/update")
async def bifocal_update():
    """Trigger a workspace map update from current camera frames."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    if not workspace_mapper.enabled:
        return JSONResponse({"ok": False, "error": "Mapper not enabled"}, status_code=409)

    # Grab snapshots from both cameras
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        import cv2

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        result = workspace_mapper.update_from_frames(left, right)
        return {"ok": True, **result}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/bifocal/workspace")
async def bifocal_workspace(max_points: int = 500):
    """Get occupied voxel positions for 3D visualization."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return {"points": [], "summary": {}}
    points = workspace_mapper.get_occupied_points(max_points)
    summary = workspace_mapper.get_occupancy_summary()
    return {"points": points, "summary": summary}


@app.post("/api/bifocal/clear")
async def bifocal_clear():
    """Clear the workspace map."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)
    workspace_mapper.clear()
    action_log.add("BIFOCAL", "Workspace map cleared", "info")
    return {"ok": True}


class CalibScaleRequest(BaseModel):
    square_size_mm: float = Field(default=25.0, gt=0)


@app.post("/api/bifocal/calibrate-scale")
async def bifocal_calibrate_scale(req: CalibScaleRequest = CalibScaleRequest()):
    """Calibrate real-world scale using a checkerboard pattern."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Failed to decode frames"}, status_code=502)

        result = workspace_mapper.calibrate_scale_from_checkerboard(left, right, req.square_size_mm)
        if result["ok"]:
            action_log.add("BIFOCAL", f"Scale calibrated: factor={result['scale_factor']}", "info")
        else:
            action_log.add(
                "BIFOCAL", f"Scale calibration failed: {result.get('error', '?')}", "warning"
            )
        return result

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class TapeMeasureRequest(BaseModel):
    known_length_mm: float = Field(gt=0)
    x1: int
    y1: int
    x2: int
    y2: int


@app.post("/api/bifocal/calibrate-tape")
async def bifocal_calibrate_tape(req: TapeMeasureRequest):
    """Calibrate scale using two points on a tape measure."""
    if not _HAS_BIFOCAL or workspace_mapper is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Failed to decode frames"}, status_code=502)

        result = workspace_mapper.calibrate_scale_from_tape_measure(
            left, right, req.known_length_mm, (req.x1, req.y1), (req.x2, req.y2)
        )
        if result["ok"]:
            action_log.add("BIFOCAL", f"Tape calibration: factor={result['scale_factor']}", "info")
        return result

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/bifocal/preview")
async def bifocal_preview():
    """Preview the current arm pose against the workspace map for collisions."""
    if not _HAS_BIFOCAL or workspace_mapper is None or collision_preview is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    current = _get_current_joints()
    result = collision_preview.preview_single_pose(current, workspace_mapper)
    arm_points = collision_preview.get_arm_envelope(current)

    return {
        "ok": True,
        "clear": result.clear,
        "summary": result.summary,
        "hits": [
            {
                "link": h.link_index,
                "point": h.link_point_mm,
                "severity": h.severity,
            }
            for h in result.hits
        ],
        "arm_points_mm": arm_points,
        "checked": result.checked_points,
        "elapsed_ms": result.elapsed_ms,
    }


@app.post("/api/bifocal/preview-target")
async def bifocal_preview_target(req: SetAllJointsRequest):
    """Preview a target pose against the workspace map for collisions."""
    if not _HAS_BIFOCAL or workspace_mapper is None or collision_preview is None:
        return JSONResponse({"ok": False, "error": "Bifocal module not available"}, status_code=501)

    target = np.array(req.angles[:6])
    result = collision_preview.preview_single_pose(target, workspace_mapper)
    arm_points = collision_preview.get_arm_envelope(target)

    return {
        "ok": True,
        "clear": result.clear,
        "summary": result.summary,
        "hits": [
            {
                "link": h.link_index,
                "point": h.link_point_mm,
                "severity": h.severity,
            }
            for h in result.hits
        ],
        "arm_points_mm": arm_points,
        "checked": result.checked_points,
        "elapsed_ms": result.elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Visual Pick endpoints — dual-camera object detection + autonomous grasp
# ---------------------------------------------------------------------------


class VisualPickRequest(BaseModel):
    target: str = Field(default="redbull", description="Object to pick: redbull, red, blue, all")
    speed: float = Field(default=0.5, ge=0.1, le=1.0)
    execute: bool = Field(default=False, description="If True, execute immediately after planning")


class VisualPickFromPositionRequest(BaseModel):
    x_mm: float = Field(description="X position in arm-base frame (mm)")
    y_mm: float = Field(description="Y position in arm-base frame (mm)")
    z_mm: float = Field(description="Z position in arm-base frame (mm)")
    label: str = Field(default="redbull")
    speed: float = Field(default=0.5, ge=0.1, le=1.0)
    execute: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Camera-based 3D localization (uses calibrated extrinsics)
# ---------------------------------------------------------------------------


@app.get("/api/camera/status")
async def camera_calibration_status():
    """Get camera calibration status."""
    return {
        "calibrated_cameras": list(camera_models.keys()),
        "details": {
            cam_id: {
                "position_m": cm.camera_position.tolist(),
                "reproj_error_px": float(
                    json.load(open(f"calibration_results/camera{cam_id}_extrinsics.json")).get(
                        "reprojection_error_mean_px", -1
                    )
                ),
            }
            for cam_id, cm in camera_models.items()
        },
    }


class LocateRequest(BaseModel):
    """Request to locate an object by pixel coords or by LLM vision."""

    pixel: Optional[list[float]] = None  # [u, v] in cam0
    camera_id: int = 0
    z_height: float = 0.0  # assumed Z height in meters (table surface)
    use_llm: bool = False  # ask Gemini to find object and return pixel
    target: str = "red bull can"  # object to find (when use_llm=True)


@app.post("/api/locate")
async def locate_object(req: LocateRequest):
    """Locate an object in 3D using calibrated camera.

    Either provide pixel coordinates directly, or set use_llm=True to have
    Gemini find the object in the camera frame.
    """
    cm = camera_models.get(req.camera_id)
    if cm is None:
        return JSONResponse(
            {"ok": False, "error": f"Camera {req.camera_id} not calibrated"},
            status_code=501,
        )

    pixel = req.pixel

    if req.use_llm and pixel is None:
        # Use Gemini to find object pixel coords
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://localhost:8081/snap/{req.camera_id}")
                if resp.status_code != 200:
                    return JSONResponse(
                        {"ok": False, "error": "Camera snapshot failed"}, status_code=502
                    )
                jpeg_bytes = resp.content

            import os

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return JSONResponse(
                    {"ok": False, "error": "GEMINI_API_KEY not set"}, status_code=501
                )

            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            import base64

            b64 = base64.b64encode(jpeg_bytes).decode()
            prompt = (
                f"Find the {req.target} in this 1920x1080 image. "
                f"Return ONLY the pixel coordinates of its center as JSON: "
                f'{{"u": <x_pixel>, "v": <y_pixel>}}'
            )
            response = model.generate_content(
                [
                    {"mime_type": "image/jpeg", "data": b64},
                    prompt,
                ]
            )

            import re

            match = re.search(r'\{[^}]*"u"\s*:\s*([\d.]+)[^}]*"v"\s*:\s*([\d.]+)', response.text)
            if not match:
                return JSONResponse(
                    {"ok": False, "error": f"LLM couldn't parse: {response.text[:200]}"},
                    status_code=422,
                )
            pixel = [float(match.group(1)), float(match.group(2))]
            action_log.add(
                "VISION",
                f"LLM located '{req.target}' at pixel ({pixel[0]:.0f}, {pixel[1]:.0f})",
                "info",
            )

        except Exception as e:
            return JSONResponse({"ok": False, "error": f"LLM locate failed: {e}"}, status_code=500)

    if pixel is None:
        return JSONResponse(
            {"ok": False, "error": "No pixel coordinates provided"}, status_code=400
        )

    # Back-project to 3D
    world_point = cm.pixel_to_world_at_z(pixel[0], pixel[1], z=req.z_height)
    if world_point is None:
        return JSONResponse({"ok": False, "error": "Ray parallel to Z plane"}, status_code=422)

    # Also get the ray for debugging
    origin, direction = cm.pixel_to_ray(pixel[0], pixel[1])

    result = {
        "ok": True,
        "pixel": pixel,
        "camera_id": req.camera_id,
        "world_position_m": [round(float(x), 4) for x in world_point],
        "world_position_mm": [round(float(x) * 1000, 1) for x in world_point],
        "z_height_m": req.z_height,
        "ray_origin_m": [round(float(x), 4) for x in origin],
        "ray_direction": [round(float(x), 4) for x in direction],
    }

    action_log.add(
        "VISION",
        f"Located at ({world_point[0]*1000:.0f}, {world_point[1]*1000:.0f}, {world_point[2]*1000:.0f})mm "
        f"from pixel ({pixel[0]:.0f}, {pixel[1]:.0f})",
        "info",
    )

    return result


# ---------------------------------------------------------------------------
# Visual servo
# ---------------------------------------------------------------------------


class ServoRequest(BaseModel):
    target: str = "red bull can"
    max_steps: int = 25


_servo_task: Optional[asyncio.Task] = None
_servo_result: Optional[dict] = None


@app.post("/api/servo/approach")
async def servo_approach(req: ServoRequest):
    """Visual servo approach to a target. Moves arm step-by-step with camera feedback."""
    global _servo_task, _servo_result
    if _servo_task and not _servo_task.done():
        return JSONResponse({"ok": False, "error": "Servo already running"}, status_code=409)

    _servo_result = None
    import os as _os

    api_key = _os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse({"ok": False, "error": "GEMINI_API_KEY not set"}, status_code=501)

    from src.control.visual_servo import VisualServo

    async def _run():
        global _servo_result
        try:
            servo = VisualServo(gemini_api_key=api_key, max_steps=req.max_steps)
            result = await servo.approach(req.target)
            _servo_result = {
                "success": result.success,
                "message": result.message,
                "steps": len(result.steps),
                "time_s": round(result.total_time_s, 1),
                "final_distance_px": round(result.final_distance_px, 1),
                "log": [
                    {
                        "step": s.step,
                        "action": s.action,
                        "distance_px": round(s.pixel_distance, 1),
                        "gripper": s.gripper_pixel,
                        "target": s.target_pixel,
                        "notes": s.notes,
                    }
                    for s in result.steps
                ],
            }
            action_log.add(
                "SERVO", f"{'Success' if result.success else 'Failed'}: {result.message}", "info"
            )
        except Exception as e:
            _servo_result = {"success": False, "message": str(e), "steps": 0}
            action_log.add("SERVO", f"Error: {e}", "error")
            import traceback

            logger.error(f"Servo error: {traceback.format_exc()}")

    _servo_task = asyncio.create_task(_run())
    return {"ok": True, "message": f"Visual servo started toward '{req.target}'"}


@app.get("/api/servo/status")
async def servo_status():
    """Get visual servo status."""
    running = _servo_task is not None and not _servo_task.done()
    return {
        "running": running,
        "result": _servo_result,
    }


@app.post("/api/servo/stop")
async def servo_stop():
    """Stop visual servo."""
    global _servo_task
    if _servo_task and not _servo_task.done():
        _servo_task.cancel()
        return {"ok": True}
    return {"ok": False, "error": "Not running"}


# ---------------------------------------------------------------------------
# Visual pick
# ---------------------------------------------------------------------------


@app.get("/api/pick/status")
async def pick_status():
    """Get visual pick system status."""
    if not _HAS_VISUAL_PICK or pick_executor is None:
        return {"available": False, "error": "Visual pick module not available"}
    status = pick_executor.get_status()
    status["available"] = True
    status["calibrated"] = arm_tracker.calibrator.is_calibrated if arm_tracker else False
    return status


@app.post("/api/pick/detect")
async def pick_detect(req: VisualPickRequest = VisualPickRequest()):
    """Detect objects using dual cameras without planning a grasp.

    Returns detected objects with their 3D positions.
    """
    if not _HAS_VISUAL_PICK or arm_tracker is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )

    # Grab camera snapshots
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        import cv2

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        result = arm_tracker.track(left, right, target_label=req.target, annotate=False)

        objects_data = []
        for obj in result.objects:
            obj_data = {
                "label": obj.label,
                "position_mm": [round(float(x), 1) for x in obj.position_mm],
                "confidence": round(obj.confidence, 3),
                "source": getattr(obj, "source", "unknown"),
            }
            if obj.bbox_cam0 is not None:
                obj_data["bbox_cam0"] = list(obj.bbox_cam0)
            if obj.bbox_cam1 is not None:
                obj_data["bbox_cam1"] = list(obj.bbox_cam1)
            if obj.centroid_cam0 is not None:
                obj_data["centroid_cam0"] = list(obj.centroid_cam0)
            if obj.centroid_cam1 is not None:
                obj_data["centroid_cam1"] = list(obj.centroid_cam1)
            objects_data.append(obj_data)

        action_log.add(
            "VISION",
            f"Detected {len(result.objects)} '{req.target}' object(s) in {result.elapsed_ms:.0f}ms",
            "info",
        )

        return {
            "ok": True,
            "objects": objects_data,
            "count": len(result.objects),
            "elapsed_ms": result.elapsed_ms,
            "status": result.status,
            "message": result.message,
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/pick/plan")
async def pick_plan(req: VisualPickRequest = VisualPickRequest()):
    """Detect object and plan a full pick trajectory.

    Uses dual cameras to find the target, plans grasp approach, and optionally
    executes the trajectory through the command smoother.
    """
    global _active_task
    if (
        not _HAS_VISUAL_PICK
        or pick_executor is None
        or arm_tracker is None
        or grasp_planner is None
    ):
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )
    if req.execute and not (smoother and smoother.arm_enabled):
        return JSONResponse(
            {"ok": False, "error": "Arm not enabled for execution"}, status_code=409
        )

    # Grab camera snapshots
    import httpx, cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        current = _get_current_joints()
        gripper = armState_gripper()

        result = pick_executor.plan_pick(
            left_frame=left,
            right_frame=right,
            arm_tracker=arm_tracker,
            grasp_planner=grasp_planner,
            current_angles_deg=current,
            current_gripper_mm=gripper,
            target_label=req.target,
            workspace_mapper=workspace_mapper,
            collision_preview=collision_preview,
        )

        response_data = {
            "ok": result.success,
            "phase": result.phase.value,
            "message": result.message,
            "elapsed_ms": result.elapsed_ms,
            "detected_count": len(result.detected_objects),
        }

        if result.success and result.trajectory is not None:
            response_data["trajectory"] = {
                "points": len(result.trajectory.points),
                "duration_s": round(result.trajectory.duration, 1),
            }
            response_data["grasp"] = {
                "approach_angles": result.approach_angles_deg,
                "grasp_angles": result.grasp_angles_deg,
                "retreat_angles": result.retreat_angles_deg,
                "gripper_open_mm": result.gripper_open_mm,
                "gripper_close_mm": result.gripper_close_mm,
            }

            if result.target_object is not None:
                response_data["target"] = {
                    "label": result.target_object.label,
                    "position_mm": [round(float(x), 1) for x in result.target_object.position_mm],
                    "depth_mm": round(result.target_object.depth_mm, 1),
                    "confidence": round(result.target_object.confidence, 3),
                }

            action_log.add(
                "PICK",
                f"Pick planned for '{req.target}': {len(result.trajectory.points)} pts, "
                f"{result.trajectory.duration:.1f}s",
                "info",
            )

            # Execute if requested
            if req.execute and smoother and smoother.arm_enabled:
                if _active_task and not _active_task.done():
                    _active_task.cancel()
                _active_task = asyncio.create_task(
                    _execute_trajectory(result.trajectory, f"Visual Pick ({req.target})")
                )
                response_data["executing"] = True
                action_log.add("PICK", f"Executing pick trajectory for '{req.target}'", "info")
        else:
            action_log.add(
                "PICK", f"Pick plan failed for '{req.target}': {result.message}", "warning"
            )

        return response_data

    except Exception as e:
        logger.exception("Pick plan error")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/pick/from-position")
async def pick_from_position(req: VisualPickFromPositionRequest):
    """Plan a pick from a known 3D position (skip camera detection).

    Useful when the object position is already known.
    """
    global _active_task
    if not _HAS_VISUAL_PICK or pick_executor is None or grasp_planner is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )
    if req.execute and not (smoother and smoother.arm_enabled):
        return JSONResponse(
            {"ok": False, "error": "Arm not enabled for execution"}, status_code=409
        )

    current = _get_current_joints()
    gripper = armState_gripper()
    obj_pos = np.array([req.x_mm, req.y_mm, req.z_mm])

    result = pick_executor.plan_pick_from_position(
        object_position_mm=obj_pos,
        grasp_planner=grasp_planner,
        current_angles_deg=current,
        current_gripper_mm=gripper,
        object_label=req.label,
    )

    response_data = {
        "ok": result.success,
        "phase": result.phase.value,
        "message": result.message,
        "elapsed_ms": result.elapsed_ms,
    }

    if result.success and result.trajectory is not None:
        response_data["trajectory"] = {
            "points": len(result.trajectory.points),
            "duration_s": round(result.trajectory.duration, 1),
        }
        response_data["grasp"] = {
            "approach_angles": result.approach_angles_deg,
            "grasp_angles": result.grasp_angles_deg,
            "retreat_angles": result.retreat_angles_deg,
            "gripper_open_mm": result.gripper_open_mm,
            "gripper_close_mm": result.gripper_close_mm,
        }
        response_data["target_position_mm"] = [req.x_mm, req.y_mm, req.z_mm]

        action_log.add(
            "PICK",
            f"Pick from position [{req.x_mm:.0f},{req.y_mm:.0f},{req.z_mm:.0f}]: "
            f"{len(result.trajectory.points)} pts",
            "info",
        )

        if req.execute and smoother and smoother.arm_enabled:
            if _active_task and not _active_task.done():
                _active_task.cancel()
            _active_task = asyncio.create_task(
                _execute_trajectory(result.trajectory, f"Pick from position ({req.label})")
            )
            response_data["executing"] = True

    return response_data


@app.post("/api/pick/calibrate-camera-arm")
async def pick_calibrate_camera_arm():
    """Calibrate camera-to-arm transform using current arm pose and visual detection.

    Place a distinctive object (e.g. red marker) at the end-effector,
    then this endpoint detects it in both cameras and computes the
    camera-to-arm translation offset.
    """
    if not _HAS_VISUAL_PICK or arm_tracker is None:
        return JSONResponse(
            {"ok": False, "error": "Visual pick module not available"}, status_code=501
        )

    import httpx, cv2

    try:
        # Get camera frames
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")

        left = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return JSONResponse({"ok": False, "error": "Camera frames failed"}, status_code=502)

        # Detect red marker at EE
        result = arm_tracker.track(left, right, target_label="red", annotate=False)
        if not result.objects:
            return JSONResponse(
                {"ok": False, "error": "No red marker detected at end-effector"},
                status_code=400,
            )

        # Get current EE position from FK
        current = _get_current_joints()
        from src.kinematics.kinematics import D1Kinematics

        kin = D1Kinematics()
        q7 = np.zeros(7)
        q7[:6] = np.deg2rad(current)
        ee_pose = kin.forward_kinematics(q7)
        ee_pos_mm = ee_pose[:3, 3] * 1000  # meters to mm

        # Use the closest detection to EE as the calibration point
        cam_pos = result.objects[0].position_mm
        arm_tracker.calibrate_cam_to_arm_from_known_point(cam_pos, ee_pos_mm)

        action_log.add(
            "CALIBRATION",
            f"Camera-to-arm calibrated: cam={[round(x,1) for x in cam_pos]}, "
            f"arm={[round(x,1) for x in ee_pos_mm]}",
            "info",
        )

        return {
            "ok": True,
            "camera_point_mm": [round(float(x), 1) for x in cam_pos],
            "arm_point_mm": [round(float(x), 1) for x in ee_pos_mm],
            "message": "Camera-to-arm transform updated",
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Vision Task Planning endpoints — look at camera, build a plan
# ---------------------------------------------------------------------------


class VisionPlanRequest(BaseModel):
    instruction: str = Field(
        ..., min_length=1, max_length=500, description="What to do, e.g. 'pick up the red object'"
    )
    camera: int = Field(default=1, ge=0, le=1, description="Primary camera (0=front, 1=overhead)")
    use_both_cameras: bool = Field(default=True, description="Use both cameras for richer scene")
    execute: bool = Field(default=False, description="Execute the plan immediately if True")


async def _grab_camera_frame(camera: int) -> Optional[np.ndarray]:
    """Fetch a snapshot from the camera server and decode it."""
    import httpx
    import cv2

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"http://localhost:8081/snap/{camera}")
        if resp.status_code != 200:
            return None
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.post("/api/vision/plan")
async def vision_plan(req: VisionPlanRequest):
    """Analyze camera feeds and build a task plan from an instruction.

    Camera layout: cam0=front/side, cam1=overhead (default primary).
    Uses both cameras by default for dual-view scene understanding.
    """
    if not _HAS_VISION_PLANNING or vision_task_planner is None or scene_analyzer is None:
        return JSONResponse(
            {"ok": False, "error": "Vision planning module not available"}, status_code=501
        )

    frame = await _grab_camera_frame(req.camera)
    if frame is None:
        return JSONResponse(
            {"ok": False, "error": f"Camera {req.camera} snapshot failed"}, status_code=502
        )

    # Grab second camera for cross-referencing
    cam0_frame = None
    if req.use_both_cameras:
        other_cam = 0 if req.camera == 1 else 1
        cam0_frame = await _grab_camera_frame(other_cam)

    import time as _time

    scene = scene_analyzer.analyze(frame, cam0_frame=cam0_frame, timestamp=_time.time())
    cameras_str = ", ".join(scene.cameras_used)
    action_log.add(
        "VISION",
        f"Scene analyzed: {scene.object_count} objects (cameras: {cameras_str})",
        "info",
    )

    current = _get_current_joints()
    plan = vision_task_planner.plan(req.instruction, scene, current)

    action_log.add(
        "VISION",
        f"Plan: {plan.action.value} -> {'OK' if plan.success else 'FAILED'}: "
        + (plan.error or (plan.task_result.message if plan.task_result else "")),
        "info" if plan.success else "warning",
    )

    global _active_task
    if req.execute and plan.success and plan.trajectory:
        if not (smoother and smoother.arm_enabled):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Arm not enabled — plan created but not executed",
                    "plan": plan.to_dict(),
                    "scene": scene.to_dict(),
                },
                status_code=409,
            )
        if _active_task and not _active_task.done():
            _active_task.cancel()
        _active_task = asyncio.create_task(
            _execute_trajectory(plan.trajectory, f"Vision: {plan.action.value}")
        )
        action_log.add("VISION", f"Executing vision plan: {plan.action.value}", "info")

    return {
        "ok": plan.success,
        "plan": plan.to_dict(),
        "scene": scene.to_dict(),
        "executed": req.execute and plan.success,
    }


@app.post("/api/vision/analyze")
async def vision_analyze(camera: int = 1, use_both: bool = True):
    """Analyze the current camera view and return a scene description.

    Camera layout: cam0=front/side, cam1=overhead (default).
    """
    if not _HAS_VISION_PLANNING or scene_analyzer is None:
        return JSONResponse(
            {"ok": False, "error": "Vision planning module not available"}, status_code=501
        )

    frame = await _grab_camera_frame(camera)
    if frame is None:
        return JSONResponse(
            {"ok": False, "error": f"Camera {camera} snapshot failed"}, status_code=502
        )

    cam0_frame = None
    if use_both:
        other_cam = 0 if camera == 1 else 1
        cam0_frame = await _grab_camera_frame(other_cam)

    import time as _time

    scene = scene_analyzer.analyze(frame, cam0_frame=cam0_frame, timestamp=_time.time())
    cameras_str = ", ".join(scene.cameras_used)
    action_log.add(
        "VISION",
        f"Scene analysis: {scene.object_count} objects (cameras: {cameras_str})",
        "info",
    )

    return {"ok": True, "scene": scene.to_dict()}


# ---------------------------------------------------------------------------
# Claw Position Prediction endpoints — visual tracking of end-effector
# ---------------------------------------------------------------------------


@app.post("/api/claw-predict/toggle")
async def claw_predict_toggle():
    """Toggle the claw position predictor on/off."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    enabled = claw_predictor.toggle()
    action_log.add("CLAW_PREDICT", f"Predictor {'enabled' if enabled else 'disabled'}", "info")
    return {"ok": True, "enabled": enabled}


@app.get("/api/claw-predict/status")
async def claw_predict_status():
    """Get claw position predictor status and last prediction."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return {"available": False}
    status = claw_predictor.get_status()
    status["available"] = True
    return status


@app.post("/api/claw-predict/update")
async def claw_predict_update():
    """Trigger a claw position prediction from current camera frames."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    if not claw_predictor.enabled:
        return JSONResponse({"ok": False, "error": "Predictor not enabled"}, status_code=409)

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp0 = await client.get("http://localhost:8081/snap/0")
            resp1 = await client.get("http://localhost:8081/snap/1")
        if resp0.status_code != 200 or resp1.status_code != 200:
            return JSONResponse({"ok": False, "error": "Camera snapshots failed"}, status_code=502)

        import cv2

        cam0 = cv2.imdecode(np.frombuffer(resp0.content, np.uint8), cv2.IMREAD_COLOR)
        cam1 = cv2.imdecode(np.frombuffer(resp1.content, np.uint8), cv2.IMREAD_COLOR)
        if cam0 is None or cam1 is None:
            return JSONResponse(
                {"ok": False, "error": "Failed to decode camera frames"}, status_code=502
            )

        # Sync scale factor from workspace mapper if available
        if workspace_mapper is not None and workspace_mapper.scale_calibrated:
            claw_predictor.set_scale_factor(workspace_mapper._scale_factor)

        # Compute FK position for comparison
        fk_pos = _get_fk_position_mm()

        result = claw_predictor.predict(cam0, cam1, fk_position_mm=fk_pos)
        return {"ok": True, "prediction": result.to_dict()}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class ClawHSVRequest(BaseModel):
    h_lower: int = Field(ge=0, le=180)
    s_lower: int = Field(ge=0, le=255)
    v_lower: int = Field(ge=0, le=255)
    h_upper: int = Field(ge=0, le=180)
    s_upper: int = Field(ge=0, le=255)
    v_upper: int = Field(ge=0, le=255)


@app.post("/api/claw-predict/set-hsv")
async def claw_predict_set_hsv(req: ClawHSVRequest):
    """Set custom HSV color range for claw detection."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.set_hsv_range(
        (req.h_lower, req.s_lower, req.v_lower),
        (req.h_upper, req.s_upper, req.v_upper),
    )
    action_log.add(
        "CLAW_PREDICT",
        f"HSV range set: [{req.h_lower},{req.s_lower},{req.v_lower}]-[{req.h_upper},{req.s_upper},{req.v_upper}]",
        "info",
    )
    return {"ok": True}


class ClawROIRequest(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


@app.post("/api/claw-predict/set-roi")
async def claw_predict_set_roi(req: ClawROIRequest):
    """Set a region of interest for claw detection."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.set_detection_roi(req.x, req.y, req.w, req.h)
    action_log.add("CLAW_PREDICT", f"ROI set: ({req.x},{req.y}) {req.w}x{req.h}", "info")
    return {"ok": True}


@app.post("/api/claw-predict/clear-roi")
async def claw_predict_clear_roi():
    """Clear the detection ROI."""
    if not _HAS_CLAW_PREDICT or claw_predictor is None:
        return JSONResponse({"ok": False, "error": "Claw predictor not available"}, status_code=501)
    claw_predictor.clear_detection_roi()
    action_log.add("CLAW_PREDICT", "ROI cleared", "info")
    return {"ok": True}


def _get_fk_position_mm() -> Optional[list[float]]:
    """Compute the current end-effector position from forward kinematics."""
    try:
        from src.kinematics import D1Kinematics

        kin = D1Kinematics()
        joints_6 = _get_current_joints()
        # FK expects 7 joints (6 arm + gripper); gripper angle is 0
        joints_7 = np.append(joints_6, 0.0)
        joint_radians = np.deg2rad(joints_7)
        T = kin.forward_kinematics(joint_radians)
        # Position in meters from FK, convert to mm
        pos_mm = T[:3, 3] * 1000.0
        return list(float(v) for v in pos_mm)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Viz Calibration endpoints
# ---------------------------------------------------------------------------

try:
    from src.vision.viz_calibrator import (
        run_calibration as _run_viz_calibration,
        load_calibration as _load_viz_calibration,
        OUTPUT_PATH as _VIZ_CALIB_PATH,
    )

    _HAS_VIZ_CALIB = True
except ImportError:
    _HAS_VIZ_CALIB = False

_viz_calib_running = False


@app.get("/api/viz/calibration")
async def get_viz_calibration():
    """Return saved visualization calibration data."""
    if not _HAS_VIZ_CALIB:
        return JSONResponse({"ok": False, "error": "Viz calibrator not available"}, status_code=501)
    data = _load_viz_calibration()
    if data is None:
        return JSONResponse({"ok": False, "error": "No calibration data found"}, status_code=404)
    return {"ok": True, **data}


@app.post("/api/viz/calibrate")
async def run_viz_calibration():
    """Run camera-based visualization calibration (moves the arm!)."""
    global _viz_calib_running
    if not _HAS_VIZ_CALIB:
        return JSONResponse({"ok": False, "error": "Viz calibrator not available"}, status_code=501)
    if _viz_calib_running:
        return JSONResponse(
            {"ok": False, "error": "Calibration already in progress"}, status_code=409
        )
    if not (smoother and smoother.arm_enabled):
        return JSONResponse({"ok": False, "error": "Arm not enabled"}, status_code=409)

    _viz_calib_running = True
    # Disable collision detection during calibration — it fights the calibrator
    _saved_collision = collision_detector
    _saved_collision_enabled = None
    if collision_detector and hasattr(collision_detector, "enabled"):
        _saved_collision_enabled = collision_detector.enabled
        collision_detector.enabled = False
    try:
        result = await _run_viz_calibration()
        action_log.add(
            "VIZ_CALIB",
            f"{'OK' if result.success else 'FAILED'} — {result.n_observations} obs, residual={result.residual}",
            "info" if result.success else "error",
        )
        import math

        def _sanitize(v):
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                return None
            return v

        return {
            "ok": result.success,
            "links_mm": {k: _sanitize(v) for k, v in result.links_mm.items()},
            "joint_viz_offsets": [_sanitize(v) for v in result.joint_viz_offsets],
            "residual": _sanitize(result.residual),
            "n_observations": result.n_observations,
            "message": result.message,
        }
    except Exception as e:
        action_log.add("VIZ_CALIB", f"Error: {e}", "error")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        _viz_calib_running = False
        # Re-enable collision detection
        if (
            collision_detector
            and hasattr(collision_detector, "enabled")
            and _saved_collision_enabled is not None
        ):
            collision_detector.enabled = _saved_collision_enabled


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------


async def _handle_collision(stall: "StallEvent", clients: list):
    """Handle a detected collision: stop, analyze, broadcast, back off."""
    global collision_events
    action_log.add(
        "COLLISION",
        f"⚠ STALL on J{stall.joint_id}: cmd={stall.commanded_deg:.1f}° actual={stall.actual_deg:.1f}° (err={stall.error_deg:.1f}°)",
        "error",
    )

    # 1. Stop arm movement
    if smoother:
        smoother._clear_targets()

    # 2. Analyze with cameras + vision (in thread to not block)
    analysis = None
    if collision_analyzer:
        try:
            analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                collision_analyzer.analyze,
                stall.joint_id,
                stall.commanded_deg,
                stall.actual_deg,
            )
            action_log.add("COLLISION", f"Analysis: {analysis.analysis_text[:120]}", "warning")
        except Exception as e:
            logger.error("Collision analysis failed: %s", e)

    # 3. Build event data
    event = {
        "type": "collision",
        "joint": stall.joint_id,
        "commanded": round(stall.commanded_deg, 1),
        "actual": round(stall.actual_deg, 1),
        "error": round(stall.error_deg, 1),
        "last_good": round(stall.last_good_position, 1),
        "timestamp": stall.timestamp,
        "analysis": analysis.analysis_text if analysis else "Analysis unavailable",
        "images": [],
    }
    if analysis:
        if analysis.cam0_path:
            event["images"].append(f"/api/collisions/{analysis.timestamp}/cam0.jpg")
        if analysis.cam1_path:
            event["images"].append(f"/api/collisions/{analysis.timestamp}/cam1.jpg")

    collision_events.append(event)
    if len(collision_events) > 100:
        collision_events = collision_events[-100:]

    # 4. Broadcast to WebSocket clients
    dead = []
    for ws in clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)

    # 5. Back off: move stalled joint 10° toward last good position
    if smoother and smoother.arm_enabled and arm:
        backoff_angle = stall.actual_deg + (
            10.0 if stall.last_good_position > stall.actual_deg else -10.0
        )
        # Clamp to last good
        if abs(backoff_angle - stall.actual_deg) > abs(stall.last_good_position - stall.actual_deg):
            backoff_angle = stall.last_good_position
        smoother.set_joint_target(stall.joint_id, backoff_angle)
        action_log.add(
            "COLLISION",
            f"Backing off J{stall.joint_id} to {backoff_angle:.1f}°",
            "info",
        )


@app.get("/api/collisions")
async def api_collisions(limit: int = 20):
    """Return recent collision events."""
    return {"events": collision_events[-limit:]}


@app.get("/api/collisions/{timestamp}/{filename}")
async def api_collision_image(timestamp: str, filename: str):
    """Serve saved collision images."""
    from fastapi.responses import FileResponse

    data_dir = Path(__file__).resolve().parent.parent / "data" / "collisions"
    img_path = data_dir / timestamp / filename
    if not img_path.exists() or not img_path.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(img_path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Telemetry viewer page route
# ---------------------------------------------------------------------------


@app.get("/telemetry")
async def telemetry_page():
    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "static" / "telemetry.html")


# ---------------------------------------------------------------------------
# ASCII video viewer page route + WebSocket stream
# ---------------------------------------------------------------------------

_ASCII_CHARSETS = {
    "standard": CHARSET_STANDARD if _HAS_ASCII else " .:-=+*#%@",
    "detailed": CHARSET_DETAILED if _HAS_ASCII else " .:-=+*#%@",
    "blocks": CHARSET_BLOCKS if _HAS_ASCII else " .:-=+*#%@",
    "minimal": CHARSET_MINIMAL if _HAS_ASCII else " .:-=+*#%@",
}


@app.get("/api/gpu/status")
async def gpu_status_endpoint():
    """Return GPU compute (OpenCL) availability and status."""
    if _HAS_GPU_PREPROCESS:
        return JSONResponse(_gpu_status())
    return JSONResponse(
        {
            "opencl_available": False,
            "opencl_enabled": False,
            "device": None,
        }
    )


@app.get("/ascii")
async def ascii_page():
    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "static" / "ascii.html")


@app.websocket("/ws/ascii")
async def ws_ascii(ws: WebSocket):
    """Stream ASCII-converted camera frames to the browser.

    The client sends JSON settings messages to control camera selection,
    charset, dimensions, color mode, and invert mode.  The server fetches
    JPEG frames from the camera server, converts them via AsciiConverter,
    and pushes the result back over the WebSocket.
    """
    await ws.accept()

    # Defaults
    cam_id = 0
    charset_name = "standard"
    width = 320
    height = 140
    color = False
    invert = True
    converter = None

    def _build_converter():
        nonlocal converter
        if not _HAS_ASCII:
            converter = None
            return
        cs = _ASCII_CHARSETS.get(charset_name, _ASCII_CHARSETS["standard"])
        converter = AsciiConverter(
            width=width, height=height, charset=cs, invert=invert, color=color
        )

    _build_converter()

    try:
        import httpx

        while True:
            # Check for incoming settings messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if data.get("type") == "settings":
                    cam_id = int(data.get("cam", cam_id))
                    charset_name = data.get("charset", charset_name)
                    width = max(20, min(600, int(data.get("width", width))))
                    height = max(10, min(400, int(data.get("height", height))))
                    color = bool(data.get("color", color))
                    invert = bool(data.get("invert", invert))
                    _build_converter()
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            if converter is None:
                await ws.send_json(
                    {
                        "type": "frame",
                        "lines": ["ASCII converter not available"],
                        "width": 40,
                        "height": 1,
                    }
                )
                await asyncio.sleep(1.0)
                continue

            # Fetch a JPEG snapshot from the camera server
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"http://localhost:8081/snap/{cam_id}")
                    if resp.status_code == 200:
                        jpeg_bytes = resp.content
                    else:
                        jpeg_bytes = None
            except Exception:
                jpeg_bytes = None

            if jpeg_bytes is None:
                await ws.send_json(
                    {
                        "type": "frame",
                        "lines": ["No camera feed available"],
                        "width": 30,
                        "height": 1,
                    }
                )
                await asyncio.sleep(0.5)
                continue

            # Convert to ASCII
            try:
                if _HAS_GPU_PREPROCESS:
                    gpu_frame = decode_jpeg_gpu(jpeg_bytes)
                    if color:
                        result = converter.frame_to_color_data(gpu_frame)
                        await ws.send_json({"type": "frame", **result})
                    else:
                        text = converter.frame_to_ascii(gpu_frame)
                        lines = text.split("\n")
                elif color:
                    result = converter.decode_jpeg_to_color_data(jpeg_bytes)
                    await ws.send_json({"type": "frame", **result})
                else:
                    text = converter.decode_jpeg_to_ascii(jpeg_bytes)
                    lines = text.split("\n")
                    await ws.send_json(
                        {"type": "frame", "lines": lines, "width": width, "height": height}
                    )
            except Exception:
                await ws.send_json(
                    {"type": "frame", "lines": ["Frame conversion error"], "width": 25, "height": 1}
                )

            # ~10 fps target for ASCII stream
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.debug("ASCII WS client disconnected normally")
    except Exception as exc:
        logger.warning("ASCII WS error: %s", exc)


# ---------------------------------------------------------------------------
# Real World 3D — Visual hull voxel reconstruction from dual cameras
# ---------------------------------------------------------------------------


@app.websocket("/ws/realworld3d")
async def ws_realworld3d(ws: WebSocket):
    """Stream voxel reconstruction data from dual camera visual hull carving.

    Fetches frames from cam0 (front→X-Y) and cam1 (overhead→X-Z),
    segments foreground via edge detection / frame differencing,
    intersects silhouettes to carve a 3D voxel grid, and sends
    non-empty voxels as JSON to the client.
    """
    await ws.accept()
    import httpx

    GRID_W, GRID_H, GRID_D = 64, 32, 64  # width, height, depth
    bg_frame0 = None  # background reference for cam0
    bg_frame1 = None  # background reference for cam1
    frame_count = 0

    try:
        while True:
            # Check for client messages (settings, bg capture commands)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if data.get("type") == "capture_bg":
                    bg_frame0 = None
                    bg_frame1 = None
                    frame_count = 0
                    await ws.send_json(
                        {
                            "type": "status",
                            "message": "Background reset, will capture on next frame",
                        }
                    )
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            # Fetch snapshots from both cameras
            frame0_bytes = None
            frame1_bytes = None
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r0, r1 = await asyncio.gather(
                        client.get("http://localhost:8081/snap/0"),
                        client.get("http://localhost:8081/snap/1"),
                        return_exceptions=True,
                    )
                    if not isinstance(r0, Exception) and r0.status_code == 200:
                        frame0_bytes = r0.content
                    if not isinstance(r1, Exception) and r1.status_code == 200:
                        frame1_bytes = r1.content
            except Exception:
                pass

            if frame0_bytes is None and frame1_bytes is None:
                await ws.send_json({"type": "status", "message": "Waiting for cameras..."})
                await asyncio.sleep(1.0)
                continue

            # Decode frames
            import cv2

            f0 = (
                cv2.imdecode(np.frombuffer(frame0_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame0_bytes
                else None
            )
            f1 = (
                cv2.imdecode(np.frombuffer(frame1_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame1_bytes
                else None
            )

            if f0 is None and f1 is None:
                await ws.send_json({"type": "status", "message": "Failed to decode frames"})
                await asyncio.sleep(0.5)
                continue

            # Wrap frames as UMat for GPU-accelerated OpenCV (OpenCL)
            if f0 is not None:
                f0 = cv2.UMat(f0)
            if f1 is not None:
                f1 = cv2.UMat(f1)

            # Capture background on first frames (stored as UMat for GPU ops)
            if bg_frame0 is None and f0 is not None:
                bg_frame0 = cv2.GaussianBlur(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if bg_frame1 is None and f1 is not None:
                bg_frame1 = cv2.GaussianBlur(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (21, 21), 0)

            frame_count += 1

            # --- Segmentation via frame differencing + edge detection ---
            def segment_foreground(frame, bg_gray, threshold=30):
                """Returns binary mask of foreground pixels."""
                gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if bg_gray is not None:
                    diff = cv2.absdiff(gray, bg_gray)
                    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                else:
                    # Fallback: use edges
                    edges = cv2.Canny(gray, 50, 150)
                    mask = cv2.dilate(edges, None, iterations=3)
                # Clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                return mask

            # Build silhouette masks
            mask0 = segment_foreground(f0, bg_frame0) if f0 is not None else None  # front: X-Y
            mask1 = segment_foreground(f1, bg_frame1) if f1 is not None else None  # top: X-Z

            # Resize masks to grid dimensions (still on GPU as UMat)
            if mask0 is not None:
                sil_front = cv2.resize(mask0, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
            else:
                sil_front = np.ones((GRID_H, GRID_W), dtype=np.uint8) * 255
            if mask1 is not None:
                sil_top = cv2.resize(mask1, (GRID_W, GRID_D), interpolation=cv2.INTER_AREA)
            else:
                sil_top = np.ones((GRID_D, GRID_W), dtype=np.uint8) * 255

            # Also resize color frames on GPU before transferring to CPU
            f0_small_u = (
                cv2.resize(f0, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
                if f0 is not None
                else None
            )
            f1_small_u = (
                cv2.resize(f1, (GRID_W, GRID_D), interpolation=cv2.INTER_AREA)
                if f1 is not None
                else None
            )

            # Transfer results from GPU to CPU for voxel carving (array indexing)
            if isinstance(sil_front, cv2.UMat):
                sil_front = sil_front.get()
            if isinstance(sil_top, cv2.UMat):
                sil_top = sil_top.get()

            # Threshold to binary
            sil_front = (sil_front > 127).astype(np.uint8)
            sil_top = (sil_top > 127).astype(np.uint8)

            # --- Visual hull intersection ---
            # front view (cam0): column x → voxel X, row y → voxel Y (flipped)
            # top view (cam1): column x → voxel X, row z → voxel Z
            # A voxel (x, y, z) is occupied if sil_front[GRID_H-1-y, x] AND sil_top[z, x]

            voxels = []
            f0_small = f0_small_u.get() if f0_small_u is not None else None
            f1_small = f1_small_u.get() if f1_small_u is not None else None

            for x in range(GRID_W):
                for y in range(GRID_H):
                    if not sil_front[GRID_H - 1 - y, x]:
                        continue
                    for z in range(GRID_D):
                        if sil_top[z, x]:
                            # Get color: blend front and top camera colors
                            r, g, b = 128, 128, 128
                            if f0_small is not None:
                                bgr = f0_small[GRID_H - 1 - y, x]
                                r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
                            if f1_small is not None:
                                bgr1 = f1_small[z, x]
                                r = (r + int(bgr1[2])) >> 1
                                g = (g + int(bgr1[1])) >> 1
                                b = (b + int(bgr1[0])) >> 1
                            voxels.append([x, y, z, r, g, b])

            # Cap voxel count for performance
            if len(voxels) > 8000:
                # Subsample — keep every Nth
                step = len(voxels) // 8000 + 1
                voxels = voxels[::step]

            await ws.send_json(
                {
                    "type": "voxels",
                    "voxels": voxels,  # [[x,y,z,r,g,b], ...]
                    "gridW": GRID_W,
                    "gridH": GRID_H,
                    "gridD": GRID_D,
                    "frame": frame_count,
                    "cam0": f0 is not None,
                    "cam1": f1 is not None,
                }
            )

            await asyncio.sleep(0.75)  # ~1.3 fps reconstruction rate

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("RealWorld3D WS error: %s", e)


# ---------------------------------------------------------------------------
# Arm 3D Pose Fusion — WebSocket + REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/arm3d/status")
async def arm3d_status():
    """Pipeline health, per-camera status, fusion quality."""
    if not _HAS_POSE_FUSION or pose_fusion is None:
        return {"available": False, "error": "Pose fusion module not available"}

    # Check cameras
    import httpx

    cam_status = {}
    for cam_id in [0, 1]:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                resp = await client.get(f"http://localhost:8081/snap/{cam_id}")
                cam_status[f"cam{cam_id}"] = {"online": resp.status_code == 200}
        except Exception:
            cam_status[f"cam{cam_id}"] = {"online": False}

    quality = pose_fusion.get_tracking_quality()
    return {
        "available": True,
        "cameras": cam_status,
        "fusion_quality": quality,
        "arm_connected": arm is not None and arm.is_connected,
    }


@app.websocket("/ws/arm3d")
async def ws_arm3d(ws: WebSocket):
    """Stream fused 3D arm positions at ~10Hz.

    Uses FK engine + optional visual joint detection from dual cameras.
    Gracefully degrades to FK-only when cameras are unavailable.
    """
    await ws.accept()

    if not _HAS_POSE_FUSION or pose_fusion is None:
        await ws.send_json({"type": "error", "message": "Pose fusion not available"})
        await ws.close()
        return

    import httpx

    try:
        while True:
            # 1. Get joint angles from arm state
            state = get_arm_state()
            joints_deg = state.get("joints", [0.0] * 6)

            # 2. Compute FK 3D positions
            fk_pos = fk_positions_fn(joints_deg)

            # 3. Try to get camera frames and run visual detection
            cam0_dets = None
            cam1_dets = None

            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    results = await asyncio.gather(
                        client.get("http://localhost:8081/snap/0"),
                        client.get("http://localhost:8081/snap/1"),
                        return_exceptions=True,
                    )

                import cv2

                for cam_id, resp in enumerate(results):
                    if isinstance(resp, Exception) or resp.status_code != 200:
                        continue
                    frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    segmenter = arm3d_segmenters.get(cam_id)
                    if segmenter is None:
                        continue

                    seg = segmenter.segment_arm(frame)

                    # Project FK to pixels for this camera (simplified — use pinhole at identity)
                    # In production, use actual camera calibration
                    fk_pixels = [(float(p[0] * 500 + 320), float(p[1] * 500 + 240)) for p in fk_pos]

                    dets = arm3d_detector.detect_joints(seg, fk_pixels)
                    if cam_id == 0:
                        cam0_dets = dets
                    else:
                        cam1_dets = dets
            except Exception:
                pass  # Cameras unavailable — FK-only mode

            # 4. Fuse
            result = pose_fusion.fuse(
                fk_pos,
                cam0_detections=cam0_dets,
                cam1_detections=cam1_dets,
                # Note: calibrations would come from saved config in production
            )

            # 5. Send
            await ws.send_json(
                {
                    "type": "arm3d",
                    "positions": [[round(c, 4) for c in p] for p in result.positions],
                    "confidence": [round(c, 3) for c in result.confidence],
                    "source": result.source.value,
                }
            )

            await asyncio.sleep(0.1)  # ~10Hz

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("Arm3D WS error: %s", e)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------

try:
    from src.calibration.calibration_runner import CalibrationRunner, CalibrationSession
    from src.calibration.pipeline import LLMCalibrationPipeline

    _HAS_CALIBRATION = True
except ImportError:
    _HAS_CALIBRATION = False

_calibration_runner: Optional[CalibrationRunner] = None
_calibration_pipeline: Optional[LLMCalibrationPipeline] = None
_calibration_task: Optional[asyncio.Task] = None
_calibration_session: Optional[CalibrationSession] = None
_calibration_error: Optional[str] = None
_calibration_report = None  # ComparisonReport from pipeline


@app.post("/api/calibration/start")
async def calibration_start():
    """Kick off the 20-pose calibration sequence with CV+LLM detection."""
    global _calibration_runner, _calibration_pipeline, _calibration_task
    global _calibration_session, _calibration_error, _calibration_report
    if not _HAS_CALIBRATION:
        return JSONResponse(
            {"ok": False, "error": "Calibration module not available"}, status_code=501
        )
    if _calibration_task and not _calibration_task.done():
        return JSONResponse({"ok": False, "error": "Calibration already running"}, status_code=409)

    _calibration_session = None
    _calibration_error = None
    _calibration_report = None

    # Create pipeline with LLM detection
    import os as _os_mod

    gemini_key = _os_mod.environ.get("GEMINI_API_KEY")
    _calibration_pipeline = LLMCalibrationPipeline(gemini_api_key=gemini_key)
    _calibration_runner = _calibration_pipeline.runner

    # Pre-generate session_id so it's available in the response
    import time as _time_mod

    _calibration_runner._session_id = f"cal_{int(_time_mod.time())}"

    async def _run():
        global _calibration_session, _calibration_error, _calibration_report
        try:
            # Run calibration (arm movement + frame capture)
            _calibration_session = await _calibration_runner.run_full_calibration()
            action_log.add(
                "CALIBRATION", f"Poses complete: {len(_calibration_session.captures)}", "info"
            )

            # Run CV + LLM detection on captured frames
            _calibration_report = await _calibration_pipeline.run_detection_only(
                _calibration_session
            )
            action_log.add(
                "CALIBRATION",
                f"Detection complete. LLM tokens: {_calibration_report.total_llm_tokens}",
                "info",
            )

            # Auto-save results to disk
            results_dir = str(Path(__file__).parent.parent / "calibration_results")
            save_path = _calibration_pipeline.save_results(
                _calibration_session, _calibration_report, results_dir
            )
            action_log.add("CALIBRATION", f"Results saved to {save_path}", "info")

            # Store report for API access
            session_id = _calibration_runner._session_id
            if session_id:
                _calib_comparison_reports[session_id] = _calibration_report

        except Exception as e:
            _calibration_error = str(e)
            action_log.add("CALIBRATION", f"Failed: {e}", "error")
            import traceback

            logger.error(f"Calibration error: {traceback.format_exc()}")

    _calibration_task = asyncio.create_task(_run())
    action_log.add("CALIBRATION", "Started with CV+LLM pipeline", "info")
    return {"ok": True, "session_id": _calibration_runner._session_id}


@app.get("/api/calibration/status")
async def calibration_status():
    """Return calibration progress."""
    if not _HAS_CALIBRATION or _calibration_runner is None:
        return {"running": False, "current_pose": -1, "total_poses": 0}
    progress = _calibration_runner.progress
    if _calibration_error:
        progress["error"] = _calibration_error
    if _calibration_task:
        progress["done"] = _calibration_task.done()
    return progress


@app.post("/api/calibration/stop")
async def calibration_stop():
    """Abort running calibration."""
    if not _HAS_CALIBRATION or _calibration_runner is None:
        return {"ok": False, "error": "No calibration running"}
    _calibration_runner.abort()
    action_log.add("CALIBRATION", "Abort requested", "warning")
    return {"ok": True}


@app.get("/api/calibration/results/{session_id}")
async def calibration_results(session_id: str):
    """Get calibration results (without raw images)."""
    if not _HAS_CALIBRATION or _calibration_session is None:
        return JSONResponse({"ok": False, "error": "No results available"}, status_code=404)
    if _calibration_runner and _calibration_runner._session_id != session_id:
        return JSONResponse({"ok": False, "error": "Session not found"}, status_code=404)
    return {
        "ok": True,
        "start_time": _calibration_session.start_time,
        "end_time": _calibration_session.end_time,
        "total_poses": _calibration_session.total_poses,
        "captures": [
            {
                "pose_index": c.pose_index,
                "commanded_angles": list(c.commanded_angles),
                "actual_angles": c.actual_angles,
                "timestamp": c.timestamp,
                "has_cam0": len(c.cam0_jpeg) > 0,
                "has_cam1": len(c.cam1_jpeg) > 0,
            }
            for c in _calibration_session.captures
        ],
    }


# ---------------------------------------------------------------------------
# Calibration Comparison Report endpoints
# ---------------------------------------------------------------------------

try:
    from src.calibration.results_reporter import CalibrationReporter as _CalibReporter

    _HAS_CALIB_REPORTER = True
except ImportError:
    _HAS_CALIB_REPORTER = False

_calib_reporter_instance = _CalibReporter() if _HAS_CALIB_REPORTER else None
_calib_comparison_reports: dict = {}  # session_id -> ComparisonReport


@app.get("/api/calibration/report/{session_id}")
async def calibration_report_md(session_id: str):
    """Return markdown comparison report for a calibration session."""
    if not _HAS_CALIB_REPORTER:
        return JSONResponse(
            {"ok": False, "error": "Reporter module not available"}, status_code=501
        )
    report = _calib_comparison_reports.get(session_id)
    if not report:
        return JSONResponse({"ok": False, "error": "Report not found"}, status_code=404)
    md = _calib_reporter_instance.generate_markdown(report)
    return JSONResponse({"ok": True, "markdown": md})


@app.get("/api/calibration/report/{session_id}/json")
async def calibration_report_json(session_id: str):
    """Return JSON comparison report for a calibration session."""
    if not _HAS_CALIB_REPORTER:
        return JSONResponse(
            {"ok": False, "error": "Reporter module not available"}, status_code=501
        )
    report = _calib_comparison_reports.get(session_id)
    if not report:
        return JSONResponse({"ok": False, "error": "Report not found"}, status_code=404)
    return _calib_reporter_instance.generate_json(report)


# ---------------------------------------------------------------------------
# Camera Extrinsics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/calibration/extrinsics")
async def get_calibration_extrinsics():
    """Return current camera extrinsics (from saved calibration file)."""
    try:
        from src.calibration.extrinsics_solver import load_extrinsics

        extrinsics_path = str(
            Path(__file__).parent.parent / "calibration_results" / "camera_extrinsics.json"
        )
        data = load_extrinsics(extrinsics_path)
        if data is None:
            return JSONResponse(
                {"ok": False, "error": "No extrinsics calibration found"}, status_code=404
            )
        return {"ok": True, **data}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/calibration/frames/{session_id}/{pose_index}/{camera_id}")
async def get_calibration_frame(session_id: str, pose_index: int, camera_id: str):
    """Return a raw JPEG frame from a calibration session."""
    if not _HAS_CALIBRATION or _calibration_session is None:
        return JSONResponse({"ok": False, "error": "No session available"}, status_code=404)
    if _calibration_runner and _calibration_runner._session_id != session_id:
        return JSONResponse({"ok": False, "error": "Session not found"}, status_code=404)
    if pose_index < 0 or pose_index >= len(_calibration_session.captures):
        return JSONResponse({"ok": False, "error": "Pose index out of range"}, status_code=404)
    cap = _calibration_session.captures[pose_index]
    jpeg = cap.cam0_jpeg if camera_id == "cam0" else cap.cam1_jpeg
    if not jpeg:
        return JSONResponse({"ok": False, "error": f"No frame for {camera_id}"}, status_code=404)
    from starlette.responses import Response

    return Response(content=jpeg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# VLA (Vision-Language-Action) endpoints
# ---------------------------------------------------------------------------

_vla_task: Optional[asyncio.Task] = None


class VLAActRequest(BaseModel):
    """Request to execute a VLA task."""

    task: str = Field(description="Natural language task, e.g. 'pick up the red bull can'")
    max_steps: int = Field(default=40, ge=1, le=100)
    settle_time_s: float = Field(default=1.5, ge=0.5, le=5.0)


class VLACollectRequest(BaseModel):
    """Request to start/stop demo collection."""

    action: str = Field(description="'start' or 'stop'")
    task: str = Field(default="", description="Task description (for start)")
    success: bool = Field(default=True, description="Whether demo was successful (for stop)")
    notes: str = Field(default="")


@app.get("/api/vla/status")
async def vla_status():
    """Get VLA system status."""
    if not _HAS_VLA or vla_controller is None:
        return {"available": False, "error": "VLA module not loaded"}

    status = vla_controller.get_status()
    status["available"] = True
    if vla_data_collector:
        status["collector"] = vla_data_collector.get_status()
    return status


@app.post("/api/vla/act")
async def vla_act(req: VLAActRequest):
    """Execute a VLA task: natural language command → arm executes.

    This is the main VLA endpoint. Send a task like "pick up the red bull can"
    and the arm will use camera vision + language understanding to execute it.
    """
    global _vla_task

    if not _HAS_VLA or vla_controller is None:
        return JSONResponse(
            {"ok": False, "error": "VLA module not available"},
            status_code=501,
        )

    if vla_controller.is_busy:
        return JSONResponse(
            {
                "ok": False,
                "error": "VLA controller is busy with another task",
                "state": vla_controller.state.value,
            },
            status_code=409,
        )

    if not (smoother and smoother._arm_enabled):
        return JSONResponse(
            {"ok": False, "error": "Arm not enabled"},
            status_code=409,
        )

    action_log.add("VLA", f"Task started: '{req.task}'", "info")

    # Configure controller
    vla_controller._max_steps = req.max_steps
    vla_controller._settle_time = req.settle_time_s

    # Run as background task so the HTTP response returns immediately
    async def _run_task():
        try:
            result = await vla_controller.execute(req.task)
            level = "info" if result.success else "warning"
            action_log.add(
                "VLA",
                f"Task {'DONE' if result.success else 'FAILED'}: '{req.task}' "
                f"({result.actions_executed} actions, {result.total_time_s:.1f}s)",
                level,
            )
        except Exception as e:
            action_log.add("VLA", f"Task error: {e}", "error")
            logger.exception("VLA task error")

    if _vla_task and not _vla_task.done():
        _vla_task.cancel()

    _vla_task = asyncio.create_task(_run_task())

    return {
        "ok": True,
        "action": "VLA_ACT",
        "task": req.task,
        "message": f"VLA task started: '{req.task}'",
        "state": vla_controller.state.value,
    }


@app.post("/api/vla/act-sync")
async def vla_act_sync(req: VLAActRequest):
    """Execute a VLA task synchronously (waits for completion).

    Same as /api/vla/act but blocks until the task finishes.
    Returns the full result including all steps taken.
    """
    if not _HAS_VLA or vla_controller is None:
        return JSONResponse(
            {"ok": False, "error": "VLA module not available"},
            status_code=501,
        )

    if vla_controller.is_busy:
        return JSONResponse(
            {"ok": False, "error": "VLA controller is busy"},
            status_code=409,
        )

    if not (smoother and smoother._arm_enabled):
        return JSONResponse(
            {"ok": False, "error": "Arm not enabled"},
            status_code=409,
        )

    action_log.add("VLA", f"Task started (sync): '{req.task}'", "info")

    vla_controller._max_steps = req.max_steps
    vla_controller._settle_time = req.settle_time_s

    result = await vla_controller.execute(req.task)

    level = "info" if result.success else "warning"
    action_log.add(
        "VLA",
        f"Task {'DONE' if result.success else 'FAILED'}: '{req.task}' "
        f"({result.actions_executed} actions, {result.total_time_s:.1f}s)",
        level,
    )

    return {
        "ok": result.success,
        "action": "VLA_ACT_SYNC",
        "task": result.task,
        "success": result.success,
        "message": result.message,
        "error": result.error,
        "total_time_s": round(result.total_time_s, 1),
        "actions_executed": result.actions_executed,
        "observations_made": result.observations_made,
        "final_phase": result.final_phase,
        "steps": [
            {
                "step": s.step_num,
                "state": s.state,
                "action": s.action,
                "phase": s.phase,
                "confidence": s.confidence,
                "execution_time_ms": round(s.execution_time_ms, 0),
                "notes": s.notes,
            }
            for s in result.steps
        ],
    }


@app.post("/api/vla/abort")
async def vla_abort():
    """Abort the current VLA task."""
    if not _HAS_VLA or vla_controller is None:
        return JSONResponse(
            {"ok": False, "error": "VLA module not available"},
            status_code=501,
        )

    vla_controller.abort()
    action_log.add("VLA", "Task abort requested", "warning")
    return {"ok": True, "action": "VLA_ABORT"}


@app.post("/api/vla/collect")
async def vla_collect(req: VLACollectRequest):
    """Start or stop demonstration collection for VLA training.

    Start: POST /api/vla/collect {"action": "start", "task": "pick up the can"}
    Stop:  POST /api/vla/collect {"action": "stop", "success": true}
    """
    if not _HAS_VLA or vla_data_collector is None:
        return JSONResponse(
            {"ok": False, "error": "VLA data collector not available"},
            status_code=501,
        )

    if req.action == "start":
        if not req.task:
            return JSONResponse(
                {"ok": False, "error": "Task description required for 'start'"},
                status_code=400,
            )
        try:
            demo_id = vla_data_collector.start(req.task, notes=req.notes)
            action_log.add("VLA_COLLECT", f"Recording started: {demo_id}", "info")
            return {"ok": True, "demo_id": demo_id, "task": req.task}
        except RuntimeError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=409)

    elif req.action == "stop":
        demo_path = vla_data_collector.stop(success=req.success, notes=req.notes)
        if demo_path:
            action_log.add(
                "VLA_COLLECT",
                f"Recording stopped: {demo_path} (success={req.success})",
                "info",
            )
            return {"ok": True, "demo_path": demo_path, "success": req.success}
        return {"ok": False, "error": "Not recording"}

    else:
        return JSONResponse(
            {"ok": False, "error": f"Unknown action: {req.action}"},
            status_code=400,
        )


@app.get("/api/vla/demos")
async def vla_demos():
    """List all recorded demonstrations."""
    if not _HAS_VLA or vla_data_collector is None:
        return {"demos": [], "error": "VLA data collector not available"}

    return {"demos": vla_data_collector.list_demos()}


# ---------------------------------------------------------------------------
# Static files — versioned UIs all pointing to the same server
# /v1/ → V1 stable base, /v2/ → V2 Cartesian controls, / → V1 (default)
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

v1_dir = static_dir / "v1"
v2_dir = static_dir / "v2"
v3_dir = static_dir / "v3"
if v1_dir.is_dir():
    app.mount("/v1", StaticFiles(directory=str(v1_dir), html=True), name="static-v1")
if v2_dir.is_dir():
    app.mount("/v2", StaticFiles(directory=str(v2_dir), html=True), name="static-v2")
if v3_dir.is_dir():
    app.mount("/v3", StaticFiles(directory=str(v3_dir), html=True), name="static-v3")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static-assets")
tools_dir = Path(__file__).parent.parent / "tools"
if tools_dir.is_dir():
    app.mount("/tools", StaticFiles(directory=str(tools_dir), html=True), name="tools")
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/")
async def serve_root():
    """Redirect root to /ui/."""
    from starlette.responses import RedirectResponse

    return RedirectResponse(url="/ui/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_pidfile():
    """Write PID file for reliable process management."""
    pidfile = Path("/tmp/th3cl4w-server.pid")
    pidfile.write_text(str(os.getpid()))


def _remove_pidfile():
    """Remove PID file on shutdown."""
    pidfile = Path("/tmp/th3cl4w-server.pid")
    try:
        pidfile.unlink(missing_ok=True)
    except Exception:
        pass


def _handle_sigterm(signum, frame):
    """Graceful shutdown on SIGTERM — lets uvicorn clean up."""
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    _remove_pidfile()
    # Raise SystemExit so uvicorn's shutdown hooks run (lifespan cleanup)
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    _write_pidfile()
    logger.info(
        "Starting th3cl4w web panel on %s:%d (simulate=%s, pid=%d)",
        args.host,
        args.port,
        args.simulate,
        os.getpid(),
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        _remove_pidfile()
