#!/usr/bin/env python3.12
"""Map Server — 3D spatial mapping for the D1 arm workspace.

Runs on port 8083. Provides real-time 3D scene (arm skeleton + environment
point cloud + objects) via REST + WebSocket.
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

# Suppress noisy httpx request logging (depth/location polls at high frequency)
logging.getLogger("httpx").setLevel(logging.WARNING)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.map.scene import Scene, ArmSkeletonData, ObjectData
from src.map.arm_model import ArmModel
from src.map.env_map import EnvMap, EnvMapConfig, MapScanManager, SCAN_DIR
from src.map.collision_map import CollisionMap
from src.map.ingest import DataIngest, IngestConfig
from src.map.ws_hub import WSHub

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAP_SERVER_PORT = int(os.getenv("MAP_SERVER_PORT", "8083"))
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "http://localhost:8080")
from src.config.camera_config import CAMERA_SERVER_URL
LOCATION_SERVER_URL = os.getenv("LOCATION_SERVER_URL", "http://localhost:8082")

ARM_POLL_HZ = float(os.getenv("ARM_POLL_HZ", "2"))
DEPTH_POLL_HZ = float(os.getenv("DEPTH_POLL_HZ", "3"))
LOCATION_POLL_HZ = float(os.getenv("LOCATION_POLL_HZ", "5"))
WS_BROADCAST_HZ = float(os.getenv("WS_BROADCAST_HZ", "15"))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
scene = Scene()
arm_model = ArmModel()
env_map = EnvMap()
collision_map = CollisionMap()
ws_hub = WSHub(scene, default_hz=WS_BROADCAST_HZ)
ingest: Optional[DataIngest] = None
scan_manager: Optional[MapScanManager] = None

_start_time = time.time()


# ---------------------------------------------------------------------------
# Ingest callbacks
# ---------------------------------------------------------------------------
async def _on_arm_state(joints: List[float], gripper_mm: float) -> None:
    """Called when new joint state arrives from main server."""
    skeleton = arm_model.update(joints, gripper_mm)
    scene.update_arm(skeleton)


async def _on_depth_frame(frame_bgr, joints: List[float]) -> None:
    """Called when new camera frame arrives for depth processing."""
    # Run depth processing in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    n_new = await loop.run_in_executor(
        None, env_map.ingest_depth_frame, frame_bgr, joints, None
    )
    if n_new > 0:
        cloud = env_map.get_cloud()
        voxels = env_map.get_voxel_centers()
        scene.update_point_cloud(cloud, voxel_centers=voxels)
        # Update collision map
        collision_map.update_from_cloud(cloud)


async def _on_objects(objects_list: List[Dict[str, Any]]) -> None:
    """Called when object list arrives from location server."""
    from src.location.reachability import classify_reach, ReachStatus
    import numpy as np

    obj_data = []
    collision_objects = []
    for obj in objects_list:
        pos = obj.get("position_mm", obj.get("position", [0, 0, 0]))
        bbox = obj.get("bbox_mm", obj.get("bbox", [50, 50, 50]))
        label = obj.get("label", obj.get("name", "unknown"))
        conf = obj.get("confidence", 0.0)
        obj_id = obj.get("id", label)

        # Reachability check
        try:
            status, _ = classify_reach(np.array(pos, dtype=float))
            reachable = status in (ReachStatus.REACHABLE, ReachStatus.MARGINAL)
        except Exception:
            reachable = False

        obj_data.append(ObjectData(
            id=obj_id,
            label=label,
            position_mm=list(pos),
            bbox_mm=list(bbox),
            confidence=conf,
            reachable=reachable,
        ))
        collision_objects.append({"position_mm": pos, "bbox_mm": bbox})

    scene.update_objects(obj_data)
    collision_map.update_from_objects(collision_objects)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingest, scan_manager

    logger.info("Map server starting on port %d", MAP_SERVER_PORT)

    # Compute reach envelope
    envelope = arm_model.compute_reach_envelope()
    scene.set_reach_envelope(envelope)

    # Start ingest
    ingest_config = IngestConfig(
        main_server_url=MAIN_SERVER_URL,
        camera_server_url=CAMERA_SERVER_URL,
        location_server_url=LOCATION_SERVER_URL,
        arm_poll_hz=ARM_POLL_HZ,
        depth_poll_hz=DEPTH_POLL_HZ,
        location_poll_hz=LOCATION_POLL_HZ,
    )
    ingest = DataIngest(ingest_config)
    ingest.on_arm_state = _on_arm_state
    ingest.on_depth_frame = _on_depth_frame
    ingest.on_objects = _on_objects

    # Scan manager (no arm command fn for now — map server is read-only)
    scan_manager = MapScanManager(
        env_map=env_map,
        camera_url=CAMERA_SERVER_URL,
    )

    await ingest.start()
    await ws_hub.start_broadcast()

    logger.info("Map server ready")
    yield

    # Shutdown
    await ws_hub.stop_broadcast()
    await ingest.stop()
    logger.info("Map server stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="th3cl4w Map Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/map/status")
async def map_status():
    """Map server health: ingest rates, scene stats, connected clients."""
    return {
        "ok": True,
        "uptime_s": round(time.time() - _start_time, 1),
        "ingest": ingest.stats.to_dict() if ingest else {},
        "scene": {
            "frame": scene.frame,
            "arm_joints": len(scene.arm.joints),
            "env_points": env_map.get_stats()["total_points"],
            "objects": len(scene.objects),
            "voxels": collision_map.get_occupied_count(),
        },
        "ws_clients": ws_hub.client_count,
    }


@app.get("/api/map/scene")
async def get_scene():
    """Full scene snapshot as JSON (for debugging)."""
    return scene.snapshot(full=True)


@app.get("/api/map/arm")
async def get_arm():
    """Current arm skeleton."""
    return {
        "joints": scene.arm.joints,
        "links": scene.arm.links,
        "gripper_mm": scene.arm.gripper_mm,
        "ee_pose": scene.arm.ee_pose,
        "joint_angles_deg": scene.arm.joint_angles_deg,
    }


@app.get("/api/map/pointcloud")
async def get_pointcloud():
    """Point cloud summary (full cloud too large for JSON — use PLY export)."""
    stats = env_map.get_stats()
    return {
        "stats": stats,
        "note": "Use /api/map/scan/result for PLY download",
    }


@app.get("/api/map/pointcloud/stats")
async def get_pointcloud_stats():
    """Point cloud statistics."""
    return env_map.get_stats()


@app.get("/api/map/voxels")
async def get_voxels():
    """Occupied voxel centers."""
    centers = collision_map.get_occupied_centers()
    # Limit for JSON response
    if len(centers) > 20000:
        idx = __import__("numpy").random.choice(len(centers), 20000, replace=False)
        centers = centers[idx]
    return {"voxels": centers.tolist(), "count": collision_map.get_occupied_count()}


@app.get("/api/map/collision/check")
async def collision_check(
    x: float = Query(...), y: float = Query(...), z: float = Query(...),
    radius: float = Query(0.0),
):
    """Check if a point/sphere is free or occupied."""
    if radius > 0:
        collides = collision_map.check_sphere([x, y, z], radius / 1000.0)
        return {"status": "occupied" if collides else "free", "point": [x, y, z], "radius": radius}
    else:
        status = collision_map.check_point([x / 1000.0, y / 1000.0, z / 1000.0])
        return {"status": status, "point": [x, y, z]}


class PathCheckRequest(BaseModel):
    points: List[List[float]]
    radius: float = 20.0  # mm


@app.post("/api/map/collision/check-path")
async def collision_check_path(req: PathCheckRequest):
    """Check a path for collisions."""
    points_m = [[p[0] / 1000, p[1] / 1000, p[2] / 1000] for p in req.points]
    collisions = collision_map.check_path(points_m, req.radius / 1000.0)
    return {"collisions": collisions, "path_clear": len(collisions) == 0}


@app.get("/api/map/objects")
async def get_objects():
    """Objects from location server currently in scene."""
    return {
        "objects": [
            {
                "id": o.id,
                "label": o.label,
                "position_mm": o.position_mm,
                "bbox_mm": o.bbox_mm,
                "confidence": o.confidence,
                "reachable": o.reachable,
            }
            for o in scene.objects
        ]
    }


class MapObjectsRequest(BaseModel):
    objects: List[Dict[str, Any]]


@app.post("/api/map/objects")
async def post_objects(req: MapObjectsRequest):
    """Receive detected objects from the main server and update the scene."""
    await _on_objects(req.objects)
    logger.info("Received %d objects via POST", len(req.objects))
    return {"ok": True, "count": len(req.objects)}


# --- Scan endpoints ---

@app.post("/api/map/scan/start")
async def scan_start():
    """Begin arm-sweep 3D scan."""
    if scan_manager:
        return await scan_manager.start_scan()
    return {"ok": False, "error": "Scan manager not initialized"}


@app.post("/api/map/scan/stop")
async def scan_stop():
    """Abort scan."""
    if scan_manager:
        return await scan_manager.stop_scan()
    return {"ok": False, "error": "Scan manager not initialized"}


@app.get("/api/map/scan/status")
async def scan_status():
    """Scan progress."""
    if scan_manager:
        return scan_manager.get_status()
    return {"running": False, "phase": "idle"}


@app.get("/api/map/scan/list")
async def scan_list():
    """Previous scans."""
    return {"scans": MapScanManager.list_scans()}


@app.get("/api/map/scan/result")
async def scan_result(scan_id: Optional[str] = None):
    """Download PLY file."""
    path = MapScanManager.get_scan_ply(scan_id)
    if path and Path(path).exists():
        return FileResponse(path, media_type="application/octet-stream", filename="scan.ply")
    return JSONResponse({"error": "No scan found"}, status_code=404)


# --- Environment config endpoints ---

@app.post("/api/map/env/clear")
async def env_clear():
    """Clear environment point cloud."""
    env_map.clear()
    scene.clear_point_cloud()
    collision_map.update_from_cloud(__import__("numpy").zeros((0, 3)))
    return {"ok": True}


class EnvConfigRequest(BaseModel):
    voxel_size_m: Optional[float] = None
    max_points: Optional[int] = None
    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None


@app.post("/api/map/env/config")
async def env_config(req: EnvConfigRequest):
    """Update environment map configuration."""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    env_map.config.update(**updates)
    if "voxel_size_m" in updates:
        collision_map.voxel_size = updates["voxel_size_m"]
    return {"ok": True, "config": env_map.config.to_dict()}


@app.get("/api/map/reach-envelope")
async def get_reach_envelope():
    """Pre-computed reach envelope mesh."""
    envelope = scene.get_reach_envelope()
    if envelope:
        return envelope
    return {"vertices": [], "faces": [], "radius_m": 0.55}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/map")
async def ws_map(ws: WebSocket):
    """Scene updates WebSocket."""
    client = await ws_hub.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await ws_hub.handle_message(ws, data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WS error: %s", e)
    finally:
        ws_hub.disconnect(ws)


# ---------------------------------------------------------------------------
# UI redirect
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"service": "th3cl4w Map Server", "port": MAP_SERVER_PORT, "ui": "/static/map3d.html"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="th3cl4w Map Server")
    parser.add_argument("--port", type=int, default=MAP_SERVER_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging to logs/map.log")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom log output directory (default: logs/)")
    args = parser.parse_args()

    from src.utils.logging_config import setup_logging
    setup_logging(server_name="map", debug=args.debug, log_dir=args.log_dir)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
