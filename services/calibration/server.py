"""
Calibration Server — Camera and joint calibration management.

Provides endpoints to run intrinsic/extrinsic camera calibration,
joint-mapping calibration, and view results. Uses the calibration
pipeline modules under this service directory.

Port: 8094 (configurable via CALIBRATION_PORT env var)

Usage:
    python -m services.calibration.server
    python -m services.calibration.server --debug
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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.calibration")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Calibration — Camera & Joint Calibration")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="calibration", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CalibrationRequest(BaseModel):
    """Request to start a calibration run."""
    calibration_type: str = Field(
        description="Type: intrinsic, extrinsic, joint_mapping, hand_eye, full_pipeline"
    )
    camera_id: Optional[int] = Field(default=None, description="Camera ID to calibrate")
    num_images: int = Field(default=20, ge=5, le=100, description="Number of calibration images")
    board_type: str = Field(default="charuco", description="Calibration board type: charuco, checkerboard")


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_calibration_runs: list[dict] = []
_latest_results: dict[str, dict] = {}
_publisher: Any = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _publisher

    try:
        from shared.bus.publisher import EventPublisher
        _publisher = EventPublisher()
        await _publisher.connect()
    except Exception as e:
        logger.info("Message bus publisher not available: %s (operating standalone)", e)

    logger.info("Calibration service ready")
    yield

    if _publisher:
        await _publisher.close()
    logger.info("Calibration shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Calibration",
    description="Camera intrinsic/extrinsic and joint-mapping calibration",
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

# Serve static files (calibration targets, etc.)
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
        "service": "calibration",
        "total_runs": len(_calibration_runs),
        "latest_results": list(_latest_results.keys()),
        "bus_connected": _publisher.is_connected if _publisher else False,
    }


# ---------------------------------------------------------------------------
# Calibration API
# ---------------------------------------------------------------------------


@app.post("/api/calibrate")
async def start_calibration(req: CalibrationRequest):
    """Start a calibration run."""
    run_id = f"cal_{uuid.uuid4().hex[:8]}"
    now = time.time()

    run = {
        "run_id": run_id,
        "type": req.calibration_type,
        "camera_id": req.camera_id,
        "num_images": req.num_images,
        "board_type": req.board_type,
        "status": "started",
        "started_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    _calibration_runs.append(run)

    logger.info("Calibration %s started: type=%s camera=%s",
                run_id, req.calibration_type, req.camera_id)
    return run


@app.get("/api/calibrations")
async def list_calibrations(limit: int = 20):
    """List calibration runs."""
    runs = _calibration_runs[-limit:]
    runs.reverse()
    return {
        "calibrations": runs,
        "count": len(runs),
        "total": len(_calibration_runs),
        "timestamp": time.time(),
    }


@app.get("/api/calibration/{run_id}")
async def get_calibration(run_id: str):
    """Get details of a specific calibration run."""
    for run in _calibration_runs:
        if run["run_id"] == run_id:
            return run
    return JSONResponse({"error": f"Run '{run_id}' not found"}, status_code=404)


@app.get("/api/results")
async def get_results():
    """Get the latest calibration results for all types."""
    return {
        "results": _latest_results,
        "timestamp": time.time(),
    }


@app.get("/api/results/{cal_type}")
async def get_result_by_type(cal_type: str):
    """Get the latest calibration result for a specific type."""
    result = _latest_results.get(cal_type)
    if result is None:
        return JSONResponse(
            {"error": f"No results for type '{cal_type}'",
             "available": list(_latest_results.keys())},
            status_code=404,
        )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.CALIBRATION_PORT
    logger.info("Starting Calibration on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
