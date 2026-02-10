#!/usr/bin/env python3.12
"""
th3cl4w — ASCII Video Processing Server

FastAPI service on port 8084 that:
- Streams live ASCII video from all cameras via WebSocket
- Captures ASCII snapshots and sends to Gemini for text-only analysis
- Manages conversation sessions with the LLM analyst
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.ascii.converter import fetch_and_convert, CHARSETS
from src.ascii.streamer import AsciiStreamer, StreamConfig
from src.ascii.session import SessionManager
from src.vision.ascii_converter import CHARSET_STANDARD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("th3cl4w.ascii_server")

# Globals
_start_time = time.time()
streamer: Optional[AsciiStreamer] = None
session_mgr = SessionManager()
analyst = None  # Lazy-loaded to avoid import error if no API key


def _get_analyst():
    global analyst
    if analyst is None:
        try:
            from src.ascii.llm_analyst import AsciiAnalyst
            analyst = AsciiAnalyst()
            logger.info("LLM analyst initialized (model: %s)", analyst._model_name)
        except Exception as e:
            logger.warning("Failed to init LLM analyst: %s", e)
            return None
    return analyst


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global streamer
    config = StreamConfig(width=120, height=60, charset_name="standard", fps=5.0)
    streamer = AsciiStreamer(cam_ids=[0, 1, 2], config=config)
    streamer.start()
    logger.info("ASCII Video Server starting on port 8084")
    yield
    streamer.stop()
    logger.info("ASCII Video Server stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="th3cl4w ASCII Video", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    cam_id: int = Field(ge=0, le=2, default=0)
    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ConfigRequest(BaseModel):
    width: Optional[int] = Field(None, ge=20, le=300)
    height: Optional[int] = Field(None, ge=10, le=150)
    charset_name: Optional[str] = None
    fps: Optional[float] = Field(None, ge=0.5, le=30)
    color: Optional[bool] = None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ascii/status")
async def get_status():
    """Health/status endpoint for heartbeat checks."""
    uptime = time.time() - _start_time
    status = {
        "service": "ascii_server",
        "status": "ok" if streamer is not None else "starting",
        "port": 8084,
        "uptime_seconds": round(uptime, 1),
    }
    if streamer is not None:
        status["cameras"] = streamer.cam_ids
        status["config"] = streamer.config.to_dict()
        a = _get_analyst()
        status["analyst_available"] = a is not None
    return status


@app.get("/api/ascii/stream/{cam_id}")
async def get_ascii_frame(cam_id: int):
    """Get the current ASCII frame for a camera."""
    if streamer is None:
        return JSONResponse({"error": "Streamer not ready"}, 503)
    frame = streamer.get_latest_frame(cam_id)
    if frame is None:
        return JSONResponse({"error": f"No frame for cam {cam_id}"}, 404)
    return {
        "cam_id": frame.cam_id,
        "lines": frame.lines,
        "ascii": "\n".join(frame.lines),
        "width": frame.width,
        "height": frame.height,
        "timestamp": round(frame.timestamp, 3),
        "frame_number": frame.frame_number,
    }


@app.get("/api/ascii/config")
async def get_config():
    """Get current ASCII streaming configuration."""
    if streamer is None:
        return JSONResponse({"error": "Streamer not ready"}, 503)
    config = streamer.config.to_dict()
    config["available_charsets"] = list(CHARSETS.keys())
    config["cam_ids"] = streamer.cam_ids
    a = _get_analyst()
    if a:
        config["analyst"] = a.get_stats()
    return config


@app.post("/api/ascii/config")
async def update_config(req: ConfigRequest):
    """Update ASCII streaming configuration."""
    if streamer is None:
        return JSONResponse({"error": "Streamer not ready"}, 503)

    updates = {}
    if req.width is not None:
        updates["width"] = req.width
    if req.height is not None:
        updates["height"] = req.height
    if req.charset_name is not None:
        if req.charset_name not in CHARSETS:
            return JSONResponse(
                {"error": f"Unknown charset. Available: {list(CHARSETS.keys())}"}, 400
            )
        updates["charset_name"] = req.charset_name
    if req.fps is not None:
        updates["fps"] = req.fps
    if req.color is not None:
        updates["color"] = req.color

    if updates:
        streamer.update_config(**updates)

    return {"ok": True, "config": streamer.config.to_dict()}


@app.post("/api/ascii/analyze")
async def analyze_frame(req: AnalyzeRequest):
    """Capture ASCII frame and send to LLM for analysis."""
    a = _get_analyst()
    if a is None:
        return JSONResponse({"error": "LLM analyst unavailable (check GEMINI_API_KEY)"}, 503)

    if streamer is None:
        return JSONResponse({"error": "Streamer not ready"}, 503)

    # Get latest frame
    frame = streamer.get_latest_frame(req.cam_id)
    if frame is None:
        return JSONResponse({"error": f"No frame available for cam {req.cam_id}"}, 404)

    ascii_text = "\n".join(frame.lines)

    # Get or create session
    session = session_mgr.get_or_create(req.session_id, cam_id=req.cam_id)
    session.add_user_message(req.question, cam_id=req.cam_id, ascii_frame=ascii_text)

    # Analyze
    result = await a.analyze(
        ascii_text=ascii_text,
        question=req.question,
        cam_id=req.cam_id,
        width=frame.width,
        height=frame.height,
        charset=streamer.config.charset,
        history=session.get_history()[:-1],  # exclude the message we just added
    )

    session.add_assistant_message(result["answer"])

    return {
        "answer": result["answer"],
        "session_id": session.id,
        "cam_id": req.cam_id,
        "tokens_used": result.get("tokens_used", 0),
        "latency_ms": result.get("latency_ms", 0),
        "model": result.get("model"),
        "frame_width": frame.width,
        "frame_height": frame.height,
    }


@app.get("/api/ascii/sessions")
async def list_sessions():
    """List active analysis sessions."""
    return {"sessions": session_mgr.list_sessions()}


@app.get("/api/ascii/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session's full history."""
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse({"error": "Session not found"}, 404)
    return session.to_dict()


@app.delete("/api/ascii/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_mgr.delete(session_id):
        return {"ok": True}
    return JSONResponse({"error": "Session not found"}, 404)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/ascii/{cam_id}")
async def ws_ascii_stream(websocket: WebSocket, cam_id: int):
    """Live ASCII video stream via WebSocket."""
    await websocket.accept()

    if streamer is None:
        await websocket.close(1011, "Streamer not ready")
        return

    stream = streamer.get_stream(cam_id)
    if stream is None:
        await websocket.close(1008, f"Unknown camera {cam_id}")
        return

    queue = stream.subscribe()
    logger.info("WS client connected for cam %d", cam_id)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, queue.get_nowait),
                    timeout=0.5
                )
                await websocket.send_text(msg)
            except (asyncio.TimeoutError, Exception):
                # Check if there's a message waiting
                try:
                    msg = queue.get_nowait()
                    await websocket.send_text(msg)
                except Exception:
                    await asyncio.sleep(0.1)

            # Check for incoming messages (config updates)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                # Client can send config updates — ignored for now
            except (asyncio.TimeoutError, WebSocketDisconnect):
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WS cam %d error: %s", cam_id, e)
    finally:
        stream.unsubscribe(queue)
        logger.info("WS client disconnected for cam %d", cam_id)


# ---------------------------------------------------------------------------
# Static files & main
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Redirect to the ASCII video UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/static/ascii_video.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8084, log_level="info")
