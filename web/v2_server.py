#!/usr/bin/env python3.12
"""
th3cl4w V2 — Thin proxy server on :8081
Proxies /api/*, /ws/* to the main server on :8080
Proxies /cam/* to camera server on :8081 (original cam server)
Serves V2 static files
"""

import asyncio
import logging
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("th3cl4w.v2proxy")

BACKEND = "http://localhost:8080"
BACKEND_WS = "ws://localhost:8080"
CAM_SERVER = "http://localhost:8082"  # camera server (if separate)

app = FastAPI(title="th3cl4w-v2-proxy")


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("%s %s -> %d", request.method, request.url.path, response.status_code)
    return response


# Shared httpx client
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _client


# ---------------------------------------------------------------------------
# API proxy — forward all /api/* requests to backend
# ---------------------------------------------------------------------------


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_api(request: Request, path: str):
    client = await get_client()
    url = f"{BACKEND}/api/{path}"

    # Forward query params
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = dict(request.headers)
    # Remove host header to avoid confusion
    headers.pop("host", None)

    try:
        resp = await client.request(
            method=request.method,
            url=url,
            content=body,
            headers={k: v for k, v in headers.items() if k.lower() not in ("host", "connection")},
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except httpx.ConnectError:
        return Response(
            content=b'{"error":"Backend unavailable"}',
            status_code=502,
            media_type="application/json",
        )


# ---------------------------------------------------------------------------
# WebSocket proxy — /ws/state and /ws/telemetry
# ---------------------------------------------------------------------------


@app.websocket("/ws/{path:path}")
async def proxy_ws(ws: WebSocket, path: str):
    await ws.accept()
    import websockets

    backend_url = f"{BACKEND_WS}/ws/{path}"
    try:
        async with websockets.connect(backend_url) as backend_ws:

            async def forward_to_client():
                try:
                    async for msg in backend_ws:
                        await ws.send_text(msg)
                except Exception:
                    pass

            async def forward_to_backend():
                try:
                    while True:
                        data = await ws.receive_text()
                        await backend_ws.send(data)
                except Exception:
                    pass

            await asyncio.gather(forward_to_client(), forward_to_backend())
    except Exception as e:
        logger.warning("WS proxy error: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Camera proxy — /cam/* streams from camera server
# ---------------------------------------------------------------------------


@app.get("/cam/{cam_id}")
async def proxy_cam_stream(request: Request, cam_id: int):
    """Proxy MJPEG stream from camera server."""
    client = await get_client()
    # Try camera server first, fall back to backend
    for base in [CAM_SERVER, BACKEND]:
        try:
            # Use streaming request for MJPEG
            req = client.build_request("GET", f"{base}/cam/{cam_id}")
            resp = await client.send(req, stream=True)

            async def stream_body():
                try:
                    async for chunk in resp.aiter_bytes(4096):
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(
                stream_body(),
                status_code=resp.status_code,
                headers={k: v for k, v in resp.headers.items() if k.lower() in ("content-type",)},
            )
        except Exception:
            continue

    return Response(content=b"Camera unavailable", status_code=502)


@app.get("/snap/{cam_id}")
async def proxy_cam_snap(cam_id: int):
    client = await get_client()
    for base in [CAM_SERVER, BACKEND]:
        try:
            resp = await client.get(f"{base}/snap/{cam_id}")
            return Response(
                content=resp.content, status_code=resp.status_code, headers=dict(resp.headers)
            )
        except Exception:
            continue
    return Response(content=b"Camera unavailable", status_code=502)


# ---------------------------------------------------------------------------
# Telemetry page proxy
# ---------------------------------------------------------------------------


@app.get("/telemetry")
async def proxy_telemetry():
    client = await get_client()
    try:
        resp = await client.get(f"{BACKEND}/telemetry")
        return Response(content=resp.content, status_code=resp.status_code, media_type="text/html")
    except Exception:
        return Response(content=b"Backend unavailable", status_code=502)


# ---------------------------------------------------------------------------
# V2 UI — serve v2.html as the root
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_v2():
    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "static" / "v2.html")


# Static files (fallback for other assets)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    logger.info("Starting th3cl4w V2 proxy on :%d → backend :8080", a.port)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
