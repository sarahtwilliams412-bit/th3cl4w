"""
Gateway Server — Reverse proxy and unified API entry point.

Routes all client requests to the appropriate backend service. Provides
a single entry point for the entire th3cl4w system, with aggregate health
checking and request proxying via httpx.

Route table:
    /api/arm/*       -> control_plane (8090)
    /api/cameras/*   -> camera (8081)
    /api/world/*     -> world_model (8082)
    /api/local/*     -> local_model (8083)
    /api/objects/*   -> object_id (8084)
    /api/plan/*      -> kinematics (8085)
    /api/map/*       -> mapping (8086)
    /api/position/*  -> positioning (8087)
    /api/tasks/*     -> tasker (8088)
    /api/sim/*       -> simulation (8089)
    /api/telemetry/* -> telemetry (8091)

Port: 8080 (configurable via GATEWAY_PORT env var)

Usage:
    python -m services.gateway.server
    python -m services.gateway.server --debug
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.config.service_registry import ServiceConfig
from shared.utils.logging_config import setup_logging

logger = logging.getLogger("th3cl4w.gateway")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="th3cl4w Gateway — Reverse Proxy & Unified Entry Point")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=None, help="Bind port (default: from service registry)")
parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")

if "pytest" not in sys.modules:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

setup_logging(server_name="gateway", debug=args.debug, log_dir=args.log_dir)

# ---------------------------------------------------------------------------
# Route table — maps URL prefixes to backend services
# ---------------------------------------------------------------------------

ROUTE_TABLE: dict[str, dict[str, Any]] = {
    "/api/arm": {
        "service": "control_plane",
        "port": ServiceConfig.CONTROL_PLANE_PORT,
        "strip_prefix": "/api/arm",
        "target_prefix": "/api",
    },
    "/api/cameras": {
        "service": "camera",
        "port": ServiceConfig.CAMERA_PORT,
        "strip_prefix": "/api/cameras",
        "target_prefix": "/api",
    },
    "/api/world": {
        "service": "world_model",
        "port": ServiceConfig.WORLD_MODEL_PORT,
        "strip_prefix": "/api/world",
        "target_prefix": "/api",
    },
    "/api/local": {
        "service": "local_model",
        "port": ServiceConfig.LOCAL_MODEL_PORT,
        "strip_prefix": "/api/local",
        "target_prefix": "/api/local",
    },
    "/api/objects": {
        "service": "object_id",
        "port": ServiceConfig.OBJECT_ID_PORT,
        "strip_prefix": "/api/objects",
        "target_prefix": "/api",
    },
    "/api/plan": {
        "service": "kinematics",
        "port": ServiceConfig.KINEMATICS_PORT,
        "strip_prefix": "/api/plan",
        "target_prefix": "/api/plan",
    },
    "/api/map": {
        "service": "mapping",
        "port": ServiceConfig.MAPPING_PORT,
        "strip_prefix": "/api/map",
        "target_prefix": "/api",
    },
    "/api/position": {
        "service": "positioning",
        "port": ServiceConfig.POSITIONING_PORT,
        "strip_prefix": "/api/position",
        "target_prefix": "/api/position",
    },
    "/api/tasks": {
        "service": "tasker",
        "port": ServiceConfig.TASKER_PORT,
        "strip_prefix": "/api/tasks",
        "target_prefix": "/api",
    },
    "/api/sim": {
        "service": "simulation",
        "port": ServiceConfig.SIMULATION_PORT,
        "strip_prefix": "/api/sim",
        "target_prefix": "/api/sim",
    },
    "/api/telemetry": {
        "service": "telemetry",
        "port": ServiceConfig.TELEMETRY_PORT,
        "strip_prefix": "/api/telemetry",
        "target_prefix": "/api",
    },
}

# All backend services for health aggregation
ALL_SERVICES: dict[str, int] = {
    "control_plane": ServiceConfig.CONTROL_PLANE_PORT,
    "camera": ServiceConfig.CAMERA_PORT,
    "world_model": ServiceConfig.WORLD_MODEL_PORT,
    "local_model": ServiceConfig.LOCAL_MODEL_PORT,
    "object_id": ServiceConfig.OBJECT_ID_PORT,
    "kinematics": ServiceConfig.KINEMATICS_PORT,
    "mapping": ServiceConfig.MAPPING_PORT,
    "positioning": ServiceConfig.POSITIONING_PORT,
    "tasker": ServiceConfig.TASKER_PORT,
    "simulation": ServiceConfig.SIMULATION_PORT,
    "telemetry": ServiceConfig.TELEMETRY_PORT,
}

# ---------------------------------------------------------------------------
# Global httpx client
# ---------------------------------------------------------------------------

_http_client: Optional[httpx.AsyncClient] = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    logger.info("Gateway ready — routing to %d backend services", len(ROUTE_TABLE))
    yield

    # Shutdown
    if _http_client:
        await _http_client.aclose()
    logger.info("Gateway shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="th3cl4w Gateway",
    description="Unified reverse proxy entry point for all th3cl4w services",
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

# Serve static files (gateway dashboard UI)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Health check — aggregate from all services
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Aggregate health check — pings all backend services."""
    results: dict[str, Any] = {}
    healthy_count = 0
    total_count = len(ALL_SERVICES)

    async def _check_service(name: str, port: int):
        nonlocal healthy_count
        host = os.getenv(f"{name.upper()}_HOST", "localhost")
        try:
            resp = await _http_client.get(
                f"http://{host}:{port}/health",
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                results[name] = {"status": "healthy", "port": port, "detail": data}
                healthy_count += 1
            else:
                results[name] = {"status": "unhealthy", "port": port, "http_status": resp.status_code}
        except Exception as e:
            results[name] = {"status": "unreachable", "port": port, "error": str(e)}

    # Check all services concurrently
    tasks = [_check_service(name, port) for name, port in ALL_SERVICES.items()]
    await asyncio.gather(*tasks)

    overall = "healthy" if healthy_count == total_count else (
        "degraded" if healthy_count > 0 else "unhealthy"
    )

    return {
        "status": overall,
        "service": "gateway",
        "healthy_services": healthy_count,
        "total_services": total_count,
        "services": results,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Route info
# ---------------------------------------------------------------------------


@app.get("/api/routes")
async def get_routes():
    """Get the gateway route table."""
    routes = {}
    for prefix, config in ROUTE_TABLE.items():
        routes[prefix] = {
            "service": config["service"],
            "port": config["port"],
            "target_prefix": config["target_prefix"],
        }
    return {"routes": routes, "total": len(routes)}


# ---------------------------------------------------------------------------
# Proxy helper
# ---------------------------------------------------------------------------


def _resolve_backend(path: str) -> Optional[tuple[str, str]]:
    """Resolve a request path to a backend URL.

    Returns (base_url, rewritten_path) or None if no route matches.
    """
    # Sort by longest prefix first so more specific routes match first
    for prefix in sorted(ROUTE_TABLE.keys(), key=len, reverse=True):
        if path.startswith(prefix):
            config = ROUTE_TABLE[prefix]
            service_name = config["service"]
            host = os.getenv(f"{service_name.upper()}_HOST", "localhost")
            port = config["port"]
            base_url = f"http://{host}:{port}"

            # Rewrite the path
            remainder = path[len(config["strip_prefix"]):]
            target_path = config["target_prefix"] + remainder
            return base_url, target_path
    return None


async def _proxy_request(request: Request) -> JSONResponse | StreamingResponse:
    """Proxy an HTTP request to the appropriate backend service."""
    path = request.url.path
    resolved = _resolve_backend(path)

    if resolved is None:
        return JSONResponse(
            {"error": "No route matched", "path": path},
            status_code=404,
        )

    base_url, target_path = resolved
    target_url = f"{base_url}{target_path}"

    # Forward query string
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Read request body
    body = await request.body()

    # Forward headers (skip hop-by-hop)
    headers = {}
    skip_headers = {"host", "connection", "transfer-encoding"}
    for key, value in request.headers.items():
        if key.lower() not in skip_headers:
            headers[key] = value

    try:
        backend_resp = await _http_client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers,
        )

        # Build response headers
        resp_headers = {}
        skip_resp = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        for key, value in backend_resp.headers.items():
            if key.lower() not in skip_resp:
                resp_headers[key] = value

        return StreamingResponse(
            content=backend_resp.iter_bytes(),
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=backend_resp.headers.get("content-type"),
        )
    except httpx.ConnectError:
        return JSONResponse(
            {"error": "Backend service unreachable", "target": target_url},
            status_code=502,
        )
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "Backend service timeout", "target": target_url},
            status_code=504,
        )
    except Exception as e:
        logger.error("Proxy error for %s: %s", target_url, e)
        return JSONResponse(
            {"error": "Gateway error", "detail": str(e)},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Catch-all proxy routes for each prefix
# ---------------------------------------------------------------------------


@app.api_route("/api/arm/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_arm(request: Request, path: str):
    """Proxy to Control Plane service."""
    return await _proxy_request(request)


@app.api_route("/api/cameras/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_cameras(request: Request, path: str):
    """Proxy to Camera service."""
    return await _proxy_request(request)


@app.api_route("/api/world/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_world(request: Request, path: str):
    """Proxy to World Model service."""
    return await _proxy_request(request)


@app.api_route("/api/local/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_local(request: Request, path: str):
    """Proxy to Local Model service."""
    return await _proxy_request(request)


@app.api_route("/api/objects/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_objects(request: Request, path: str):
    """Proxy to Object ID service."""
    return await _proxy_request(request)


@app.api_route("/api/plan/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_plan(request: Request, path: str):
    """Proxy to Kinematics App service."""
    return await _proxy_request(request)


@app.api_route("/api/map/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_map(request: Request, path: str):
    """Proxy to Mapping service."""
    return await _proxy_request(request)


@app.api_route("/api/position/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_position(request: Request, path: str):
    """Proxy to Positioning service."""
    return await _proxy_request(request)


@app.api_route("/api/tasks/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_tasks(request: Request, path: str):
    """Proxy to Tasker service."""
    return await _proxy_request(request)


@app.api_route("/api/sim/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_sim(request: Request, path: str):
    """Proxy to Simulation service."""
    return await _proxy_request(request)


@app.api_route("/api/telemetry/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_telemetry(request: Request, path: str):
    """Proxy to Telemetry service."""
    return await _proxy_request(request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = args.port or ServiceConfig.GATEWAY_PORT
    logger.info("Starting Gateway on port %d", port)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
    )
