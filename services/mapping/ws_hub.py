"""WebSocket connection manager for map server.

Broadcasts scene snapshots at configurable rate to connected viewers.
Supports per-client layer subscriptions and update rate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from src.map.scene import Scene

logger = logging.getLogger(__name__)


class ClientState:
    """Per-client subscription state."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.layers: Set[str] = {"arm", "env", "objects", "collision", "waypoints"}
        self.update_hz: float = 15.0
        self.needs_full_scene: bool = True  # Send full on first connect
        self.last_send: float = 0.0


class WSHub:
    """WebSocket hub that broadcasts scene updates."""

    def __init__(self, scene: Scene, default_hz: float = 15.0):
        self._scene = scene
        self._clients: Dict[int, ClientState] = {}
        self._default_hz = default_hz
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, ws: WebSocket) -> ClientState:
        """Register a new WebSocket client."""
        await ws.accept()
        client = ClientState(ws)
        client.update_hz = self._default_hz
        self._clients[id(ws)] = client
        logger.info("WS client connected (%d total)", len(self._clients))
        return client

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket client."""
        self._clients.pop(id(ws), None)
        logger.info("WS client disconnected (%d remaining)", len(self._clients))

    async def handle_message(self, ws: WebSocket, data: str) -> None:
        """Handle incoming client message."""
        client = self._clients.get(id(ws))
        if not client:
            return

        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "subscribe":
            layers = msg.get("layers", [])
            if layers:
                client.layers = set(layers)
                logger.debug("Client subscribed to layers: %s", client.layers)

        elif msg_type == "set_update_rate":
            hz = msg.get("hz", self._default_hz)
            client.update_hz = max(1.0, min(30.0, float(hz)))

        elif msg_type == "request_full_scene":
            client.needs_full_scene = True

    async def start_broadcast(self) -> None:
        """Start the broadcast loop."""
        if self._running:
            return
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop_broadcast(self) -> None:
        """Stop the broadcast loop."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

    async def _broadcast_loop(self) -> None:
        """Main broadcast loop — sends updates to each client at their rate."""
        min_interval = 1.0 / 30.0  # Cap at 30Hz tick rate

        while self._running:
            now = time.time()

            if not self._clients:
                await asyncio.sleep(0.1)
                continue

            # Collect clients that need an update
            to_send: list = []
            for cid, client in list(self._clients.items()):
                interval = 1.0 / client.update_hz
                if now - client.last_send >= interval:
                    to_send.append(client)

            if to_send:
                for client in to_send:
                    try:
                        full = client.needs_full_scene
                        snapshot = self._scene.snapshot(full=full, layers=client.layers)
                        msg = json.dumps({"type": "scene_update", "data": snapshot})
                        await client.ws.send_text(msg)
                        client.last_send = now
                        client.needs_full_scene = False
                    except Exception:
                        # Client likely disconnected
                        self._clients.pop(id(client.ws), None)

            await asyncio.sleep(min_interval)

    async def broadcast_immediate(self, message: Dict[str, Any]) -> None:
        """Send a one-off message to all clients."""
        text = json.dumps(message)
        dead = []
        for cid, client in self._clients.items():
            try:
                await client.ws.send_text(text)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self._clients.pop(cid, None)
