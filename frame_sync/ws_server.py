"""
WebSocket server for receiving ASCII frames from the browser.

Accepts connections on ws://0.0.0.0:9100 and parses incoming messages
containing ASCII frame data from the dual-camera renderer.

Message format (per frame):
  char 0:      Camera ID — 'T' (top-down) or 'P' (profile)
  chars 1-8:   Zero-padded millisecond timestamp
  chars 9+:    128×128 = 16384 ASCII characters (row-major)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np

try:
    import websockets
    import websockets.server
except ImportError:
    websockets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GRID_SIZE = 128
EXPECTED_PAYLOAD_LEN = GRID_SIZE * GRID_SIZE  # 16384
HEADER_LEN = 9  # 1 char camera_id + 8 char timestamp
EXPECTED_MSG_LEN = HEADER_LEN + EXPECTED_PAYLOAD_LEN  # 16393

WS_HOST = "0.0.0.0"
WS_PORT = 9100


class FrameWSServer:
    """WebSocket server that receives ASCII frames from the browser renderer.

    Parameters
    ----------
    on_frame : callable
        Called with (camera_id: str, timestamp_ms: int, grid: np.ndarray)
        for each valid frame received. grid is uint8 [128, 128].
    host : str
        Bind address.
    port : int
        Bind port.
    """

    def __init__(
        self,
        on_frame: Callable[[str, int, np.ndarray], None],
        host: str = WS_HOST,
        port: int = WS_PORT,
    ):
        if websockets is None:
            raise RuntimeError("websockets library is required: pip install websockets")
        self.on_frame = on_frame
        self.host = host
        self.port = port
        self._server = None
        self._connection_count = 0
        self._frame_count = 0

    async def _handle_connection(self, ws) -> None:
        """Handle a single WebSocket client connection."""
        self._connection_count += 1
        remote = ws.remote_address
        logger.info("Client connected: %s (total: %d)", remote, self._connection_count)

        try:
            async for message in ws:
                self._process_message(message)
        except Exception as e:
            logger.debug("Connection closed: %s (%s)", remote, e)
        finally:
            self._connection_count -= 1
            logger.info("Client disconnected: %s (remaining: %d)", remote, self._connection_count)

    def _process_message(self, message: str | bytes) -> None:
        """Parse and dispatch a single frame message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        if len(message) < EXPECTED_MSG_LEN:
            logger.debug(
                "Message too short: got %d chars, expected %d", len(message), EXPECTED_MSG_LEN
            )
            return

        camera_id = message[0]
        if camera_id not in ("T", "P"):
            logger.debug("Invalid camera ID: %r", camera_id)
            return

        try:
            timestamp_ms = int(message[1:9])
        except ValueError:
            logger.debug("Invalid timestamp: %r", message[1:9])
            return

        # Extract payload and convert to uint8 grid of ASCII codepoints
        payload = message[9 : 9 + EXPECTED_PAYLOAD_LEN]
        ascii_codes = np.array([ord(c) for c in payload], dtype=np.uint8)
        grid = ascii_codes.reshape(GRID_SIZE, GRID_SIZE)

        self._frame_count += 1
        self.on_frame(camera_id, timestamp_ms, grid)

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.server.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info("Frame WebSocket server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Frame WebSocket server stopped")

    @property
    def frame_count(self) -> int:
        return self._frame_count


async def run_server(on_frame: Callable[[str, int, np.ndarray], None]) -> None:
    """Convenience: run the WebSocket server until cancelled."""
    server = FrameWSServer(on_frame=on_frame)
    await server.start()
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        await server.stop()
