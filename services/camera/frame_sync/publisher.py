"""
ZeroMQ publisher for synchronized frame pairs.

Subscribes to matched FramePairs from pair_matcher and publishes them
on tcp://localhost:5555 in a compact binary format.

Message format (32776 bytes):
  - 8 bytes: uint64 timestamp_ms (little-endian)
  - 16384 bytes: top_down grid (uint8 [128, 128], row-major)
  - 16384 bytes: profile grid (uint8 [128, 128], row-major)

Run as: python -m frame_sync.publisher
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time

import numpy as np

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore[assignment]

from frame_sync.pair_matcher import FramePair, PairMatcher
from frame_sync.ws_server import FrameWSServer

logger = logging.getLogger(__name__)

ZMQ_PUB_ADDR = "tcp://*:5555"
WS_PORT = 9100
STATS_INTERVAL_S = 5.0


class FramePublisher:
    """Publishes synchronized FramePairs over ZeroMQ.

    Parameters
    ----------
    zmq_addr : str
        ZeroMQ PUB socket bind address.
    ws_port : int
        WebSocket server port for receiving browser frames.
    """

    def __init__(self, zmq_addr: str = ZMQ_PUB_ADDR, ws_port: int = WS_PORT):
        if zmq is None:
            raise RuntimeError("pyzmq is required: pip install pyzmq")

        self.zmq_addr = zmq_addr
        self.ws_port = ws_port

        # ZMQ setup
        self._zmq_ctx = zmq.Context()
        self._pub_socket = self._zmq_ctx.socket(zmq.PUB)
        self._pub_socket.bind(self.zmq_addr)

        # Frame pair matcher
        self._matcher = PairMatcher(on_pair=self._on_pair)

        # Stats
        self._publish_count = 0
        self._last_stats_time = time.monotonic()
        self._stats_frame_count = 0

    def _on_frame(self, camera_id: str, timestamp_ms: int, grid: np.ndarray) -> None:
        """Callback from WebSocket server — feed into pair matcher."""
        self._matcher.feed(camera_id, timestamp_ms, grid)

    def _on_pair(self, pair: FramePair) -> None:
        """Callback from pair matcher — serialize and publish."""
        # Pack: 8-byte timestamp + 16384-byte top_down + 16384-byte profile
        message = (
            struct.pack("<Q", pair.timestamp_ms)
            + pair.top_down.tobytes()
            + pair.profile.tobytes()
        )
        self._pub_socket.send(message, zmq.NOBLOCK)
        self._publish_count += 1
        self._stats_frame_count += 1

        # Periodic stats logging
        now = time.monotonic()
        elapsed = now - self._last_stats_time
        if elapsed >= STATS_INTERVAL_S:
            fps = self._stats_frame_count / elapsed
            stats = self._matcher.stats
            logger.info(
                "Frame sync: %.1f fps, %d pairs total, "
                "top_q=%d prof_q=%d dropped=%d",
                fps,
                self._publish_count,
                stats["top_queue_size"],
                stats["prof_queue_size"],
                stats["frames_dropped"],
            )
            self._last_stats_time = now
            self._stats_frame_count = 0

    async def run(self) -> None:
        """Start WebSocket server and run forever."""
        ws_server = FrameWSServer(
            on_frame=self._on_frame,
            port=self.ws_port,
        )
        await ws_server.start()
        logger.info("Frame publisher started — ZMQ PUB on %s", self.zmq_addr)

        try:
            # Log stats periodically while running
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await ws_server.stop()
            self._pub_socket.close()
            self._zmq_ctx.term()
            logger.info("Frame publisher stopped")

    def close(self) -> None:
        """Clean up ZMQ resources."""
        self._pub_socket.close()
        self._zmq_ctx.term()


def main() -> None:
    """Entry point for python -m frame_sync.publisher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    publisher = FramePublisher()
    try:
        asyncio.run(publisher.run())
    except KeyboardInterrupt:
        logger.info("Shutting down frame publisher")


if __name__ == "__main__":
    main()
