"""
ASCII Video Streamer — manages live ASCII feeds from all cameras.

Background threads fetch frames from the camera server and convert to ASCII.
WebSocket clients subscribe to per-camera feeds.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .ascii_converter import AsciiConverter, CHARSET_STANDARD
from .converter import fetch_jpeg, CHARSETS

logger = logging.getLogger("th3cl4w.ascii.streamer")


@dataclass
class StreamConfig:
    """Global streaming configuration."""

    width: int = 120
    height: int = 60
    charset_name: str = "standard"
    fps: float = 5.0
    color: bool = False

    @property
    def charset(self) -> str:
        return CHARSETS.get(self.charset_name, CHARSET_STANDARD)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "charset_name": self.charset_name,
            "charset": self.charset,
            "fps": self.fps,
            "color": self.color,
        }


@dataclass
class AsciiStreamFrame:
    """A single ASCII frame ready for streaming."""

    cam_id: int
    lines: list[str]
    width: int
    height: int
    timestamp: float
    frame_number: int
    color_data: Optional[dict] = None

    def to_json(self) -> str:
        d = {
            "cam_id": self.cam_id,
            "lines": self.lines,
            "width": self.width,
            "height": self.height,
            "timestamp": round(self.timestamp, 3),
            "frame_number": self.frame_number,
        }
        if self.color_data:
            d["colors"] = self.color_data.get("colors")
        return json.dumps(d)


class CameraStream:
    """Background thread capturing and converting frames for one camera."""

    def __init__(self, cam_id: int, config: StreamConfig):
        self.cam_id = cam_id
        self.config = config
        self._converter: Optional[AsciiConverter] = None
        self._rebuild_converter()
        self._latest: Optional[AsciiStreamFrame] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_number = 0
        self._subscribers: set = set()  # asyncio.Queue set

    def _rebuild_converter(self):
        self._converter = AsciiConverter(
            width=self.config.width,
            height=self.config.height,
            charset=self.config.charset,
            invert=True,
            color=self.config.color,
        )

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Camera stream %d started (%dx%d @ %.1f fps)",
            self.cam_id,
            self.config.width,
            self.config.height,
            self.config.fps,
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _capture_loop(self):
        while self._running:
            t0 = time.monotonic()
            try:
                jpeg = fetch_jpeg(self.cam_id, timeout=2.0)
                if jpeg:
                    ascii_text = self._converter.decode_jpeg_to_ascii(jpeg)
                    color_data = None
                    if self.config.color:
                        color_data = self._converter.decode_jpeg_to_color_data(jpeg)

                    self._frame_number += 1
                    frame = AsciiStreamFrame(
                        cam_id=self.cam_id,
                        lines=ascii_text.split("\n"),
                        width=self.config.width,
                        height=self.config.height,
                        timestamp=time.time(),
                        frame_number=self._frame_number,
                        color_data=color_data,
                    )

                    with self._lock:
                        self._latest = frame

                    # Notify subscribers (non-blocking)
                    msg = frame.to_json()
                    dead = set()
                    for q in self._subscribers:
                        try:
                            q.put_nowait(msg)
                        except asyncio.QueueFull:
                            pass  # Drop frame for slow consumers
                        except Exception:
                            dead.add(q)
                    if dead:
                        self._subscribers -= dead

            except Exception as e:
                logger.error("Stream %d error: %s", self.cam_id, e)

            elapsed = time.monotonic() - t0
            sleep_time = (1.0 / self.config.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest(self) -> Optional[AsciiStreamFrame]:
        with self._lock:
            return self._latest

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=3)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    def update_config(self, config: StreamConfig):
        self.config = config
        self._rebuild_converter()


class AsciiStreamer:
    """Manages ASCII streams for all cameras."""

    def __init__(self, cam_ids: list[int] = None, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self.cam_ids = cam_ids or [0, 1, 2]
        self.streams: dict[int, CameraStream] = {}

    def start(self):
        for cam_id in self.cam_ids:
            stream = CameraStream(cam_id, self.config)
            stream.start()
            self.streams[cam_id] = stream
        logger.info("ASCII streamer started for cameras: %s", self.cam_ids)

    def stop(self):
        for stream in self.streams.values():
            stream.stop()

    def get_stream(self, cam_id: int) -> Optional[CameraStream]:
        return self.streams.get(cam_id)

    def get_latest_frame(self, cam_id: int) -> Optional[AsciiStreamFrame]:
        stream = self.streams.get(cam_id)
        return stream.get_latest() if stream else None

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        for stream in self.streams.values():
            stream.update_config(self.config)
        logger.info("Streamer config updated: %s", self.config.to_dict())
