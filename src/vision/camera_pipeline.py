"""
Camera Pipeline — Dual camera capture with configurable ASCII conversion.

Manages one or two camera inputs, converts each frame to ASCII at configurable
resolution and frequency, and provides frames to downstream consumers (VLA model,
video recorder, scene modeler).

Camera layout:
  cam0 (front/side): provides object color, height (Z) from vertical position
  cam1 (overhead):   provides object X/Y position on workspace table

Configuration parameters:
  - ascii_width / ascii_height: character grid resolution
  - capture_fps: how often frames are grabbed from hardware
  - ascii_fps: how often ASCII conversion runs (can be lower than capture_fps)
  - charset: character ramp for brightness mapping
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from .ascii_converter import AsciiConverter, CHARSET_STANDARD, CHARSET_DETAILED

logger = logging.getLogger("th3cl4w.vision.camera_pipeline")


@dataclass
class AsciiFrame:
    """A single ASCII-converted frame with metadata."""

    camera_id: int
    ascii_text: str
    color_data: Optional[dict] = None
    grid_width: int = 120
    grid_height: int = 40
    timestamp: float = 0.0
    frame_number: int = 0
    raw_frame: Optional[np.ndarray] = None  # original BGR frame if retained

    @property
    def lines(self) -> list[str]:
        return self.ascii_text.split("\n")

    @property
    def grid(self) -> list[list[str]]:
        """Return the ASCII frame as a 2D character grid for coordinate-based access."""
        return [list(line.ljust(self.grid_width)) for line in self.lines]

    def char_at(self, col: int, row: int) -> str:
        """Get the ASCII character at a grid position."""
        grid = self.grid
        if 0 <= row < len(grid) and 0 <= col < len(grid[row]):
            return grid[row][col]
        return " "

    def to_dict(self) -> dict:
        result: dict = {
            "camera_id": self.camera_id,
            "ascii_text": self.ascii_text,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "timestamp": round(self.timestamp, 4),
            "frame_number": self.frame_number,
        }
        if self.color_data is not None:
            result["color_data"] = self.color_data
        return result


@dataclass
class StereoAsciiFrame:
    """A synchronized pair of ASCII frames from both cameras."""

    cam0: Optional[AsciiFrame] = None
    cam1: Optional[AsciiFrame] = None
    timestamp: float = 0.0
    frame_number: int = 0

    @property
    def has_both(self) -> bool:
        return self.cam0 is not None and self.cam1 is not None

    def to_dict(self) -> dict:
        return {
            "cam0": self.cam0.to_dict() if self.cam0 else None,
            "cam1": self.cam1.to_dict() if self.cam1 else None,
            "timestamp": round(self.timestamp, 4),
            "frame_number": self.frame_number,
            "has_both": self.has_both,
        }


@dataclass
class PipelineConfig:
    """Configuration for the camera pipeline."""

    # ASCII conversion parameters
    ascii_width: int = 120
    ascii_height: int = 40
    charset: str = CHARSET_STANDARD
    invert: bool = True
    color: bool = True

    # Capture and conversion rates
    capture_fps: int = 15
    ascii_fps: float = 5.0  # how often ASCII conversion runs

    # Frame retention
    retain_raw_frames: bool = True  # keep BGR frames for recording/annotation

    # Camera device indices
    cam0_device: int = 0
    cam1_device: Optional[int] = 4  # None for single-camera mode

    # Image capture resolution
    capture_width: int = 1920
    capture_height: int = 1080
    jpeg_quality: int = 92

    def validate(self):
        if self.ascii_width < 10 or self.ascii_width > 500:
            raise ValueError(f"ascii_width must be 10-500, got {self.ascii_width}")
        if self.ascii_height < 5 or self.ascii_height > 200:
            raise ValueError(f"ascii_height must be 5-200, got {self.ascii_height}")
        if self.ascii_fps <= 0 or self.ascii_fps > self.capture_fps:
            raise ValueError(
                f"ascii_fps must be >0 and <= capture_fps ({self.capture_fps})"
            )


# Callback type: called with each new StereoAsciiFrame
FrameCallback = Callable[[StereoAsciiFrame], None]


class CameraPipeline:
    """Manages dual camera capture and continuous ASCII conversion.

    Pipeline flow:
      Camera hardware → BGR frames → ASCII conversion → callbacks

    Supports configurable resolution, frame rate, and character set.
    Consumers register callbacks to receive ASCII frames as they're produced.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        if cv2 is None:
            raise RuntimeError("opencv-python (cv2) is required for CameraPipeline")

        self.config = config or PipelineConfig()
        self.config.validate()

        # ASCII converters (one per camera for thread safety)
        self._converter_cam0 = AsciiConverter(
            width=self.config.ascii_width,
            height=self.config.ascii_height,
            charset=self.config.charset,
            invert=self.config.invert,
            color=self.config.color,
        )
        self._converter_cam1 = AsciiConverter(
            width=self.config.ascii_width,
            height=self.config.ascii_height,
            charset=self.config.charset,
            invert=self.config.invert,
            color=self.config.color,
        )

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_number = 0
        self._callbacks: list[FrameCallback] = []

        # Latest frames
        self._latest_stereo: Optional[StereoAsciiFrame] = None
        self._latest_raw_cam0: Optional[np.ndarray] = None
        self._latest_raw_cam1: Optional[np.ndarray] = None

        # Camera sources (can be CameraThread instances or cv2.VideoCapture)
        self._cam0_source = None
        self._cam1_source = None

        # Stats
        self._ascii_frame_count = 0
        self._start_time = 0.0

    def attach_cameras(self, cam0, cam1=None):
        """Attach camera sources (CameraThread instances from camera_server).

        Args:
            cam0: Camera source with get_raw_frame() method.
            cam1: Optional second camera source.
        """
        self._cam0_source = cam0
        self._cam1_source = cam1

    def on_frame(self, callback: FrameCallback):
        """Register a callback to receive each new StereoAsciiFrame."""
        self._callbacks.append(callback)

    def start(self):
        """Start the ASCII conversion pipeline in a background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._conversion_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Camera pipeline started: %dx%d ASCII @ %.1f fps",
            self.config.ascii_width,
            self.config.ascii_height,
            self.config.ascii_fps,
        )

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            "Camera pipeline stopped after %d ASCII frames",
            self._ascii_frame_count,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[StereoAsciiFrame]:
        """Get the most recent stereo ASCII frame (thread-safe)."""
        with self._lock:
            return self._latest_stereo

    def get_latest_raw(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the most recent raw BGR frames from both cameras."""
        with self._lock:
            return self._latest_raw_cam0, self._latest_raw_cam1

    def update_config(
        self,
        ascii_width: Optional[int] = None,
        ascii_height: Optional[int] = None,
        ascii_fps: Optional[float] = None,
        charset: Optional[str] = None,
    ):
        """Update pipeline configuration on-the-fly.

        Rebuilds the ASCII converters with new parameters.
        """
        if ascii_width is not None:
            self.config.ascii_width = ascii_width
        if ascii_height is not None:
            self.config.ascii_height = ascii_height
        if ascii_fps is not None:
            self.config.ascii_fps = ascii_fps
        if charset is not None:
            self.config.charset = charset

        self.config.validate()

        # Rebuild converters
        self._converter_cam0 = AsciiConverter(
            width=self.config.ascii_width,
            height=self.config.ascii_height,
            charset=self.config.charset,
            invert=self.config.invert,
            color=self.config.color,
        )
        self._converter_cam1 = AsciiConverter(
            width=self.config.ascii_width,
            height=self.config.ascii_height,
            charset=self.config.charset,
            invert=self.config.invert,
            color=self.config.color,
        )
        logger.info(
            "Pipeline config updated: %dx%d @ %.1f fps, charset=%d chars",
            self.config.ascii_width,
            self.config.ascii_height,
            self.config.ascii_fps,
            len(self.config.charset),
        )

    def convert_single_frame(self, frame: np.ndarray, camera_id: int = 0) -> AsciiFrame:
        """Convert a single BGR frame to ASCII (synchronous, for one-off use)."""
        converter = self._converter_cam0 if camera_id == 0 else self._converter_cam1
        ascii_text = converter.frame_to_ascii(frame)
        color_data = None
        if self.config.color:
            color_data = converter.frame_to_color_data(frame)

        return AsciiFrame(
            camera_id=camera_id,
            ascii_text=ascii_text,
            color_data=color_data,
            grid_width=self.config.ascii_width,
            grid_height=self.config.ascii_height,
            timestamp=time.monotonic(),
            frame_number=self._frame_number,
            raw_frame=frame.copy() if self.config.retain_raw_frames else None,
        )

    def _conversion_loop(self):
        """Main conversion loop running in a background thread."""
        interval = 1.0 / self.config.ascii_fps

        while self._running:
            t0 = time.monotonic()

            try:
                stereo = self._capture_and_convert()
                if stereo is not None:
                    with self._lock:
                        self._latest_stereo = stereo
                    self._ascii_frame_count += 1

                    # Notify callbacks
                    for cb in self._callbacks:
                        try:
                            cb(stereo)
                        except Exception as e:
                            logger.warning("Frame callback error: %s", e)
            except Exception as e:
                logger.error("Pipeline conversion error: %s", e)

            # Maintain target frame rate
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _capture_and_convert(self) -> Optional[StereoAsciiFrame]:
        """Capture frames from cameras and convert to ASCII."""
        now = time.monotonic()
        self._frame_number += 1

        cam0_frame = self._get_raw_frame(self._cam0_source)
        cam1_frame = self._get_raw_frame(self._cam1_source)

        if cam0_frame is None and cam1_frame is None:
            return None

        # Store raw frames
        with self._lock:
            self._latest_raw_cam0 = cam0_frame
            self._latest_raw_cam1 = cam1_frame

        # Convert to ASCII
        ascii_cam0 = None
        ascii_cam1 = None

        if cam0_frame is not None:
            ascii_text = self._converter_cam0.frame_to_ascii(cam0_frame)
            color_data = None
            if self.config.color:
                color_data = self._converter_cam0.frame_to_color_data(cam0_frame)

            ascii_cam0 = AsciiFrame(
                camera_id=0,
                ascii_text=ascii_text,
                color_data=color_data,
                grid_width=self.config.ascii_width,
                grid_height=self.config.ascii_height,
                timestamp=now,
                frame_number=self._frame_number,
                raw_frame=cam0_frame.copy() if self.config.retain_raw_frames else None,
            )

        if cam1_frame is not None:
            ascii_text = self._converter_cam1.frame_to_ascii(cam1_frame)
            color_data = None
            if self.config.color:
                color_data = self._converter_cam1.frame_to_color_data(cam1_frame)

            ascii_cam1 = AsciiFrame(
                camera_id=1,
                ascii_text=ascii_text,
                color_data=color_data,
                grid_width=self.config.ascii_width,
                grid_height=self.config.ascii_height,
                timestamp=now,
                frame_number=self._frame_number,
                raw_frame=cam1_frame.copy() if self.config.retain_raw_frames else None,
            )

        return StereoAsciiFrame(
            cam0=ascii_cam0,
            cam1=ascii_cam1,
            timestamp=now,
            frame_number=self._frame_number,
        )

    def _get_raw_frame(self, source) -> Optional[np.ndarray]:
        """Get a raw BGR frame from a camera source."""
        if source is None:
            return None
        try:
            if hasattr(source, "get_raw_frame"):
                return source.get_raw_frame()
            elif hasattr(source, "read"):
                ret, frame = source.read()
                return frame if ret else None
        except Exception as e:
            logger.debug("Failed to get frame: %s", e)
        return None

    def get_stats(self) -> dict:
        """Get pipeline performance statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "ascii_frame_count": self._ascii_frame_count,
            "elapsed_s": round(elapsed, 1),
            "effective_fps": round(self._ascii_frame_count / max(elapsed, 0.001), 2),
            "config": {
                "ascii_width": self.config.ascii_width,
                "ascii_height": self.config.ascii_height,
                "ascii_fps": self.config.ascii_fps,
                "charset_length": len(self.config.charset),
                "color": self.config.color,
                "retain_raw": self.config.retain_raw_frames,
            },
            "callbacks_registered": len(self._callbacks),
        }
