"""
Video Recorder — Local recording, screenshots, and training data annotation.

Records video from the camera pipeline to local files, captures screenshots
at configurable intervals, and annotates frames with scene analysis data
for use as training data.

Output structure:
  recordings/
    session_YYYYMMDD_HHMMSS/
      cam0.avi                   # raw video from camera 0
      cam1.avi                   # raw video from camera 1
      screenshots/
        frame_000001_cam0.jpg    # annotated screenshot
        frame_000001_cam1.jpg
      annotations/
        frame_000001.json        # scene description + ASCII + metadata
      ascii_log/
        frame_000001_cam0.txt    # ASCII art frame
        frame_000001_cam1.txt
      session_metadata.json      # session config, timestamps, stats
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from .camera_pipeline import StereoAsciiFrame

logger = logging.getLogger("th3cl4w.vision.video_recorder")


@dataclass
class RecorderConfig:
    """Configuration for the video recorder."""

    output_dir: str = "recordings"
    record_video: bool = True
    capture_screenshots: bool = True
    capture_ascii: bool = True
    capture_annotations: bool = True

    # Screenshot frequency: every N frames
    screenshot_interval: int = 30  # ~1 per second at 30fps, ~6s at 5fps ASCII

    # Video codec and quality
    video_codec: str = "XVID"
    video_fps: float = 5.0
    jpeg_quality: int = 95

    # Annotation settings
    annotate_bboxes: bool = True
    annotate_grid: bool = True  # overlay ASCII grid coordinates
    annotate_measurements: bool = True

    # Storage limits
    max_session_frames: int = 100000  # stop recording after this many frames
    max_screenshots: int = 5000


@dataclass
class AnnotationRecord:
    """Training data annotation for a single frame pair."""

    frame_number: int
    timestamp: float
    cam0_ascii: Optional[str] = None
    cam1_ascii: Optional[str] = None
    ascii_width: int = 0
    ascii_height: int = 0
    scene_objects: list[dict] = field(default_factory=list)
    arm_state: Optional[dict] = None
    action_label: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4),
            "cam0_ascii": self.cam0_ascii,
            "cam1_ascii": self.cam1_ascii,
            "ascii_resolution": {
                "width": self.ascii_width,
                "height": self.ascii_height,
            },
            "scene_objects": self.scene_objects,
            "arm_state": self.arm_state,
            "action_label": self.action_label,
            "notes": self.notes,
        }


class VideoRecorder:
    """Records video, captures screenshots, and generates training annotations.

    Consumes StereoAsciiFrame from the CameraPipeline and writes:
    - Raw video files (AVI) from both cameras
    - Annotated screenshot JPEGs at configurable intervals
    - JSON annotation files with scene descriptions and ASCII data
    - ASCII text logs of each frame
    """

    def __init__(self, config: Optional[RecorderConfig] = None):
        self.config = config or RecorderConfig()
        self._running = False
        self._session_dir: Optional[Path] = None
        self._lock = threading.Lock()

        # Video writers
        self._writer_cam0: Optional[cv2.VideoWriter] = None
        self._writer_cam1: Optional[cv2.VideoWriter] = None

        # Counters
        self._frame_count = 0
        self._screenshot_count = 0
        self._annotation_count = 0

        # Scene analyzer reference for annotation
        self._scene_analyzer = None

        # Session metadata
        self._session_start = 0.0
        self._session_id = ""

    def set_scene_analyzer(self, analyzer):
        """Attach a SceneAnalyzer for annotating screenshots."""
        self._scene_analyzer = analyzer

    def start_session(self, session_id: Optional[str] = None):
        """Start a new recording session, creating output directories."""
        if cv2 is None:
            raise RuntimeError("opencv-python required for VideoRecorder")

        self._session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(self.config.output_dir) / f"session_{self._session_id}"

        # Create directory structure
        self._session_dir.mkdir(parents=True, exist_ok=True)
        if self.config.capture_screenshots:
            (self._session_dir / "screenshots").mkdir(exist_ok=True)
        if self.config.capture_annotations:
            (self._session_dir / "annotations").mkdir(exist_ok=True)
        if self.config.capture_ascii:
            (self._session_dir / "ascii_log").mkdir(exist_ok=True)

        self._frame_count = 0
        self._screenshot_count = 0
        self._annotation_count = 0
        self._session_start = time.monotonic()
        self._running = True

        logger.info("Recording session started: %s", self._session_dir)

    def stop_session(self):
        """Stop recording and close all file handles."""
        self._running = False

        with self._lock:
            if self._writer_cam0 is not None:
                self._writer_cam0.release()
                self._writer_cam0 = None
            if self._writer_cam1 is not None:
                self._writer_cam1.release()
                self._writer_cam1 = None

        # Write session metadata
        if self._session_dir is not None:
            self._write_session_metadata()

        logger.info(
            "Recording session stopped: %d frames, %d screenshots, %d annotations",
            self._frame_count,
            self._screenshot_count,
            self._annotation_count,
        )

    @property
    def is_recording(self) -> bool:
        return self._running

    def process_frame(
        self,
        stereo: StereoAsciiFrame,
        scene_objects: Optional[list[dict]] = None,
        arm_state: Optional[dict] = None,
        action_label: str = "",
    ):
        """Process a stereo ASCII frame: record video, take screenshots, annotate.

        This is the main entry point, called for each new StereoAsciiFrame.

        Args:
            stereo: The stereo ASCII frame from the camera pipeline.
            scene_objects: Optional list of detected object dicts for annotation.
            arm_state: Optional arm joint state dict for annotation.
            action_label: Optional action label for training data.
        """
        if not self._running or self._session_dir is None:
            return
        if self._frame_count >= self.config.max_session_frames:
            return

        self._frame_count += 1

        with self._lock:
            # Record raw video
            if self.config.record_video:
                self._record_video_frame(stereo)

            # Capture screenshot at interval
            if (
                self.config.capture_screenshots
                and self._frame_count % self.config.screenshot_interval == 0
                and self._screenshot_count < self.config.max_screenshots
            ):
                self._capture_screenshot(stereo, scene_objects)

            # Save ASCII log
            if self.config.capture_ascii:
                self._save_ascii_frame(stereo)

            # Save annotation
            if self.config.capture_annotations:
                self._save_annotation(
                    stereo, scene_objects, arm_state, action_label
                )

    def _record_video_frame(self, stereo: StereoAsciiFrame):
        """Write raw BGR frames to video files."""
        if stereo.cam0 is not None and stereo.cam0.raw_frame is not None:
            if self._writer_cam0 is None:
                h, w = stereo.cam0.raw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
                path = str(self._session_dir / "cam0.avi")
                self._writer_cam0 = cv2.VideoWriter(
                    path, fourcc, self.config.video_fps, (w, h)
                )
            self._writer_cam0.write(stereo.cam0.raw_frame)

        if stereo.cam1 is not None and stereo.cam1.raw_frame is not None:
            if self._writer_cam1 is None:
                h, w = stereo.cam1.raw_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
                path = str(self._session_dir / "cam1.avi")
                self._writer_cam1 = cv2.VideoWriter(
                    path, fourcc, self.config.video_fps, (w, h)
                )
            self._writer_cam1.write(stereo.cam1.raw_frame)

    def _capture_screenshot(
        self,
        stereo: StereoAsciiFrame,
        scene_objects: Optional[list[dict]] = None,
    ):
        """Capture annotated screenshots from both cameras."""
        frame_id = f"frame_{self._frame_count:06d}"

        for cam_id, ascii_frame in [(0, stereo.cam0), (1, stereo.cam1)]:
            if ascii_frame is None or ascii_frame.raw_frame is None:
                continue

            annotated = self._annotate_frame(
                ascii_frame.raw_frame.copy(), ascii_frame, scene_objects
            )

            path = self._session_dir / "screenshots" / f"{frame_id}_cam{cam_id}.jpg"
            cv2.imwrite(
                str(path),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
            )

        self._screenshot_count += 1

    def _annotate_frame(
        self,
        frame: np.ndarray,
        ascii_frame,
        scene_objects: Optional[list[dict]] = None,
    ) -> np.ndarray:
        """Draw annotations on a frame for training data."""
        h, w = frame.shape[:2]

        if self.config.annotate_grid:
            # Draw ASCII grid overlay
            cell_w = w / ascii_frame.grid_width
            cell_h = h / ascii_frame.grid_height

            # Draw grid lines (light, semi-transparent)
            for col in range(0, ascii_frame.grid_width + 1, 10):
                x = int(col * cell_w)
                cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
                if col % 20 == 0:
                    cv2.putText(
                        frame,
                        str(col),
                        (x + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (100, 100, 100),
                        1,
                    )

            for row in range(0, ascii_frame.grid_height + 1, 10):
                y = int(row * cell_h)
                cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)
                if row % 20 == 0:
                    cv2.putText(
                        frame,
                        str(row),
                        (2, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (100, 100, 100),
                        1,
                    )

        if self.config.annotate_bboxes and scene_objects:
            for obj in scene_objects:
                bbox = obj.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, bw, bh = bbox
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    label = obj.get("label", "")
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        if self.config.annotate_measurements and scene_objects:
            for obj in scene_objects:
                pos = obj.get("centroid_3d")
                if pos and len(pos) == 3:
                    centroid_2d = obj.get("centroid_2d")
                    if centroid_2d:
                        cx, cy = centroid_2d
                        text = f"({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})mm"
                        cv2.putText(
                            frame,
                            text,
                            (cx, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 200, 0),
                            1,
                        )

        # Timestamp overlay
        ts_text = f"F{ascii_frame.frame_number} cam{ascii_frame.camera_id}"
        cv2.putText(
            frame, ts_text, (w - 200, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        return frame

    def _save_ascii_frame(self, stereo: StereoAsciiFrame):
        """Save ASCII text representation of each frame."""
        frame_id = f"frame_{self._frame_count:06d}"

        if stereo.cam0 is not None:
            path = self._session_dir / "ascii_log" / f"{frame_id}_cam0.txt"
            path.write_text(stereo.cam0.ascii_text)

        if stereo.cam1 is not None:
            path = self._session_dir / "ascii_log" / f"{frame_id}_cam1.txt"
            path.write_text(stereo.cam1.ascii_text)

    def _save_annotation(
        self,
        stereo: StereoAsciiFrame,
        scene_objects: Optional[list[dict]],
        arm_state: Optional[dict],
        action_label: str,
    ):
        """Save a JSON annotation for training data."""
        record = AnnotationRecord(
            frame_number=self._frame_count,
            timestamp=stereo.timestamp,
            cam0_ascii=stereo.cam0.ascii_text if stereo.cam0 else None,
            cam1_ascii=stereo.cam1.ascii_text if stereo.cam1 else None,
            ascii_width=stereo.cam0.grid_width if stereo.cam0 else 0,
            ascii_height=stereo.cam0.grid_height if stereo.cam0 else 0,
            scene_objects=scene_objects or [],
            arm_state=arm_state,
            action_label=action_label,
        )

        frame_id = f"frame_{self._frame_count:06d}"
        path = self._session_dir / "annotations" / f"{frame_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))
        self._annotation_count += 1

    def _write_session_metadata(self):
        """Write session summary metadata."""
        elapsed = time.monotonic() - self._session_start
        metadata = {
            "session_id": self._session_id,
            "total_frames": self._frame_count,
            "total_screenshots": self._screenshot_count,
            "total_annotations": self._annotation_count,
            "duration_s": round(elapsed, 1),
            "config": {
                "record_video": self.config.record_video,
                "screenshot_interval": self.config.screenshot_interval,
                "video_codec": self.config.video_codec,
                "video_fps": self.config.video_fps,
                "annotate_bboxes": self.config.annotate_bboxes,
                "annotate_grid": self.config.annotate_grid,
                "annotate_measurements": self.config.annotate_measurements,
            },
        }

        path = self._session_dir / "session_metadata.json"
        path.write_text(json.dumps(metadata, indent=2))
        logger.info("Session metadata written to %s", path)

    def get_stats(self) -> dict:
        """Get recording statistics."""
        elapsed = time.monotonic() - self._session_start if self._session_start else 0
        return {
            "recording": self._running,
            "session_id": self._session_id,
            "session_dir": str(self._session_dir) if self._session_dir else None,
            "frames_recorded": self._frame_count,
            "screenshots_captured": self._screenshot_count,
            "annotations_saved": self._annotation_count,
            "elapsed_s": round(elapsed, 1),
        }
