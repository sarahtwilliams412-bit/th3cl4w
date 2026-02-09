"""Tests for the video recorder module."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

import cv2

from src.vision.video_recorder import (
    VideoRecorder,
    RecorderConfig,
    AnnotationRecord,
)
from src.vision.camera_pipeline import AsciiFrame, StereoAsciiFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_OUTPUT_DIR = "/tmp/th3cl4w_test_recordings"


def _make_frame(w=320, h=240, brightness=128) -> np.ndarray:
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_stereo(frame_num=1, include_raw=True) -> StereoAsciiFrame:
    """Create a test stereo ASCII frame."""
    raw = _make_frame() if include_raw else None

    cam0 = AsciiFrame(
        camera_id=0,
        ascii_text="###\n...\n###",
        grid_width=3,
        grid_height=3,
        timestamp=time.monotonic(),
        frame_number=frame_num,
        raw_frame=raw,
    )
    cam1 = AsciiFrame(
        camera_id=1,
        ascii_text="@@@\n   \n@@@",
        grid_width=3,
        grid_height=3,
        timestamp=time.monotonic(),
        frame_number=frame_num,
        raw_frame=raw,
    )
    return StereoAsciiFrame(
        cam0=cam0,
        cam1=cam1,
        timestamp=time.monotonic(),
        frame_number=frame_num,
    )


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    """Clean up test output directory before and after tests."""
    if Path(TEST_OUTPUT_DIR).exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    if Path(TEST_OUTPUT_DIR).exists():
        shutil.rmtree(TEST_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# AnnotationRecord tests
# ---------------------------------------------------------------------------


class TestAnnotationRecord:
    def test_to_dict(self):
        record = AnnotationRecord(
            frame_number=1,
            timestamp=1.5,
            cam0_ascii="###",
            cam1_ascii="@@@",
            ascii_width=3,
            ascii_height=1,
            scene_objects=[{"label": "red", "bbox": [10, 20, 30, 40]}],
            action_label="pick",
        )
        d = record.to_dict()
        assert d["frame_number"] == 1
        assert d["cam0_ascii"] == "###"
        assert d["action_label"] == "pick"
        assert len(d["scene_objects"]) == 1


# ---------------------------------------------------------------------------
# RecorderConfig tests
# ---------------------------------------------------------------------------


class TestRecorderConfig:
    def test_defaults(self):
        config = RecorderConfig()
        assert config.record_video is True
        assert config.capture_screenshots is True
        assert config.screenshot_interval == 30

    def test_custom(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            screenshot_interval=5,
            video_fps=10.0,
        )
        assert config.screenshot_interval == 5


# ---------------------------------------------------------------------------
# VideoRecorder tests
# ---------------------------------------------------------------------------


class TestVideoRecorder:
    def test_construction(self):
        recorder = VideoRecorder()
        assert recorder.is_recording is False

    def test_start_stop_session(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_001")
        assert recorder.is_recording is True

        recorder.stop_session()
        assert recorder.is_recording is False

    def test_session_directory_created(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_dirs")

        session_dir = Path(TEST_OUTPUT_DIR) / "session_test_dirs"
        assert session_dir.exists()
        assert (session_dir / "screenshots").exists()
        assert (session_dir / "annotations").exists()
        assert (session_dir / "ascii_log").exists()

        recorder.stop_session()

    def test_process_frame_saves_ascii(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
            capture_screenshots=False,
            capture_annotations=False,
            capture_ascii=True,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_ascii")

        stereo = _make_stereo(frame_num=1)
        recorder.process_frame(stereo)

        session_dir = Path(TEST_OUTPUT_DIR) / "session_test_ascii"
        cam0_ascii = session_dir / "ascii_log" / "frame_000001_cam0.txt"
        cam1_ascii = session_dir / "ascii_log" / "frame_000001_cam1.txt"
        assert cam0_ascii.exists()
        assert cam1_ascii.exists()
        assert cam0_ascii.read_text() == "###\n...\n###"

        recorder.stop_session()

    def test_process_frame_saves_annotations(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
            capture_screenshots=False,
            capture_ascii=False,
            capture_annotations=True,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_annot")

        stereo = _make_stereo(frame_num=1)
        scene = [{"label": "red", "bbox": [10, 20, 30, 40]}]
        recorder.process_frame(stereo, scene_objects=scene, action_label="grab")

        session_dir = Path(TEST_OUTPUT_DIR) / "session_test_annot"
        annot_file = session_dir / "annotations" / "frame_000001.json"
        assert annot_file.exists()

        data = json.loads(annot_file.read_text())
        assert data["frame_number"] == 1
        assert data["action_label"] == "grab"
        assert len(data["scene_objects"]) == 1

        recorder.stop_session()

    def test_screenshot_interval(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
            capture_screenshots=True,
            capture_ascii=False,
            capture_annotations=False,
            screenshot_interval=3,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_screenshot")

        # Process 6 frames — screenshots at 3 and 6
        for i in range(1, 7):
            stereo = _make_stereo(frame_num=i)
            recorder.process_frame(stereo)

        session_dir = Path(TEST_OUTPUT_DIR) / "session_test_screenshot"
        screenshots = list((session_dir / "screenshots").glob("*.jpg"))
        # Should have screenshots at frames 3 and 6 (2 cameras each)
        assert len(screenshots) == 4  # 2 screenshots * 2 cameras

        recorder.stop_session()

    def test_session_metadata_written(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
            capture_screenshots=False,
            capture_ascii=False,
            capture_annotations=False,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_meta")

        stereo = _make_stereo()
        recorder.process_frame(stereo)

        recorder.stop_session()

        session_dir = Path(TEST_OUTPUT_DIR) / "session_test_meta"
        meta_file = session_dir / "session_metadata.json"
        assert meta_file.exists()

        meta = json.loads(meta_file.read_text())
        assert meta["session_id"] == "test_meta"
        assert meta["total_frames"] == 1

    def test_get_stats(self):
        config = RecorderConfig(output_dir=TEST_OUTPUT_DIR, record_video=False)
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_stats")

        stats = recorder.get_stats()
        assert stats["recording"] is True
        assert stats["session_id"] == "test_stats"
        assert stats["frames_recorded"] == 0

        recorder.stop_session()

    def test_max_frames_limit(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
            capture_screenshots=False,
            capture_ascii=False,
            capture_annotations=False,
            max_session_frames=3,
        )
        recorder = VideoRecorder(config=config)
        recorder.start_session("test_limit")

        # Process 5 frames, but limit is 3
        for i in range(5):
            stereo = _make_stereo(frame_num=i)
            recorder.process_frame(stereo)

        stats = recorder.get_stats()
        assert stats["frames_recorded"] == 3

        recorder.stop_session()

    def test_no_processing_when_stopped(self):
        config = RecorderConfig(
            output_dir=TEST_OUTPUT_DIR,
            record_video=False,
        )
        recorder = VideoRecorder(config=config)
        # Don't start session
        stereo = _make_stereo()
        recorder.process_frame(stereo)  # should be a no-op

        stats = recorder.get_stats()
        assert stats["frames_recorded"] == 0
