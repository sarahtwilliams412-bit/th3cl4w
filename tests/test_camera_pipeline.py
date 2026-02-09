"""Tests for the camera pipeline module."""

import time
import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

import cv2

from src.vision.camera_pipeline import (
    AsciiFrame,
    StereoAsciiFrame,
    PipelineConfig,
    CameraPipeline,
)
from src.vision.ascii_converter import CHARSET_STANDARD, CHARSET_DETAILED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(w=640, h=480, brightness=128) -> np.ndarray:
    """Create a solid BGR frame."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_gradient_frame(w=640, h=480) -> np.ndarray:
    """Create a BGR frame with a horizontal brightness gradient."""
    gray = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(gray, (h, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class FakeCamera:
    """Fake camera source for testing."""

    def __init__(self, frame=None):
        self._frame = frame if frame is not None else _make_gradient_frame()

    def get_raw_frame(self):
        return self._frame.copy()


# ---------------------------------------------------------------------------
# AsciiFrame tests
# ---------------------------------------------------------------------------


class TestAsciiFrame:
    def test_lines_property(self):
        af = AsciiFrame(
            camera_id=0,
            ascii_text="AB\nCD\nEF",
            grid_width=2,
            grid_height=3,
        )
        assert af.lines == ["AB", "CD", "EF"]

    def test_grid_property(self):
        af = AsciiFrame(
            camera_id=0,
            ascii_text="AB\nCD",
            grid_width=2,
            grid_height=2,
        )
        grid = af.grid
        assert grid[0] == ["A", "B"]
        assert grid[1] == ["C", "D"]

    def test_char_at(self):
        af = AsciiFrame(
            camera_id=0,
            ascii_text="ABC\nDEF",
            grid_width=3,
            grid_height=2,
        )
        assert af.char_at(0, 0) == "A"
        assert af.char_at(2, 1) == "F"
        assert af.char_at(99, 99) == " "  # out of bounds

    def test_to_dict(self):
        af = AsciiFrame(
            camera_id=1,
            ascii_text="test",
            grid_width=4,
            grid_height=1,
            timestamp=1.5,
            frame_number=10,
        )
        d = af.to_dict()
        assert d["camera_id"] == 1
        assert d["grid_width"] == 4
        assert d["frame_number"] == 10


class TestStereoAsciiFrame:
    def test_has_both(self):
        cam0 = AsciiFrame(camera_id=0, ascii_text="X", grid_width=1, grid_height=1)
        cam1 = AsciiFrame(camera_id=1, ascii_text="Y", grid_width=1, grid_height=1)
        stereo = StereoAsciiFrame(cam0=cam0, cam1=cam1)
        assert stereo.has_both is True

    def test_single_camera(self):
        cam0 = AsciiFrame(camera_id=0, ascii_text="X", grid_width=1, grid_height=1)
        stereo = StereoAsciiFrame(cam0=cam0, cam1=None)
        assert stereo.has_both is False


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.ascii_width == 120
        assert config.ascii_height == 40
        assert config.ascii_fps == 5.0

    def test_validate_good(self):
        config = PipelineConfig(ascii_width=80, ascii_height=24, ascii_fps=3.0)
        config.validate()  # should not raise

    def test_validate_bad_width(self):
        config = PipelineConfig(ascii_width=5)
        with pytest.raises(ValueError, match="ascii_width"):
            config.validate()

    def test_validate_bad_height(self):
        config = PipelineConfig(ascii_height=2)
        with pytest.raises(ValueError, match="ascii_height"):
            config.validate()

    def test_validate_bad_fps(self):
        config = PipelineConfig(ascii_fps=0)
        with pytest.raises(ValueError, match="ascii_fps"):
            config.validate()


# ---------------------------------------------------------------------------
# CameraPipeline tests
# ---------------------------------------------------------------------------


class TestCameraPipeline:
    def test_construction(self):
        pipeline = CameraPipeline()
        assert pipeline.is_running is False
        assert pipeline.get_latest_frame() is None

    def test_custom_config(self):
        config = PipelineConfig(ascii_width=80, ascii_height=24)
        pipeline = CameraPipeline(config=config)
        assert pipeline.config.ascii_width == 80

    def test_convert_single_frame(self):
        pipeline = CameraPipeline()
        frame = _make_gradient_frame()
        result = pipeline.convert_single_frame(frame, camera_id=0)
        assert isinstance(result, AsciiFrame)
        assert result.camera_id == 0
        assert len(result.lines) == 40
        assert result.grid_width == 120

    def test_convert_with_color(self):
        config = PipelineConfig(color=True)
        pipeline = CameraPipeline(config=config)
        frame = _make_gradient_frame()
        result = pipeline.convert_single_frame(frame, camera_id=1)
        assert result.color_data is not None
        assert "colors" in result.color_data

    def test_attach_cameras(self):
        pipeline = CameraPipeline()
        cam0 = FakeCamera()
        cam1 = FakeCamera()
        pipeline.attach_cameras(cam0, cam1)

    def test_update_config(self):
        pipeline = CameraPipeline()
        pipeline.update_config(ascii_width=60, ascii_height=20, ascii_fps=2.0)
        assert pipeline.config.ascii_width == 60
        assert pipeline.config.ascii_height == 20
        assert pipeline.config.ascii_fps == 2.0

    def test_update_config_charset(self):
        pipeline = CameraPipeline()
        pipeline.update_config(charset=CHARSET_DETAILED)
        assert pipeline.config.charset == CHARSET_DETAILED

    def test_callback_registration(self):
        pipeline = CameraPipeline()
        received = []
        pipeline.on_frame(lambda f: received.append(f))
        # Callbacks are registered but not called until pipeline runs

    def test_get_stats(self):
        pipeline = CameraPipeline()
        stats = pipeline.get_stats()
        assert stats["running"] is False
        assert stats["ascii_frame_count"] == 0
        assert "config" in stats

    def test_start_stop_with_fake_cameras(self):
        config = PipelineConfig(ascii_fps=10.0, retain_raw_frames=True)
        pipeline = CameraPipeline(config=config)
        cam0 = FakeCamera()
        cam1 = FakeCamera()
        pipeline.attach_cameras(cam0, cam1)

        received = []
        pipeline.on_frame(lambda f: received.append(f))

        pipeline.start()
        assert pipeline.is_running is True

        # Let it run for a few frames
        time.sleep(0.5)

        pipeline.stop()
        assert pipeline.is_running is False

        # Should have received some frames
        assert len(received) > 0
        assert isinstance(received[0], StereoAsciiFrame)
        assert received[0].has_both is True

    def test_get_latest_after_processing(self):
        config = PipelineConfig(ascii_fps=10.0)
        pipeline = CameraPipeline(config=config)
        cam0 = FakeCamera()
        pipeline.attach_cameras(cam0)

        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        latest = pipeline.get_latest_frame()
        assert latest is not None
        assert latest.cam0 is not None

    def test_single_camera_mode(self):
        config = PipelineConfig(ascii_fps=10.0)
        pipeline = CameraPipeline(config=config)
        cam0 = FakeCamera()
        pipeline.attach_cameras(cam0, None)

        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()

        latest = pipeline.get_latest_frame()
        assert latest is not None
        assert latest.cam0 is not None
        assert latest.cam1 is None
