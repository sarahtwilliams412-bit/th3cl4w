"""Tests for the ASCII video converter module."""

import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

import cv2

from src.vision.ascii_converter import (
    AsciiConverter,
    CHARSET_STANDARD,
    CHARSET_DETAILED,
    CHARSET_BLOCKS,
    CHARSET_MINIMAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gradient_frame(w=640, h=480) -> np.ndarray:
    """Create a BGR frame with a horizontal brightness gradient."""
    gray = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(gray, (h, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_solid_frame(brightness: int, w=640, h=480) -> np.ndarray:
    """Create a solid-color BGR frame."""
    frame = np.full((h, w, 3), brightness, dtype=np.uint8)
    return frame


def _make_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAsciiConverterInit:
    def test_default_construction(self):
        conv = AsciiConverter()
        assert conv.width == 120
        assert conv.height == 40
        assert conv.charset == CHARSET_STANDARD
        assert conv.invert is True
        assert conv.color is False

    def test_custom_dimensions(self):
        conv = AsciiConverter(width=80, height=24)
        assert conv.width == 80
        assert conv.height == 24

    def test_custom_charset(self):
        conv = AsciiConverter(charset=CHARSET_DETAILED)
        assert conv.charset == CHARSET_DETAILED

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            AsciiConverter(width=0, height=10)
        with pytest.raises(ValueError):
            AsciiConverter(width=10, height=-1)

    def test_invalid_charset(self):
        with pytest.raises(ValueError):
            AsciiConverter(charset="X")  # only 1 character


# ---------------------------------------------------------------------------
# frame_to_ascii tests
# ---------------------------------------------------------------------------


class TestFrameToAscii:
    def test_output_dimensions(self):
        conv = AsciiConverter(width=60, height=20)
        frame = _make_gradient_frame()
        text = conv.frame_to_ascii(frame)
        lines = text.split("\n")
        assert len(lines) == 20
        assert all(len(line) == 60 for line in lines)

    def test_black_frame_inverted(self):
        """With invert=True, a black frame should produce the darkest characters."""
        conv = AsciiConverter(width=10, height=5, invert=True)
        frame = _make_solid_frame(0)
        text = conv.frame_to_ascii(frame)
        # First char in charset is the darkest
        for line in text.split("\n"):
            assert all(c == CHARSET_STANDARD[0] for c in line)

    def test_white_frame_inverted(self):
        """With invert=True, a white frame should produce the brightest characters."""
        conv = AsciiConverter(width=10, height=5, invert=True)
        frame = _make_solid_frame(255)
        text = conv.frame_to_ascii(frame)
        # Last char in charset is the brightest
        for line in text.split("\n"):
            assert all(c == CHARSET_STANDARD[-1] for c in line)

    def test_non_inverted_mode(self):
        """With invert=False, a white frame should produce the darkest characters."""
        conv = AsciiConverter(width=10, height=5, invert=False)
        frame = _make_solid_frame(255)
        text = conv.frame_to_ascii(frame)
        for line in text.split("\n"):
            assert all(c == CHARSET_STANDARD[0] for c in line)

    def test_gradient_produces_variety(self):
        """A gradient should produce multiple different characters."""
        conv = AsciiConverter(width=80, height=5)
        frame = _make_gradient_frame()
        text = conv.frame_to_ascii(frame)
        unique_chars = set(text.replace("\n", ""))
        assert len(unique_chars) > 3

    def test_all_charsets_work(self):
        frame = _make_gradient_frame()
        for charset in [CHARSET_STANDARD, CHARSET_DETAILED, CHARSET_BLOCKS, CHARSET_MINIMAL]:
            conv = AsciiConverter(width=40, height=15, charset=charset)
            text = conv.frame_to_ascii(frame)
            lines = text.split("\n")
            assert len(lines) == 15
            assert all(len(line) == 40 for line in lines)


# ---------------------------------------------------------------------------
# frame_to_color_data tests
# ---------------------------------------------------------------------------


class TestFrameToColorData:
    def test_structure_no_color(self):
        conv = AsciiConverter(width=30, height=10, color=False)
        frame = _make_gradient_frame()
        result = conv.frame_to_color_data(frame)
        assert result["width"] == 30
        assert result["height"] == 10
        assert len(result["lines"]) == 10
        assert all(len(line) == 30 for line in result["lines"])
        assert "colors" not in result

    def test_structure_with_color(self):
        conv = AsciiConverter(width=30, height=10, color=True)
        frame = _make_gradient_frame()
        result = conv.frame_to_color_data(frame)
        assert "colors" in result
        assert len(result["colors"]) == 10
        assert len(result["colors"][0]) == 30
        # Each color entry is [r, g, b]
        c = result["colors"][0][0]
        assert isinstance(c, list)
        assert len(c) == 3
        assert all(isinstance(v, int) for v in c)

    def test_color_values_in_range(self):
        conv = AsciiConverter(width=20, height=10, color=True)
        frame = _make_gradient_frame()
        result = conv.frame_to_color_data(frame)
        for row in result["colors"]:
            for r, g, b in row:
                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# JPEG decode tests
# ---------------------------------------------------------------------------


class TestJpegDecode:
    def test_decode_jpeg_to_ascii(self):
        conv = AsciiConverter(width=40, height=15)
        frame = _make_gradient_frame()
        jpeg = _make_jpeg(frame)
        text = conv.decode_jpeg_to_ascii(jpeg)
        lines = text.split("\n")
        assert len(lines) == 15
        assert all(len(line) == 40 for line in lines)

    def test_decode_jpeg_to_color_data(self):
        conv = AsciiConverter(width=40, height=15, color=True)
        frame = _make_gradient_frame()
        jpeg = _make_jpeg(frame)
        result = conv.decode_jpeg_to_color_data(jpeg)
        assert result["width"] == 40
        assert result["height"] == 15
        assert "colors" in result

    def test_decode_invalid_jpeg_raises(self):
        conv = AsciiConverter(width=40, height=15)
        with pytest.raises(ValueError, match="Failed to decode"):
            conv.decode_jpeg_to_ascii(b"not a jpeg")

    def test_decode_invalid_jpeg_color_raises(self):
        conv = AsciiConverter(width=40, height=15)
        with pytest.raises(ValueError, match="Failed to decode"):
            conv.decode_jpeg_to_color_data(b"not a jpeg")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tiny_output(self):
        conv = AsciiConverter(width=1, height=1)
        frame = _make_solid_frame(128)
        text = conv.frame_to_ascii(frame)
        assert len(text) == 1

    def test_large_output(self):
        conv = AsciiConverter(width=300, height=120)
        frame = _make_gradient_frame()
        text = conv.frame_to_ascii(frame)
        lines = text.split("\n")
        assert len(lines) == 120
        assert all(len(line) == 300 for line in lines)

    def test_small_input_frame(self):
        """Converter should handle very small input frames."""
        conv = AsciiConverter(width=40, height=15)
        frame = _make_solid_frame(200, w=2, h=2)
        text = conv.frame_to_ascii(frame)
        lines = text.split("\n")
        assert len(lines) == 15

    def test_non_square_frame(self):
        conv = AsciiConverter(width=80, height=20)
        frame = np.zeros((100, 800, 3), dtype=np.uint8)
        text = conv.frame_to_ascii(frame)
        lines = text.split("\n")
        assert len(lines) == 20
        assert all(len(line) == 80 for line in lines)
