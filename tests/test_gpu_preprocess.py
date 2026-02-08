"""Tests for src.vision.gpu_preprocess — GPU-accelerated frame utilities."""

import cv2
import numpy as np
import pytest

from src.vision.gpu_preprocess import (
    decode_jpeg_gpu,
    gpu_status,
    resize_gpu,
    to_grayscale_gpu,
    to_numpy,
)


def _make_test_jpeg(width=64, height=48):
    """Create a small test JPEG image and return the bytes."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes(), img


class TestDecodeJpegGpu:
    def test_decode_returns_frame(self):
        jpeg, _ = _make_test_jpeg()
        frame = decode_jpeg_gpu(jpeg)
        arr = to_numpy(frame)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3 and arr.shape[2] == 3

    def test_decode_invalid_raises(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_jpeg_gpu(b"not a jpeg")


class TestResizeGpu:
    def test_resize_dimensions(self):
        jpeg, _ = _make_test_jpeg(100, 80)
        frame = decode_jpeg_gpu(jpeg)
        resized = resize_gpu(frame, 50, 40)
        arr = to_numpy(resized)
        assert arr.shape[:2] == (40, 50)


class TestToGrayscaleGpu:
    def test_single_channel(self):
        jpeg, _ = _make_test_jpeg()
        frame = decode_jpeg_gpu(jpeg)
        gray = to_grayscale_gpu(frame)
        arr = to_numpy(gray)
        assert arr.ndim == 2


class TestToNumpy:
    def test_numpy_passthrough(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        assert to_numpy(arr) is arr

    def test_umat_conversion(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        umat = cv2.UMat(arr)
        result = to_numpy(umat)
        assert isinstance(result, np.ndarray)


class TestGpuStatus:
    def test_returns_expected_keys(self):
        status = gpu_status()
        assert "opencl_available" in status
        assert "opencl_enabled" in status
        assert "device" in status
        assert isinstance(status["opencl_available"], bool)


class TestAsciiConverterWithUMat:
    """Ensure AsciiConverter still works with both numpy and UMat inputs."""

    def test_frame_to_ascii_with_numpy(self):
        from src.vision.ascii_converter import AsciiConverter

        converter = AsciiConverter(width=20, height=10)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = converter.frame_to_ascii(frame)
        lines = result.split("\n")
        assert len(lines) == 10
        assert all(len(l) == 20 for l in lines)

    def test_frame_to_ascii_with_umat(self):
        from src.vision.ascii_converter import AsciiConverter

        converter = AsciiConverter(width=20, height=10)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        umat = cv2.UMat(frame)
        result = converter.frame_to_ascii(umat)
        lines = result.split("\n")
        assert len(lines) == 10
        assert all(len(l) == 20 for l in lines)

    def test_frame_to_color_data_with_umat(self):
        from src.vision.ascii_converter import AsciiConverter

        converter = AsciiConverter(width=20, height=10, color=True)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        umat = cv2.UMat(frame)
        result = converter.frame_to_color_data(umat)
        assert len(result["lines"]) == 10
        assert len(result["colors"]) == 10
        assert len(result["colors"][0]) == 20
