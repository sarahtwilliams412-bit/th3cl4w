"""Tests for th3cl4w camera streaming server.

Requires opencv-python (cv2) — skipped if not installed.
"""

import json
import threading
import time
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Skip entire module if cv2 is not available (camera server needs real cv2)
pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.camera_server import CameraThread, CameraHandler, ThreadedHTTPServer, cameras

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_frame(w=640, h=480):
    """Create a fake BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def mock_camera():
    """Create a CameraThread with mocked OpenCV capture."""
    cam = CameraThread(device_id=99, width=640, height=480, fps=15)
    return cam


@pytest.fixture
def running_server():
    """Start the camera server with mock cameras for integration tests."""
    cameras.clear()

    # Create cameras with mock capture
    for idx in (0, 1):
        cam = CameraThread(device_id=idx, width=640, height=480, fps=15)
        # Pre-load a frame so we don't need real hardware
        import cv2

        frame = _make_fake_frame()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cam._frame = buf.tobytes()
        cam._connected = True
        cameras[idx] = cam

    server = ThreadedHTTPServer(("127.0.0.1", 0), CameraHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)
    yield port
    server.shutdown()
    cameras.clear()


# ---------------------------------------------------------------------------
# Unit tests — CameraThread
# ---------------------------------------------------------------------------


class TestCameraThread:
    def test_no_signal_frame_is_valid_jpeg(self, mock_camera):
        """NO SIGNAL frame should be a valid JPEG."""
        frame = mock_camera._no_signal_frame
        assert frame[:2] == b"\xff\xd8"  # JPEG SOI marker
        assert len(frame) > 100

    def test_get_frame_returns_no_signal_when_no_capture(self, mock_camera):
        """Before any capture, get_frame returns NO SIGNAL."""
        frame = mock_camera.get_frame()
        assert frame[:2] == b"\xff\xd8"

    def test_connected_initially_false(self, mock_camera):
        assert mock_camera.connected is False

    @patch("web.camera_server.cv2.VideoCapture")
    def test_capture_loop_opens_camera(self, mock_vc, mock_camera):
        """Capture loop should attempt to open the camera device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, _make_fake_frame())] * 3 + [KeyboardInterrupt]
        mock_vc.return_value = mock_cap

        mock_camera._running = True

        def stop_after():
            time.sleep(0.3)
            mock_camera._running = False

        threading.Thread(target=stop_after, daemon=True).start()

        try:
            mock_camera._capture_loop()
        except KeyboardInterrupt:
            pass

        import cv2 as _cv2

        mock_vc.assert_called_with(99, _cv2.CAP_V4L2)
        assert mock_camera._connected is True

    @patch("web.camera_server.cv2.VideoCapture")
    def test_capture_loop_handles_disconnect(self, mock_vc, mock_camera):
        """When read() fails, camera should reconnect."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # First read succeeds, second fails
        # After disconnect, the loop reopens — provide enough reads for the reconnect cycle
        mock_cap.read.side_effect = [(True, _make_fake_frame()), (False, None)] + [
            (True, _make_fake_frame())
        ] * 20
        mock_vc.return_value = mock_cap

        mock_camera._running = True

        def stop_after():
            time.sleep(0.5)
            mock_camera._running = False

        threading.Thread(target=stop_after, daemon=True).start()
        mock_camera._capture_loop()

        # Should have released and tried to reopen
        assert mock_cap.release.called


# ---------------------------------------------------------------------------
# Integration tests — HTTP endpoints
# ---------------------------------------------------------------------------


class TestHTTPEndpoints:
    def test_snapshot_returns_jpeg(self, running_server):
        """GET /snap/0 should return a JPEG image."""
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/snap/0")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "image/jpeg"
        data = resp.read()
        assert data[:2] == b"\xff\xd8"
        conn.close()

    def test_snapshot_cam1(self, running_server):
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/snap/1")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "image/jpeg"
        conn.close()

    def test_snapshot_invalid_cam_404(self, running_server):
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/snap/9")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_mjpeg_stream_returns_multipart(self, running_server):
        """GET /cam/0 should return multipart MJPEG content."""
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/cam/0")
        resp = conn.getresponse()
        assert resp.status == 200
        ct = resp.getheader("Content-Type")
        assert "multipart/x-mixed-replace" in ct
        assert "boundary=frame" in ct
        # Read a chunk — should contain JPEG data
        chunk = resp.read(8192)
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk
        conn.close()

    def test_status_endpoint(self, running_server):
        """GET /status should return JSON with camera info."""
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/status")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "0" in data
        assert "1" in data
        assert data["0"]["connected"] is True
        assert data["0"]["width"] == 640
        conn.close()

    def test_cors_headers(self, running_server):
        """Responses should include CORS headers."""
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/snap/0")
        resp = conn.getresponse()
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        conn.close()

    def test_404_for_unknown_path(self, running_server):
        conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
        conn.request("GET", "/unknown")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()
