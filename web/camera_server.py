#!/usr/bin/env python3.12
"""
th3cl4w — Dual Camera MJPEG Streaming Server
Serves MJPEG streams and JPEG snapshots from two cameras.
Runs standalone on port 8081 alongside the main web server.
"""

import argparse
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("th3cl4w.camera")

# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class CameraThread:
    """Captures frames from a camera in a background thread."""

    def __init__(self, device_id: int, width: int = 640, height: int = 480, fps: int = 15):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._no_signal_frame = self._make_no_signal_frame()

    def _make_no_signal_frame(self) -> bytes:
        """Generate a 'NO SIGNAL' placeholder frame."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (20, 20, 30)
        # Draw border
        cv2.rectangle(img, (2, 2), (self.width - 3, self.height - 3), (40, 40, 80), 2)
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "NO SIGNAL"
        (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
        cx, cy = (self.width - tw) // 2, (self.height + th) // 2
        cv2.putText(img, text, (cx, cy), font, 1.2, (80, 80, 140), 2, cv2.LINE_AA)
        # Device info
        info = f"dev {self.device_id}"
        cv2.putText(img, info, (cx + 20, cy + 35), font, 0.5, (60, 60, 100), 1, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _capture_loop(self):
        cap = None
        reconnect_delay = 1.0
        while self._running:
            # Open camera
            if cap is None or not cap.isOpened():
                self._connected = False
                logger.info("Opening camera /dev/video%d ...", self.device_id)
                cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._connected = True
                    reconnect_delay = 1.0
                    logger.info("Camera /dev/video%d opened (%dx%d @ %dfps)",
                                self.device_id, self.width, self.height, self.fps)
                else:
                    logger.warning("Failed to open /dev/video%d, retrying in %.0fs", self.device_id, reconnect_delay)
                    with self._lock:
                        self._frame = self._no_signal_frame
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)
                    continue

            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera /dev/video%d read failed, reconnecting", self.device_id)
                cap.release()
                cap = None
                with self._lock:
                    self._frame = self._no_signal_frame
                continue

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame = buf.tobytes()

            time.sleep(1.0 / self.fps)

        if cap and cap.isOpened():
            cap.release()

    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame if self._frame else self._no_signal_frame

    @property
    def connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

cameras: dict[int, CameraThread] = {}

class CameraHandler(BaseHTTPRequestHandler):
    """Serves MJPEG streams and JPEG snapshots."""

    def log_message(self, format, *args):
        # Suppress default logging for stream requests
        pass

    def do_GET(self):
        if self.path.startswith('/cam/'):
            self._handle_mjpeg()
        elif self.path.startswith('/snap/'):
            self._handle_snapshot()
        elif self.path == '/status':
            self._handle_status()
        else:
            self.send_error(404)

    def _get_cam_index(self) -> Optional[int]:
        parts = self.path.strip('/').split('/')
        if len(parts) >= 2:
            try:
                idx = int(parts[1])
                if idx in cameras:
                    return idx
            except ValueError:
                pass
        return None

    def _handle_mjpeg(self):
        idx = self._get_cam_index()
        if idx is None:
            self.send_error(404, "Camera not found")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        cam = cameras[idx]
        try:
            while True:
                frame = cam.get_frame()
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(frame)}\r\n'.encode())
                self.wfile.write(b'\r\n')
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
                time.sleep(1.0 / cam.fps)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_snapshot(self):
        idx = self._get_cam_index()
        if idx is None:
            self.send_error(404, "Camera not found")
            return

        frame = cameras[idx].get_frame()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(frame)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(frame)

    def _handle_status(self):
        import json
        status = {}
        for idx, cam in cameras.items():
            status[str(idx)] = {
                "connected": cam.connected,
                "device_id": cam.device_id,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread."""
    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread, args=(request, client_address), daemon=True)
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="th3cl4w Camera Streaming Server")
    parser.add_argument("--cam0", type=int, default=0, help="Device index for camera 0 (default: 0)")
    parser.add_argument("--cam1", type=int, default=4, help="Device index for camera 1 (default: 4)")
    parser.add_argument("--port", type=int, default=8081, help="HTTP port (default: 8081)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS (default: 15)")
    args = parser.parse_args()

    # Start camera threads
    cameras[0] = CameraThread(args.cam0, args.width, args.height, args.fps)
    cameras[1] = CameraThread(args.cam1, args.width, args.height, args.fps)

    for cam in cameras.values():
        cam.start()

    logger.info("Cameras started: cam0=/dev/video%d, cam1=/dev/video%d", args.cam0, args.cam1)

    # Start HTTP server
    server = ThreadedHTTPServer(('0.0.0.0', args.port), CameraHandler)
    logger.info("Camera server listening on http://0.0.0.0:%d", args.port)
    logger.info("  MJPEG streams: /cam/0, /cam/1")
    logger.info("  Snapshots:     /snap/0, /snap/1")
    logger.info("  Status:        /status")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        for cam in cameras.values():
            cam.stop()
        server.server_close()


if __name__ == "__main__":
    main()
