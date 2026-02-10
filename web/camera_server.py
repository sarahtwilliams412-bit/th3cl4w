#!/usr/bin/env python3.12
"""
th3cl4w — Dual Camera MJPEG Streaming Server
Serves MJPEG streams and JPEG snapshots from two cameras.
Runs standalone on port 8081 alongside the main web server.
"""

import argparse
import logging
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from src.telemetry.camera_monitor import CameraHealthMonitor

    _HAS_MONITOR = True
except ImportError:
    _HAS_MONITOR = False

try:
    from src.telemetry import get_collector

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False

try:
    from src.vision.startup_scanner import StartupScanner
    from src.vision.world_model import WorldModel

    _HAS_SCANNER = True
except ImportError:
    _HAS_SCANNER = False

# Logging configured in main() via setup_logging(); fall back for import-time usage
logger = logging.getLogger("th3cl4w.camera")

# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------


class CameraThread:
    """Captures frames from a camera in a background thread."""

    def __init__(
        self,
        device_id: int,
        width: int = 1920,
        height: int = 1080,
        fps: int = 15,
        jpeg_quality: int = 92,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._no_signal_frame = self._make_no_signal_frame()
        self._frame_count = 0
        self._health: Optional["CameraHealthMonitor"] = None
        if _HAS_MONITOR:
            self._health = CameraHealthMonitor(camera_id=str(device_id), target_fps=float(fps))

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
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
                    # Request MJPEG format for high-res throughput
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Log actual resolution
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._connected = True
                    reconnect_delay = 1.0
                    logger.info(
                        "Camera /dev/video%d opened (requested %dx%d, actual %dx%d @ %dfps)",
                        self.device_id,
                        self.width,
                        self.height,
                        actual_w,
                        actual_h,
                        self.fps,
                    )
                else:
                    logger.warning(
                        "Failed to open /dev/video%d, retrying in %.0fs",
                        self.device_id,
                        reconnect_delay,
                    )
                    with self._lock:
                        self._frame = self._no_signal_frame
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)
                    continue

            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera /dev/video%d read failed, reconnecting", self.device_id)
                if self._health:
                    self._health.on_drop()
                    self._health.on_frame(resolution=(self.width, self.height), connected=False)
                cap.release()
                cap = None
                with self._lock:
                    self._frame = self._no_signal_frame
                continue

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            with self._lock:
                self._frame = buf.tobytes()

            if self._health:
                h, w = frame.shape[:2]
                self._health.on_frame(resolution=(w, h), connected=True)
                self._frame_count += 1
                if self._frame_count % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self._health.compute_motion(gray)
                # Log camera health every ~5 seconds
                if self._frame_count % (self.fps * 5) == 0 and _HAS_TELEMETRY:
                    try:
                        tc = get_collector()
                        if tc is not None:
                            tc.log_camera_health(str(self.device_id), self._health.stats)
                    except Exception:
                        pass

            time.sleep(1.0 / self.fps)

        if cap and cap.isOpened():
            cap.release()

    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame if self._frame else self._no_signal_frame

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame as a decoded numpy BGR array.

        Returns None if no frame is available or camera is disconnected.
        Used by the arm tracker and grasp planner for vision processing.
        """
        jpeg_bytes = self.get_frame()
        if jpeg_bytes is None or jpeg_bytes == self._no_signal_frame:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        return frame

    @property
    def connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

cameras: dict[int, CameraThread] = {}
startup_scanner: Optional["StartupScanner"] = None


class CameraHandler(BaseHTTPRequestHandler):
    """Serves MJPEG streams and JPEG snapshots."""

    def log_message(self, format, *args):
        # Suppress default logging for stream requests
        pass

    def do_GET(self):
        if self.path.startswith("/cam/"):
            self._handle_mjpeg()
        elif self.path.startswith("/snap/"):
            self._handle_snapshot()
        elif self.path == "/status":
            self._handle_status()
        elif self.path == "/world":
            self._handle_world_model()
        elif self.path == "/scan":
            self._handle_scan_report()
        else:
            self.send_error(404)

    def _get_cam_index(self) -> Optional[int]:
        parts = self.path.strip("/").split("/")
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
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        cam = cameras[idx]
        try:
            while True:
                frame = cam.get_frame()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
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
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(frame)

    def _handle_status(self):
        import json

        status = {}
        for idx, cam in cameras.items():
            cam_status = {
                "connected": cam.connected,
                "device_id": cam.device_id,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }
            if cam._health:
                cam_status["health"] = cam._health.stats
            status[str(idx)] = cam_status
        # Include scanner phase if running
        if startup_scanner is not None:
            status["scanner"] = {
                "phase": startup_scanner.phase.value,
                "running": startup_scanner.is_running,
            }
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_world_model(self):
        """Serve the current world model snapshot as JSON."""
        import json

        if startup_scanner is None:
            body = json.dumps({"error": "Scanner not available"}).encode()
        else:
            model = startup_scanner.get_world_model()
            snap = model.snapshot()
            body = json.dumps(snap.to_dict()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _handle_scan_report(self):
        """Serve the startup scan report as JSON."""
        import json

        if startup_scanner is None:
            body = json.dumps({"error": "Scanner not available"}).encode()
        else:
            report = startup_scanner.get_report()
            body = json.dumps(report.to_dict()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread."""

    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(
            target=self.process_request_thread, args=(request, client_address), daemon=True
        )
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


def _on_world_model_ready(snapshot):
    """Callback fired when the startup scanner's initial world model is built."""
    logger.info(
        "World model ready: %d objects, confidence=%.2f grade=%s, "
        "reachable_targets=%d, obstacles=%d, free_zone=%.0f%%",
        len(snapshot.objects),
        snapshot.model_confidence,
        snapshot.model_grade,
        snapshot.reachable_targets,
        snapshot.obstacles_detected,
        snapshot.free_zone_pct,
    )
    for obj in snapshot.objects:
        dims = obj.dimensions_mm
        logger.info(
            "  %s: %.0fx%.0fx%.0fmm at (%.0f,%.0f,%.0f)mm — %s, %s, grade=%s",
            obj.object_id,
            dims[0],
            dims[1],
            dims[2],
            obj.position_mm[0],
            obj.position_mm[1],
            obj.position_mm[2],
            obj.category.value,
            obj.reach_status.value,
            obj.grade,
        )


def main():
    global startup_scanner

    parser = argparse.ArgumentParser(description="th3cl4w Camera Streaming Server")
    parser.add_argument(
        "--cam0", type=int, default=0, help="Device index for camera 0 (default: 0)"
    )
    parser.add_argument(
        "--cam1", type=int, default=4, help="Device index for camera 1 (default: 4)"
    )
    parser.add_argument(
        "--cam2", type=int, default=6, help="Device index for camera 2 (default: 6)"
    )
    parser.add_argument("--port", type=int, default=8081, help="HTTP port (default: 8081)")
    parser.add_argument("--width", type=int, default=1920, help="Capture width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Capture height (default: 1080)")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS (default: 15)")
    parser.add_argument(
        "--jpeg-quality", type=int, default=92, help="JPEG quality 1-100 (default: 92)"
    )
    parser.add_argument(
        "--no-scan", action="store_true", help="Disable startup environment scanning"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging to logs/camera.log")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom log output directory (default: logs/)")
    args = parser.parse_args()

    from src.utils.logging_config import setup_logging
    setup_logging(server_name="camera", debug=args.debug, log_dir=args.log_dir)

    # Start camera threads
    cameras[0] = CameraThread(args.cam0, args.width, args.height, args.fps, args.jpeg_quality)
    cameras[1] = CameraThread(args.cam1, args.width, args.height, args.fps, args.jpeg_quality)

    # Try to open cam2 — don't crash if it fails
    try:
        cam2 = CameraThread(args.cam2, args.width, args.height, args.fps, args.jpeg_quality)
        cameras[2] = cam2
    except Exception as e:
        logger.warning("Failed to create camera 2 (dev %d): %s — continuing with 2 cameras", args.cam2, e)

    for cam in cameras.values():
        cam.start()

    cam_list = ", ".join(f"cam{i}=/dev/video{cameras[i].device_id}" for i in sorted(cameras))
    logger.info("Cameras started: %s", cam_list)

    # Launch startup scanner — immediately begins assessing the environment
    if _HAS_SCANNER and not args.no_scan:
        scanner_kwargs = {"cam0": cameras[0], "cam1": cameras[1]}
        if 2 in cameras:
            scanner_kwargs["cam2"] = cameras[2]
        startup_scanner = StartupScanner(**{k: v for k, v in scanner_kwargs.items() if k in ("cam0", "cam1")})
        startup_scanner.on_model_ready(_on_world_model_ready)
        startup_scanner.start()
        logger.info("Startup scanner launched — building world model from camera feeds")
    elif not _HAS_SCANNER:
        logger.warning("Startup scanner unavailable (vision modules not installed)")

    # Start HTTP server
    server = ThreadedHTTPServer(("0.0.0.0", args.port), CameraHandler)
    logger.info("Camera server listening on http://0.0.0.0:%d", args.port)
    cam_ids = sorted(cameras.keys())
    logger.info("  MJPEG streams: %s", ", ".join(f"/cam/{i}" for i in cam_ids))
    logger.info("  Snapshots:     %s", ", ".join(f"/snap/{i}" for i in cam_ids))
    logger.info("  Status:        /status")
    logger.info("  World model:   /world")
    logger.info("  Scan report:   /scan")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if startup_scanner is not None:
            startup_scanner.stop()
        for cam in cameras.values():
            cam.stop()
        server.server_close()


if __name__ == "__main__":
    main()
