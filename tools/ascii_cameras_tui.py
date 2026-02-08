#!/usr/bin/env python3
"""
Dual ASCII Camera TUI Viewer

Shows both camera feeds (front + overhead) as live ASCII art side-by-side
in the terminal using curses.

Usage:
    python tools/ascii_cameras_tui.py                  # HTTP polling mode
    python tools/ascii_cameras_tui.py --websocket      # WebSocket mode
    python tools/ascii_cameras_tui.py --url http://host:8081  # Custom server

Quit: q or Ctrl+C
"""

import argparse
import curses
import io
import json
import sys
import time
import threading
from urllib.request import urlopen
from urllib.error import URLError

# Add project root to path for imports
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.vision.ascii_converter import AsciiConverter


def fetch_frame_http(url: str, timeout: float = 2.0) -> bytes | None:
    """Fetch a JPEG frame from the camera HTTP endpoint."""
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except (URLError, OSError, TimeoutError):
        return None


class CameraFeed:
    """Background thread that continuously fetches frames and converts to ASCII."""

    def __init__(self, cam_id: int, snap_url: str, width: int, height: int):
        self.cam_id = cam_id
        self.snap_url = snap_url
        self.width = width
        self.height = height
        self.converter = AsciiConverter(width=width, height=height)
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.running = True
        self.error: str | None = None
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def resize(self, width: int, height: int):
        with self.lock:
            self.width = width
            self.height = height
            self.converter = AsciiConverter(width=width, height=height)

    def _poll_loop(self):
        while self.running:
            data = fetch_frame_http(self.snap_url)
            if data:
                try:
                    ascii_str = self.converter.decode_jpeg_to_ascii(data)
                    with self.lock:
                        self.lines = ascii_str.split("\n")
                        self.error = None
                except Exception as e:
                    with self.lock:
                        self.error = str(e)
            else:
                with self.lock:
                    self.error = f"No response from {self.snap_url}"
            time.sleep(0.1)  # ~10 FPS

    def get_lines(self) -> tuple[list[str], str | None]:
        with self.lock:
            return list(self.lines), self.error

    def stop(self):
        self.running = False


class WebSocketFeed:
    """Background thread using WebSocket /ws/ascii endpoint."""

    def __init__(self, cam_id: int, ws_url: str, width: int, height: int):
        self.cam_id = cam_id
        self.ws_url = ws_url
        self.width = width
        self.height = height
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.running = True
        self.error: str | None = None
        self._ws = None
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def resize(self, width: int, height: int):
        with self.lock:
            self.width = width
            self.height = height
        # Send new settings to server
        if self._ws:
            try:
                self._ws.send(json.dumps({
                    "cam": self.cam_id,
                    "width": width,
                    "height": height,
                }))
            except Exception:
                pass

    def _ws_loop(self):
        try:
            import websockets.sync.client as wsc
        except ImportError:
            with self.lock:
                self.error = "pip install websockets"
            return

        while self.running:
            try:
                with wsc.connect(self.ws_url) as ws:
                    self._ws = ws
                    # Send initial settings
                    ws.send(json.dumps({
                        "cam": self.cam_id,
                        "width": self.width,
                        "height": self.height,
                        "color": False,
                    }))
                    while self.running:
                        msg = ws.recv(timeout=2)
                        data = json.loads(msg)
                        if "lines" in data:
                            with self.lock:
                                self.lines = data["lines"]
                                self.error = None
                        elif "ascii" in data:
                            with self.lock:
                                self.lines = data["ascii"].split("\n")
                                self.error = None
            except Exception as e:
                with self.lock:
                    self.error = f"WS error: {e}"
                self._ws = None
                time.sleep(1)

    def get_lines(self) -> tuple[list[str], str | None]:
        with self.lock:
            return list(self.lines), self.error

    def stop(self):
        self.running = False


LABELS = {0: "CAM 0: Front", 1: "CAM 1: Overhead"}


def main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 100ms for ~10 FPS refresh

    base_url = args.url.rstrip("/")
    max_h, max_w = stdscr.getmaxyx()

    # Each panel gets half the width (minus 1 for divider)
    panel_w = (max_w - 1) // 2
    panel_h = max_h - 2  # 1 for label, 1 for status bar

    feeds = []
    for cam_id in (0, 1):
        if args.websocket:
            ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/ascii"
            feeds.append(WebSocketFeed(cam_id, ws_url, panel_w, panel_h))
        else:
            snap_url = f"{base_url}/snap/{cam_id}"
            feeds.append(CameraFeed(cam_id, snap_url, panel_w, panel_h))

    frame_count = 0
    fps = 0.0
    fps_timer = time.monotonic()
    fps_frames = 0

    try:
        while True:
            ch = stdscr.getch()
            if ch == ord("q") or ch == ord("Q"):
                break
            if ch == curses.KEY_RESIZE:
                max_h, max_w = stdscr.getmaxyx()
                panel_w = (max_w - 1) // 2
                panel_h = max_h - 2
                for f in feeds:
                    f.resize(panel_w, panel_h)
                stdscr.clear()

            # FPS tracking
            fps_frames += 1
            now = time.monotonic()
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            stdscr.erase()

            for i, feed in enumerate(feeds):
                x_off = i * (panel_w + 1)
                lines, err = feed.get_lines()

                # Label
                label = LABELS.get(i, f"CAM {i}")
                try:
                    stdscr.addnstr(0, x_off, label, panel_w, curses.A_BOLD)
                except curses.error:
                    pass

                if err and not lines:
                    # Show error centered
                    msg = err[:panel_w]
                    try:
                        stdscr.addnstr(panel_h // 2 + 1, x_off, msg, panel_w)
                    except curses.error:
                        pass
                else:
                    for row_idx, line in enumerate(lines[:panel_h]):
                        try:
                            stdscr.addnstr(row_idx + 1, x_off, line[:panel_w], panel_w)
                        except curses.error:
                            pass

            # Divider
            for y in range(max_h - 1):
                try:
                    stdscr.addch(y, panel_w, "│")
                except curses.error:
                    pass

            # Status bar
            mode = "WS" if args.websocket else "HTTP"
            status = f" [{mode}] {fps:.1f} FPS | q=quit "
            try:
                stdscr.addnstr(max_h - 1, 0, status, max_w - 1, curses.A_REVERSE)
                # Pad status bar
                remaining = max_w - 1 - len(status)
                if remaining > 0:
                    stdscr.addstr(max_h - 1, len(status), " " * remaining, curses.A_REVERSE)
            except curses.error:
                pass

            stdscr.refresh()

    finally:
        for f in feeds:
            f.stop()


def run():
    parser = argparse.ArgumentParser(description="Dual ASCII Camera TUI Viewer")
    parser.add_argument("--url", default="http://localhost:8081", help="Camera server base URL")
    parser.add_argument("--websocket", action="store_true", help="Use WebSocket /ws/ascii instead of HTTP polling")
    args = parser.parse_args()
    curses.wrapper(lambda stdscr: main(stdscr, args))


if __name__ == "__main__":
    run()
