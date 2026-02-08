"""
Calibration Frame Capture Tool

Subscribes to ZMQ frame pairs and displays ASCII frames in the terminal.
User places a checkerboard at various positions and saves captures with
keypress.

Usage:
    python -m calibration.capture_frames

Controls:
    s — Save current frame pair
    q — Quit
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GRID_SIZE = 128
CAPTURE_DIR = Path(__file__).parent / "captures"


def load_config() -> dict:
    """Load calibration config from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    if yaml is None:
        raise RuntimeError("pyyaml is required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def receive_frame_pair(socket) -> tuple[int, np.ndarray, np.ndarray]:
    """Receive a single frame pair from ZMQ.

    Returns
    -------
    tuple
        (timestamp_ms, top_down[128,128], profile[128,128])
    """
    msg = socket.recv()
    timestamp_ms = struct.unpack("<Q", msg[:8])[0]
    top_data = np.frombuffer(msg[8 : 8 + GRID_SIZE * GRID_SIZE], dtype=np.uint8)
    prof_data = np.frombuffer(
        msg[8 + GRID_SIZE * GRID_SIZE : 8 + 2 * GRID_SIZE * GRID_SIZE], dtype=np.uint8
    )
    top_grid = top_data.reshape(GRID_SIZE, GRID_SIZE)
    prof_grid = prof_data.reshape(GRID_SIZE, GRID_SIZE)
    return timestamp_ms, top_grid, prof_grid


def display_grid(grid: np.ndarray, label: str, display_size: int = 40) -> None:
    """Print an ASCII grid to terminal at reduced size for readability."""
    step = max(1, GRID_SIZE // display_size)
    print(f"\n--- {label} (sampled {display_size}x{display_size}) ---")
    for y in range(0, GRID_SIZE, step):
        row = ""
        for x in range(0, GRID_SIZE, step):
            row += chr(grid[y, x]) if 32 <= grid[y, x] < 127 else "?"
        print(row)


def main() -> None:
    """Interactive frame capture for calibration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if zmq is None:
        print("ERROR: pyzmq is required. Install with: pip install pyzmq")
        sys.exit(1)

    config = load_config()
    zmq_source = config.get("zmq_frame_source", "tcp://localhost:5555")

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Count existing captures
    existing = list(CAPTURE_DIR.glob("frame_*.npz"))
    capture_idx = len(existing)

    print(f"Calibration Frame Capture")
    print(f"  ZMQ source: {zmq_source}")
    print(f"  Capture dir: {CAPTURE_DIR}")
    print(f"  Existing captures: {capture_idx}")
    print(f"\nControls:")
    print(f"  s — Save current frame pair")
    print(f"  q — Quit")
    print(f"\nPlace checkerboard at 30+ positions across the workspace.")
    print(f"Waiting for frames...")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(zmq_source)
    sock.subscribe(b"")
    sock.setsockopt(zmq.RCVTIMEO, 2000)

    try:
        import termios
        import tty

        # Set terminal to raw mode for single-keypress detection
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        raw_mode = True
    except (ImportError, termios.error):
        raw_mode = False
        print("(Terminal raw mode unavailable — type command + Enter)")

    try:
        while True:
            try:
                ts, top, prof = receive_frame_pair(sock)
            except zmq.error.Again:
                print(".", end="", flush=True)
                continue

            display_grid(top, f"Top-Down (ts={ts})")
            display_grid(prof, f"Profile (ts={ts})")
            print(f"\n[Capture #{capture_idx}] Press 's' to save, 'q' to quit: ", end="", flush=True)

            if raw_mode:
                ch = sys.stdin.read(1)
            else:
                ch = input().strip().lower()[:1]

            if ch == "s":
                path = CAPTURE_DIR / f"frame_{capture_idx:04d}.npz"
                np.savez(str(path), top_down=top, profile=prof, timestamp_ms=ts)
                print(f"  Saved: {path}")
                capture_idx += 1
            elif ch == "q":
                break

    except KeyboardInterrupt:
        pass
    finally:
        if raw_mode:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sock.close()
        ctx.term()

    print(f"\nDone. Total captures: {capture_idx}")


if __name__ == "__main__":
    main()
