"""
Visual Hull Pipeline — Main Loop

Subscribes to synchronized ASCII frame pairs on ZMQ :5555,
runs visual hull reconstruction, applies temporal filtering,
and publishes the 128^3 occupancy grid on ZMQ :5556.

Publish format:
  8 bytes: uint64 timestamp_ms (little-endian)
  4,194,304 bytes: float16[128, 128, 128] occupancy grid

Run as: python -m visual_hull.pipeline
"""

from __future__ import annotations

import json
import logging
import struct
import time
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

from visual_hull.hull_reconstructor import VisualHullReconstructor
from visual_hull.temporal_filter import TemporalFilter

logger = logging.getLogger(__name__)

GRID_SIZE = 128
FRAME_PAIR_MSG_SIZE = 8 + GRID_SIZE * GRID_SIZE * 2  # 32776 bytes
STATS_INTERVAL_S = 5.0


def load_config() -> dict:
    """Load visual hull config."""
    config_path = Path(__file__).parent / "config.yaml"
    if yaml is None:
        raise RuntimeError("pyyaml required")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_density_lut(config: dict) -> np.ndarray:
    """Load density LUT from file or generate default."""
    lut_path = config.get("density_lut_file", "calibration/density_lut.npy")
    try:
        lut = np.load(lut_path)
        logger.info("Loaded density LUT from %s", lut_path)
        return lut.astype(np.float32)
    except FileNotFoundError:
        logger.warning("Density LUT not found at %s, generating default", lut_path)
        from calibration.ascii_to_grayscale import build_density_lut
        return build_density_lut()


def load_calibration(config: dict) -> dict:
    """Load calibration.json."""
    cal_path = config.get("calibration_file", "calibration/calibration.json")
    try:
        with open(cal_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Calibration file not found at %s, using defaults", cal_path)
        return {"shared_axis": "x"}


def receive_frame_pair(socket) -> tuple[int, np.ndarray, np.ndarray]:
    """Receive and decode a frame pair from ZMQ.

    Returns
    -------
    tuple
        (timestamp_ms, top_down[128,128], profile[128,128])
    """
    msg = socket.recv()
    timestamp_ms = struct.unpack("<Q", msg[:8])[0]
    top_data = np.frombuffer(msg[8 : 8 + GRID_SIZE**2], dtype=np.uint8)
    prof_data = np.frombuffer(msg[8 + GRID_SIZE**2 : 8 + 2 * GRID_SIZE**2], dtype=np.uint8)
    return (
        timestamp_ms,
        top_data.reshape(GRID_SIZE, GRID_SIZE).copy(),
        prof_data.reshape(GRID_SIZE, GRID_SIZE).copy(),
    )


def publish_grid(socket, grid: np.ndarray, timestamp_ms: int) -> None:
    """Publish occupancy grid on ZMQ.

    Uses float16 to halve bandwidth (128^3 × 2 = 4,194,304 bytes).
    """
    message = struct.pack("<Q", timestamp_ms) + grid.astype(np.float16).tobytes()
    socket.send(message, zmq.NOBLOCK)


def run_pipeline() -> None:
    """Main visual hull pipeline loop."""
    if zmq is None:
        raise RuntimeError("pyzmq required: pip install pyzmq")

    config = load_config()
    density_lut = load_density_lut(config)
    calibration = load_calibration(config)

    shared_axis = calibration.get("shared_axis", "x")
    alpha = config.get("temporal_alpha", 0.7)
    coarse_threshold = config.get("coarse_occupancy_threshold", 0.1)

    # Initialize components
    reconstructor = VisualHullReconstructor(
        density_lut=density_lut,
        shared_axis=shared_axis,
        coarse_threshold=coarse_threshold,
    )
    temporal_filter = TemporalFilter(alpha=alpha)

    # ZMQ setup
    ctx = zmq.Context()

    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(config.get("zmq_input", "tcp://localhost:5555"))
    sub_socket.subscribe(b"")

    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind(config.get("zmq_output", "tcp://*:5556"))

    logger.info(
        "Visual hull pipeline started — "
        "SUB %s → PUB %s, shared_axis=%s, alpha=%.2f",
        config.get("zmq_input"),
        config.get("zmq_output"),
        shared_axis,
        alpha,
    )

    # Stats tracking
    frame_count = 0
    stats_frame_count = 0
    last_stats_time = time.monotonic()

    try:
        while True:
            timestamp_ms, top_frame, prof_frame = receive_frame_pair(sub_socket)

            t0 = time.monotonic()

            # Reconstruct occupancy grid
            grid = reconstructor.reconstruct_coarse_to_fine(top_frame, prof_frame)

            # Apply temporal filtering
            grid = temporal_filter.update(grid)

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            # Publish
            publish_grid(pub_socket, grid, timestamp_ms)

            frame_count += 1
            stats_frame_count += 1

            # Periodic stats
            now = time.monotonic()
            stats_elapsed = now - last_stats_time
            if stats_elapsed >= STATS_INTERVAL_S:
                fps = stats_frame_count / stats_elapsed
                occupied_pct = (
                    np.count_nonzero(grid > config.get("occupancy_threshold", 0.15))
                    / grid.size
                    * 100.0
                )
                logger.info(
                    "Visual hull: %.1f fps, %.1fms/frame, %.1f%% voxels occupied",
                    fps,
                    elapsed_ms,
                    occupied_pct,
                )
                last_stats_time = now
                stats_frame_count = 0

    except KeyboardInterrupt:
        logger.info("Shutting down visual hull pipeline")
    finally:
        sub_socket.close()
        pub_socket.close()
        ctx.term()


def main() -> None:
    """Entry point for python -m visual_hull.pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
