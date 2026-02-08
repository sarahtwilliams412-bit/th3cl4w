"""
Self-Filter Pipeline — Main Loop

Subscribes to occupancy grids on ZMQ :5556, reads D1 arm joint angles,
computes arm voxelization, subtracts arm volume, and publishes
obstacle-only grid with distance field on ZMQ :5557.

Publish format on tcp://*:5557:
  64 bytes: JSON header (padded with spaces)
    {"ts": 1234567, "min_dist_mm": 45.2, "obs_count": 1832}
  2,097,152 bytes: obstacle_binary as uint8[128^3]
  4,194,304 bytes: distance_field as float16[128^3]

Run as: python -m self_filter.pipeline
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

from self_filter.arm_voxelizer import ArmVoxelizer
from self_filter.d1_state_reader import D1StateReader
from self_filter.forward_kinematics import ForwardKinematics
from self_filter.obstacle_extractor import ObstacleExtractor

logger = logging.getLogger(__name__)

GRID_SIZE = 128
OCCUPANCY_MSG_HEADER = 8  # uint64 timestamp
OCCUPANCY_GRID_SIZE = GRID_SIZE**3 * 2  # float16
STATS_INTERVAL_S = 5.0
JSON_HEADER_SIZE = 64


def load_config() -> dict:
    """Load self-filter config."""
    config_path = Path(__file__).parent / "config.yaml"
    if yaml is None:
        raise RuntimeError("pyyaml required")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_calibration(config: dict) -> dict:
    """Load calibration.json for grid parameters."""
    cal_path = config.get("calibration_file", "calibration/calibration.json")
    try:
        with open(cal_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Calibration file not found at %s, using defaults", cal_path)
        return {
            "cell_size_mm": 7.8,
            "workspace_bounds_mm": {"x": [0, 1000], "y": [0, 1000], "z": [0, 600]},
        }


def receive_occupancy_grid(socket) -> tuple[int, np.ndarray]:
    """Receive occupancy grid from ZMQ.

    Returns
    -------
    tuple
        (timestamp_ms, float32[128, 128, 128] occupancy grid)
    """
    msg = socket.recv()
    timestamp_ms = struct.unpack("<Q", msg[:OCCUPANCY_MSG_HEADER])[0]
    grid_f16 = np.frombuffer(
        msg[OCCUPANCY_MSG_HEADER : OCCUPANCY_MSG_HEADER + OCCUPANCY_GRID_SIZE],
        dtype=np.float16,
    )
    grid = grid_f16.astype(np.float32).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    return timestamp_ms, grid


def publish_result(socket, result: dict, timestamp_ms: int) -> None:
    """Publish self-filtered result on ZMQ.

    Format:
      64 bytes JSON header + obstacle_binary (uint8) + distance_field (float16)
    """
    # JSON header (padded to 64 bytes)
    header = json.dumps({
        "ts": timestamp_ms,
        "min_dist_mm": round(result["min_obstacle_distance_mm"], 1),
        "obs_count": result["obstacle_voxel_count"],
    })
    header_padded = header.ljust(JSON_HEADER_SIZE)[:JSON_HEADER_SIZE]

    # Binary data
    obstacle_bytes = result["obstacle_binary"].astype(np.uint8).tobytes()
    distance_bytes = result["distance_field"].astype(np.float16).tobytes()

    message = header_padded.encode("utf-8") + obstacle_bytes + distance_bytes
    socket.send(message, zmq.NOBLOCK)


def compute_grid_origin(calibration: dict) -> list[float]:
    """Compute the grid origin from calibration data."""
    bounds = calibration.get("workspace_bounds_mm", {})
    x_min = bounds.get("x", [0, 1000])[0]
    y_min = bounds.get("y", [0, 1000])[0]
    z_min = bounds.get("z", [0, 600])[0]
    return [float(x_min), float(y_min), float(z_min)]


def run_pipeline() -> None:
    """Main self-filter pipeline loop."""
    if zmq is None:
        raise RuntimeError("pyzmq required: pip install pyzmq")

    config = load_config()
    calibration = load_calibration(config)

    # Extract configuration
    d1_cfg = config.get("d1_connection", {})
    dh_params = config.get("dh_parameters", [])
    link_radii = config.get("link_radii_mm", [40, 35, 30, 30, 25, 25, 20])
    safety_margin = config.get("arm_safety_margin_mm", 15)
    obstacle_threshold = config.get("obstacle_threshold", 0.3)
    cell_size_mm = calibration.get("cell_size_mm", 7.8)
    grid_origin = compute_grid_origin(calibration)

    # Initialize components
    state_reader = D1StateReader(
        ip=d1_cfg.get("ip", "192.168.123.100"),
        port=d1_cfg.get("port", 8082),
        timeout_ms=d1_cfg.get("timeout_ms", 100),
    )
    state_reader.connect()

    fk = ForwardKinematics(dh_params)

    voxelizer = ArmVoxelizer(
        link_radii_mm=link_radii,
        safety_margin_mm=safety_margin,
        grid_resolution=GRID_SIZE,
        cell_size_mm=cell_size_mm,
        grid_origin_mm=grid_origin,
    )

    extractor = ObstacleExtractor(
        obstacle_threshold=obstacle_threshold,
        cell_size_mm=cell_size_mm,
    )

    # ZMQ setup
    ctx = zmq.Context()

    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(config.get("zmq_input", "tcp://localhost:5556"))
    sub_socket.subscribe(b"")

    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind(config.get("zmq_output", "tcp://*:5557"))

    logger.info(
        "Self-filter pipeline started — SUB %s → PUB %s",
        config.get("zmq_input"),
        config.get("zmq_output"),
    )

    # State
    last_known_angles = np.zeros(len(dh_params), dtype=np.float64)
    frame_count = 0
    stats_frame_count = 0
    last_stats_time = time.monotonic()

    try:
        while True:
            timestamp_ms, occupancy = receive_occupancy_grid(sub_socket)
            t0 = time.monotonic()

            # Read joint angles
            joint_angles = state_reader.read_joint_angles()
            if joint_angles is not None:
                last_known_angles[: len(joint_angles)] = joint_angles[: len(dh_params)]
            else:
                logger.debug("Using last known joint angles")

            # Forward kinematics → link segments
            segments = fk.link_endpoints(last_known_angles)

            # Voxelize arm
            arm_mask = voxelizer.voxelize_arm(segments)

            # Extract obstacles
            result = extractor.extract(occupancy, arm_mask)

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            # Publish
            publish_result(pub_socket, result, timestamp_ms)

            frame_count += 1
            stats_frame_count += 1

            # Periodic stats
            now = time.monotonic()
            stats_elapsed = now - last_stats_time
            if stats_elapsed >= STATS_INTERVAL_S:
                fps = stats_frame_count / stats_elapsed
                logger.info(
                    "Self-filter: %.1f fps, %.1fms/frame, "
                    "obstacles=%d voxels, nearest=%.0fmm",
                    fps,
                    elapsed_ms,
                    result["obstacle_voxel_count"],
                    result["min_obstacle_distance_mm"],
                )
                last_stats_time = now
                stats_frame_count = 0

    except KeyboardInterrupt:
        logger.info("Shutting down self-filter pipeline")
    finally:
        state_reader.disconnect()
        sub_socket.close()
        pub_socket.close()
        ctx.term()


def main() -> None:
    """Entry point for python -m self_filter.pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
