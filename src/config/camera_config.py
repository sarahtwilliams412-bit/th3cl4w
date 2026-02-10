"""Camera service configuration — single source of truth."""
import os

CAMERA_SERVER_URL = os.getenv("CAMERA_SERVER_URL", "http://localhost:8081")


def snap_url(cam_id: int) -> str:
    return f"{CAMERA_SERVER_URL}/snap/{cam_id}"


def latest_url(cam_id: int) -> str:
    return f"{CAMERA_SERVER_URL}/latest/{cam_id}"


def cameras_url() -> str:
    return f"{CAMERA_SERVER_URL}/cameras"


# Camera IDs
CAM_SIDE = 0
CAM_ARM = 1
CAM_OVERHEAD = 2
