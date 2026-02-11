"""Camera device-to-role assignment configuration.

Manages which physical video device is assigned to which camera role.
Persisted in data/camera_config.json under the 'assignments' key.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("th3cl4w.camera.assignments")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = _PROJECT_ROOT / "data" / "camera_assignments.json"

CAMERA_ROLES = ["side_profile", "end_effector", "overhead"]

ROLE_METADATA = {
    "side_profile": {
        "cam_id": 0,
        "name": "Side",
        "mount": "fixed",
        "description": "Fixed side-view camera. Used for height (Z) estimation of objects on workspace.",
    },
    "end_effector": {
        "cam_id": 1,
        "name": "Arm-mounted",
        "mount": "arm",
        "description": "Camera attached to end-effector. Moves with arm. Used for close-up inspection and visual servo.",
    },
    "overhead": {
        "cam_id": 2,
        "name": "Overhead",
        "mount": "fixed",
        "description": "Overhead camera looking straight down. Primary camera for object detection X/Y positioning.",
    },
}


def _default_config() -> dict:
    return {
        "assignments": {},
        "configured": False,
        "detected_devices": {},
    }


def load_config() -> dict:
    """Load camera assignment config. Returns default if missing/invalid."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            # Ensure required keys
            if "assignments" not in cfg:
                cfg["assignments"] = {}
            if "configured" not in cfg:
                cfg["configured"] = False
            return cfg
        except Exception as e:
            logger.warning("Failed to load camera assignments: %s", e)
    return _default_config()


def save_config(cfg: dict) -> None:
    """Save camera assignment config."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Camera assignments saved to %s", CONFIG_PATH)


def is_configured() -> bool:
    """Check if cameras have been assigned to roles."""
    cfg = load_config()
    return bool(cfg.get("configured", False))


def get_assignments() -> dict[str, str]:
    """Return role -> device path mapping. E.g. {'side_profile': '/dev/video0'}."""
    cfg = load_config()
    return cfg.get("assignments", {})


def get_device_index_for_role(role: str) -> Optional[int]:
    """Get the device index for a given role. Returns None if not assigned."""
    assignments = get_assignments()
    dev_path = assignments.get(role)
    if dev_path and dev_path.startswith("/dev/video"):
        try:
            return int(dev_path.replace("/dev/video", ""))
        except ValueError:
            pass
    return None


def get_cam_id_assignments() -> dict[int, int]:
    """Return {cam_id: device_index} for use by camera threads.
    
    Maps role -> cam_id -> device_index based on ROLE_METADATA.
    """
    assignments = get_assignments()
    result = {}
    for role, dev_path in assignments.items():
        meta = ROLE_METADATA.get(role)
        if meta and dev_path and dev_path.startswith("/dev/video"):
            try:
                dev_idx = int(dev_path.replace("/dev/video", ""))
                result[meta["cam_id"]] = dev_idx
            except ValueError:
                pass
    return result


def check_device_changes(current_devices: dict[int, str]) -> bool:
    """Check if connected devices differ from what was configured.
    
    Returns True if devices changed (config should be invalidated).
    """
    cfg = load_config()
    saved_devices = cfg.get("detected_devices", {})
    
    # Compare device sets (convert keys to strings for JSON compat)
    current_set = {str(k): v for k, v in current_devices.items()}
    
    if not saved_devices:
        return False  # No saved state to compare against
    
    if current_set != saved_devices:
        logger.warning(
            "USB camera devices changed! Saved: %s, Current: %s — resetting configuration",
            saved_devices, current_set,
        )
        cfg["configured"] = False
        save_config(cfg)
        return True
    
    return False
