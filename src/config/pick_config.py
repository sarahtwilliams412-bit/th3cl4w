"""
Central configuration for pick task parameters.

Singleton config loader that reads from data/pick_config.json.
All modules should use `get_pick_config()` instead of hardcoding values.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CONFIG_FILE = _CONFIG_DIR / "pick_config.json"

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "detection": {
        "redbull_hsv_ranges": [
            {"lower": [0, 100, 80], "upper": [10, 255, 255]},
            {"lower": [160, 100, 80], "upper": [180, 255, 255]},
        ],
        "redbull_blue_hsv": {
            "lower": [100, 120, 60],
            "upper": [130, 255, 255],
        },
        "color_presets": {
            "red": [
                {"lower": [0, 100, 80], "upper": [10, 255, 255]},
                {"lower": [160, 100, 80], "upper": [180, 255, 255]},
            ],
            "blue": [
                {"lower": [100, 120, 60], "upper": [130, 255, 255]},
            ],
            "green": [
                {"lower": [35, 80, 60], "upper": [85, 255, 255]},
            ],
            "yellow": [
                {"lower": [20, 100, 100], "upper": [35, 255, 255]},
            ],
        },
        "min_contour_area": 500,
        "morph_kernel_size": 5,
        "gripper_hsv_lower": [0, 0, 180],
        "gripper_hsv_upper": [180, 40, 255],
        "gripper_min_area": 300,
    },
    "reach": {
        "max_mm": 550.0,
        "safe_mm": 500.0,
        "min_mm": 80.0,
    },
    "pick": {
        "approach_height_mm": 120.0,
        "grasp_height_offset_mm": 10.0,
        "lift_height_mm": 80.0,
        "z_step_mm": 5.0,
        "z_tolerance_mm": 5.0,
        "max_steps_per_phase": 30,
        "phase_timeouts": {
            "A": 15.0,
            "B": 10.0,
            "C": 10.0,
            "D": 10.0,
            "E": 5.0,
            "F": 8.0,
        },
    },
    "servo": {
        "convergence_threshold_px": 20.0,
        "max_iterations": 30,
        "kp": 0.3,
        "max_step_mm": 30.0,
        "settle_time_s": 0.3,
        "fallback_px_to_mm": 0.3,
        "llm_max_steps": 25,
        "llm_close_enough_px": 80,
        "llm_step_deg": 5.0,
        "llm_stall_limit": 3,
    },
    "gripper": {
        "open_mm": 65.0,
        "close_mm": 0.0,
        "default_mm": 30.0,
        "pick_open_mm": 60.0,
        "pick_close_mm": 5.0,
        "max_mm": 65.0,
        "min_mm": 0.0,
    },
    "safety": {
        "torque_proxy_limit": 100.0,
        "torque_j2_factor": 0.7,
        "stall_check_delay_s": 3.0,
        "stall_threshold_deg": 5.0,
        "collision_position_error_deg": 3.0,
        "collision_stall_duration_s": 0.5,
        "collision_cooldown_s": 5.0,
    },
    "tracker": {
        "fast_scan_interval_s": 2.0,
        "deep_scan_interval_s": 180.0,
        "verify_interval_s": 5.0,
        "stale_sweep_interval_s": 10.0,
        "overhead_mm_per_px": 0.4167,
        "side_mm_per_px": 0.5556,
    },
    "side_height": {
        "redbull_hsv_ranges": [
            {"lower": [0, 100, 80], "upper": [10, 255, 255]},
            {"lower": [160, 100, 80], "upper": [180, 255, 255]},
        ],
        "gripper_hsv_ranges": [
            {"lower": [0, 0, 0], "upper": [180, 80, 60]},
        ],
        "neon_tape_hsv": {
            "lower": [35, 100, 100],
            "upper": [85, 255, 255],
        },
        "min_contour_area": 300,
    },
    "arm_camera": {
        "redbull_hsv_ranges": [
            {"lower": [0, 80, 60], "upper": [12, 255, 255]},
            {"lower": [158, 80, 60], "upper": [180, 255, 255]},
            {"lower": [100, 80, 60], "upper": [130, 255, 255]},
        ],
        "px_to_mm": 0.18,
        "tolerance_px": 30,
        "min_contour_area": 200,
    },
    "contact": {
        "poll_hz": 10.0,
        "stable_duration_s": 0.2,
        "stable_tolerance_mm": 2.0,
        "close_target_mm": 10.0,
        "object_min_mm": 25.0,
        "timeout_s": 3.0,
        "redbull_contact_min_mm": 48.0,
        "redbull_contact_max_mm": 54.0,
    },
}


class PickConfig:
    """Singleton configuration for pick task parameters.

    Thread-safe. Auto-saves on change.
    """

    _instance: Optional[PickConfig] = None
    _lock = threading.Lock()

    def __new__(cls) -> PickConfig:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self._initialized = True
        self._load()

    def _load(self):
        """Load config from disk, merging with defaults."""
        if _CONFIG_FILE.exists():
            try:
                saved = json.loads(_CONFIG_FILE.read_text())
                self._merge(self._data, saved)
                logger.info("Loaded pick config from %s", _CONFIG_FILE)
            except Exception as e:
                logger.warning("Failed to load pick config: %s", e)

    def _merge(self, base: dict, overlay: dict):
        """Deep-merge overlay into base."""
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge(base[k], v)
            else:
                base[k] = v

    def _save(self):
        """Save current config to disk."""
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _CONFIG_FILE.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            logger.warning("Failed to save pick config: %s", e)

    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get a config value. If key is None, returns the whole section."""
        with self._lock:
            sec = self._data.get(section, {})
            if key is None:
                return copy.deepcopy(sec)
            return copy.deepcopy(sec.get(key))

    def set(self, section: str, key: str, value: Any):
        """Set a config value and auto-save."""
        with self._lock:
            if section not in self._data:
                self._data[section] = {}
            self._data[section][key] = value
            self._save()

    def get_all(self) -> dict[str, Any]:
        """Return full config."""
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, data: dict[str, Any]):
        """Deep-merge updates and save."""
        with self._lock:
            self._merge(self._data, data)
            self._save()

    def reset(self):
        """Reset to defaults and save."""
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)
            self._save()

    def get_defaults(self) -> dict[str, Any]:
        """Return default config."""
        return copy.deepcopy(DEFAULTS)

    def diff(self) -> dict[str, Any]:
        """Return values that differ from defaults."""
        return self._diff_dict(DEFAULTS, self._data)

    def _diff_dict(self, defaults: dict, current: dict) -> dict:
        result = {}
        for k, v in current.items():
            if k not in defaults:
                result[k] = v
            elif isinstance(v, dict) and isinstance(defaults[k], dict):
                d = self._diff_dict(defaults[k], v)
                if d:
                    result[k] = d
            elif v != defaults[k]:
                result[k] = v
        return result


def get_pick_config() -> PickConfig:
    """Get the singleton PickConfig instance."""
    return PickConfig()
