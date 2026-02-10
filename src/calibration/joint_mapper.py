"""
Joint Mapping Calibration Module

Maps UI joint indices (J0-J5) to DDS joint indices (angle0-angle5).
Handles the case where the DDS indices don't correspond to the expected
physical joints (e.g., J0 in UI might actually control DDS angle2).

The mapping is stored in data/joint_mapping.json and loaded at startup.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAPPING_PATH = _PROJECT_ROOT / "data" / "joint_mapping.json"

# Default: identity mapping (UI index == DDS index)
_DEFAULT_MAPPING = {
    "ui_to_dds": {str(i): i for i in range(6)},
    "labels": {
        "0": "J0 — base yaw",
        "1": "J1 — shoulder pitch",
        "2": "J2 — elbow pitch",
        "3": "J3 — elbow roll",
        "4": "J4 — wrist pitch",
        "5": "J5 — wrist roll",
    },
    "calibrated": False,
    "timestamp": None,
}


class JointMapper:
    """Bidirectional mapping between UI joint IDs and DDS joint IDs."""

    def __init__(self, mapping_path: Optional[Path] = None):
        self._path = mapping_path or _MAPPING_PATH
        self._ui_to_dds: dict[int, int] = {i: i for i in range(6)}
        self._dds_to_ui: dict[int, int] = {i: i for i in range(6)}
        self._labels: dict[int, str] = {}
        self._calibrated = False
        self._load()

    def _load(self):
        """Load mapping from JSON file, or use identity mapping."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                u2d = data.get("ui_to_dds", {})
                self._ui_to_dds = {int(k): int(v) for k, v in u2d.items()}
                self._dds_to_ui = {v: k for k, v in self._ui_to_dds.items()}
                self._labels = {int(k): v for k, v in data.get("labels", {}).items()}
                self._calibrated = data.get("calibrated", False)
                logger.info("Joint mapping loaded from %s (calibrated=%s)", self._path, self._calibrated)
            except Exception as e:
                logger.warning("Failed to load joint mapping: %s — using identity", e)
                self._reset_to_identity()
        else:
            self._reset_to_identity()

    def _reset_to_identity(self):
        self._ui_to_dds = {i: i for i in range(6)}
        self._dds_to_ui = {i: i for i in range(6)}
        self._calibrated = False

    def ui_to_dds(self, ui_joint_id: int) -> int:
        """Convert a UI joint index to the corresponding DDS joint index."""
        return self._ui_to_dds.get(ui_joint_id, ui_joint_id)

    def dds_to_ui(self, dds_joint_id: int) -> int:
        """Convert a DDS joint index to the corresponding UI joint index."""
        return self._dds_to_ui.get(dds_joint_id, dds_joint_id)

    def remap_angles_ui_to_dds(self, ui_angles: list[float]) -> list[float]:
        """Remap a list of 6 angles from UI order to DDS order."""
        dds_angles = [0.0] * 6
        for ui_id in range(6):
            dds_id = self.ui_to_dds(ui_id)
            dds_angles[dds_id] = ui_angles[ui_id]
        return dds_angles

    def remap_angles_dds_to_ui(self, dds_angles: list[float]) -> list[float]:
        """Remap a list of 6 angles from DDS order to UI order."""
        ui_angles = [0.0] * 6
        for dds_id in range(6):
            ui_id = self.dds_to_ui(dds_id)
            ui_angles[ui_id] = dds_angles[dds_id]
        return ui_angles

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def save(self, ui_to_dds: dict[int, int], labels: dict[int, str]):
        """Save a new mapping to disk."""
        import time
        self._ui_to_dds = {int(k): int(v) for k, v in ui_to_dds.items()}
        self._dds_to_ui = {v: k for k, v in self._ui_to_dds.items()}
        self._labels = labels
        self._calibrated = True
        data = {
            "ui_to_dds": {str(k): v for k, v in self._ui_to_dds.items()},
            "labels": {str(k): v for k, v in self._labels.items()},
            "calibrated": True,
            "timestamp": time.time(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Joint mapping saved to %s", self._path)

    def get_mapping_info(self) -> dict:
        """Return current mapping as a dict for API responses."""
        return {
            "ui_to_dds": {str(k): v for k, v in self._ui_to_dds.items()},
            "dds_to_ui": {str(k): v for k, v in self._dds_to_ui.items()},
            "labels": {str(k): v for k, v in self._labels.items()},
            "calibrated": self._calibrated,
        }


# Singleton instance
_instance: Optional[JointMapper] = None


def get_joint_mapper() -> JointMapper:
    """Get the global JointMapper singleton."""
    global _instance
    if _instance is None:
        _instance = JointMapper()
    return _instance
