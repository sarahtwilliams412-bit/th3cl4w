"""Consolidated world model — single source of truth for object locations.

This replaces src/vision/world_model.py as the canonical spatial model.
Maintains a live map of all tracked objects with position, dimensions,
confidence, timestamps, and reachability status.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .reachability import ReachStatus, classify_reach, is_reachable

logger = logging.getLogger("th3cl4w.location.world_model")


class ObjectCategory(Enum):
    TARGET = "target"
    OBSTACLE = "obstacle"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


@dataclass
class TrackedObject:
    """A tracked object in the world model."""

    id: str
    label: str
    position_mm: np.ndarray          # (3,) XYZ relative to arm base
    dimensions_mm: np.ndarray         # (3,) width, height, depth
    confidence: float
    category: ObjectCategory
    reach_status: ReachStatus
    reach_distance_mm: float
    graspable: bool

    # Pixel-level info from last detection
    bbox_px: Optional[tuple[int, int, int, int]] = None
    camera_id: Optional[int] = None

    # Tracking
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_verified: float = 0.0
    observation_count: int = 0
    stable: bool = False
    stale: bool = False

    # Detection source
    source: str = "cv"  # "cv", "llm", "both"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "position_mm": [round(v, 1) for v in self.position_mm.tolist()],
            "dimensions_mm": [round(v, 1) for v in self.dimensions_mm.tolist()],
            "confidence": round(self.confidence, 3),
            "category": self.category.value,
            "reach_status": self.reach_status.value,
            "reach_distance_mm": round(self.reach_distance_mm, 1),
            "graspable": self.graspable,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "last_verified": round(self.last_verified, 3),
            "observation_count": self.observation_count,
            "stable": self.stable,
            "stale": self.stale,
            "source": self.source,
        }


# How long before an object is considered stale (seconds)
STALE_TIMEOUT_S = 30.0
# How long before a stale object is removed entirely
REMOVE_TIMEOUT_S = 120.0
# Minimum observations to be considered stable
MIN_STABLE_OBS = 3
# Position smoothing factor (0-1, lower = more smoothing)
POSITION_ALPHA = 0.3
# Distance threshold for matching detections to existing objects (mm)
MATCH_DISTANCE_MM = 80.0


class LocationWorldModel:
    """Thread-safe world model for the location server.

    Objects are:
    - Added/updated when detected
    - Matched to existing objects by label + proximity
    - Smoothed in position over multiple observations
    - Flagged stale if not seen recently
    - Removed if stale for too long
    """

    def __init__(self):
        self._objects: dict[str, TrackedObject] = {}
        self._lock = threading.Lock()
        self._scan_count = 0
        self._last_scan_time = 0.0
        self._label_counters: dict[str, int] = {}
        self._update_callbacks: list = []

    def register_callback(self, cb):
        """Register a callback called on every update with list of changed objects."""
        self._update_callbacks.append(cb)

    def _notify(self, changed: list[TrackedObject]):
        for cb in self._update_callbacks:
            try:
                cb(changed)
            except Exception:
                logger.exception("Update callback error")

    def upsert(
        self,
        label: str,
        position_mm: np.ndarray,
        dimensions_mm: np.ndarray = None,
        confidence: float = 0.5,
        source: str = "cv",
        bbox_px: tuple = None,
        camera_id: int = None,
        graspable: bool = False,
    ) -> TrackedObject:
        """Add or update an object in the world model.

        Matches by label + proximity. If a matching object exists within
        MATCH_DISTANCE_MM, updates it. Otherwise creates a new entry.
        """
        now = time.time()
        if dimensions_mm is None:
            dimensions_mm = np.array([30.0, 30.0, 30.0])

        reach_status, reach_dist = classify_reach(position_mm)
        category = self._classify(label, graspable, reach_status, dimensions_mm)

        with self._lock:
            # Try to match existing object
            match_id = self._find_match(label, position_mm)

            if match_id:
                obj = self._objects[match_id]
                # Smooth position
                obj.position_mm = obj.position_mm * (1 - POSITION_ALPHA) + position_mm * POSITION_ALPHA
                obj.dimensions_mm = dimensions_mm
                obj.confidence = max(obj.confidence, confidence)  # keep best confidence
                obj.reach_status = reach_status
                obj.reach_distance_mm = reach_dist
                obj.category = category
                obj.graspable = graspable
                obj.last_seen = now
                obj.last_verified = now
                obj.observation_count += 1
                obj.stable = obj.observation_count >= MIN_STABLE_OBS
                obj.stale = False
                obj.source = source if obj.source == source else "both"
                if bbox_px:
                    obj.bbox_px = bbox_px
                    obj.camera_id = camera_id
            else:
                obj_id = self._make_id(label)
                obj = TrackedObject(
                    id=obj_id,
                    label=label,
                    position_mm=position_mm.copy(),
                    dimensions_mm=dimensions_mm,
                    confidence=confidence,
                    category=category,
                    reach_status=reach_status,
                    reach_distance_mm=reach_dist,
                    graspable=graspable,
                    bbox_px=bbox_px,
                    camera_id=camera_id,
                    first_seen=now,
                    last_seen=now,
                    last_verified=now,
                    observation_count=1,
                    source=source,
                )
                self._objects[obj_id] = obj

        self._notify([obj])
        return obj

    def _find_match(self, label: str, position_mm: np.ndarray) -> Optional[str]:
        """Find an existing object matching this label within proximity threshold."""
        best_id = None
        best_dist = MATCH_DISTANCE_MM

        for obj_id, obj in self._objects.items():
            if obj.label != label:
                continue
            dist = float(np.linalg.norm(obj.position_mm - position_mm))
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        return best_id

    def _make_id(self, label: str) -> str:
        count = self._label_counters.get(label, 0)
        self._label_counters[label] = count + 1
        return f"{label}_{count}"

    def _classify(
        self, label: str, graspable: bool, reach: ReachStatus, dims: np.ndarray
    ) -> ObjectCategory:
        if graspable and reach in (ReachStatus.REACHABLE, ReachStatus.MARGINAL):
            return ObjectCategory.TARGET
        max_dim = float(np.max(dims)) if dims is not None else 0
        if max_dim > 200:
            return ObjectCategory.BOUNDARY
        return ObjectCategory.UNKNOWN

    def mark_scan(self):
        """Record that a scan pass completed."""
        self._scan_count += 1
        self._last_scan_time = time.time()

    def sweep_stale(self):
        """Flag stale objects, remove very old ones."""
        now = time.time()
        to_remove = []

        with self._lock:
            for obj_id, obj in self._objects.items():
                age = now - obj.last_seen
                if age > REMOVE_TIMEOUT_S:
                    to_remove.append(obj_id)
                elif age > STALE_TIMEOUT_S:
                    obj.stale = True
                    obj.confidence *= 0.9

            for obj_id in to_remove:
                logger.info("Removing stale object: %s", obj_id)
                del self._objects[obj_id]

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_all_objects(self) -> list[TrackedObject]:
        with self._lock:
            return list(self._objects.values())

    def get_reachable(self) -> list[TrackedObject]:
        with self._lock:
            return [
                o for o in self._objects.values()
                if o.reach_status in (ReachStatus.REACHABLE, ReachStatus.MARGINAL)
                and not o.stale
            ]

    def get_object(self, obj_id: str) -> Optional[TrackedObject]:
        with self._lock:
            return self._objects.get(obj_id)

    def clear(self):
        with self._lock:
            self._objects.clear()
            self._scan_count = 0
            self._label_counters.clear()

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def last_scan_time(self) -> float:
        return self._last_scan_time

    @property
    def object_count(self) -> int:
        with self._lock:
            return len(self._objects)
