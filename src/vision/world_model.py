"""
World Model — Spatial understanding of the arm's operating environment.

Built from dimension estimates and scene analysis, this model provides:
1. Object catalog: position, dimensions, confidence for each detected object
2. Obstacle zones: regions the arm must avoid
3. Reachability map: what's within the arm's reach envelope
4. Free-space assessment: where the arm can safely move
5. Continuous re-assessment: improves as new frames arrive

Designed to be populated in just a couple of camera frames during startup,
giving the arm an immediate understanding of its surroundings.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .dimension_estimator import DimensionEstimate

logger = logging.getLogger("th3cl4w.vision.world_model")

# Unitree D1 workspace constants
ARM_MAX_REACH_MM = 550.0
ARM_MIN_REACH_MM = 80.0
ARM_BASE_POSITION = np.array([0.0, 0.0, 0.0])  # arm base at origin


class ObjectCategory(Enum):
    """Categorize objects by their role in the world model."""
    TARGET = "target"       # something the arm might pick up
    OBSTACLE = "obstacle"   # something the arm must avoid
    BOUNDARY = "boundary"   # edge of workspace / wall
    UNKNOWN = "unknown"     # detected but not yet classified


class ReachStatus(Enum):
    """Whether an object is within the arm's operating envelope."""
    REACHABLE = "reachable"
    OUT_OF_RANGE = "out_of_range"
    TOO_CLOSE = "too_close"
    MARGINAL = "marginal"  # near the edge of reachability


@dataclass
class WorldObject:
    """A physical object in the world model with position, dimensions, and metadata."""

    object_id: str  # unique identifier (e.g. "red_0", "blue_1")
    label: str
    position_mm: np.ndarray  # (3,) XYZ in arm-base frame
    dimensions_mm: np.ndarray  # (3,) width, height, depth
    confidence: float
    grade: str
    category: ObjectCategory
    reach_status: ReachStatus
    reach_distance_mm: float  # XY distance from arm base
    graspable: bool

    # Safety envelope: expanded bounding box for collision avoidance
    safety_min_mm: np.ndarray  # (3,) min corner of safety zone
    safety_max_mm: np.ndarray  # (3,) max corner of safety zone

    # Tracking
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0
    stable: bool = False  # True if consistently observed across frames

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position_mm": [round(v, 1) for v in self.position_mm.tolist()],
            "dimensions_mm": [round(v, 1) for v in self.dimensions_mm.tolist()],
            "confidence": round(self.confidence, 3),
            "grade": self.grade,
            "category": self.category.value,
            "reach_status": self.reach_status.value,
            "reach_distance_mm": round(self.reach_distance_mm, 1),
            "graspable": self.graspable,
            "safety_zone": {
                "min_mm": [round(v, 1) for v in self.safety_min_mm.tolist()],
                "max_mm": [round(v, 1) for v in self.safety_max_mm.tolist()],
            },
            "stable": self.stable,
            "observation_count": self.observation_count,
        }


@dataclass
class WorldModelSnapshot:
    """Immutable snapshot of the world model at a point in time."""

    objects: list[WorldObject]
    scan_count: int  # how many scan passes have been completed
    total_observations: int
    timestamp: float
    build_time_ms: float  # time to construct this snapshot

    # Summaries
    reachable_targets: int
    obstacles_detected: int
    free_zone_pct: float  # percentage of workspace that's clear

    # Overall assessment
    model_confidence: float  # aggregate confidence in the world model
    model_grade: str

    def to_dict(self) -> dict:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "scan_count": self.scan_count,
            "total_observations": self.total_observations,
            "timestamp": round(self.timestamp, 3),
            "build_time_ms": round(self.build_time_ms, 1),
            "reachable_targets": self.reachable_targets,
            "obstacles_detected": self.obstacles_detected,
            "free_zone_pct": round(self.free_zone_pct, 1),
            "model_confidence": round(self.model_confidence, 3),
            "model_grade": self.model_grade,
        }


class WorldModel:
    """Builds and maintains a spatial world model from dimension estimates.

    The model is continuously refined as new observations arrive.
    It starts usable after a single scan pass (2 frames from 2 cameras)
    and improves with each additional observation.

    Key design decisions:
    - Objects below grade D are excluded from the model entirely
    - Safety zones are inflated by 30mm around each object for collision margin
    - Objects not seen in the last 5 scan passes are marked stale and downgraded
    - The model supports concurrent read access via snapshots
    """

    SAFETY_MARGIN_MM = 30.0  # inflate bounding boxes for collision avoidance
    STALE_AFTER_SCANS = 5    # mark objects stale after N scans without observation
    MIN_GRADE_FOR_INCLUSION = "D"  # objects below this grade are excluded

    # Grade ordering for comparison
    _GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

    def __init__(
        self,
        arm_max_reach_mm: float = ARM_MAX_REACH_MM,
        arm_min_reach_mm: float = ARM_MIN_REACH_MM,
        safety_margin_mm: float = SAFETY_MARGIN_MM,
    ):
        self.arm_max_reach = arm_max_reach_mm
        self.arm_min_reach = arm_min_reach_mm
        self.safety_margin = safety_margin_mm

        self._objects: dict[str, WorldObject] = {}
        self._lock = threading.Lock()
        self._scan_count = 0
        self._total_observations = 0
        self._label_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Updating the model
    # ------------------------------------------------------------------

    def update(
        self,
        estimates: list[DimensionEstimate],
        positions: Optional[dict[str, np.ndarray]] = None,
    ):
        """Integrate new dimension estimates into the world model.

        Args:
            estimates: Dimension estimates from ObjectDimensionEstimator.
            positions: Optional dict mapping label→position (3,) in mm.
                       If not provided, position is estimated from bounding box.
        """
        t0 = time.monotonic()
        now = time.monotonic()

        with self._lock:
            self._scan_count += 1

            for est in estimates:
                # Filter by grade
                if self._GRADE_ORDER.get(est.grade, 0) < self._GRADE_ORDER.get(
                    self.MIN_GRADE_FOR_INCLUSION, 0
                ):
                    continue

                obj_id = self._get_or_create_id(est.label)

                # Position: use provided position or estimate from bbox center
                if positions and est.label in positions:
                    pos = positions[est.label].copy()
                else:
                    pos = self._estimate_position_from_bbox(est)

                dims = np.array([est.width_mm, est.height_mm, est.depth_mm])
                reach_dist = float(np.linalg.norm(pos[:2]))
                reach_status = self._classify_reach(reach_dist)
                category = self._classify_category(est, reach_status)

                # Safety zone: bounding box inflated by margin
                half_dims = dims / 2.0 + self.safety_margin
                safety_min = pos - half_dims
                safety_max = pos + half_dims

                if obj_id in self._objects:
                    # Update existing object
                    existing = self._objects[obj_id]
                    existing.position_mm = self._smooth_position(
                        existing.position_mm, pos, alpha=0.3
                    )
                    existing.dimensions_mm = self._smooth_position(
                        existing.dimensions_mm, dims, alpha=0.3
                    )
                    existing.confidence = est.confidence
                    existing.grade = est.grade
                    existing.category = category
                    existing.reach_status = reach_status
                    existing.reach_distance_mm = reach_dist
                    existing.graspable = est.graspable
                    existing.safety_min_mm = safety_min
                    existing.safety_max_mm = safety_max
                    existing.last_seen = now
                    existing.observation_count += 1
                    existing.stable = existing.observation_count >= 3
                else:
                    # Create new object
                    self._objects[obj_id] = WorldObject(
                        object_id=obj_id,
                        label=est.label,
                        position_mm=pos,
                        dimensions_mm=dims,
                        confidence=est.confidence,
                        grade=est.grade,
                        category=category,
                        reach_status=reach_status,
                        reach_distance_mm=reach_dist,
                        graspable=est.graspable,
                        safety_min_mm=safety_min,
                        safety_max_mm=safety_max,
                        first_seen=now,
                        last_seen=now,
                        observation_count=1,
                        stable=False,
                    )

                self._total_observations += 1

            # Mark stale objects
            self._mark_stale()

        elapsed_ms = (time.monotonic() - t0) * 1000
        n_objs = len(self._objects)
        logger.info(
            "World model updated: scan=%d, objects=%d, elapsed=%.1fms",
            self._scan_count, n_objs, elapsed_ms,
        )

    def _get_or_create_id(self, label: str) -> str:
        """Get existing ID for label or create a new one."""
        # Check for existing object with this label
        for obj_id, obj in self._objects.items():
            if obj.label == label:
                return obj_id

        # Create new
        count = self._label_counters.get(label, 0)
        self._label_counters[label] = count + 1
        return f"{label}_{count}"

    def _estimate_position_from_bbox(self, est: DimensionEstimate) -> np.ndarray:
        """Rough position estimate when no 3D position is provided.

        Uses bounding box center in image space mapped to approximate
        workspace coordinates. This is intentionally rough — the world
        model will refine as more data arrives.
        """
        # Default: place in front of the arm at a mid-range distance
        x = 250.0  # forward
        y = 0.0    # centered
        z = est.height_mm / 2.0  # half-height above table

        if est.bbox_cam1 is not None:
            # Overhead view: map normalized bbox center to workspace
            bx, by, bw, bh = est.bbox_cam1
            # Assume 1920x1080 image covers -400 to +400mm workspace
            cx_norm = (bx + bw / 2.0) / 1920.0
            cy_norm = (by + bh / 2.0) / 1080.0
            x = -400.0 + cx_norm * 800.0
            y = -400.0 + cy_norm * 800.0

        return np.array([x, y, z])

    def _classify_reach(self, xy_distance_mm: float) -> ReachStatus:
        """Classify reachability based on XY distance from arm base."""
        if xy_distance_mm < self.arm_min_reach:
            return ReachStatus.TOO_CLOSE
        elif xy_distance_mm > self.arm_max_reach:
            return ReachStatus.OUT_OF_RANGE
        elif xy_distance_mm > self.arm_max_reach * 0.85:
            return ReachStatus.MARGINAL
        else:
            return ReachStatus.REACHABLE

    def _classify_category(
        self, est: DimensionEstimate, reach: ReachStatus
    ) -> ObjectCategory:
        """Classify whether an object is a target, obstacle, or boundary."""
        if est.graspable and reach in (ReachStatus.REACHABLE, ReachStatus.MARGINAL):
            return ObjectCategory.TARGET
        elif est.width_mm > 200 or est.depth_mm > 200:
            return ObjectCategory.BOUNDARY
        else:
            return ObjectCategory.OBSTACLE

    def _smooth_position(
        self, old: np.ndarray, new: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """Exponential moving average for position smoothing."""
        return old * (1 - alpha) + new * alpha

    def _mark_stale(self):
        """Downgrade objects not seen in recent scans."""
        to_remove = []
        for obj_id, obj in self._objects.items():
            scans_since_seen = self._scan_count - int(
                obj.observation_count
            )
            # If object hasn't been observed recently relative to total scans
            if self._scan_count > self.STALE_AFTER_SCANS:
                # Check if last seen was too long ago
                age = time.monotonic() - obj.last_seen
                if age > 30.0:  # 30 seconds without observation
                    obj.stable = False
                    obj.confidence *= 0.5
                    if obj.confidence < 0.05:
                        to_remove.append(obj_id)

        for obj_id in to_remove:
            logger.info("Removing stale object: %s", obj_id)
            del self._objects[obj_id]

    # ------------------------------------------------------------------
    # Querying the model
    # ------------------------------------------------------------------

    def snapshot(self) -> WorldModelSnapshot:
        """Take a thread-safe snapshot of the current world model."""
        t0 = time.monotonic()
        with self._lock:
            objects = list(self._objects.values())
            scan_count = self._scan_count
            total_obs = self._total_observations

        reachable_targets = sum(
            1 for o in objects
            if o.category == ObjectCategory.TARGET
            and o.reach_status == ReachStatus.REACHABLE
        )
        obstacles = sum(
            1 for o in objects
            if o.category in (ObjectCategory.OBSTACLE, ObjectCategory.BOUNDARY)
        )

        # Free zone: estimate percentage of workspace not blocked
        free_pct = self._estimate_free_zone(objects)

        # Aggregate model confidence
        if objects:
            model_conf = sum(o.confidence for o in objects) / len(objects)
            stable_count = sum(1 for o in objects if o.stable)
            stability_factor = stable_count / len(objects)
            model_conf = model_conf * 0.6 + stability_factor * 0.4
        else:
            model_conf = 0.0

        # Grade the model itself
        grade_thresholds = {"A": 0.7, "B": 0.5, "C": 0.3, "D": 0.15, "F": 0.0}
        model_grade = "F"
        for g, thresh in grade_thresholds.items():
            if model_conf >= thresh:
                model_grade = g
                break

        build_ms = (time.monotonic() - t0) * 1000

        return WorldModelSnapshot(
            objects=objects,
            scan_count=scan_count,
            total_observations=total_obs,
            timestamp=time.monotonic(),
            build_time_ms=build_ms,
            reachable_targets=reachable_targets,
            obstacles_detected=obstacles,
            free_zone_pct=free_pct,
            model_confidence=model_conf,
            model_grade=model_grade,
        )

    def _estimate_free_zone(self, objects: list[WorldObject]) -> float:
        """Estimate what percentage of the reachable workspace is clear.

        Uses a simple grid sampling approach over the arm's reachable area.
        """
        if not objects:
            return 100.0

        # Sample points in the XY workspace plane
        n_samples = 100
        angles = np.linspace(0, 2 * math.pi, n_samples, endpoint=False)
        radii = np.linspace(self.arm_min_reach, self.arm_max_reach, 5)

        total = 0
        blocked = 0

        for r in radii:
            for theta in angles:
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                total += 1
                # Check if this point is inside any object's safety zone
                for obj in objects:
                    if (obj.safety_min_mm[0] <= x <= obj.safety_max_mm[0] and
                            obj.safety_min_mm[1] <= y <= obj.safety_max_mm[1]):
                        blocked += 1
                        break

        if total == 0:
            return 100.0
        return 100.0 * (1 - blocked / total)

    def get_reachable_targets(self) -> list[WorldObject]:
        """Get all objects that are targets within arm reach, sorted by confidence."""
        with self._lock:
            targets = [
                o for o in self._objects.values()
                if o.category == ObjectCategory.TARGET
                and o.reach_status in (ReachStatus.REACHABLE, ReachStatus.MARGINAL)
            ]
        targets.sort(key=lambda o: o.confidence, reverse=True)
        return targets

    def get_obstacles(self) -> list[WorldObject]:
        """Get all detected obstacles and boundaries."""
        with self._lock:
            return [
                o for o in self._objects.values()
                if o.category in (ObjectCategory.OBSTACLE, ObjectCategory.BOUNDARY)
            ]

    def check_collision(
        self, point_mm: np.ndarray, radius_mm: float = 30.0
    ) -> list[WorldObject]:
        """Check if a point would collide with any known object's safety zone.

        Args:
            point_mm: (3,) XYZ point to check.
            radius_mm: Clearance radius around the point.

        Returns:
            List of WorldObjects whose safety zones are violated.
        """
        collisions = []
        with self._lock:
            for obj in self._objects.values():
                # Inflate safety zone by the check radius
                if (point_mm[0] >= obj.safety_min_mm[0] - radius_mm and
                        point_mm[0] <= obj.safety_max_mm[0] + radius_mm and
                        point_mm[1] >= obj.safety_min_mm[1] - radius_mm and
                        point_mm[1] <= obj.safety_max_mm[1] + radius_mm and
                        point_mm[2] >= obj.safety_min_mm[2] - radius_mm and
                        point_mm[2] <= obj.safety_max_mm[2] + radius_mm):
                    collisions.append(obj)
        return collisions

    def check_path_collisions(
        self, path_points_mm: list[np.ndarray], radius_mm: float = 30.0
    ) -> list[dict]:
        """Check a sequence of points (trajectory) for collisions.

        Returns a list of collision events with the point index, position,
        and which object was hit.
        """
        events = []
        for i, pt in enumerate(path_points_mm):
            hits = self.check_collision(pt, radius_mm)
            for obj in hits:
                events.append({
                    "path_index": i,
                    "point_mm": pt.tolist(),
                    "object_id": obj.object_id,
                    "object_label": obj.label,
                    "object_category": obj.category.value,
                })
        return events

    def clear(self):
        """Reset the world model."""
        with self._lock:
            self._objects.clear()
            self._scan_count = 0
            self._total_observations = 0
            self._label_counters.clear()
        logger.info("World model cleared")
