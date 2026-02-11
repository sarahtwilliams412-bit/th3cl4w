"""Vision scene data types — extracted from src/vision/scene_analyzer.py.

Pure dataclasses for structured scene descriptions from camera analysis.
No OpenCV or NumPy dependencies — only stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SpatialRelation(Enum):
    """Spatial relationship between two objects."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"


@dataclass
class SceneObject:
    """An object in the scene with position and spatial context."""

    label: str
    color: str
    centroid_2d: tuple[int, int]  # primary camera pixel coords
    centroid_3d: Optional[tuple[float, float, float]]  # workspace XYZ (mm)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) from primary camera
    area: float
    depth_mm: float  # distance from arm base (X/Y distance on workspace)
    confidence: float
    # Relative position in the overhead frame (normalized 0-1)
    normalized_x: float  # 0=left, 1=right
    normalized_y: float  # 0=top, 1=bottom
    # Qualitative position on workspace
    region: str  # e.g. "center", "top-left", "bottom-right"
    # Which cameras detected this object
    source: str = "cam1"  # "cam0", "cam1", or "both"

    @property
    def size_category(self) -> str:
        """Categorize object by area."""
        if self.area < 2000:
            return "small"
        elif self.area < 10000:
            return "medium"
        else:
            return "large"

    @property
    def has_workspace_position(self) -> bool:
        return self.centroid_3d is not None


@dataclass
class ObjectRelationship:
    """A spatial relationship between two scene objects."""

    subject: str  # e.g. "red object #0"
    relation: SpatialRelation
    target: str  # e.g. "blue object #1"


@dataclass
class SceneDescription:
    """Complete structured description of a scene from camera input."""

    objects: list[SceneObject] = field(default_factory=list)
    relationships: list[ObjectRelationship] = field(default_factory=list)
    frame_width: int = 1920
    frame_height: int = 1080
    timestamp: float = 0.0
    summary: str = ""
    # Which cameras were used
    cameras_used: list[str] = field(default_factory=lambda: ["cam1"])

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def has_objects(self) -> bool:
        return len(self.objects) > 0

    def objects_by_color(self, color: str) -> list[SceneObject]:
        """Get all objects of a given color."""
        return [o for o in self.objects if o.color.lower() == color.lower()]

    def largest_object(self) -> Optional[SceneObject]:
        """Get the largest object by area."""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.area)

    def nearest_object(self) -> Optional[SceneObject]:
        """Get the object closest to the arm base (smallest workspace distance)."""
        with_pos = [o for o in self.objects if o.centroid_3d is not None]
        if not with_pos:
            return self.largest_object()
        return min(with_pos, key=lambda o: math.sqrt(o.centroid_3d[0] ** 2 + o.centroid_3d[1] ** 2))

    def leftmost_object(self) -> Optional[SceneObject]:
        """Get the leftmost object in the frame."""
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: o.normalized_x)

    def rightmost_object(self) -> Optional[SceneObject]:
        """Get the rightmost object in the frame."""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.normalized_x)
