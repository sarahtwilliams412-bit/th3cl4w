"""
Scene Analyzer — Structured scene understanding from camera feeds.

Captures a frame from the camera, runs object detection, computes spatial
relationships, and produces a SceneDescription that the VisionTaskPlanner
can reason about to build executable arm plans.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .object_detection import ObjectDetector, DetectedObject, ColorRange, COLOR_PRESETS

logger = logging.getLogger("th3cl4w.vision.scene_analyzer")


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
    centroid_2d: tuple[int, int]
    centroid_3d: Optional[tuple[float, float, float]]
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    depth_mm: float
    confidence: float
    # Relative position in the frame (normalized 0-1)
    normalized_x: float  # 0=left, 1=right
    normalized_y: float  # 0=top, 1=bottom
    # Qualitative position
    region: str  # e.g. "center", "top-left", "bottom-right"

    @property
    def size_category(self) -> str:
        """Categorize object by area."""
        if self.area < 2000:
            return "small"
        elif self.area < 10000:
            return "medium"
        else:
            return "large"


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
    frame_width: int = 640
    frame_height: int = 480
    timestamp: float = 0.0
    summary: str = ""

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
        """Get the nearest object (smallest depth)."""
        with_depth = [o for o in self.objects if o.depth_mm > 0]
        if not with_depth:
            return self.largest_object()
        return min(with_depth, key=lambda o: o.depth_mm)

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

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "object_count": self.object_count,
            "objects": [
                {
                    "label": o.label,
                    "color": o.color,
                    "centroid_2d": list(o.centroid_2d),
                    "centroid_3d": list(o.centroid_3d) if o.centroid_3d else None,
                    "bbox": list(o.bbox),
                    "area": round(o.area, 1),
                    "depth_mm": round(o.depth_mm, 1),
                    "confidence": round(o.confidence, 3),
                    "normalized_x": round(o.normalized_x, 3),
                    "normalized_y": round(o.normalized_y, 3),
                    "region": o.region,
                    "size_category": o.size_category,
                }
                for o in self.objects
            ],
            "relationships": [
                {
                    "subject": r.subject,
                    "relation": r.relation.value,
                    "target": r.target,
                }
                for r in self.relationships
            ],
            "frame_size": [self.frame_width, self.frame_height],
            "summary": self.summary,
        }


class SceneAnalyzer:
    """Analyzes camera frames to produce structured scene descriptions.

    Uses ObjectDetector for detection, then adds spatial reasoning
    on top: regions, relationships, and a human-readable summary.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        depth_threshold_near_mm: float = 300.0,
        depth_threshold_far_mm: float = 800.0,
    ):
        self.detector = detector or ObjectDetector(min_area=300)
        self.depth_near = depth_threshold_near_mm
        self.depth_far = depth_threshold_far_mm

    def analyze(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> SceneDescription:
        """Analyze an image and produce a structured scene description.

        Args:
            image: BGR image from camera.
            depth_map: Optional depth map (mm) for 3D positioning.
            Q: Optional disparity-to-depth matrix.
            timestamp: Frame timestamp.

        Returns:
            SceneDescription with detected objects and spatial relationships.
        """
        h, w = image.shape[:2]

        # Run object detection
        detections = self.detector.detect(image, depth_map=depth_map, Q=Q)

        # Convert detections to scene objects with spatial context
        scene_objects = []
        for det in detections:
            cx, cy = det.centroid_2d
            norm_x = cx / max(w, 1)
            norm_y = cy / max(h, 1)
            region = self._classify_region(norm_x, norm_y)

            scene_obj = SceneObject(
                label=det.label,
                color=det.label,  # color name from detector
                centroid_2d=det.centroid_2d,
                centroid_3d=det.centroid_3d,
                bbox=det.bbox,
                area=det.area,
                depth_mm=det.depth_mm,
                confidence=det.confidence,
                normalized_x=norm_x,
                normalized_y=norm_y,
                region=region,
            )
            scene_objects.append(scene_obj)

        # Compute spatial relationships
        relationships = self._compute_relationships(scene_objects)

        # Build summary
        summary = self._build_summary(scene_objects, relationships)

        return SceneDescription(
            objects=scene_objects,
            relationships=relationships,
            frame_width=w,
            frame_height=h,
            timestamp=timestamp,
            summary=summary,
        )

    def _classify_region(self, norm_x: float, norm_y: float) -> str:
        """Classify a normalized position into a named region."""
        # 3x3 grid
        if norm_x < 0.33:
            h_label = "left"
        elif norm_x > 0.66:
            h_label = "right"
        else:
            h_label = "center"

        if norm_y < 0.33:
            v_label = "top"
        elif norm_y > 0.66:
            v_label = "bottom"
        else:
            v_label = "center"

        if h_label == "center" and v_label == "center":
            return "center"
        elif v_label == "center":
            return h_label
        elif h_label == "center":
            return v_label
        else:
            return f"{v_label}-{h_label}"

    def _object_id(self, obj: SceneObject, index: int) -> str:
        """Create a readable identifier for an object."""
        return f"{obj.color} object #{index}"

    def _compute_relationships(
        self, objects: list[SceneObject]
    ) -> list[ObjectRelationship]:
        """Compute pairwise spatial relationships between objects."""
        relationships = []

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    continue

                id_a = self._object_id(obj_a, i)
                id_b = self._object_id(obj_b, j)

                # Horizontal relationship
                dx = obj_b.normalized_x - obj_a.normalized_x
                if abs(dx) > 0.1:
                    if dx > 0:
                        relationships.append(
                            ObjectRelationship(id_a, SpatialRelation.LEFT_OF, id_b)
                        )
                    else:
                        relationships.append(
                            ObjectRelationship(id_a, SpatialRelation.RIGHT_OF, id_b)
                        )

                # Vertical relationship
                dy = obj_b.normalized_y - obj_a.normalized_y
                if abs(dy) > 0.1:
                    if dy > 0:
                        relationships.append(
                            ObjectRelationship(id_a, SpatialRelation.ABOVE, id_b)
                        )
                    else:
                        relationships.append(
                            ObjectRelationship(id_a, SpatialRelation.BELOW, id_b)
                        )

                # Depth relationship (if available)
                if obj_a.depth_mm > 0 and obj_b.depth_mm > 0:
                    depth_diff = obj_b.depth_mm - obj_a.depth_mm
                    if abs(depth_diff) > 50:
                        if depth_diff > 0:
                            relationships.append(
                                ObjectRelationship(
                                    id_a, SpatialRelation.IN_FRONT_OF, id_b
                                )
                            )
                        else:
                            relationships.append(
                                ObjectRelationship(id_a, SpatialRelation.BEHIND, id_b)
                            )

                # Proximity
                dist_2d = math.sqrt(
                    (obj_a.normalized_x - obj_b.normalized_x) ** 2
                    + (obj_a.normalized_y - obj_b.normalized_y) ** 2
                )
                if dist_2d < 0.15:
                    relationships.append(
                        ObjectRelationship(id_a, SpatialRelation.NEAR, id_b)
                    )

        return relationships

    def _build_summary(
        self,
        objects: list[SceneObject],
        relationships: list[ObjectRelationship],
    ) -> str:
        """Build a human-readable scene summary."""
        if not objects:
            return "No objects detected in the scene."

        parts = []

        # Count by color
        color_counts: dict[str, int] = {}
        for obj in objects:
            color_counts[obj.color] = color_counts.get(obj.color, 0) + 1

        count_strs = []
        for color, count in color_counts.items():
            if count == 1:
                count_strs.append(f"1 {color} object")
            else:
                count_strs.append(f"{count} {color} objects")

        parts.append(f"Scene contains {', '.join(count_strs)}.")

        # Describe positions
        for i, obj in enumerate(objects):
            depth_str = ""
            if obj.depth_mm > 0:
                depth_str = f" at {obj.depth_mm:.0f}mm depth"
            parts.append(
                f"  {self._object_id(obj, i)}: {obj.region}, "
                f"{obj.size_category}{depth_str}"
            )

        # Key relationships
        for rel in relationships[:5]:
            parts.append(
                f"  {rel.subject} is {rel.relation.value.replace('_', ' ')} {rel.target}"
            )

        return "\n".join(parts)

    def annotate_frame(
        self, image: np.ndarray, scene: SceneDescription
    ) -> np.ndarray:
        """Draw scene analysis annotations on a frame (returns copy)."""
        vis = image.copy()

        for i, obj in enumerate(scene.objects):
            x, y, w, h = obj.bbox
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # Centroid
            cv2.circle(vis, obj.centroid_2d, 5, (0, 0, 255), -1)

            # Label with region info
            label = f"{obj.color} [{obj.region}]"
            if obj.depth_mm > 0:
                label += f" {obj.depth_mm:.0f}mm"
            cv2.putText(
                vis,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return vis
