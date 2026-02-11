"""
Scene Analyzer — Structured scene understanding from dual independent cameras.

Analyzes frames from cam0 (front/side) and cam1 (overhead) independently,
runs object detection on each, cross-references detections by color label,
and produces a SceneDescription with workspace positions derived from
camera calibration data.

Camera layout:
  cam0 (front/side): provides object color, height (Z) from vertical position
  cam1 (overhead):   provides object X/Y position on workspace table

No stereo pair or stereo matching required.
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
from .calibration import CameraCalibration

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
                    "source": o.source,
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
            "cameras_used": self.cameras_used,
            "summary": self.summary,
        }


class SceneAnalyzer:
    """Analyzes independent camera feeds to produce structured scene descriptions.

    Uses ObjectDetector on each camera independently, then merges detections
    by color label. Overhead camera (cam1) provides the primary spatial layout;
    front camera (cam0) adds height information when available.

    Can also work with a single camera frame for simpler setups.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
    ):
        self.detector = detector or ObjectDetector(min_area=300)
        self.cal_cam0 = cal_cam0
        self.cal_cam1 = cal_cam1

    def set_calibration(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
    ):
        """Update camera calibrations."""
        if cal_cam0 is not None:
            self.cal_cam0 = cal_cam0
        if cal_cam1 is not None:
            self.cal_cam1 = cal_cam1

    def analyze(
        self,
        image: np.ndarray,
        cam0_frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> SceneDescription:
        """Analyze camera frame(s) and produce a structured scene description.

        Args:
            image: Primary frame — overhead (cam1) preferred, or any single cam.
            cam0_frame: Optional front camera frame for height estimation.
            timestamp: Frame timestamp.

        Returns:
            SceneDescription with detected objects and spatial relationships.
        """
        h, w = image.shape[:2]
        cameras_used = ["cam1"]

        # Detect objects in the primary (overhead) frame
        detections = self.detector.detect(image)

        # If front camera frame provided, detect there too for cross-referencing
        cam0_detections: list[DetectedObject] = []
        if cam0_frame is not None:
            cam0_detections = self.detector.detect(cam0_frame)
            cameras_used.append("cam0")

        # Build a lookup of cam0 detections by color for height cross-reference
        cam0_by_color: dict[str, list[DetectedObject]] = {}
        for det in cam0_detections:
            cam0_by_color.setdefault(det.label, []).append(det)

        # Convert detections to scene objects with spatial context
        scene_objects = []
        for det in detections:
            cx, cy = det.centroid_2d
            norm_x = cx / max(w, 1)
            norm_y = cy / max(h, 1)
            region = self._classify_region(norm_x, norm_y)

            # Try to get workspace 3D position from calibration
            centroid_3d = None
            depth_mm = 0.0
            source = "cam1"

            if self.cal_cam1 is not None and self.cal_cam1.cam_to_workspace is not None:
                ws_pos = self.cal_cam1.pixel_to_workspace(float(cx), float(cy), known_z=0.0)
                if ws_pos is not None:
                    # X/Y from overhead, Z=0 (table surface) by default
                    z_mm = 0.0
                    # Cross-reference with front camera for height
                    if det.label in cam0_by_color and cam0_by_color[det.label]:
                        cam0_det = cam0_by_color[det.label][0]
                        z_mm = self._estimate_height_from_front(cam0_det)
                        source = "both"
                    centroid_3d = (float(ws_pos[0]), float(ws_pos[1]), z_mm)
                    depth_mm = float(math.sqrt(ws_pos[0] ** 2 + ws_pos[1] ** 2))
            elif det.label in cam0_by_color:
                source = "both"

            scene_obj = SceneObject(
                label=det.label,
                color=det.label,
                centroid_2d=det.centroid_2d,
                centroid_3d=centroid_3d,
                bbox=det.bbox,
                area=det.area,
                depth_mm=depth_mm,
                confidence=det.confidence,
                normalized_x=norm_x,
                normalized_y=norm_y,
                region=region,
                source=source,
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
            cameras_used=cameras_used,
        )

    def _estimate_height_from_front(self, cam0_det: DetectedObject) -> float:
        """Estimate object height (Z) from front camera vertical position.

        Objects lower in the front camera frame are on the table (Z=0).
        Objects higher in the frame are taller. Uses cam0 calibration
        or falls back to a simple linear estimate.
        """
        if self.cal_cam0 is not None:
            _, cy = cam0_det.centroid_2d
            _, bbox_y, _, bbox_h = cam0_det.bbox
            # Bottom of bounding box is roughly where the object meets the table
            # Height is estimated from bbox size scaled by mm_per_pixel
            img_h = self.cal_cam0.image_size[1]
            # Rough: objects near bottom of frame are at table level
            # Height proportional to how far centroid is above bbox bottom
            bbox_bottom = bbox_y + bbox_h
            table_fraction = bbox_bottom / max(img_h, 1)
            # Very rough estimate: ~300mm visible height in frame
            height_mm = max(0.0, (1.0 - table_fraction) * 300.0)
            return height_mm

        return 0.0  # no calibration, assume table level

    def _classify_region(self, norm_x: float, norm_y: float) -> str:
        """Classify a normalized position into a named region."""
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

    def _compute_relationships(self, objects: list[SceneObject]) -> list[ObjectRelationship]:
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
                        relationships.append(ObjectRelationship(id_a, SpatialRelation.ABOVE, id_b))
                    else:
                        relationships.append(ObjectRelationship(id_a, SpatialRelation.BELOW, id_b))

                # Workspace distance relationship (if both have 3D positions)
                if obj_a.centroid_3d is not None and obj_b.centroid_3d is not None:
                    dist_ws = math.sqrt(
                        (obj_a.centroid_3d[0] - obj_b.centroid_3d[0]) ** 2
                        + (obj_a.centroid_3d[1] - obj_b.centroid_3d[1]) ** 2
                    )
                    if dist_ws > 100:  # >100mm apart in workspace
                        # Determine front/behind from Y axis in workspace
                        dy_ws = obj_b.centroid_3d[1] - obj_a.centroid_3d[1]
                        if abs(dy_ws) > 50:
                            if dy_ws > 0:
                                relationships.append(
                                    ObjectRelationship(id_a, SpatialRelation.IN_FRONT_OF, id_b)
                                )
                            else:
                                relationships.append(
                                    ObjectRelationship(id_a, SpatialRelation.BEHIND, id_b)
                                )

                # Proximity (in normalized image space)
                dist_2d = math.sqrt(
                    (obj_a.normalized_x - obj_b.normalized_x) ** 2
                    + (obj_a.normalized_y - obj_b.normalized_y) ** 2
                )
                if dist_2d < 0.15:
                    relationships.append(ObjectRelationship(id_a, SpatialRelation.NEAR, id_b))

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
            pos_str = ""
            if obj.centroid_3d is not None:
                x, y, z = obj.centroid_3d
                pos_str = f" at ({x:.0f}, {y:.0f}, {z:.0f})mm"
            parts.append(
                f"  {self._object_id(obj, i)}: {obj.region}, " f"{obj.size_category}{pos_str}"
            )

        # Key relationships
        for rel in relationships[:5]:
            parts.append(f"  {rel.subject} is {rel.relation.value.replace('_', ' ')} {rel.target}")

        return "\n".join(parts)

    def annotate_frame(self, image: np.ndarray, scene: SceneDescription) -> np.ndarray:
        """Draw scene analysis annotations on a frame (returns copy)."""
        vis = image.copy()

        for i, obj in enumerate(scene.objects):
            x, y, w, h = obj.bbox
            color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # Centroid
            cv2.circle(vis, obj.centroid_2d, 5, (0, 0, 255), -1)

            # Label with region and workspace position
            label = f"{obj.color} [{obj.region}]"
            if obj.centroid_3d is not None:
                label += f" ({obj.centroid_3d[0]:.0f},{obj.centroid_3d[1]:.0f})mm"
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
