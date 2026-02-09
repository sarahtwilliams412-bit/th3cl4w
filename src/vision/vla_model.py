"""
Video Language Action Model (VLA) — Continuous ASCII scene analysis and 3D modeling.

Runs a continuous loop that:
1. Receives ASCII frames from the camera pipeline
2. Identifies objects in the ASCII representation
3. Uses the ASCII character grid as a measurement coordinate system
4. Estimates object dimensions from ASCII cell occupancy
5. Builds digital 3D representations of detected objects
6. Identifies reachable objects around the arm
7. Feeds results to the digital twin for Factory 3D integration

The VLA can operate in two modes:
  - Vision model mode: sends ASCII frames to a vision-language model (Gemini, etc.)
  - LLM text mode: sends ASCII text to a text-only LLM for analysis

The ASCII grid serves as a natural coordinate system: each character position
maps to a physical region in the camera's field of view, enabling measurement
through character counting.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .camera_pipeline import AsciiFrame, StereoAsciiFrame

logger = logging.getLogger("th3cl4w.vision.vla_model")

# Arm workspace constants (Unitree D1)
ARM_MAX_REACH_MM = 550.0
ARM_MIN_REACH_MM = 80.0


class VLAMode(Enum):
    """Operating mode for the VLA model."""
    VISION = "vision"    # send frames to a vision-language model
    TEXT = "text"        # send ASCII text to a text LLM
    LOCAL = "local"      # local rule-based analysis (no external API)


class ObjectShape(Enum):
    """Rough shape classification from ASCII silhouette."""
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


@dataclass
class AsciiMeasurement:
    """A measurement taken from the ASCII grid.

    Each cell in the ASCII grid maps to a physical area. By counting
    occupied cells, we estimate physical dimensions.
    """
    label: str
    # ASCII grid bounds (col, row coordinates)
    grid_min_col: int = 0
    grid_max_col: int = 0
    grid_min_row: int = 0
    grid_max_row: int = 0

    # Number of occupied cells
    occupied_cells: int = 0
    total_cells: int = 0  # within bounding rect

    # Physical dimensions estimated from grid
    width_mm: float = 0.0
    height_mm: float = 0.0
    depth_mm: float = 0.0

    # Centroid in grid coordinates
    centroid_col: float = 0.0
    centroid_row: float = 0.0

    # Shape classification
    shape: ObjectShape = ObjectShape.UNKNOWN
    fill_ratio: float = 0.0  # fraction of bounding rect that's filled

    # Confidence in the measurement
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "grid_bounds": {
                "min_col": self.grid_min_col,
                "max_col": self.grid_max_col,
                "min_row": self.grid_min_row,
                "max_row": self.grid_max_row,
            },
            "occupied_cells": self.occupied_cells,
            "centroid": {
                "col": round(self.centroid_col, 1),
                "row": round(self.centroid_row, 1),
            },
            "dimensions_mm": {
                "width": round(self.width_mm, 1),
                "height": round(self.height_mm, 1),
                "depth": round(self.depth_mm, 1),
            },
            "shape": self.shape.value,
            "fill_ratio": round(self.fill_ratio, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DetectedObject3D:
    """A 3D digital representation of a detected object."""

    object_id: str
    label: str
    position_mm: np.ndarray     # (3,) XYZ in arm-base frame
    dimensions_mm: np.ndarray   # (3,) width, height, depth
    shape: ObjectShape
    confidence: float
    reachable: bool
    reach_distance_mm: float

    # ASCII measurement source
    measurement: Optional[AsciiMeasurement] = None

    # Camera source
    source_camera: str = "cam1"  # "cam0", "cam1", or "both"

    # Mesh data (for Factory 3D)
    mesh_vertices: Optional[list] = None
    mesh_faces: Optional[list] = None

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position_mm": [round(v, 1) for v in self.position_mm.tolist()],
            "dimensions_mm": [round(v, 1) for v in self.dimensions_mm.tolist()],
            "shape": self.shape.value,
            "confidence": round(self.confidence, 3),
            "reachable": self.reachable,
            "reach_distance_mm": round(self.reach_distance_mm, 1),
            "source_camera": self.source_camera,
            "has_mesh": self.mesh_vertices is not None,
        }


@dataclass
class VLAAnalysisResult:
    """Result of a single VLA analysis cycle."""

    timestamp: float
    frame_number: int
    objects: list[DetectedObject3D] = field(default_factory=list)
    measurements: list[AsciiMeasurement] = field(default_factory=list)
    reachable_objects: list[str] = field(default_factory=list)
    scene_description: str = ""
    analysis_time_ms: float = 0.0
    mode: str = "local"

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_number": self.frame_number,
            "objects": [o.to_dict() for o in self.objects],
            "measurements": [m.to_dict() for m in self.measurements],
            "reachable_objects": self.reachable_objects,
            "scene_description": self.scene_description,
            "analysis_time_ms": round(self.analysis_time_ms, 1),
            "mode": self.mode,
        }


# Callback type for VLA results
VLAResultCallback = Callable[[VLAAnalysisResult], None]


class VLAModel:
    """Continuous Video Language Action model for ASCII scene analysis.

    Runs a background loop that analyzes ASCII frames to:
    - Identify objects by density patterns in the ASCII character grid
    - Measure objects using grid cell counting
    - Classify object shapes from their ASCII silhouettes
    - Estimate 3D positions from dual-camera views
    - Generate simple mesh representations for Factory 3D
    - Track which objects are within the arm's reach envelope
    """

    # Density threshold: characters denser than this are considered "filled"
    # Based on CHARSET_STANDARD = " .:-=+*#%@"
    DENSE_CHARS = set("=+*#%@")
    MEDIUM_CHARS = set(":-")
    EMPTY_CHARS = set(" .")

    # Physical scale factors (mm per ASCII cell)
    # These map the ASCII grid to physical workspace dimensions.
    # Default assumes overhead camera covers ~800x500mm workspace at 120x40 chars.
    DEFAULT_MM_PER_COL = 6.67   # 800mm / 120 cols
    DEFAULT_MM_PER_ROW = 12.5   # 500mm / 40 rows

    def __init__(
        self,
        mode: VLAMode = VLAMode.LOCAL,
        analysis_fps: float = 2.0,
        mm_per_col: float = DEFAULT_MM_PER_COL,
        mm_per_row: float = DEFAULT_MM_PER_ROW,
        density_threshold: float = 0.3,
        min_object_cells: int = 8,
    ):
        self.mode = mode
        self.analysis_fps = analysis_fps
        self.mm_per_col = mm_per_col
        self.mm_per_row = mm_per_row
        self.density_threshold = density_threshold
        self.min_object_cells = min_object_cells

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._callbacks: list[VLAResultCallback] = []
        self._latest_result: Optional[VLAAnalysisResult] = None
        self._analysis_count = 0

        # Latest frame input (set by pipeline callback)
        self._pending_frame: Optional[StereoAsciiFrame] = None
        self._frame_event = threading.Event()

        # Object tracking across frames
        self._tracked_objects: dict[str, DetectedObject3D] = {}
        self._object_counter = 0

    def on_result(self, callback: VLAResultCallback):
        """Register a callback to receive analysis results."""
        self._callbacks.append(callback)

    def feed_frame(self, stereo: StereoAsciiFrame):
        """Feed a new stereo ASCII frame for analysis.

        Called by the camera pipeline's frame callback.
        """
        with self._lock:
            self._pending_frame = stereo
        self._frame_event.set()

    def start(self):
        """Start the continuous analysis loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        logger.info("VLA model started in %s mode @ %.1f fps", self.mode.value, self.analysis_fps)

    def stop(self):
        """Stop the analysis loop."""
        self._running = False
        self._frame_event.set()  # wake up the thread
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("VLA model stopped after %d analyses", self._analysis_count)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest_result(self) -> Optional[VLAAnalysisResult]:
        """Get the most recent analysis result."""
        with self._lock:
            return self._latest_result

    def get_tracked_objects(self) -> list[DetectedObject3D]:
        """Get all currently tracked 3D objects."""
        with self._lock:
            return list(self._tracked_objects.values())

    def get_reachable_objects(self) -> list[DetectedObject3D]:
        """Get only objects within the arm's reach envelope."""
        with self._lock:
            return [o for o in self._tracked_objects.values() if o.reachable]

    def update_scale(self, mm_per_col: float, mm_per_row: float):
        """Update the physical scale mapping for the ASCII grid."""
        self.mm_per_col = mm_per_col
        self.mm_per_row = mm_per_row
        logger.info("VLA scale updated: %.2f mm/col, %.2f mm/row", mm_per_col, mm_per_row)

    # ------------------------------------------------------------------
    # Analysis loop
    # ------------------------------------------------------------------

    def _analysis_loop(self):
        """Main analysis loop running in a background thread."""
        interval = 1.0 / self.analysis_fps

        while self._running:
            # Wait for a new frame or timeout
            self._frame_event.wait(timeout=interval)
            self._frame_event.clear()

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            t0 = time.monotonic()
            try:
                result = self._analyze_frame(frame)
                elapsed_ms = (time.monotonic() - t0) * 1000
                result.analysis_time_ms = elapsed_ms

                with self._lock:
                    self._latest_result = result
                self._analysis_count += 1

                # Notify callbacks
                for cb in self._callbacks:
                    try:
                        cb(result)
                    except Exception as e:
                        logger.warning("VLA callback error: %s", e)

            except Exception as e:
                logger.error("VLA analysis error: %s", e)

    def _analyze_frame(self, stereo: StereoAsciiFrame) -> VLAAnalysisResult:
        """Analyze a stereo ASCII frame to detect and measure objects."""
        result = VLAAnalysisResult(
            timestamp=stereo.timestamp,
            frame_number=stereo.frame_number,
            mode=self.mode.value,
        )

        # Analyze the overhead camera (cam1) for X/Y workspace positions
        measurements_cam1 = []
        if stereo.cam1 is not None:
            measurements_cam1 = self._measure_from_ascii(stereo.cam1, "cam1")
            result.measurements.extend(measurements_cam1)

        # Analyze the front camera (cam0) for height (Z) information
        measurements_cam0 = []
        if stereo.cam0 is not None:
            measurements_cam0 = self._measure_from_ascii(stereo.cam0, "cam0")
            result.measurements.extend(measurements_cam0)

        # Fuse measurements into 3D objects
        objects_3d = self._fuse_measurements(
            measurements_cam1, measurements_cam0, stereo
        )
        result.objects = objects_3d

        # Track reachable objects
        result.reachable_objects = [o.object_id for o in objects_3d if o.reachable]

        # Update tracking state
        with self._lock:
            for obj in objects_3d:
                self._tracked_objects[obj.object_id] = obj

            # Expire stale objects (not seen in last 30 analyses)
            stale_ids = []
            for oid, obj in self._tracked_objects.items():
                if oid not in {o.object_id for o in objects_3d}:
                    stale_ids.append(oid)
            # Only remove if they've been missing for a while
            # (simple approach: remove after one cycle for now)
            if self._analysis_count % 30 == 0:
                for sid in stale_ids:
                    del self._tracked_objects[sid]

        # Build scene description
        result.scene_description = self._describe_scene(objects_3d)

        return result

    # ------------------------------------------------------------------
    # ASCII measurement
    # ------------------------------------------------------------------

    def _measure_from_ascii(
        self, ascii_frame: AsciiFrame, source: str
    ) -> list[AsciiMeasurement]:
        """Detect and measure objects from an ASCII frame using density analysis.

        Scans the ASCII grid for connected regions of dense characters,
        then measures each region's extent in grid cells.
        """
        grid = ascii_frame.grid
        if not grid:
            return []

        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        # Build a binary density map
        density_map = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(min(cols, len(grid[r]))):
                ch = grid[r][c]
                density_map[r, c] = ch in self.DENSE_CHARS

        # Find connected components via flood fill
        visited = np.zeros((rows, cols), dtype=bool)
        regions: list[list[tuple[int, int]]] = []

        for r in range(rows):
            for c in range(cols):
                if density_map[r, c] and not visited[r, c]:
                    region = self._flood_fill(density_map, visited, r, c, rows, cols)
                    if len(region) >= self.min_object_cells:
                        regions.append(region)

        # Convert regions to measurements
        measurements = []
        for i, region in enumerate(regions):
            m = self._region_to_measurement(region, f"obj_{source}_{i}", source)
            measurements.append(m)

        return measurements

    def _flood_fill(
        self,
        density_map: np.ndarray,
        visited: np.ndarray,
        start_r: int,
        start_c: int,
        rows: int,
        cols: int,
    ) -> list[tuple[int, int]]:
        """Flood fill to find a connected region of dense characters."""
        stack = [(start_r, start_c)]
        region = []

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r, c] or not density_map[r, c]:
                continue

            visited[r, c] = True
            region.append((r, c))

            # 4-connected neighbors
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

        return region

    def _region_to_measurement(
        self,
        region: list[tuple[int, int]],
        label: str,
        source: str,
    ) -> AsciiMeasurement:
        """Convert a connected region to an ASCII measurement."""
        rows = [r for r, c in region]
        cols = [c for r, c in region]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        grid_width = max_col - min_col + 1
        grid_height = max_row - min_row + 1
        total_cells = grid_width * grid_height
        occupied = len(region)
        fill_ratio = occupied / max(total_cells, 1)

        centroid_row = sum(rows) / len(rows)
        centroid_col = sum(cols) / len(cols)

        # Estimate physical dimensions
        if source == "cam1":  # overhead: width and depth
            width_mm = grid_width * self.mm_per_col
            depth_mm = grid_height * self.mm_per_row
            height_mm = 0.0  # unknown from overhead
        else:  # front: width and height
            width_mm = grid_width * self.mm_per_col
            height_mm = grid_height * self.mm_per_row
            depth_mm = width_mm  # assume roughly symmetric

        # Classify shape
        shape = self._classify_shape(fill_ratio, grid_width, grid_height)

        # Confidence based on size and fill
        confidence = min(1.0, 0.3 + 0.3 * fill_ratio + 0.4 * min(occupied / 50.0, 1.0))

        return AsciiMeasurement(
            label=label,
            grid_min_col=min_col,
            grid_max_col=max_col,
            grid_min_row=min_row,
            grid_max_row=max_row,
            occupied_cells=occupied,
            total_cells=total_cells,
            width_mm=width_mm,
            height_mm=height_mm,
            depth_mm=depth_mm,
            centroid_col=centroid_col,
            centroid_row=centroid_row,
            shape=shape,
            fill_ratio=fill_ratio,
            confidence=confidence,
        )

    def _classify_shape(
        self, fill_ratio: float, grid_w: int, grid_h: int
    ) -> ObjectShape:
        """Classify shape from fill ratio and aspect ratio."""
        aspect = grid_w / max(grid_h, 1)

        if fill_ratio > 0.85:
            return ObjectShape.RECTANGULAR
        elif fill_ratio > 0.7:
            if 0.7 < aspect < 1.4:
                return ObjectShape.CYLINDRICAL
            else:
                return ObjectShape.RECTANGULAR
        elif fill_ratio > 0.5:
            if 0.8 < aspect < 1.2:
                return ObjectShape.SPHERICAL
            else:
                return ObjectShape.IRREGULAR
        else:
            return ObjectShape.IRREGULAR

    # ------------------------------------------------------------------
    # 3D fusion
    # ------------------------------------------------------------------

    def _fuse_measurements(
        self,
        cam1_measurements: list[AsciiMeasurement],
        cam0_measurements: list[AsciiMeasurement],
        stereo: StereoAsciiFrame,
    ) -> list[DetectedObject3D]:
        """Fuse overhead and front camera measurements into 3D objects.

        Overhead camera (cam1) provides X/Y positions.
        Front camera (cam0) provides height (Z) information.
        """
        objects_3d: list[DetectedObject3D] = []

        # Map cam0 measurements by rough column position for cross-reference
        cam0_by_col: dict[int, AsciiMeasurement] = {}
        for m in cam0_measurements:
            col_center = int(m.centroid_col)
            cam0_by_col[col_center] = m

        grid_w = stereo.cam1.grid_width if stereo.cam1 else 120
        grid_h = stereo.cam1.grid_height if stereo.cam1 else 40

        for m_cam1 in cam1_measurements:
            # Map grid centroid to workspace position
            # Overhead view: col maps to X, row maps to Y (arm workspace)
            norm_x = m_cam1.centroid_col / max(grid_w, 1)
            norm_y = m_cam1.centroid_row / max(grid_h, 1)

            # Map to workspace coordinates (mm)
            # Workspace centered at arm base, ranging [-400, 400] x [-300, 300]
            x_mm = (norm_x - 0.5) * 800.0
            y_mm = (norm_y - 0.5) * 600.0
            z_mm = 0.0  # default table surface

            # Cross-reference with front camera for height
            source = "cam1"
            height_mm = m_cam1.height_mm

            col_key = int(m_cam1.centroid_col)
            # Search nearby columns for a matching front-view object
            for dc in range(-5, 6):
                if (col_key + dc) in cam0_by_col:
                    m_cam0 = cam0_by_col[col_key + dc]
                    z_mm = m_cam0.height_mm / 2.0  # mid-height
                    height_mm = m_cam0.height_mm
                    source = "both"
                    del cam0_by_col[col_key + dc]
                    break

            # Compute reach distance
            reach_dist = math.sqrt(x_mm ** 2 + y_mm ** 2)
            reachable = ARM_MIN_REACH_MM <= reach_dist <= ARM_MAX_REACH_MM

            self._object_counter += 1
            obj_id = f"vla_obj_{self._object_counter}"

            # Generate simple box mesh for Factory 3D
            dims = np.array([m_cam1.width_mm, height_mm, m_cam1.depth_mm])
            vertices, faces = self._generate_box_mesh(
                np.array([x_mm, y_mm, z_mm]), dims
            )

            obj_3d = DetectedObject3D(
                object_id=obj_id,
                label=m_cam1.label,
                position_mm=np.array([x_mm, y_mm, z_mm]),
                dimensions_mm=dims,
                shape=m_cam1.shape,
                confidence=m_cam1.confidence,
                reachable=reachable,
                reach_distance_mm=reach_dist,
                measurement=m_cam1,
                source_camera=source,
                mesh_vertices=vertices,
                mesh_faces=faces,
            )
            objects_3d.append(obj_3d)

        # Handle cam0-only objects (no overhead match)
        for m_cam0 in cam0_by_col.values():
            norm_x = m_cam0.centroid_col / max(grid_w, 1)
            x_mm = (norm_x - 0.5) * 800.0
            y_mm = 250.0  # assume mid-range depth from front view
            z_mm = m_cam0.height_mm / 2.0

            reach_dist = math.sqrt(x_mm ** 2 + y_mm ** 2)
            reachable = ARM_MIN_REACH_MM <= reach_dist <= ARM_MAX_REACH_MM

            self._object_counter += 1
            obj_id = f"vla_obj_{self._object_counter}"

            dims = np.array([m_cam0.width_mm, m_cam0.height_mm, m_cam0.depth_mm])
            vertices, faces = self._generate_box_mesh(
                np.array([x_mm, y_mm, z_mm]), dims
            )

            obj_3d = DetectedObject3D(
                object_id=obj_id,
                label=m_cam0.label,
                position_mm=np.array([x_mm, y_mm, z_mm]),
                dimensions_mm=dims,
                shape=m_cam0.shape,
                confidence=m_cam0.confidence * 0.5,  # lower confidence for single-cam
                reachable=reachable,
                reach_distance_mm=reach_dist,
                measurement=m_cam0,
                source_camera="cam0",
                mesh_vertices=vertices,
                mesh_faces=faces,
            )
            objects_3d.append(obj_3d)

        return objects_3d

    def _generate_box_mesh(
        self, center: np.ndarray, dims: np.ndarray
    ) -> tuple[list, list]:
        """Generate a simple box mesh (8 vertices, 12 triangles).

        Suitable for sending to Factory 3D WebGL visualization.
        """
        half = dims / 2.0
        cx, cy, cz = center

        # 8 corners of the box
        vertices = [
            [cx - half[0], cy - half[1], cz - half[2]],
            [cx + half[0], cy - half[1], cz - half[2]],
            [cx + half[0], cy + half[1], cz - half[2]],
            [cx - half[0], cy + half[1], cz - half[2]],
            [cx - half[0], cy - half[1], cz + half[2]],
            [cx + half[0], cy - half[1], cz + half[2]],
            [cx + half[0], cy + half[1], cz + half[2]],
            [cx - half[0], cy + half[1], cz + half[2]],
        ]

        # 12 triangles (2 per face)
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ]

        return vertices, faces

    def _describe_scene(self, objects: list[DetectedObject3D]) -> str:
        """Build a human-readable scene description from 3D objects."""
        if not objects:
            return "No objects detected in the ASCII frame."

        reachable = [o for o in objects if o.reachable]
        unreachable = [o for o in objects if not o.reachable]

        parts = [f"Detected {len(objects)} objects in ASCII frame."]

        if reachable:
            parts.append(f"  {len(reachable)} within arm reach:")
            for obj in reachable:
                dims = obj.dimensions_mm
                parts.append(
                    f"    {obj.object_id}: {obj.shape.value} "
                    f"{dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}mm "
                    f"at ({obj.position_mm[0]:.0f},{obj.position_mm[1]:.0f},{obj.position_mm[2]:.0f})mm "
                    f"reach={obj.reach_distance_mm:.0f}mm"
                )

        if unreachable:
            parts.append(f"  {len(unreachable)} out of reach")

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get VLA model statistics."""
        with self._lock:
            tracked = len(self._tracked_objects)
            reachable = sum(1 for o in self._tracked_objects.values() if o.reachable)

        return {
            "running": self._running,
            "mode": self.mode.value,
            "analysis_count": self._analysis_count,
            "analysis_fps": self.analysis_fps,
            "tracked_objects": tracked,
            "reachable_objects": reachable,
            "scale": {
                "mm_per_col": round(self.mm_per_col, 2),
                "mm_per_row": round(self.mm_per_row, 2),
            },
        }
