"""
Dual-Camera Arm Tracker — Independent camera views for 3D localization.

Uses two independent cameras (no stereo pair):
- cam0 (front/side): Detects objects, estimates height (Z) from vertical position.
- cam1 (overhead): Detects objects, gives X/Y workspace position.

Cross-references detections by label and workspace geometry to produce
3D positions without stereo triangulation.
"""

import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraCalibration
from .object_detection import ObjectDetector, DetectedObject, COLOR_PRESETS

logger = logging.getLogger("th3cl4w.vision.arm_tracker")


@dataclass
class TrackedObject:
    """An object tracked in 3D workspace coordinates from dual cameras."""

    label: str
    position_mm: np.ndarray  # (3,) XYZ in workspace/arm-base frame (mm)
    confidence: float  # 0-1
    bbox_cam0: Optional[tuple[int, int, int, int]] = None  # bbox in front camera
    bbox_cam1: Optional[tuple[int, int, int, int]] = None  # bbox in overhead camera
    centroid_cam0: Optional[tuple[int, int]] = None  # pixel centroid in front camera
    centroid_cam1: Optional[tuple[int, int]] = None  # pixel centroid in overhead camera
    source: str = "both"  # "cam0", "cam1", or "both"
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ArmTrackingResult:
    """Result of a dual-camera tracking pass."""

    objects: list[TrackedObject]
    cam0_frame: Optional[np.ndarray]  # annotated front camera frame
    cam1_frame: Optional[np.ndarray]  # annotated overhead camera frame
    elapsed_ms: float
    status: str  # "ok", "no_calibration", "no_frames", "error"
    message: str = ""


class DualCameraArmTracker:
    """Track arm and objects using two independent camera views.

    Architecture:
    - Overhead camera (cam1) gives X/Y position on the workspace plane.
    - Front camera (cam0) gives rough Z (height) from vertical pixel position.
    - Objects matched between views by label and spatial consistency.
    """

    # Default: arm base is at workspace origin.
    # cam0 looks from the front, cam1 looks from above.
    DEFAULT_TABLE_HEIGHT_MM = 0.0  # table surface = Z=0

    def __init__(
        self,
        cal_cam0: Optional[CameraCalibration] = None,
        cal_cam1: Optional[CameraCalibration] = None,
        table_height_mm: float = 0.0,
        target_labels: Optional[list[str]] = None,
    ):
        self.cal_cam0 = cal_cam0
        self.cal_cam1 = cal_cam1
        self.table_height_mm = table_height_mm

        self._build_detectors(target_labels)

        self._lock = threading.Lock()
        self._last_result: Optional[ArmTrackingResult] = None
        self._tracking_count = 0

    def _build_detectors(self, target_labels: Optional[list[str]] = None):
        """Set up color-based object detectors."""
        if target_labels is None:
            target_labels = ["red", "blue", "green"]

        self._detectors: dict[str, ObjectDetector] = {}

        from .object_detection import ColorRange

        # Red Bull can
        redbull_colors = [
            ColorRange("redbull", np.array([0, 120, 80]), np.array([10, 255, 255])),
            ColorRange("redbull", np.array([160, 120, 80]), np.array([180, 255, 255])),
        ]
        self._detectors["redbull"] = ObjectDetector(
            color_ranges=redbull_colors, min_area=800, max_area=150000, morph_iterations=2,
        )
        self._detectors["red"] = ObjectDetector(
            color_ranges=[COLOR_PRESETS["red_low"], COLOR_PRESETS["red_high"]],
            min_area=500, max_area=150000,
        )
        self._detectors["blue"] = ObjectDetector(
            color_ranges=[COLOR_PRESETS["blue"]], min_area=500, max_area=150000,
        )
        self._detectors["green"] = ObjectDetector(
            color_ranges=[COLOR_PRESETS["green"]], min_area=500, max_area=150000,
        )
        self._detectors["all"] = ObjectDetector(min_area=500)
        self._active_labels = target_labels

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

    def _estimate_xy_from_overhead(
        self, det: DetectedObject
    ) -> Optional[np.ndarray]:
        """Get X/Y workspace position from overhead camera detection."""
        if self.cal_cam1 is None or self.cal_cam1.cam_to_workspace is None:
            return None
        cx, cy = det.centroid_2d
        pos = self.cal_cam1.pixel_to_workspace(
            float(cx), float(cy), known_z=self.table_height_mm
        )
        return pos

    def _estimate_z_from_front(
        self, det: DetectedObject, default_depth_mm: float = 400.0
    ) -> float:
        """Estimate object height (Z) from front camera vertical position.

        Objects higher in the image (smaller v) are higher in the workspace.
        Uses the calibrated front camera if available, otherwise heuristic.
        """
        if self.cal_cam0 is not None and self.cal_cam0.cam_to_workspace is not None:
            cx, cy = det.centroid_2d
            # Project onto a vertical plane at an assumed distance
            pos = self.cal_cam0.pixel_to_workspace(
                float(cx), float(cy), known_z=self.table_height_mm
            )
            if pos is not None:
                return float(pos[2])

        # Heuristic fallback: linear mapping from pixel row to height
        # Bottom of image (v=480) = table surface (Z=0)
        # Top of image (v=0) = ~300mm above table
        _, cy = det.centroid_2d
        image_h = self.cal_cam0.image_size[1] if self.cal_cam0 else 480
        z_mm = (1.0 - cy / image_h) * 300.0
        return max(0.0, z_mm)

    def _match_detections(
        self,
        dets_cam0: list[DetectedObject],
        dets_cam1: list[DetectedObject],
    ) -> list[TrackedObject]:
        """Match detections between cameras and produce 3D tracked objects.

        Strategy:
        1. Use cam1 (overhead) for X/Y position on table plane.
        2. Use cam0 (front) for Z (height) estimation.
        3. Match by label. If only one camera sees it, use partial info.
        """
        tracked: list[TrackedObject] = []
        used_cam0: set[int] = set()
        used_cam1: set[int] = set()

        # Try to match by label
        for i, d1 in enumerate(dets_cam1):
            best_j = None
            best_score = float("inf")
            for j, d0 in enumerate(dets_cam0):
                if j in used_cam0:
                    continue
                if d0.label != d1.label:
                    continue
                # Simple area-ratio score (both should see similar-sized object)
                ratio = min(d0.area, d1.area) / max(d0.area, d1.area)
                score = 1.0 - ratio
                if score < best_score:
                    best_score = score
                    best_j = j

            xy_pos = self._estimate_xy_from_overhead(d1)
            if xy_pos is None:
                xy_pos = np.array([0.0, 0.0, 0.0])

            if best_j is not None:
                d0 = dets_cam0[best_j]
                z = self._estimate_z_from_front(d0)
                used_cam0.add(best_j)
                used_cam1.add(i)
                pos = np.array([xy_pos[0], xy_pos[1], z])
                tracked.append(TrackedObject(
                    label=d1.label,
                    position_mm=pos,
                    confidence=min(1.0, (d0.confidence + d1.confidence) / 2 * 1.3),
                    bbox_cam0=d0.bbox,
                    bbox_cam1=d1.bbox,
                    centroid_cam0=d0.centroid_2d,
                    centroid_cam1=d1.centroid_2d,
                    source="both",
                ))
            else:
                # Only overhead camera saw it — assume on table
                used_cam1.add(i)
                pos = np.array([xy_pos[0], xy_pos[1], self.table_height_mm])
                tracked.append(TrackedObject(
                    label=d1.label,
                    position_mm=pos,
                    confidence=d1.confidence * 0.7,
                    bbox_cam1=d1.bbox,
                    centroid_cam1=d1.centroid_2d,
                    source="cam1",
                ))

        # Objects only seen in front camera
        for j, d0 in enumerate(dets_cam0):
            if j in used_cam0:
                continue
            z = self._estimate_z_from_front(d0)
            pos = np.array([0.0, 0.0, z])  # X/Y unknown
            tracked.append(TrackedObject(
                label=d0.label,
                position_mm=pos,
                confidence=d0.confidence * 0.5,
                bbox_cam0=d0.bbox,
                centroid_cam0=d0.centroid_2d,
                source="cam0",
            ))

        return tracked

    def track(
        self,
        cam0_frame: np.ndarray,
        cam1_frame: np.ndarray,
        target_label: str = "red",
        annotate: bool = True,
    ) -> ArmTrackingResult:
        """Run a full tracking pass on both camera frames.

        Args:
            cam0_frame: BGR image from front camera.
            cam1_frame: BGR image from overhead camera.
            target_label: Which detector to use.
            annotate: Whether to draw annotations.

        Returns:
            ArmTrackingResult with detected and localized objects.
        """
        t0 = time.monotonic()

        detector = self._detectors.get(target_label, self._detectors.get("all"))
        if detector is None:
            detector = self._detectors["all"]

        # Detect independently in each camera
        dets_cam0 = detector.detect(cam0_frame)
        dets_cam1 = detector.detect(cam1_frame)

        # Match and fuse
        tracked = self._match_detections(dets_cam0, dets_cam1)

        # Annotate
        vis0 = cam0_frame
        vis1 = cam1_frame
        if annotate and tracked:
            vis0 = self._annotate_frame(cam0_frame.copy(), tracked, "cam0")
            vis1 = self._annotate_frame(cam1_frame.copy(), tracked, "cam1")

        elapsed_ms = (time.monotonic() - t0) * 1000

        result = ArmTrackingResult(
            objects=tracked,
            cam0_frame=vis0 if annotate else None,
            cam1_frame=vis1 if annotate else None,
            elapsed_ms=round(elapsed_ms, 1),
            status="ok",
            message=f"Tracked {len(tracked)} object(s)",
        )

        with self._lock:
            self._last_result = result
            self._tracking_count += 1

        return result

    def _annotate_frame(
        self, frame: np.ndarray, objects: list[TrackedObject], cam: str,
    ) -> np.ndarray:
        """Draw tracking annotations on a camera frame."""
        for obj in objects:
            bbox = obj.bbox_cam0 if cam == "cam0" else obj.bbox_cam1
            centroid = obj.centroid_cam0 if cam == "cam0" else obj.centroid_cam1
            if bbox is None or centroid is None:
                continue

            x, y, w, h = bbox
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.drawMarker(frame, centroid, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

            pos = obj.position_mm
            label = f"{obj.label} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})mm"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
            info = f"src={obj.source} conf={obj.confidence:.2f}"
            cv2.putText(
                frame, info, (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1, cv2.LINE_AA,
            )

        cv2.putText(
            frame, cam.upper(), (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return frame

    def get_last_result(self) -> Optional[ArmTrackingResult]:
        with self._lock:
            return self._last_result

    @property
    def tracking_count(self) -> int:
        with self._lock:
            return self._tracking_count
