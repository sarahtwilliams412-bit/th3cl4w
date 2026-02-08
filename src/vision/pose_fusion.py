"""
Pose fusion engine: combine FK predictions with visual joint detections.

Weighted fusion of forward-kinematics 3D positions with back-projected
visual detections from one or two cameras. Gracefully degrades when
cameras are offline.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .joint_detector import JointDetection

logger = logging.getLogger("th3cl4w.vision.pose_fusion")

# Disagreement threshold in meters (20mm)
DISAGREEMENT_THRESHOLD_M = 0.020


class FusionSource(Enum):
    FUSED = "fused"
    FK_ONLY = "fk_only"
    VISION_ONLY = "vision_only"


@dataclass
class FusionResult:
    """Result of fusing FK and visual joint positions."""
    positions: list[list[float]]          # [[x,y,z], ...] in meters
    confidence: list[float]               # per-joint confidence 0-1
    source: FusionSource
    per_joint_source: list[FusionSource] = field(default_factory=list)
    disagreements: list[float] = field(default_factory=list)  # per-joint FK-visual distance (m)
    timestamp: float = 0.0


@dataclass
class CameraCalib:
    """Minimal camera calibration for back-projection."""
    fx: float
    fy: float
    cx: float
    cy: float
    rvec: list[float]  # Rodrigues rotation vector
    tvec: list[float]  # Translation vector

    def rotation_matrix(self) -> np.ndarray:
        rv = np.array(self.rvec, dtype=np.float64)
        angle = np.linalg.norm(rv)
        if angle < 1e-8:
            return np.eye(3)
        k = rv / angle
        c, s = math.cos(angle), math.sin(angle)
        v = 1 - c
        return np.array([
            [k[0]*k[0]*v + c,      k[0]*k[1]*v - k[2]*s, k[0]*k[2]*v + k[1]*s],
            [k[1]*k[0]*v + k[2]*s, k[1]*k[1]*v + c,      k[1]*k[2]*v - k[0]*s],
            [k[2]*k[0]*v - k[1]*s, k[2]*k[1]*v + k[0]*s, k[2]*k[2]*v + c],
        ])


class PoseFusion:
    """Fuse FK predictions with visual joint detections from up to two cameras."""

    def __init__(
        self,
        fk_base_confidence: float = 0.5,
        visual_weight_scale: float = 1.0,
    ):
        self.fk_base_confidence = fk_base_confidence
        self.visual_weight_scale = visual_weight_scale
        self._last_result: Optional[FusionResult] = None
        self._quality_history: list[dict] = []  # rolling window

    def fuse(
        self,
        fk_positions_3d: list[list[float]],
        cam0_detections: Optional[list[JointDetection]] = None,
        cam1_detections: Optional[list[JointDetection]] = None,
        cam0_calib: Optional[CameraCalib] = None,
        cam1_calib: Optional[CameraCalib] = None,
    ) -> FusionResult:
        """Fuse FK 3D positions with visual detections.

        Back-projects visual pixel detections to 3D using FK-predicted depth,
        then blends with FK positions weighted by visual confidence.
        """
        n_joints = len(fk_positions_3d)
        has_cam0 = cam0_detections is not None and cam0_calib is not None and len(cam0_detections) > 0
        has_cam1 = cam1_detections is not None and cam1_calib is not None and len(cam1_detections) > 0

        # FK-only mode
        if not has_cam0 and not has_cam1:
            result = FusionResult(
                positions=[list(p) for p in fk_positions_3d],
                confidence=[self.fk_base_confidence] * n_joints,
                source=FusionSource.FK_ONLY,
                per_joint_source=[FusionSource.FK_ONLY] * n_joints,
                disagreements=[0.0] * n_joints,
                timestamp=time.time(),
            )
            self._last_result = result
            self._record_quality(result)
            return result

        # Build detection maps indexed by joint_index
        cam0_map: dict[int, JointDetection] = {}
        cam1_map: dict[int, JointDetection] = {}
        if has_cam0:
            for d in cam0_detections:
                cam0_map[d.joint_index] = d
        if has_cam1:
            for d in cam1_detections:
                cam1_map[d.joint_index] = d

        positions = []
        confidences = []
        sources = []
        disagreements = []

        for i in range(n_joints):
            fk_pos = np.array(fk_positions_3d[i], dtype=np.float64)

            # Back-project visual detections to 3D
            visual_positions = []
            visual_weights = []

            if i in cam0_map and cam0_calib is not None:
                det = cam0_map[i]
                bp = self._backproject_to_depth_plane(
                    det.pixel_pos, cam0_calib, fk_pos[2]  # intersect at FK Z
                )
                if bp is not None:
                    visual_positions.append(bp)
                    visual_weights.append(det.confidence * self.visual_weight_scale)

            if i in cam1_map and cam1_calib is not None:
                det = cam1_map[i]
                bp = self._backproject_to_height_plane(
                    det.pixel_pos, cam1_calib, fk_pos[2]  # project at FK height
                )
                if bp is not None:
                    visual_positions.append(bp)
                    visual_weights.append(det.confidence * self.visual_weight_scale)

            if not visual_positions:
                # No visual data for this joint
                positions.append(list(fk_pos))
                confidences.append(self.fk_base_confidence)
                sources.append(FusionSource.FK_ONLY)
                disagreements.append(0.0)
                continue

            # Average visual positions weighted by confidence
            total_w = sum(visual_weights)
            if total_w < 1e-8:
                visual_3d = visual_positions[0]
            else:
                visual_3d = np.zeros(3)
                for vp, vw in zip(visual_positions, visual_weights):
                    visual_3d += vp * (vw / total_w)

            # Compute disagreement
            dist = float(np.linalg.norm(visual_3d - fk_pos))
            disagreements.append(dist)

            if dist > DISAGREEMENT_THRESHOLD_M:
                logger.warning(
                    "Joint %d: FK-visual disagreement %.1fmm (threshold %.1fmm)",
                    i, dist * 1000, DISAGREEMENT_THRESHOLD_M * 1000,
                )

            # Weighted fusion: α depends on average visual confidence
            avg_visual_conf = total_w / len(visual_weights)
            # Higher visual confidence → lower α (more weight on visual)
            alpha = 1.0 - avg_visual_conf
            alpha = max(0.1, min(0.9, alpha))

            fused_pos = alpha * fk_pos + (1.0 - alpha) * visual_3d
            fused_conf = min(1.0, self.fk_base_confidence + avg_visual_conf * 0.5)

            positions.append(list(fused_pos))
            confidences.append(fused_conf)
            sources.append(FusionSource.FUSED)

        # Overall source
        unique_sources = set(sources)
        if len(unique_sources) == 1:
            overall = unique_sources.pop()
        elif FusionSource.FUSED in unique_sources:
            overall = FusionSource.FUSED
        else:
            overall = FusionSource.FK_ONLY

        result = FusionResult(
            positions=positions,
            confidence=confidences,
            source=overall,
            per_joint_source=sources,
            disagreements=disagreements,
            timestamp=time.time(),
        )
        self._last_result = result
        self._record_quality(result)
        return result

    def get_tracking_quality(self) -> dict:
        """Per-joint agreement metrics from recent fusion results."""
        if not self._quality_history:
            return {"available": False}

        n_joints = len(self._quality_history[0]["disagreements"]) if self._quality_history else 0
        joint_metrics = {}
        for j in range(n_joints):
            dists = [h["disagreements"][j] for h in self._quality_history if j < len(h["disagreements"])]
            if dists:
                joint_metrics[f"joint_{j}"] = {
                    "mean_disagreement_mm": round(float(np.mean(dists)) * 1000, 1),
                    "max_disagreement_mm": round(float(np.max(dists)) * 1000, 1),
                    "samples": len(dists),
                }

        fused_count = sum(
            1 for h in self._quality_history if h["source"] == FusionSource.FUSED.value
        )
        total = len(self._quality_history)

        return {
            "available": True,
            "fusion_rate": round(fused_count / total, 2) if total > 0 else 0.0,
            "total_samples": total,
            "joints": joint_metrics,
        }

    def _record_quality(self, result: FusionResult) -> None:
        self._quality_history.append({
            "disagreements": result.disagreements,
            "source": result.source.value,
            "timestamp": result.timestamp,
        })
        # Keep last 100 entries
        if len(self._quality_history) > 100:
            self._quality_history = self._quality_history[-100:]

    def _backproject_to_depth_plane(
        self,
        pixel: tuple[float, float],
        calib: CameraCalib,
        target_z: float,
    ) -> Optional[np.ndarray]:
        """Back-project a pixel through cam0 to intersect with a Z=target_z plane.

        cam0 is the front camera: pixel → ray in 3D → intersect at FK depth.
        """
        R = calib.rotation_matrix()
        t = np.array(calib.tvec, dtype=np.float64)
        R_inv = R.T
        t_inv = -R_inv @ t

        # Ray direction in camera frame
        dx = (pixel[0] - calib.cx) / calib.fx
        dy = (pixel[1] - calib.cy) / calib.fy
        ray_cam = np.array([dx, dy, 1.0])

        # Transform to world frame
        ray_world = R_inv @ ray_cam
        origin = t_inv

        # Intersect with Z = target_z
        if abs(ray_world[2]) < 1e-8:
            return None
        t_param = (target_z - origin[2]) / ray_world[2]
        if t_param < 0:
            return None
        return origin + t_param * ray_world

    def _backproject_to_height_plane(
        self,
        pixel: tuple[float, float],
        calib: CameraCalib,
        target_z: float,
    ) -> Optional[np.ndarray]:
        """Back-project a pixel through cam1 to intersect at FK-predicted height.

        cam1 is overhead: project onto workspace XY plane at FK-predicted Z.
        Same math as depth plane intersection but semantically for overhead cam.
        """
        # Same algorithm — intersect ray with Z=target_z plane
        return self._backproject_to_depth_plane(pixel, calib, target_z)
