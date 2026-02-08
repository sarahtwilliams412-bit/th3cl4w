"""Tests for the pose fusion engine."""

import math
import logging
import pytest
import numpy as np

from src.vision.pose_fusion import (
    PoseFusion,
    FusionResult,
    FusionSource,
    CameraCalib,
    DISAGREEMENT_THRESHOLD_M,
)
from src.vision.joint_detector import JointDetection, DetectionSource


def _make_calib(
    fx=500.0, fy=500.0, cx=320.0, cy=240.0,
    rvec=None, tvec=None,
) -> CameraCalib:
    """Create a simple camera calibration (identity rotation, offset translation).

    With identity R and tvec=[0,0,1]: world origin is at camera Z=1,
    so camera is at world (0,0,-1) looking along +Z. Points at Z=0.5
    are at camera Z=1.5 (in front of camera).
    """
    return CameraCalib(
        fx=fx, fy=fy, cx=cx, cy=cy,
        rvec=rvec or [0.0, 0.0, 0.0],
        tvec=tvec or [0.0, 0.0, 1.0],  # camera 1m in front of origin along Z
    )


def _make_detection(joint_index: int, px: float, py: float, confidence: float) -> JointDetection:
    return JointDetection(
        joint_index=joint_index,
        pixel_pos=(px, py),
        confidence=confidence,
        source=DetectionSource.GOLD,
    )


# --- FK-only mode ---

class TestFKOnly:
    def test_no_visual_data_returns_fk_positions(self):
        fusion = PoseFusion()
        fk = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.12], [0.2, 0.0, 0.3]]
        result = fusion.fuse(fk)
        assert result.source == FusionSource.FK_ONLY
        assert len(result.positions) == 3
        for i in range(3):
            assert result.positions[i] == pytest.approx(fk[i])
        assert all(s == FusionSource.FK_ONLY for s in result.per_joint_source)

    def test_empty_detections_returns_fk(self):
        fusion = PoseFusion()
        fk = [[0.0, 0.0, 0.1]]
        result = fusion.fuse(fk, cam0_detections=[], cam1_detections=[])
        assert result.source == FusionSource.FK_ONLY

    def test_detections_without_calib_returns_fk(self):
        fusion = PoseFusion()
        fk = [[0.0, 0.0, 0.1]]
        det = [_make_detection(0, 320, 240, 0.9)]
        # Detections but no calibration → FK only
        result = fusion.fuse(fk, cam0_detections=det, cam0_calib=None)
        assert result.source == FusionSource.FK_ONLY


# --- Fusion with visual data ---

class TestFusion:
    def test_high_confidence_shifts_toward_visual(self):
        """High-confidence visual detection should pull result away from FK."""
        fusion = PoseFusion()
        calib = _make_calib()

        # FK says joint is at [0, 0, 0.5]
        fk = [[0.0, 0.0, 0.5]]

        # Visual detection at camera center → back-projects through (0,0,z)
        # With identity rotation and tvec=[0,0,-1], camera is at world (0,0,1)
        # Pixel at center → ray along (0,0,1) → intersects Z=0.5 at (0,0,0.5)
        # So center pixel matches FK. Use an off-center pixel to create offset.
        # Pixel (370, 240) → dx = 50/500 = 0.1, dy = 0
        # Ray: (0.1, 0, 1), origin (0, 0, 1), target Z=0.5
        # t = (0.5 - 1) / 1 = -0.5 → point = (0 + 0.1*(-0.5), 0, 0.5) = (-0.05, 0, 0.5)
        det = [_make_detection(0, 370.0, 240.0, 0.9)]
        result = fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)

        assert result.source == FusionSource.FUSED
        # With high confidence (0.9), alpha ≈ 0.1, so result should be close to visual
        # Visual X ≈ 0.15, FK X = 0.0
        # Fused X ≈ 0.1 * 0.0 + 0.9 * 0.15 = 0.135
        assert result.positions[0][0] > 0.01  # shifted toward visual (positive X)

    def test_low_confidence_stays_near_fk(self):
        """Low-confidence visual detection should keep result near FK."""
        fusion = PoseFusion()
        calib = _make_calib()
        fk = [[0.0, 0.0, 0.5]]

        # Same offset pixel but low confidence
        det = [_make_detection(0, 370.0, 240.0, 0.1)]
        result = fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)

        # With low confidence (0.1), alpha ≈ 0.9, result stays near FK
        # Visual X ≈ 0.15, Fused X ≈ 0.9 * 0.0 + 0.1 * 0.15 = 0.015
        assert abs(result.positions[0][0]) < 0.02  # stays near FK


# --- Disagreement detection ---

class TestDisagreement:
    def test_large_disagreement_logs_warning(self, caplog):
        """Disagreement > 20mm should trigger a warning."""
        fusion = PoseFusion()
        calib = _make_calib()
        fk = [[0.0, 0.0, 0.5]]

        # Create a detection that back-projects far from FK
        # Pixel (820, 240) → dx = 500/500 = 1.0 → ray (1, 0, 1)
        # origin (0, 0, 1), t = -0.5, point = (-0.5, 0, 0.5) → 500mm from FK
        det = [_make_detection(0, 820.0, 240.0, 0.5)]

        with caplog.at_level(logging.WARNING, logger="th3cl4w.vision.pose_fusion"):
            result = fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)

        assert result.disagreements[0] > DISAGREEMENT_THRESHOLD_M
        assert any("disagreement" in r.message.lower() for r in caplog.records)

    def test_small_disagreement_no_warning(self, caplog):
        """Small disagreement should not trigger warning."""
        fusion = PoseFusion()
        calib = _make_calib()
        fk = [[0.0, 0.0, 0.5]]

        # Center pixel → back-projects to (0, 0, 0.5) → 0 disagreement
        det = [_make_detection(0, 320.0, 240.0, 0.5)]

        with caplog.at_level(logging.WARNING, logger="th3cl4w.vision.pose_fusion"):
            result = fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)

        assert result.disagreements[0] < DISAGREEMENT_THRESHOLD_M
        assert not any("disagreement" in r.message.lower() for r in caplog.records)


# --- Tracking quality ---

class TestTrackingQuality:
    def test_quality_after_multiple_fusions(self):
        fusion = PoseFusion()
        calib = _make_calib()
        fk = [[0.0, 0.0, 0.5], [0.1, 0.0, 0.3]]

        # Run several fusions
        for _ in range(5):
            det = [_make_detection(0, 320.0, 240.0, 0.8)]
            fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)

        quality = fusion.get_tracking_quality()
        assert quality["available"] is True
        assert quality["total_samples"] == 5
        assert "joint_0" in quality["joints"]

    def test_quality_empty(self):
        fusion = PoseFusion()
        quality = fusion.get_tracking_quality()
        assert quality["available"] is False


# --- Graceful degradation ---

class TestDegradation:
    def test_single_camera_fusion(self):
        """One camera + FK should still produce fused result."""
        fusion = PoseFusion()
        calib = _make_calib()
        fk = [[0.0, 0.0, 0.5]]
        det = [_make_detection(0, 320.0, 240.0, 0.7)]

        # Only cam0, no cam1
        result = fusion.fuse(fk, cam0_detections=det, cam0_calib=calib)
        assert result.per_joint_source[0] == FusionSource.FUSED

    def test_cameras_offline_fk_only(self):
        """Both cameras offline → FK-only mode."""
        fusion = PoseFusion()
        fk = [[0.0, 0.0, 0.5]]
        result = fusion.fuse(fk, cam0_detections=None, cam1_detections=None)
        assert result.source == FusionSource.FK_ONLY
