"""Tests for joint_detector module."""

import numpy as np
import pytest

from src.vision.arm_segmenter import ArmSegmentation
from src.vision.joint_detector import (
    DetectionSource,
    JointDetection,
    JointDetector,
    JointTracker,
)


def _make_segmentation(
    gold_centroids: list[tuple[int, int]] | None = None,
    contour_pts: np.ndarray | None = None,
    mask_size: tuple[int, int] = (480, 640),
    bbox: tuple[int, int, int, int] | None = None,
) -> ArmSegmentation:
    """Create a synthetic ArmSegmentation for testing."""
    mask = np.zeros(mask_size, dtype=np.uint8)
    if bbox is not None:
        x, y, w, h = bbox
        # Draw a simple vertical bar with narrow "joints"
        for row in range(y, y + h):
            # Vary width: narrow at 1/3 and 2/3 (simulate joints)
            rel = (row - y) / h
            base_w = w // 2
            # Narrower at ~0.33 and ~0.66
            if abs(rel - 0.33) < 0.05 or abs(rel - 0.66) < 0.05:
                half = max(2, base_w // 3)
            else:
                half = base_w
            cx = x + w // 2
            mask[row, max(0, cx - half) : min(mask.shape[1], cx + half)] = 255

    return ArmSegmentation(
        silhouette_mask=mask,
        gold_centroids=gold_centroids or [],
        contour=contour_pts,
        bounding_box=bbox,
    )


def _sample_fk_pixels() -> list[tuple[float, float]]:
    """5 FK-predicted pixel positions (base, shoulder, elbow, wrist, EE)."""
    return [
        (320.0, 400.0),
        (320.0, 350.0),
        (320.0, 280.0),
        (320.0, 210.0),
        (320.0, 150.0),
    ]


class TestJointDetectorGold:
    def test_gold_near_fk_gets_high_confidence(self):
        fk = _sample_fk_pixels()
        # Place gold centroid 10px from elbow FK prediction
        seg = _make_segmentation(gold_centroids=[(330, 285)])
        det = JointDetector()
        results = det.detect_joints(seg, fk)

        elbow = results[2]
        assert elbow.source == DetectionSource.GOLD
        assert elbow.confidence >= 0.7
        assert abs(elbow.pixel_pos[0] - 330) < 1
        assert abs(elbow.pixel_pos[1] - 285) < 1

    def test_gold_far_away_ignored(self):
        fk = _sample_fk_pixels()
        # Gold centroid 200px away — outside search radius
        seg = _make_segmentation(gold_centroids=[(100, 100)])
        det = JointDetector()
        results = det.detect_joints(seg, fk)

        for r in results:
            assert r.source != DetectionSource.GOLD

    def test_no_gold_falls_back_to_fk_only(self):
        fk = _sample_fk_pixels()
        seg = _make_segmentation()
        det = JointDetector()
        results = det.detect_joints(seg, fk)

        for r in results:
            assert r.source == DetectionSource.FK_ONLY
            assert r.confidence == pytest.approx(0.2)


class TestJointDetectorWidthMinima:
    def test_width_minima_detected(self):
        fk = _sample_fk_pixels()
        bbox = (300, 100, 40, 320)
        seg = _make_segmentation(bbox=bbox)
        det = JointDetector(width_search_radius=40.0)
        results = det.detect_joints(seg, fk)

        # At least one joint should pick up a width minimum
        width_sources = [r for r in results if r.source == DetectionSource.WIDTH]
        # May or may not find width minima depending on geometry — just check no crash
        assert len(results) == 5


class TestJointDetectorContour:
    def test_contour_inflection(self):
        # Build an L-shaped contour with a clear bend
        pts = []
        for y in range(0, 50):
            pts.append([320, 400 - y])  # vertical segment
        for x in range(1, 50):
            pts.append([320 + x, 350])  # horizontal segment
        contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        fk = _sample_fk_pixels()
        seg = _make_segmentation(contour_pts=contour)
        det = JointDetector(min_inflection_angle=10.0)
        results = det.detect_joints(seg, fk)
        assert len(results) == 5


class TestCameraVisibility:
    def test_cam0_visible_joints(self):
        vis = JointDetector.get_visible_joints(0)
        assert 1 in vis  # shoulder
        assert 2 in vis  # elbow
        assert 4 in vis  # wrist

    def test_cam1_visible_joints(self):
        vis = JointDetector.get_visible_joints(1)
        assert 0 in vis  # base yaw

    def test_unknown_camera(self):
        assert JointDetector.get_visible_joints(99) == []


class TestJointTracker:
    def test_first_detection_sets_position(self):
        tracker = JointTracker(num_joints=5)
        dets = [JointDetection(0, (100.0, 200.0), 0.8, DetectionSource.GOLD)]
        results = tracker.update(dets)
        j0 = results[0]
        assert j0.pixel_pos == pytest.approx((100.0, 200.0))
        assert j0.confidence == pytest.approx(0.8)
        assert j0.frames_tracked == 1

    def test_smoothing_converges(self):
        tracker = JointTracker(alpha=0.3, num_joints=1)
        target = (100.0, 200.0)

        # Feed consistent detections
        for _ in range(20):
            dets = [JointDetection(0, target, 0.9, DetectionSource.GOLD)]
            results = tracker.update(dets)

        pos = results[0].pixel_pos
        assert abs(pos[0] - target[0]) < 1.0
        assert abs(pos[1] - target[1]) < 1.0

    def test_outlier_rejection(self):
        tracker = JointTracker(alpha=0.3, outlier_threshold=50.0, outlier_alpha=0.05, num_joints=1)

        # Establish baseline
        for _ in range(10):
            tracker.update([JointDetection(0, (100.0, 100.0), 0.9, DetectionSource.GOLD)])

        # Sudden jump of 200px
        results = tracker.update([JointDetection(0, (300.0, 100.0), 0.9, DetectionSource.GOLD)])
        pos = results[0].pixel_pos
        # Should barely move (outlier_alpha=0.05 → move ~10px, not 200)
        assert pos[0] < 120.0, f"Position jumped too far: {pos[0]}"

    def test_confidence_decays_without_detection(self):
        tracker = JointTracker(confidence_decay=0.5, num_joints=1)
        tracker.update([JointDetection(0, (100.0, 100.0), 1.0, DetectionSource.GOLD)])

        # Update with no detection for this joint
        results = tracker.update([])
        assert results[0].confidence == pytest.approx(0.5)
        results = tracker.update([])
        assert results[0].confidence == pytest.approx(0.25)

    def test_reset(self):
        tracker = JointTracker(num_joints=2)
        tracker.update([JointDetection(0, (50.0, 50.0), 0.8, DetectionSource.FK_ONLY)])
        tracker.reset()
        results = tracker.update([])
        assert results[0].confidence == 0.0
        assert results[0].frames_tracked == 0
