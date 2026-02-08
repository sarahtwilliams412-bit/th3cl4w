"""Tests for the CV vs LLM detection comparison engine."""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.vision.detection_comparator import (
    DetectionComparator,
    ComparisonResult,
    ComparisonReport,
    JointComparison,
    PoseCapture,
    _pixel_dist,
)
from src.vision.joint_detector import JointDetector, JointDetection, DetectionSource, JOINT_NAMES
from src.vision.arm_segmenter import ArmSegmentation


# ── Helpers ───────────────────────────────────────────────────────────

FK_PIXELS = [
    (960.0, 900.0), (960.0, 700.0), (800.0, 500.0),
    (600.0, 400.0), (500.0, 300.0),
]


def _make_segmentation():
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    return ArmSegmentation(
        silhouette_mask=mask, gold_centroids=[(960, 540)],
        contour=None, bounding_box=(400, 200, 1100, 700),
    )


def _make_cv_detector(detect_map=None):
    """Mock CV detector. detect_map: {joint_idx: (pixel, source)}."""
    det = MagicMock(spec=JointDetector)

    def detect_joints(seg, fk_pixels):
        results = []
        for i, fk in enumerate(fk_pixels):
            if detect_map and i in detect_map:
                pos, src = detect_map[i]
                results.append(JointDetection(i, pos, 0.8, src))
            else:
                results.append(JointDetection(i, fk, 0.2, DetectionSource.FK_ONLY))
        return results

    det.detect_joints = detect_joints
    return det


def _make_llm_detector(joints_response=None):
    """Mock LLM detector returning normalized coords."""
    det = MagicMock()

    async def detect_joints(frame_bytes, camera_id):
        return {
            "joints": joints_response or [],
            "input_tokens": 800, "output_tokens": 200,
            "latency_ms": 2500.0, "cost_usd": 0.00012,
        }

    det.detect_joints = detect_joints
    return det


def _make_comparator(cv_map=None, llm_joints=None):
    """Build comparator with mock detectors and fixed FK."""
    return DetectionComparator(
        cv_detector=_make_cv_detector(cv_map),
        llm_detector=_make_llm_detector(llm_joints) if llm_joints is not None else None,
        fk_engine_func=lambda angles: FK_PIXELS,
    )


def _llm_joints_close():
    """LLM joints close to FK ground truth."""
    return [
        {"name": n, "x": (FK_PIXELS[i][0] + 3) / 1920, "y": (FK_PIXELS[i][1] + 3) / 1080}
        for i, n in enumerate(JOINT_NAMES)
    ]


def _cv_map_all_gold():
    """CV detects all joints via GOLD, offset +5px."""
    return {
        i: ((FK_PIXELS[i][0] + 5, FK_PIXELS[i][1] + 5), DetectionSource.GOLD)
        for i in range(5)
    }


# ── Tests ─────────────────────────────────────────────────────────────

class TestPixelDist:
    def test_zero(self):
        assert _pixel_dist((0, 0), (0, 0)) == 0.0

    def test_diagonal(self):
        assert abs(_pixel_dist((0, 0), (3, 4)) - 5.0) < 1e-9

    def test_none(self):
        assert _pixel_dist(None, (1, 1)) is None
        assert _pixel_dist((1, 1), None) is None


class TestCompareSingle:
    def test_both_detect_all(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=_llm_joints_close())
        result = comp.compare_single(
            frame_bytes=b"fake", camera_id=0, joint_angles=[0]*6,
            segmentation=_make_segmentation(), pose_index=0,
        )
        assert isinstance(result, ComparisonResult)
        assert len(result.joints) == 5
        assert result.llm_tokens == 1000
        for j in result.joints:
            assert j.cv_pixel is not None
            assert j.llm_pixel is not None
            assert j.cv_error_px < 20.0
            assert j.llm_error_px < 20.0
            assert j.agreement_px is not None

    def test_no_llm_detector(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=None)
        result = comp.compare_single(
            frame_bytes=b"fake", camera_id=0, joint_angles=[0]*6,
            segmentation=_make_segmentation(),
        )
        for j in result.joints:
            assert j.llm_pixel is None

    def test_no_cv_detections(self):
        comp = _make_comparator(cv_map={}, llm_joints=[{"name": "base", "x": 0.5, "y": 0.83}])
        result = comp.compare_single(
            frame_bytes=b"fake", camera_id=0, joint_angles=[0]*6,
            segmentation=_make_segmentation(),
        )
        assert result.joints[0].cv_pixel is None  # FK_ONLY not counted
        assert result.joints[0].llm_pixel is not None

    def test_graceful_empty(self):
        comp = _make_comparator(cv_map={}, llm_joints=[])
        result = comp.compare_single(
            frame_bytes=b"fake", camera_id=0, joint_angles=[0]*6,
            segmentation=_make_segmentation(),
        )
        assert len(result.joints) == 5
        for j in result.joints:
            assert j.cv_pixel is None
            assert j.llm_pixel is None
            assert j.agreement_px is None


class TestBatchAndReport:
    def _make_poses(self, n=5):
        return [
            PoseCapture(
                pose_index=i, camera_id=0, frame_bytes=b"fake",
                joint_angles=[0]*6, segmentation=_make_segmentation(),
            )
            for i in range(n)
        ]

    def test_batch_aggregation(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=_llm_joints_close())
        report = comp.compare_batch(self._make_poses(5))
        assert isinstance(report, ComparisonReport)
        assert len(report.results) == 5
        assert report.cv_detection_rate == 1.0
        assert report.llm_detection_rate == 1.0
        assert report.total_llm_tokens == 5000

    def test_recommendation_continue(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=_llm_joints_close())
        report = comp.compare_batch(self._make_poses(5))
        assert report.recommendation == "continue"

    def test_recommendation_archive_no_llm(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=[])
        report = comp.compare_batch(self._make_poses(5))
        assert report.recommendation == "archive"
        assert report.llm_detection_rate == 0.0

    def test_recommendation_archive_high_error(self):
        # LLM detects but far from FK
        bad_llm = [
            {"name": n, "x": (FK_PIXELS[i][0] + 150) / 1920, "y": (FK_PIXELS[i][1] + 150) / 1080}
            for i, n in enumerate(JOINT_NAMES)
        ]
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=bad_llm)
        report = comp.compare_batch(self._make_poses(5))
        assert report.recommendation == "archive"

    def test_empty_batch(self):
        comp = _make_comparator(cv_map={}, llm_joints=[])
        report = comp.compare_batch([])
        assert report.recommendation == "inconclusive"


class TestSummary:
    def test_summary_keys(self):
        comp = _make_comparator(cv_map=_cv_map_all_gold(), llm_joints=_llm_joints_close())
        report = comp.compare_batch([
            PoseCapture(0, 0, b"fake", [0]*6, _make_segmentation()),
        ])
        summary = comp.summary(report)
        expected = {
            "num_poses", "cv_detection_rate", "llm_detection_rate",
            "cv_mean_error_px", "llm_mean_error_px", "agreement_rate",
            "total_llm_tokens", "total_llm_cost_usd", "recommendation",
        }
        assert set(summary.keys()) == expected
        assert summary["num_poses"] == 1
        assert summary["recommendation"] == "continue"


class TestRecommendation:
    def test_low_detection(self):
        assert DetectionComparator._recommend(0.20, 30.0, 0.70) == "archive"

    def test_high_error(self):
        assert DetectionComparator._recommend(0.60, 120.0, 0.70) == "archive"

    def test_good(self):
        assert DetectionComparator._recommend(0.60, 40.0, 0.70) == "continue"

    def test_borderline(self):
        assert DetectionComparator._recommend(0.55, 45.0, 0.50) == "archive"
