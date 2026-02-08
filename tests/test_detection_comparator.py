"""Tests for detection_comparator module."""

import math
import pytest
from unittest.mock import MagicMock, patch

from src.vision.detection_comparator import (
    JointComparison, ComparisonResult, ComparisonReport,
    compute_joint_comparison, compute_report, _pixel_dist,
    DetectionComparator,
)
from src.vision.joint_detector import JointDetection, DetectionSource


# ── Helpers ──────────────────────────────────────────────────────────

def _make_cv_det(idx, px, py, source=DetectionSource.GOLD, conf=0.8):
    return JointDetection(joint_index=idx, pixel_pos=(px, py), confidence=conf, source=source)


def _make_result(
    pose_index=0, camera_id=0,
    cv_pixels=None, llm_pixels=None, fk_pixels=None,
    cv_sources=None, llm_tokens=0,
):
    """Build a ComparisonResult from simple pixel lists."""
    names = ["base", "shoulder", "elbow", "wrist", "end_effector"]
    joints = []
    for i, name in enumerate(names):
        fk = fk_pixels[i] if fk_pixels and i < len(fk_pixels) else (100.0, 100.0)
        cv = cv_pixels[i] if cv_pixels and i < len(cv_pixels) else None
        llm = llm_pixels[i] if llm_pixels and i < len(llm_pixels) else None
        src = cv_sources[i] if cv_sources and i < len(cv_sources) else "gold"

        joints.append(JointComparison(
            name=name, fk_pixel=fk, cv_pixel=cv, llm_pixel=llm,
            cv_error_px=_pixel_dist(cv, fk),
            llm_error_px=_pixel_dist(llm, fk),
            agreement_px=_pixel_dist(cv, llm),
            cv_source=src, llm_confidence=0.7 if llm else None,
        ))
    return ComparisonResult(
        pose_index=pose_index, camera_id=camera_id,
        joint_angles=[0.0]*6, joints=joints,
        cv_latency_ms=5.0, llm_latency_ms=2000.0, llm_tokens=llm_tokens,
    )


# ── Tests ────────────────────────────────────────────────────────────

class TestPixelDist:
    def test_basic(self):
        assert _pixel_dist((0, 0), (3, 4)) == 5.0

    def test_none(self):
        assert _pixel_dist(None, (1, 1)) is None
        assert _pixel_dist((1, 1), None) is None


class TestComputeJointComparison:
    def test_all_detected(self):
        cv = _make_cv_det(0, 105, 100)
        jc = compute_joint_comparison("base", (100, 100), cv, (110, 100))
        assert jc.cv_error_px == pytest.approx(5.0)
        assert jc.llm_error_px == pytest.approx(10.0)
        assert jc.agreement_px == pytest.approx(5.0)

    def test_fk_only_not_counted(self):
        """FK_ONLY CV detections should not count as real detections."""
        cv = _make_cv_det(0, 100, 100, source=DetectionSource.FK_ONLY)
        jc = compute_joint_comparison("base", (100, 100), cv, (110, 100))
        assert jc.cv_pixel is None  # FK_ONLY filtered out
        assert jc.cv_error_px is None

    def test_no_llm(self):
        cv = _make_cv_det(0, 105, 100)
        jc = compute_joint_comparison("base", (100, 100), cv, None)
        assert jc.llm_pixel is None
        assert jc.llm_error_px is None
        assert jc.agreement_px is None

    def test_no_cv(self):
        jc = compute_joint_comparison("base", (100, 100), None, (110, 100))
        assert jc.cv_pixel is None
        assert jc.cv_error_px is None
        assert jc.llm_error_px == pytest.approx(10.0)


class TestComputeReport:
    def test_empty(self):
        r = compute_report([])
        assert r.recommendation == "inconclusive"
        assert r.cv_detection_rate == 0.0

    def test_archive_low_detection(self):
        """<30% LLM detection → archive."""
        # 5 joints, only 1 has LLM → 20%
        r = _make_result(
            cv_pixels=[(101,100),(102,100),(103,100),(104,100),(105,100)],
            llm_pixels=[(110,100), None, None, None, None],
            fk_pixels=[(100,100)]*5,
        )
        report = compute_report([r])
        assert report.llm_detection_rate == pytest.approx(0.2)
        assert report.recommendation == "archive"

    def test_archive_high_error(self):
        """LLM mean error >100px → archive."""
        # All detected but 150px off
        llm = [(250, 100)]*5
        r = _make_result(
            cv_pixels=[(105,100)]*5,
            llm_pixels=llm,
            fk_pixels=[(100,100)]*5,
        )
        report = compute_report([r])
        assert report.llm_mean_error_px > 100
        assert report.recommendation == "archive"

    def test_continue_good_results(self):
        """>50% det, <50px error, >60% agreement → continue."""
        # All 5 joints detected by both, close to FK and each other
        cv = [(105,100),(102,100),(108,100),(103,100),(106,100)]
        llm = [(107,100),(104,100),(110,100),(105,100),(108,100)]
        fk = [(100,100)]*5
        r = _make_result(cv_pixels=cv, llm_pixels=llm, fk_pixels=fk)
        report = compute_report([r])
        assert report.llm_detection_rate == 1.0
        assert report.llm_mean_error_px < 50
        assert report.recommendation == "continue"

    def test_batch_aggregation(self):
        """Multiple results aggregate correctly."""
        fk = [(100,100)]*5
        r1 = _make_result(
            pose_index=0,
            cv_pixels=[(110,100)]*5, llm_pixels=[(115,100)]*5,
            fk_pixels=fk, llm_tokens=100,
        )
        r2 = _make_result(
            pose_index=1,
            cv_pixels=[(120,100)]*5, llm_pixels=[(125,100)]*5,
            fk_pixels=fk, llm_tokens=200,
        )
        report = compute_report([r1, r2])
        assert report.total_llm_tokens == 300
        assert report.cv_detection_rate == 1.0
        assert report.llm_detection_rate == 1.0
        # Mean errors: r1=15, r2=25 → mean=20
        assert report.llm_mean_error_px == pytest.approx(20.0)
        # CV: r1=10, r2=20 → mean=15
        assert report.cv_mean_error_px == pytest.approx(15.0)

    def test_missing_llm_cv_only(self):
        """LLM offline → all None LLM, CV-only comparison."""
        r = _make_result(
            cv_pixels=[(105,100)]*5, llm_pixels=[None]*5,
            fk_pixels=[(100,100)]*5,
        )
        report = compute_report([r])
        assert report.llm_detection_rate == 0.0
        assert report.cv_detection_rate == 1.0
        assert report.recommendation == "archive"  # <30% LLM

    def test_missing_cv_detections(self):
        """CV returns nothing, only LLM has detections."""
        r = _make_result(
            cv_pixels=[None]*5,
            llm_pixels=[(110,100)]*5,
            fk_pixels=[(100,100)]*5,
        )
        report = compute_report([r])
        assert report.cv_detection_rate == 0.0
        assert report.llm_detection_rate == 1.0
        assert report.llm_mean_error_px == pytest.approx(10.0)

    def test_inconclusive_middle_ground(self):
        """Between thresholds → inconclusive."""
        # 3/5 LLM detected (60%), but high error (80px) and low agreement
        llm = [(180,100),(180,100),(180,100), None, None]
        cv = [(105,100),(105,100),(105,100),(105,100),(105,100)]
        fk = [(100,100)]*5
        r = _make_result(cv_pixels=cv, llm_pixels=llm, fk_pixels=fk)
        report = compute_report([r])
        assert report.llm_detection_rate == pytest.approx(0.6)
        assert report.llm_mean_error_px > 50
        assert report.recommendation == "inconclusive"

    def test_agreement_rate_threshold(self):
        """Agreement requires <50px between CV and LLM."""
        # Both detect all, but they're 60px apart → low agreement
        cv = [(100,100)]*5
        llm = [(160,100)]*5  # 60px from CV
        fk = [(130,100)]*5   # both ~30px from FK
        r = _make_result(cv_pixels=cv, llm_pixels=llm, fk_pixels=fk)
        report = compute_report([r])
        assert report.agreement_rate == 0.0  # all >50px apart
        # 30px error, 100% det, but 0% agreement → inconclusive
        assert report.recommendation == "inconclusive"
