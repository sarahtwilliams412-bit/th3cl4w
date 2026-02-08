"""Tests for CalibrationReporter."""

import json
import os
import shutil
import tempfile
import pytest

from src.calibration.results_reporter import (
    CalibrationReporter,
    ComparisonReport,
    ComparisonResult,
    JointComparison,
    CalibrationSession,
    JOINT_NAMES,
)


def _jc(name, cv_px=(100, 200), llm_px=(110, 210), cv_err=10.0, llm_err=15.0):
    """Helper to make a JointComparison."""
    return JointComparison(
        name=name,
        fk_pixel=(100, 200),
        cv_pixel=cv_px,
        llm_pixel=llm_px,
        cv_error_px=cv_err,
        llm_error_px=llm_err,
        agreement_px=5.0 if cv_px and llm_px else None,
        cv_source="centroid",
        llm_confidence=0.8,
    )


def _make_results(n=5):
    results = []
    for i in range(n):
        joints = [
            _jc("base", cv_px=(100, 200), llm_px=None, cv_err=10.0, llm_err=None),
            _jc("shoulder", cv_px=(150, 180), llm_px=(155, 185), cv_err=12.0, llm_err=18.0),
            _jc("elbow", cv_px=None, llm_px=(200, 150), cv_err=None, llm_err=25.0),
            _jc("wrist", cv_px=(250, 120), llm_px=(260, 130), cv_err=20.0, llm_err=35.0),
            _jc("end_effector", cv_px=(300, 100), llm_px=(310, 105), cv_err=8.0, llm_err=12.0),
        ]
        results.append(ComparisonResult(
            pose_index=i,
            camera_id=0,
            joint_angles=[10.0 * i, 20.0, 30.0],
            joints=joints,
            cv_latency_ms=5.0,
            llm_latency_ms=2500.0,
            llm_tokens=1200,
        ))
    return results


def _make_report(**kwargs):
    defaults = dict(
        results=_make_results(),
        cv_detection_rate=0.8,
        llm_detection_rate=0.8,
        cv_mean_error_px=12.5,
        llm_mean_error_px=22.5,
        agreement_rate=0.6,
        total_llm_tokens=6000,
        total_llm_cost_usd=0.000675,
        recommendation="",
    )
    defaults.update(kwargs)
    return ComparisonReport(**defaults)


def _make_session():
    return CalibrationSession(
        session_id="test_session_001",
        start_time=1000.0,
        end_time=1055.0,
        num_poses=20,
        status="complete",
    )


class TestMarkdownGeneration:
    def test_basic_generation(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report(), _make_session())
        assert "# Calibration Comparison Report" in md
        assert "test_session_001" in md
        assert "## Summary" in md
        assert "## Per-Joint Analysis" in md
        assert "## Cost Analysis" in md
        assert "## Recommendation" in md
        assert "## Visualizations" in md

    def test_contains_joint_table(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report())
        assert "base" in md
        assert "shoulder" in md
        assert "end_effector" in md

    def test_contains_per_pose_detail(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report())
        assert "## Per-Pose Detail" in md

    def test_contains_cost_info(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report())
        assert "$0.0007" in md or "0.000675" in md
        assert "Cost per pose" in md

    def test_llm_helped_section(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report())
        # In test data, CV missed elbow but LLM found it
        assert "LLM Helped" in md
        assert "elbow" in md

    def test_llm_failed_section(self):
        # Create result where LLM detected but error > 50px
        joints = [
            _jc("base", llm_px=(200, 300), llm_err=80.0),
            _jc("shoulder", llm_px=None, llm_err=None),
            _jc("elbow", llm_px=(200, 150), llm_err=25.0),
            _jc("wrist", llm_px=(260, 130), llm_err=60.0),
            _jc("end_effector", llm_px=(310, 105), llm_err=12.0),
        ]
        results = [ComparisonResult(pose_index=0, camera_id=0, joint_angles=[], joints=joints,
                                     cv_latency_ms=5.0, llm_latency_ms=2500.0, llm_tokens=1200)]
        report = _make_report(results=results)
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(report)
        assert "LLM Failed" in md

    def test_duration_with_session(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report(), _make_session())
        assert "55.0s" in md


class TestRecommendation:
    def _report_with_joints(self, cv_rates, llm_rates, llm_errors):
        """Build a report where each joint has controlled detection rates and errors."""
        results = []
        n_poses = 10
        for pi in range(n_poses):
            joints = []
            for ji, name in enumerate(JOINT_NAMES):
                cv_det = (pi / n_poses) < cv_rates[ji]
                llm_det = (pi / n_poses) < llm_rates[ji]
                joints.append(JointComparison(
                    name=name,
                    fk_pixel=(100, 100),
                    cv_pixel=(100, 100) if cv_det else None,
                    llm_pixel=(110, 110) if llm_det else None,
                    cv_error_px=10.0 if cv_det else None,
                    llm_error_px=llm_errors[ji] if llm_det else None,
                    agreement_px=None, cv_source=None, llm_confidence=None,
                ))
            results.append(ComparisonResult(
                pose_index=pi, camera_id=0, joint_angles=[],
                joints=joints, cv_latency_ms=5.0, llm_latency_ms=2000.0, llm_tokens=1000,
            ))
        return _make_report(results=results)

    def test_archive_low_llm_detection(self):
        reporter = CalibrationReporter()
        report = self._report_with_joints(
            cv_rates=[0.8, 0.7, 0.6, 0.5, 0.8],
            llm_rates=[0.1, 0.2, 0.1, 0.1, 0.2],
            llm_errors=[20.0, 25.0, 30.0, 20.0, 15.0],
        )
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_archive_high_llm_error(self):
        reporter = CalibrationReporter()
        report = self._report_with_joints(
            cv_rates=[0.8, 0.7, 0.6, 0.5, 0.8],
            llm_rates=[0.6, 0.6, 0.6, 0.6, 0.6],
            llm_errors=[120.0, 110.0, 130.0, 105.0, 115.0],
        )
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_archive_cv_sufficient(self):
        reporter = CalibrationReporter()
        report = self._report_with_joints(
            cv_rates=[0.9, 0.95, 0.88, 0.92, 0.87],
            llm_rates=[0.6, 0.6, 0.6, 0.6, 0.6],
            llm_errors=[20.0, 25.0, 22.0, 18.0, 15.0],
        )
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_continue_llm_helps_weak_cv(self):
        reporter = CalibrationReporter()
        report = self._report_with_joints(
            cv_rates=[0.8, 0.7, 0.3, 0.5, 0.8],
            llm_rates=[0.6, 0.5, 0.7, 0.5, 0.8],
            llm_errors=[20.0, 25.0, 22.0, 18.0, 15.0],
        )
        rec = reporter._compute_recommendation(report)
        assert rec == "continue"

    def test_continue_low_llm_error(self):
        reporter = CalibrationReporter()
        report = self._report_with_joints(
            cv_rates=[0.8, 0.7, 0.6, 0.5, 0.8],
            llm_rates=[0.5, 0.5, 0.5, 0.5, 0.5],
            llm_errors=[20.0, 25.0, 22.0, 18.0, 15.0],
        )
        rec = reporter._compute_recommendation(report)
        assert rec == "continue"

    def test_inconclusive_empty(self):
        reporter = CalibrationReporter()
        report = _make_report(results=[])
        rec = reporter._compute_recommendation(report)
        assert rec == "inconclusive"


class TestJsonOutput:
    def test_required_fields(self):
        reporter = CalibrationReporter()
        js = reporter.generate_json(_make_report())
        assert "session_id" in js
        assert "total_poses" in js
        assert "recommendation" in js
        assert "joints" in js
        assert "per_pose" in js
        assert "total_llm_cost_usd" in js
        assert "total_llm_tokens" in js

    def test_joint_count(self):
        reporter = CalibrationReporter()
        js = reporter.generate_json(_make_report())
        assert len(js["joints"]) == 5

    def test_json_serializable(self):
        reporter = CalibrationReporter()
        js = reporter.generate_json(_make_report())
        serialized = json.dumps(js)
        assert isinstance(serialized, str)


class TestEmptyPartialReports:
    def test_empty_results(self):
        reporter = CalibrationReporter()
        report = _make_report(results=[])
        md = reporter.generate_markdown(report)
        assert "# Calibration Comparison Report" in md

    def test_no_joints_in_results(self):
        reporter = CalibrationReporter()
        results = [ComparisonResult(pose_index=0, camera_id=0, joint_angles=[],
                                     joints=[], cv_latency_ms=0, llm_latency_ms=0, llm_tokens=0)]
        report = _make_report(results=results)
        md = reporter.generate_markdown(report)
        assert "# Calibration Comparison Report" in md
        js = reporter.generate_json(report)
        assert js["joints"] == []

    def test_all_none_detections(self):
        """Results with no detections at all."""
        reporter = CalibrationReporter()
        joints = [JointComparison(name="base", fk_pixel=(100, 100), cv_pixel=None, llm_pixel=None,
                                   cv_error_px=None, llm_error_px=None, agreement_px=None,
                                   cv_source=None, llm_confidence=None)]
        results = [ComparisonResult(pose_index=0, camera_id=0, joint_angles=[],
                                     joints=joints, cv_latency_ms=0, llm_latency_ms=0, llm_tokens=0)]
        report = _make_report(results=results)
        md = reporter.generate_markdown(report)
        assert "0/1" in md


class TestAsciiVisualizations:
    def test_bar_chart_renders(self):
        reporter = CalibrationReporter()
        from src.calibration.results_reporter import _joint_stats
        jstats = _joint_stats(_make_report())
        chart = reporter._ascii_detection_bar_chart(jstats)
        assert "█" in chart or "▓" in chart
        assert "CV" in chart
        assert "LLM" in chart

    def test_scatter_renders(self):
        reporter = CalibrationReporter()
        scatter = reporter._ascii_error_scatter(_make_report())
        assert "cv" in scatter.lower()

    def test_scatter_empty_data(self):
        reporter = CalibrationReporter()
        report = _make_report(results=[])
        scatter = reporter._ascii_error_scatter(report)
        assert "no paired error data" in scatter

    def test_bar_chart_no_joints(self):
        reporter = CalibrationReporter()
        chart = reporter._ascii_detection_bar_chart([])
        assert "```" in chart


class TestSaveReport:
    def test_save_creates_files(self):
        reporter = CalibrationReporter()
        report = _make_report()
        session = _make_session()
        tmpdir = tempfile.mkdtemp()
        try:
            path = reporter.save_report(report, session, output_dir=tmpdir)
            assert os.path.exists(os.path.join(path, "report.md"))
            assert os.path.exists(os.path.join(path, "summary.json"))
            assert os.path.exists(os.path.join(path, "raw_results.json"))
            with open(os.path.join(path, "summary.json")) as f:
                data = json.load(f)
            assert "recommendation" in data
        finally:
            shutil.rmtree(tmpdir)
