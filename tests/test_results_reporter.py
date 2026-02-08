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


def _make_joint_comparisons(
    cv_rates=None, llm_rates=None, cv_errors=None, llm_errors=None, agreements=None
):
    cv_rates = cv_rates or [0.8, 0.7, 0.6, 0.5, 0.9]
    llm_rates = llm_rates or [0.6, 0.5, 0.7, 0.4, 0.8]
    cv_errors = cv_errors or [10.0, 15.0, 20.0, 25.0, 8.0]
    llm_errors = llm_errors or [30.0, 35.0, 25.0, 40.0, 20.0]
    agreements = agreements or [0.5, 0.4, 0.6, 0.3, 0.7]
    return [
        JointComparison(
            joint_name=JOINT_NAMES[i],
            cv_detection_rate=cv_rates[i],
            llm_detection_rate=llm_rates[i],
            cv_mean_error_px=cv_errors[i],
            llm_mean_error_px=llm_errors[i],
            agreement_rate=agreements[i],
        )
        for i in range(5)
    ]


def _make_results(n=5):
    results = []
    for i in range(n):
        results.append(
            ComparisonResult(
                pose_index=i,
                camera_id=0,
                timestamp=1000.0 + i,
                joint_angles=[10.0 * i, 20.0, 30.0],
                cv_errors_px=[10.0, 15.0, None, 20.0, 8.0],
                llm_errors_px=[30.0, None, 25.0, 40.0, 20.0],
                cv_detected=[True, True, False, True, True],
                llm_detected=[False, False, True, True, True],
                llm_input_tokens=1000,
                llm_output_tokens=200,
                llm_cost_usd=0.000135,
                cv_latency_ms=5.0,
                llm_latency_ms=2500.0,
            )
        )
    return results


def _make_report(**kwargs):
    defaults = dict(
        session_id="test_session_001",
        created_at="2026-02-08T13:00:00",
        total_poses=5,
        results=_make_results(),
        joint_comparisons=_make_joint_comparisons(),
        cv_mean_latency_ms=5.0,
        llm_mean_latency_ms=2500.0,
        total_llm_input_tokens=5000,
        total_llm_output_tokens=1000,
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
        # In our test data, CV missed elbow but LLM found it
        assert "LLM Helped" in md
        assert "elbow" in md

    def test_llm_failed_section(self):
        # Create result where LLM detected but error > 50px
        results = _make_results(1)
        results[0].llm_errors_px = [80.0, None, 25.0, 60.0, 20.0]
        results[0].llm_detected = [True, False, True, True, True]
        report = _make_report(results=results)
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(report)
        assert "LLM Failed" in md

    def test_duration_with_session(self):
        reporter = CalibrationReporter()
        md = reporter.generate_markdown(_make_report(), _make_session())
        assert "55.0s" in md


class TestRecommendation:
    def test_archive_low_llm_detection(self):
        reporter = CalibrationReporter()
        jcs = _make_joint_comparisons(llm_rates=[0.1, 0.2, 0.15, 0.1, 0.2])
        report = _make_report(joint_comparisons=jcs)
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_archive_high_llm_error(self):
        reporter = CalibrationReporter()
        jcs = _make_joint_comparisons(llm_errors=[120.0, 110.0, 130.0, 105.0, 115.0])
        report = _make_report(joint_comparisons=jcs)
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_archive_cv_sufficient(self):
        reporter = CalibrationReporter()
        jcs = _make_joint_comparisons(cv_rates=[0.9, 0.95, 0.88, 0.92, 0.87])
        report = _make_report(joint_comparisons=jcs)
        rec = reporter._compute_recommendation(report)
        assert rec == "archive"

    def test_continue_llm_helps_weak_cv(self):
        reporter = CalibrationReporter()
        # CV low on elbow, LLM high
        jcs = _make_joint_comparisons(
            cv_rates=[0.8, 0.7, 0.3, 0.5, 0.8],
            llm_rates=[0.6, 0.5, 0.7, 0.5, 0.8],
        )
        report = _make_report(joint_comparisons=jcs)
        rec = reporter._compute_recommendation(report)
        assert rec == "continue"

    def test_continue_low_llm_error(self):
        reporter = CalibrationReporter()
        jcs = _make_joint_comparisons(
            cv_rates=[0.8, 0.7, 0.6, 0.5, 0.8],
            llm_rates=[0.5, 0.5, 0.5, 0.5, 0.5],
            llm_errors=[20.0, 25.0, 22.0, 18.0, 15.0],
        )
        report = _make_report(joint_comparisons=jcs)
        rec = reporter._compute_recommendation(report)
        assert rec == "continue"

    def test_inconclusive_empty(self):
        reporter = CalibrationReporter()
        report = _make_report(joint_comparisons=[])
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
        report = _make_report(results=[], total_poses=0)
        md = reporter.generate_markdown(report)
        assert "# Calibration Comparison Report" in md

    def test_no_joint_comparisons(self):
        reporter = CalibrationReporter()
        report = _make_report(joint_comparisons=[], results=[])
        md = reporter.generate_markdown(report)
        assert "# Calibration Comparison Report" in md
        js = reporter.generate_json(report)
        assert js["joints"] == []

    def test_partial_detections(self):
        """Results with some None errors."""
        reporter = CalibrationReporter()
        results = [
            ComparisonResult(
                pose_index=0,
                cv_errors_px=[None, None, None, None, None],
                llm_errors_px=[None, None, None, None, None],
                cv_detected=[False, False, False, False, False],
                llm_detected=[False, False, False, False, False],
            )
        ]
        report = _make_report(results=results, total_poses=1)
        md = reporter.generate_markdown(report)
        assert "0/5" in md


class TestAsciiVisualizations:
    def test_bar_chart_renders(self):
        reporter = CalibrationReporter()
        report = _make_report()
        chart = reporter._ascii_detection_bar_chart(report)
        assert "█" in chart or "▓" in chart
        assert "CV" in chart
        assert "LLM" in chart

    def test_scatter_renders(self):
        reporter = CalibrationReporter()
        report = _make_report()
        scatter = reporter._ascii_error_scatter(report)
        assert "CV" in scatter.lower() or "cv" in scatter.lower()

    def test_scatter_empty_data(self):
        reporter = CalibrationReporter()
        report = _make_report(results=[])
        scatter = reporter._ascii_error_scatter(report)
        assert "no paired error data" in scatter

    def test_bar_chart_no_joints(self):
        reporter = CalibrationReporter()
        report = _make_report(joint_comparisons=[])
        chart = reporter._ascii_detection_bar_chart(report)
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
            # Verify JSON is valid
            with open(os.path.join(path, "summary.json")) as f:
                data = json.load(f)
            assert data["session_id"] == "test_session_001"
        finally:
            shutil.rmtree(tmpdir)
