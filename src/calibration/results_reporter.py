"""Calibration Results Reporter — generates comparison reports between CV and LLM detection pipelines."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Stub dataclasses (compatible with detection_comparator.py when it lands)
# ---------------------------------------------------------------------------

try:
    from src.vision.detection_comparator import (
        ComparisonReport,
        ComparisonResult,
        JointComparison,
    )
except ImportError:

    @dataclass
    class JointComparison:
        joint_name: str = ""
        cv_detection_rate: float = 0.0
        llm_detection_rate: float = 0.0
        cv_mean_error_px: Optional[float] = None
        llm_mean_error_px: Optional[float] = None
        agreement_rate: float = 0.0  # both detected same joint

    @dataclass
    class ComparisonResult:
        pose_index: int = 0
        camera_id: int = 0
        timestamp: float = 0.0
        joint_angles: list = field(default_factory=list)
        fk_pixels: list = field(default_factory=list)
        cv_detections: list = field(default_factory=list)
        cv_latency_ms: float = 0.0
        llm_raw_response: str = ""
        llm_normalized_coords: list = field(default_factory=list)
        llm_pixels: list = field(default_factory=list)
        llm_latency_ms: float = 0.0
        llm_input_tokens: int = 0
        llm_output_tokens: int = 0
        llm_cost_usd: float = 0.0
        cv_errors_px: list = field(default_factory=list)
        llm_errors_px: list = field(default_factory=list)
        cv_detected: list = field(default_factory=list)
        llm_detected: list = field(default_factory=list)

    @dataclass
    class ComparisonReport:
        session_id: str = ""
        created_at: str = ""
        total_poses: int = 0
        results: list = field(default_factory=list)  # list[ComparisonResult]
        joint_comparisons: list = field(default_factory=list)  # list[JointComparison]
        cv_mean_latency_ms: float = 0.0
        llm_mean_latency_ms: float = 0.0
        total_llm_input_tokens: int = 0
        total_llm_output_tokens: int = 0
        total_llm_cost_usd: float = 0.0
        recommendation: str = ""  # "continue" | "archive" | "inconclusive"


@dataclass
class CalibrationSession:
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    num_poses: int = 20
    status: str = "pending"  # pending | running | complete | failed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "end_effector"]

# Kill / continue thresholds (from comparison plan)
KILL_DETECTION_RATE = 0.30
KILL_MEAN_ERROR = 100.0
KILL_API_ERROR_RATE = 0.20
CV_SUFFICIENT_RATE = 0.85

CONTINUE_LLM_DETECTION = 0.60
CONTINUE_CV_LOW = 0.50
CONTINUE_LLM_ERROR = 30.0
CONTINUE_LLM_MIN_DETECTION = 0.40
CONTINUE_LLM_MAX_ERROR = 50.0


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class CalibrationReporter:

    def __init__(self):
        pass

    # ---- Markdown report ----

    def generate_markdown(
        self,
        report: ComparisonReport,
        session: Optional[CalibrationSession] = None,
    ) -> str:
        lines: list[str] = []
        _a = lines.append

        _a(f"# Calibration Comparison Report")
        _a(f"**Session:** {report.session_id}")
        _a(f"**Date:** {report.created_at or datetime.now().isoformat()}")
        if session:
            elapsed = session.end_time - session.start_time if session.end_time else 0
            _a(f"**Duration:** {elapsed:.1f}s")
        _a("")

        # -- Summary --
        _a("## Summary")
        _a("")
        _a(f"- **Total poses:** {report.total_poses}")

        cv_det_rates = [j.cv_detection_rate for j in report.joint_comparisons]
        llm_det_rates = [j.llm_detection_rate for j in report.joint_comparisons]
        cv_errors = [j.cv_mean_error_px for j in report.joint_comparisons if j.cv_mean_error_px is not None]
        llm_errors = [j.llm_mean_error_px for j in report.joint_comparisons if j.llm_mean_error_px is not None]

        _a(f"- **CV mean detection rate:** {_pct(cv_det_rates)}")
        _a(f"- **LLM mean detection rate:** {_pct(llm_det_rates)}")
        _a(f"- **CV mean error:** {_avg(cv_errors):.1f} px" if cv_errors else "- **CV mean error:** N/A")
        _a(f"- **LLM mean error:** {_avg(llm_errors):.1f} px" if llm_errors else "- **LLM mean error:** N/A")
        _a(f"- **CV mean latency:** {report.cv_mean_latency_ms:.1f} ms")
        _a(f"- **LLM mean latency:** {report.llm_mean_latency_ms:.1f} ms")
        rec = self._compute_recommendation(report)
        _a(f"- **Recommendation:** `{rec}`")
        _a("")

        # -- Per-Joint Analysis --
        _a("## Per-Joint Analysis")
        _a("")
        _a("| Joint | CV Det% | LLM Det% | CV Err(px) | LLM Err(px) | Agreement |")
        _a("|-------|---------|----------|------------|-------------|-----------|")
        for jc in report.joint_comparisons:
            cv_e = f"{jc.cv_mean_error_px:.1f}" if jc.cv_mean_error_px is not None else "—"
            llm_e = f"{jc.llm_mean_error_px:.1f}" if jc.llm_mean_error_px is not None else "—"
            _a(f"| {jc.joint_name:<14} | {jc.cv_detection_rate*100:5.1f}% | {jc.llm_detection_rate*100:6.1f}% | {cv_e:>10} | {llm_e:>11} | {jc.agreement_rate*100:5.1f}% |")
        _a("")

        # -- Per-Pose Detail --
        if report.results:
            _a("## Per-Pose Detail")
            _a("")
            _a("| Pose | Cam | CV Det | LLM Det | CV Err(mean) | LLM Err(mean) |")
            _a("|------|-----|--------|---------|--------------|---------------|")
            for r in report.results:
                cv_count = sum(1 for d in r.cv_detected if d)
                llm_count = sum(1 for d in r.llm_detected if d)
                cv_me = _avg([e for e in r.cv_errors_px if e is not None])
                llm_me = _avg([e for e in r.llm_errors_px if e is not None])
                cv_me_s = f"{cv_me:.1f}" if cv_me > 0 else "—"
                llm_me_s = f"{llm_me:.1f}" if llm_me > 0 else "—"
                angles_s = ",".join(f"{a:.0f}" for a in (r.joint_angles or [])[:3])
                _a(f"| {r.pose_index:>4} | {r.camera_id} | {cv_count}/5 | {llm_count}/5 | {cv_me_s:>12} | {llm_me_s:>13} |")
            _a("")

        # -- Cost Analysis --
        _a("## Cost Analysis")
        _a("")
        total_tokens = report.total_llm_input_tokens + report.total_llm_output_tokens
        _a(f"- **Total LLM tokens:** {total_tokens:,} (in: {report.total_llm_input_tokens:,}, out: {report.total_llm_output_tokens:,})")
        _a(f"- **Estimated cost:** ${report.total_llm_cost_usd:.4f}")
        if report.total_poses > 0:
            _a(f"- **Cost per pose:** ${report.total_llm_cost_usd / report.total_poses:.6f}")
        n_llm_det = sum(sum(1 for d in r.llm_detected if d) for r in report.results) if report.results else 0
        if n_llm_det > 0:
            _a(f"- **Cost per detection:** ${report.total_llm_cost_usd / n_llm_det:.6f}")
        _a("")

        # -- Recommendation --
        _a("## Recommendation")
        _a("")
        _a(f"**Verdict: {rec}**")
        _a("")

        # Areas where LLM helped
        helped = self._find_llm_helped(report)
        if helped:
            _a("### Where LLM Helped (CV missed, LLM found)")
            _a("")
            for h in helped:
                _a(f"- Pose {h[0]}, joint `{h[1]}`")
            _a("")

        # Areas where LLM failed
        failed = self._find_llm_failed(report)
        if failed:
            _a("### Where LLM Failed")
            _a("")
            for f_ in failed:
                _a(f"- Pose {f_[0]}, joint `{f_[1]}`: error {f_[2]:.1f}px")
            _a("")

        # -- ASCII Visualizations --
        _a("## Visualizations")
        _a("")
        _a(self._ascii_detection_bar_chart(report))
        _a("")
        _a(self._ascii_error_scatter(report))

        return "\n".join(lines)

    # ---- JSON report ----

    def generate_json(self, report: ComparisonReport) -> dict:
        rec = self._compute_recommendation(report)
        return {
            "session_id": report.session_id,
            "created_at": report.created_at,
            "total_poses": report.total_poses,
            "recommendation": rec,
            "cv_mean_latency_ms": report.cv_mean_latency_ms,
            "llm_mean_latency_ms": report.llm_mean_latency_ms,
            "total_llm_cost_usd": report.total_llm_cost_usd,
            "total_llm_tokens": report.total_llm_input_tokens + report.total_llm_output_tokens,
            "joints": [
                {
                    "name": jc.joint_name,
                    "cv_detection_rate": jc.cv_detection_rate,
                    "llm_detection_rate": jc.llm_detection_rate,
                    "cv_mean_error_px": jc.cv_mean_error_px,
                    "llm_mean_error_px": jc.llm_mean_error_px,
                    "agreement_rate": jc.agreement_rate,
                }
                for jc in report.joint_comparisons
            ],
            "per_pose": [
                {
                    "pose_index": r.pose_index,
                    "camera_id": r.camera_id,
                    "cv_detected": sum(1 for d in r.cv_detected if d),
                    "llm_detected": sum(1 for d in r.llm_detected if d),
                }
                for r in (report.results or [])
            ],
        }

    # ---- Save ----

    def save_report(
        self,
        report: ComparisonReport,
        session: CalibrationSession,
        output_dir: str = "calibration_results",
    ):
        out = Path(output_dir) / report.session_id
        out.mkdir(parents=True, exist_ok=True)

        md = self.generate_markdown(report, session)
        (out / "report.md").write_text(md)

        js = self.generate_json(report)
        (out / "summary.json").write_text(json.dumps(js, indent=2))

        # Raw data
        raw = []
        for r in report.results or []:
            try:
                raw.append(asdict(r))
            except Exception:
                raw.append({"pose_index": r.pose_index, "camera_id": r.camera_id})
        (out / "raw_results.json").write_text(json.dumps(raw, indent=2))

        return str(out)

    # ---- Internal helpers ----

    def _compute_recommendation(self, report: ComparisonReport) -> str:
        if not report.joint_comparisons:
            return "inconclusive"

        llm_rates = [j.llm_detection_rate for j in report.joint_comparisons]
        cv_rates = [j.cv_detection_rate for j in report.joint_comparisons]
        llm_errors = [j.llm_mean_error_px for j in report.joint_comparisons if j.llm_mean_error_px is not None]

        # Kill criteria
        if _avg(llm_rates) < KILL_DETECTION_RATE:
            return "archive"
        if llm_errors and _avg(llm_errors) > KILL_MEAN_ERROR:
            return "archive"
        if all(r >= CV_SUFFICIENT_RATE for r in cv_rates):
            return "archive"

        # Continue criteria
        llm_helps_low_cv = any(
            jc.llm_detection_rate > CONTINUE_LLM_DETECTION
            for jc in report.joint_comparisons
            if jc.cv_detection_rate < CONTINUE_CV_LOW
        )
        if llm_helps_low_cv:
            return "continue"
        if llm_errors and _avg(llm_errors) < CONTINUE_LLM_ERROR:
            return "continue"

        # Middle ground
        if _avg(llm_rates) >= CONTINUE_LLM_MIN_DETECTION:
            if not llm_errors or _avg(llm_errors) <= CONTINUE_LLM_MAX_ERROR:
                return "continue"

        return "inconclusive"

    def _find_llm_helped(self, report: ComparisonReport) -> list[tuple]:
        """Return (pose_index, joint_name) where CV missed but LLM found."""
        found = []
        for r in report.results or []:
            for i, name in enumerate(JOINT_NAMES):
                if i < len(r.cv_detected) and i < len(r.llm_detected):
                    if not r.cv_detected[i] and r.llm_detected[i]:
                        found.append((r.pose_index, name))
        return found

    def _find_llm_failed(self, report: ComparisonReport) -> list[tuple]:
        """Return (pose_index, joint_name, error) where LLM detected but error > 50px."""
        found = []
        for r in report.results or []:
            for i, name in enumerate(JOINT_NAMES):
                if i < len(r.llm_detected) and r.llm_detected[i]:
                    if i < len(r.llm_errors_px) and r.llm_errors_px[i] is not None:
                        if r.llm_errors_px[i] > 50.0:
                            found.append((r.pose_index, name, r.llm_errors_px[i]))
        return found

    def _ascii_detection_bar_chart(self, report: ComparisonReport) -> str:
        """ASCII bar chart of detection rates per joint."""
        lines = ["### Detection Rate by Joint", "```"]
        max_bar = 40
        for jc in report.joint_comparisons:
            name = jc.joint_name[:12].ljust(12)
            cv_len = int(jc.cv_detection_rate * max_bar)
            llm_len = int(jc.llm_detection_rate * max_bar)
            lines.append(f"{name} CV  |{'█' * cv_len}{'░' * (max_bar - cv_len)}| {jc.cv_detection_rate*100:.0f}%")
            lines.append(f"{'':12} LLM |{'▓' * llm_len}{'░' * (max_bar - llm_len)}| {jc.llm_detection_rate*100:.0f}%")
            lines.append("")
        lines.append("```")
        return "\n".join(lines)

    def _ascii_error_scatter(self, report: ComparisonReport) -> str:
        """ASCII scatter plot of CV vs LLM errors."""
        lines = ["### CV vs LLM Error (px)", "```"]
        # Collect paired errors
        pairs = []
        for r in report.results or []:
            for i in range(min(len(r.cv_errors_px), len(r.llm_errors_px))):
                cv_e = r.cv_errors_px[i]
                llm_e = r.llm_errors_px[i]
                if cv_e is not None and llm_e is not None:
                    pairs.append((cv_e, llm_e))

        if not pairs:
            lines.append("  (no paired error data)")
            lines.append("```")
            return "\n".join(lines)

        max_val = max(max(p[0], p[1]) for p in pairs)
        if max_val == 0:
            max_val = 1.0
        h, w = 15, 40

        grid = [[' '] * (w + 1) for _ in range(h + 1)]
        for cv_e, llm_e in pairs:
            x = min(int(cv_e / max_val * w), w)
            y = min(int(llm_e / max_val * h), h)
            grid[h - y][x] = '●'

        # Diagonal
        for i in range(min(h, w) + 1):
            r_idx = h - int(i * h / min(h, w)) if min(h, w) > 0 else 0
            c_idx = int(i * w / min(h, w)) if min(h, w) > 0 else 0
            if 0 <= r_idx <= h and 0 <= c_idx <= w and grid[r_idx][c_idx] == ' ':
                grid[r_idx][c_idx] = '·'

        lines.append(f"LLM err ^  (max={max_val:.0f}px)")
        for row in grid:
            lines.append("  |" + "".join(row))
        lines.append("  +" + "─" * (w + 1) + "> CV err")
        lines.append("```")
        return "\n".join(lines)


def _avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pct(vals: list) -> str:
    if not vals:
        return "N/A"
    return f"{_avg(vals)*100:.1f}%"
