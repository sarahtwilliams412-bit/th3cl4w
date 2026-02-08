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
# Import actual dataclasses from detection_comparator, with stubs as fallback
# ---------------------------------------------------------------------------

try:
    from src.vision.detection_comparator import (
        ComparisonReport,
        ComparisonResult,
        JointComparison,
    )
    _USING_REAL_CLASSES = True
except ImportError:
    _USING_REAL_CLASSES = False

    @dataclass
    class JointComparison:
        name: str = ""
        fk_pixel: tuple | None = None
        cv_pixel: tuple | None = None
        llm_pixel: tuple | None = None
        cv_error_px: float | None = None
        llm_error_px: float | None = None
        agreement_px: float | None = None
        cv_source: str | None = None
        llm_confidence: float | None = None

    @dataclass
    class ComparisonResult:
        pose_index: int = 0
        camera_id: int = 0
        joint_angles: list = field(default_factory=list)
        joints: list = field(default_factory=list)  # list[JointComparison]
        cv_latency_ms: float = 0.0
        llm_latency_ms: float = 0.0
        llm_tokens: int = 0

    @dataclass
    class ComparisonReport:
        results: list = field(default_factory=list)  # list[ComparisonResult]
        cv_detection_rate: float = 0.0
        llm_detection_rate: float = 0.0
        cv_mean_error_px: float = 0.0
        llm_mean_error_px: float = 0.0
        agreement_rate: float = 0.0
        total_llm_tokens: int = 0
        total_llm_cost_usd: float = 0.0
        recommendation: str = ""


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
CV_SUFFICIENT_RATE = 0.85

CONTINUE_LLM_DETECTION = 0.60
CONTINUE_CV_LOW = 0.50
CONTINUE_LLM_ERROR = 30.0
CONTINUE_LLM_MIN_DETECTION = 0.40
CONTINUE_LLM_MAX_ERROR = 50.0

# Gemini Flash pricing
COST_PER_TOKEN = 0.15 / 1_000_000  # blended estimate


# ---------------------------------------------------------------------------
# Helper: extract per-joint stats from ComparisonReport
# ---------------------------------------------------------------------------

def _joint_stats(report: ComparisonReport) -> list[dict]:
    """Aggregate per-joint stats from results."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"cv_det": 0, "llm_det": 0, "cv_errs": [], "llm_errs": [], "agree": 0, "total": 0})

    for r in report.results:
        for jc in r.joints:
            name = jc.name
            s = stats[name]
            s["total"] += 1
            if jc.cv_pixel is not None:
                s["cv_det"] += 1
            if jc.llm_pixel is not None:
                s["llm_det"] += 1
            if jc.cv_error_px is not None:
                s["cv_errs"].append(jc.cv_error_px)
            if jc.llm_error_px is not None:
                s["llm_errs"].append(jc.llm_error_px)
            if jc.cv_pixel is not None and jc.llm_pixel is not None:
                s["agree"] += 1

    result = []
    for name in JOINT_NAMES:
        if name not in stats:
            continue
        s = stats[name]
        total = s["total"] or 1
        result.append({
            "name": name,
            "cv_detection_rate": s["cv_det"] / total,
            "llm_detection_rate": s["llm_det"] / total,
            "cv_mean_error_px": _avg(s["cv_errs"]) if s["cv_errs"] else None,
            "llm_mean_error_px": _avg(s["llm_errs"]) if s["llm_errs"] else None,
            "agreement_rate": s["agree"] / total,
        })
    return result


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

        session_id = getattr(session, 'session_id', '') or 'unknown'
        _a("# Calibration Comparison Report")
        _a(f"**Session:** {session_id}")
        _a(f"**Date:** {datetime.now().isoformat()}")
        if session:
            elapsed = session.end_time - session.start_time if session.end_time else 0
            _a(f"**Duration:** {elapsed:.1f}s")
        _a("")

        total_poses = len(report.results)
        jstats = _joint_stats(report)

        # -- Summary --
        _a("## Summary")
        _a("")
        _a(f"- **Total poses:** {total_poses}")
        _a(f"- **CV mean detection rate:** {report.cv_detection_rate*100:.1f}%")
        _a(f"- **LLM mean detection rate:** {report.llm_detection_rate*100:.1f}%")
        _a(f"- **CV mean error:** {report.cv_mean_error_px:.1f} px")
        _a(f"- **LLM mean error:** {report.llm_mean_error_px:.1f} px")

        cv_lats = [r.cv_latency_ms for r in report.results]
        llm_lats = [r.llm_latency_ms for r in report.results]
        _a(f"- **CV mean latency:** {_avg(cv_lats):.1f} ms")
        _a(f"- **LLM mean latency:** {_avg(llm_lats):.1f} ms")
        rec = self._compute_recommendation(report)
        _a(f"- **Recommendation:** `{rec}`")
        _a("")

        # -- Per-Joint Analysis --
        _a("## Per-Joint Analysis")
        _a("")
        _a("| Joint | CV Det% | LLM Det% | CV Err(px) | LLM Err(px) | Agreement |")
        _a("|-------|---------|----------|------------|-------------|-----------|")
        for js in jstats:
            cv_e = f"{js['cv_mean_error_px']:.1f}" if js['cv_mean_error_px'] is not None else "—"
            llm_e = f"{js['llm_mean_error_px']:.1f}" if js['llm_mean_error_px'] is not None else "—"
            _a(f"| {js['name']:<14} | {js['cv_detection_rate']*100:5.1f}% | {js['llm_detection_rate']*100:6.1f}% | {cv_e:>10} | {llm_e:>11} | {js['agreement_rate']*100:5.1f}% |")
        _a("")

        # -- Per-Pose Detail --
        if report.results:
            _a("## Per-Pose Detail")
            _a("")
            _a("| Pose | Cam | CV Det | LLM Det | CV Err(mean) | LLM Err(mean) |")
            _a("|------|-----|--------|---------|--------------|---------------|")
            for r in report.results:
                n_joints = len(r.joints)
                cv_count = sum(1 for j in r.joints if j.cv_pixel is not None)
                llm_count = sum(1 for j in r.joints if j.llm_pixel is not None)
                cv_errs = [j.cv_error_px for j in r.joints if j.cv_error_px is not None]
                llm_errs = [j.llm_error_px for j in r.joints if j.llm_error_px is not None]
                cv_me_s = f"{_avg(cv_errs):.1f}" if cv_errs else "—"
                llm_me_s = f"{_avg(llm_errs):.1f}" if llm_errs else "—"
                _a(f"| {r.pose_index:>4} | {r.camera_id} | {cv_count}/{n_joints} | {llm_count}/{n_joints} | {cv_me_s:>12} | {llm_me_s:>13} |")
            _a("")

        # -- Cost Analysis --
        _a("## Cost Analysis")
        _a("")
        _a(f"- **Total LLM tokens:** {report.total_llm_tokens:,}")
        _a(f"- **Estimated cost:** ${report.total_llm_cost_usd:.4f}")
        if total_poses > 0:
            _a(f"- **Cost per pose:** ${report.total_llm_cost_usd / total_poses:.6f}")
        n_llm_det = sum(1 for r in report.results for j in r.joints if j.llm_pixel is not None)
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
        _a(self._ascii_detection_bar_chart(jstats))
        _a("")
        _a(self._ascii_error_scatter(report))

        return "\n".join(lines)

    # ---- JSON report ----

    def generate_json(self, report: ComparisonReport) -> dict:
        rec = self._compute_recommendation(report)
        jstats = _joint_stats(report)
        return {
            "session_id": "",
            "total_poses": len(report.results),
            "recommendation": rec,
            "cv_detection_rate": report.cv_detection_rate,
            "llm_detection_rate": report.llm_detection_rate,
            "cv_mean_error_px": report.cv_mean_error_px,
            "llm_mean_error_px": report.llm_mean_error_px,
            "total_llm_cost_usd": report.total_llm_cost_usd,
            "total_llm_tokens": report.total_llm_tokens,
            "joints": jstats,
            "per_pose": [
                {
                    "pose_index": r.pose_index,
                    "camera_id": r.camera_id,
                    "cv_detected": sum(1 for j in r.joints if j.cv_pixel is not None),
                    "llm_detected": sum(1 for j in r.joints if j.llm_pixel is not None),
                }
                for r in report.results
            ],
        }

    # ---- Save ----

    def save_report(
        self,
        report: ComparisonReport,
        session: CalibrationSession,
        output_dir: str = "calibration_results",
    ):
        sid = session.session_id or "unknown"
        out = Path(output_dir) / sid
        out.mkdir(parents=True, exist_ok=True)

        md = self.generate_markdown(report, session)
        (out / "report.md").write_text(md)

        js = self.generate_json(report)
        (out / "summary.json").write_text(json.dumps(js, indent=2))

        # Raw data
        raw = []
        for r in report.results:
            try:
                raw.append(asdict(r))
            except Exception:
                raw.append({"pose_index": r.pose_index, "camera_id": r.camera_id})
        (out / "raw_results.json").write_text(json.dumps(raw, indent=2))

        return str(out)

    # ---- Internal helpers ----

    def _compute_recommendation(self, report: ComparisonReport) -> str:
        jstats = _joint_stats(report)
        if not jstats:
            return "inconclusive"

        llm_rates = [j["llm_detection_rate"] for j in jstats]
        cv_rates = [j["cv_detection_rate"] for j in jstats]
        llm_errors = [j["llm_mean_error_px"] for j in jstats if j["llm_mean_error_px"] is not None]

        # Kill criteria
        if _avg(llm_rates) < KILL_DETECTION_RATE:
            return "archive"
        if llm_errors and _avg(llm_errors) > KILL_MEAN_ERROR:
            return "archive"
        if all(r >= CV_SUFFICIENT_RATE for r in cv_rates):
            return "archive"

        # Continue criteria
        llm_helps_low_cv = any(
            j["llm_detection_rate"] > CONTINUE_LLM_DETECTION
            for j in jstats
            if j["cv_detection_rate"] < CONTINUE_CV_LOW
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
        for r in report.results:
            for jc in r.joints:
                if jc.cv_pixel is None and jc.llm_pixel is not None:
                    found.append((r.pose_index, jc.name))
        return found

    def _find_llm_failed(self, report: ComparisonReport) -> list[tuple]:
        """Return (pose_index, joint_name, error) where LLM detected but error > 50px."""
        found = []
        for r in report.results:
            for jc in r.joints:
                if jc.llm_pixel is not None and jc.llm_error_px is not None:
                    if jc.llm_error_px > 50.0:
                        found.append((r.pose_index, jc.name, jc.llm_error_px))
        return found

    def _ascii_detection_bar_chart(self, jstats: list[dict]) -> str:
        """ASCII bar chart of detection rates per joint."""
        lines = ["### Detection Rate by Joint", "```"]
        max_bar = 40
        for js in jstats:
            name = js["name"][:12].ljust(12)
            cv_len = int(js["cv_detection_rate"] * max_bar)
            llm_len = int(js["llm_detection_rate"] * max_bar)
            lines.append(f"{name} CV  |{'█' * cv_len}{'░' * (max_bar - cv_len)}| {js['cv_detection_rate']*100:.0f}%")
            lines.append(f"{'':12} LLM |{'▓' * llm_len}{'░' * (max_bar - llm_len)}| {js['llm_detection_rate']*100:.0f}%")
            lines.append("")
        lines.append("```")
        return "\n".join(lines)

    def _ascii_error_scatter(self, report: ComparisonReport) -> str:
        """ASCII scatter plot of CV vs LLM errors."""
        lines = ["### CV vs LLM Error (px)", "```"]
        pairs = []
        for r in report.results:
            for jc in r.joints:
                if jc.cv_error_px is not None and jc.llm_error_px is not None:
                    pairs.append((jc.cv_error_px, jc.llm_error_px))

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
