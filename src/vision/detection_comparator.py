"""
CV vs LLM detection comparison engine.

Runs both CV and LLM pipelines on the same frames, computes metrics
against FK ground truth, and generates comparison reports.
"""

import asyncio
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .joint_detector import JointDetector, JointDetection, DetectionSource, JOINT_NAMES
from .arm_segmenter import ArmSegmenter, ArmSegmentation
from .fk_engine import fk_positions, project_to_camera_pinhole

try:
    from .llm_detector import LLMJointDetector, LLMDetectionResult
except ImportError:
    LLMJointDetector = None
    LLMDetectionResult = None

logger = logging.getLogger("th3cl4w.vision.detection_comparator")


@dataclass
class JointComparison:
    """Per-joint comparison between CV, LLM, and FK ground truth."""
    name: str
    fk_pixel: Optional[tuple[float, float]]
    cv_pixel: Optional[tuple[float, float]]
    llm_pixel: Optional[tuple[float, float]]
    cv_error_px: Optional[float]
    llm_error_px: Optional[float]
    agreement_px: Optional[float]  # distance between CV and LLM
    cv_source: Optional[str]  # DetectionSource value
    llm_confidence: Optional[float]


@dataclass
class ComparisonResult:
    """Result of comparing both pipelines on a single frame/pose."""
    pose_index: int
    camera_id: int
    joint_angles: list[float]
    joints: list[JointComparison]
    cv_latency_ms: float
    llm_latency_ms: float
    llm_tokens: int


@dataclass
class ComparisonReport:
    """Aggregated report across all poses."""
    results: list[ComparisonResult]
    cv_detection_rate: float
    llm_detection_rate: float
    cv_mean_error_px: float
    llm_mean_error_px: float
    agreement_rate: float
    total_llm_tokens: int
    total_llm_cost_usd: float
    recommendation: str  # "archive", "continue", or "inconclusive"


def _pixel_dist(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_joint_comparison(
    name: str,
    fk_pixel: Optional[tuple[float, float]],
    cv_det: Optional[JointDetection],
    llm_pixel: Optional[tuple[float, float]],
    llm_confidence: Optional[float] = None,
) -> JointComparison:
    """Compute metrics for a single joint."""
    cv_pixel = cv_det.pixel_pos if cv_det else None
    cv_source = cv_det.source.value if cv_det else None

    # Don't count FK_ONLY as a real CV detection
    if cv_det and cv_det.source == DetectionSource.FK_ONLY:
        cv_pixel_for_error = None
    else:
        cv_pixel_for_error = cv_pixel

    return JointComparison(
        name=name,
        fk_pixel=fk_pixel,
        cv_pixel=cv_pixel_for_error,
        llm_pixel=llm_pixel,
        cv_error_px=_pixel_dist(cv_pixel_for_error, fk_pixel),
        llm_error_px=_pixel_dist(llm_pixel, fk_pixel),
        agreement_px=_pixel_dist(cv_pixel_for_error, llm_pixel),
        cv_source=cv_source,
        llm_confidence=llm_confidence,
    )


def compute_report(results: list[ComparisonResult], total_llm_cost_usd: float = 0.0) -> ComparisonReport:
    """Aggregate comparison results into a report with recommendation."""
    if not results:
        return ComparisonReport(
            results=[], cv_detection_rate=0.0, llm_detection_rate=0.0,
            cv_mean_error_px=0.0, llm_mean_error_px=0.0, agreement_rate=0.0,
            total_llm_tokens=0, total_llm_cost_usd=0.0, recommendation="inconclusive",
        )

    total_joints = 0
    cv_detected = 0
    llm_detected = 0
    cv_errors: list[float] = []
    llm_errors: list[float] = []
    agreements = 0
    agreement_total = 0
    total_tokens = 0

    for r in results:
        total_tokens += r.llm_tokens
        for jc in r.joints:
            total_joints += 1
            if jc.cv_pixel is not None:
                cv_detected += 1
                if jc.cv_error_px is not None:
                    cv_errors.append(jc.cv_error_px)
            if jc.llm_pixel is not None:
                llm_detected += 1
                if jc.llm_error_px is not None:
                    llm_errors.append(jc.llm_error_px)
            # Agreement: both detected
            if jc.cv_pixel is not None and jc.llm_pixel is not None:
                agreement_total += 1
                if jc.agreement_px is not None and jc.agreement_px < 50.0:
                    agreements += 1

    cv_det_rate = cv_detected / total_joints if total_joints > 0 else 0.0
    llm_det_rate = llm_detected / total_joints if total_joints > 0 else 0.0
    cv_mean = float(np.mean(cv_errors)) if cv_errors else 0.0
    llm_mean = float(np.mean(llm_errors)) if llm_errors else 0.0
    agree_rate = agreements / agreement_total if agreement_total > 0 else 0.0

    # Recommendation logic
    recommendation = "inconclusive"
    if llm_det_rate < 0.3 or llm_mean > 100.0:
        recommendation = "archive"
    elif llm_det_rate > 0.5 and llm_mean < 50.0 and agree_rate > 0.6:
        recommendation = "continue"

    return ComparisonReport(
        results=results,
        cv_detection_rate=cv_det_rate,
        llm_detection_rate=llm_det_rate,
        cv_mean_error_px=cv_mean,
        llm_mean_error_px=llm_mean,
        agreement_rate=agree_rate,
        total_llm_tokens=total_tokens,
        total_llm_cost_usd=total_llm_cost_usd,
        recommendation=recommendation,
    )


class DetectionComparator:
    """Run CV and LLM pipelines on frames and compare against FK ground truth."""

    def __init__(
        self,
        segmenter: ArmSegmenter,
        joint_detector: JointDetector,
        llm_detector=None,  # LLMJointDetector or None
        camera_intrinsics: Optional[dict] = None,
    ):
        self.segmenter = segmenter
        self.joint_detector = joint_detector
        self.llm_detector = llm_detector
        self.camera_intrinsics = camera_intrinsics or {}
        self._results: list[ComparisonResult] = []
        self._total_llm_cost: float = 0.0

    def get_fk_pixels(
        self, joint_angles: list[float], camera_id: int,
    ) -> list[Optional[tuple[float, float]]]:
        """Get FK ground truth projected to pixel space."""
        params = self.camera_intrinsics.get(camera_id)
        if params is None:
            return [None] * 5
        positions = fk_positions(joint_angles)
        return project_to_camera_pinhole(
            positions,
            fx=params["fx"], fy=params["fy"],
            cx=params["cx"], cy=params["cy"],
            rvec=params["rvec"], tvec=params["tvec"],
        )

    def compare_single(
        self,
        frame: np.ndarray,
        joint_angles: list[float],
        camera_id: int,
        pose_index: int,
        llm_result: Optional[dict] = None,
    ) -> ComparisonResult:
        """Compare CV and LLM on a single frame (synchronous).

        Args:
            frame: BGR image
            joint_angles: degrees
            camera_id: camera index
            pose_index: pose number
            llm_result: pre-computed LLM result dict with 'joints', 'latency_ms', 'tokens', 'cost'
        """
        fk_pixels = self.get_fk_pixels(joint_angles, camera_id)

        # CV pipeline
        t0 = time.monotonic()
        seg = self.segmenter.segment_arm(frame)
        fk_for_cv = [(p[0], p[1]) if p else (0.0, 0.0) for p in fk_pixels]
        cv_dets = self.joint_detector.detect_joints(seg, fk_for_cv)
        cv_latency = (time.monotonic() - t0) * 1000

        # Parse LLM results
        llm_pixels: list[Optional[tuple[float, float]]] = [None] * 5
        llm_confidences: list[Optional[float]] = [None] * 5
        llm_latency = 0.0
        llm_tokens = 0

        if llm_result:
            llm_latency = llm_result.get("latency_ms", 0.0)
            llm_tokens = llm_result.get("input_tokens", 0) + llm_result.get("output_tokens", 0)
            self._total_llm_cost += llm_result.get("cost_usd", 0.0)
            h, w = frame.shape[:2]
            for j in llm_result.get("joints", []):
                idx = _joint_name_to_index(j.get("name", ""))
                if idx is not None and j.get("x") is not None and j.get("y") is not None:
                    llm_pixels[idx] = (j["x"] * w, j["y"] * h)
                    llm_confidences[idx] = j.get("confidence")

        # Build joint comparisons
        joints: list[JointComparison] = []
        for i, name in enumerate(JOINT_NAMES):
            cv_det = cv_dets[i] if i < len(cv_dets) else None
            joints.append(compute_joint_comparison(
                name, fk_pixels[i], cv_det, llm_pixels[i], llm_confidences[i],
            ))

        result = ComparisonResult(
            pose_index=pose_index,
            camera_id=camera_id,
            joint_angles=joint_angles,
            joints=joints,
            cv_latency_ms=cv_latency,
            llm_latency_ms=llm_latency,
            llm_tokens=llm_tokens,
        )
        self._results.append(result)
        return result

    def generate_report(self) -> ComparisonReport:
        """Generate aggregated comparison report from all collected results."""
        return compute_report(self._results, self._total_llm_cost)

    def reset(self) -> None:
        self._results.clear()
        self._total_llm_cost = 0.0


def _joint_name_to_index(name: str) -> Optional[int]:
    try:
        return JOINT_NAMES.index(name)
    except ValueError:
        return None
