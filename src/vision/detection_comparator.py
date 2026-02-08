"""
CV vs LLM detection comparison engine.

Runs both CV and LLM joint detection pipelines on the same frames,
computes metrics against FK ground truth, and generates comparison reports.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .joint_detector import JointDetector, JointDetection, DetectionSource, JOINT_NAMES
from .arm_segmenter import ArmSegmenter, ArmSegmentation
from .fk_engine import fk_positions, project_to_camera_pinhole

logger = logging.getLogger("th3cl4w.vision.detection_comparator")


@dataclass
class JointComparison:
    """Per-joint comparison metrics."""
    name: str
    fk_pixel: Optional[tuple[float, float]]
    cv_pixel: Optional[tuple[float, float]]
    llm_pixel: Optional[tuple[float, float]]
    cv_error_px: Optional[float]
    llm_error_px: Optional[float]
    agreement_px: Optional[float]
    cv_source: Optional[str]
    llm_confidence: Optional[str]


@dataclass
class ComparisonResult:
    """Result of comparing CV and LLM on a single frame."""
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
    recommendation: str  # "continue", "archive", or "inconclusive"


@dataclass
class PoseCapture:
    """Input data for a single pose comparison."""
    pose_index: int
    camera_id: int
    frame_bytes: bytes
    joint_angles: list[float]
    segmentation: object = None  # ArmSegmentation


def _pixel_dist(
    a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]
) -> Optional[float]:
    """Euclidean pixel distance, or None if either point is missing."""
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint_name_to_index(name: str) -> Optional[int]:
    try:
        return JOINT_NAMES.index(name)
    except ValueError:
        return None


class DetectionComparator:
    """Compare CV and LLM joint detection pipelines against FK ground truth."""

    AGREE_THRESHOLD_PX = 50.0

    # Gemini 2.0 Flash pricing
    COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
    COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

    def __init__(
        self,
        cv_detector: JointDetector,
        llm_detector=None,
        fk_engine_func: Optional[Callable] = None,
        camera_resolution: tuple[int, int] = (1920, 1080),
    ):
        """
        Args:
            cv_detector: CV-based JointDetector.
            llm_detector: LLM-based detector with async detect_joints(frame_bytes, camera_id).
            fk_engine_func: Callable(joint_angles) -> list[(px, py)] FK ground truth.
            camera_resolution: (width, height) for converting LLM normalized coords.
        """
        self.cv_detector = cv_detector
        self.llm_detector = llm_detector
        self.fk_engine_func = fk_engine_func
        self.camera_resolution = camera_resolution

    def compare_single(
        self,
        frame_bytes: bytes,
        camera_id: int,
        joint_angles: list[float],
        segmentation: ArmSegmentation,
        pose_index: int = 0,
    ) -> ComparisonResult:
        """Run both pipelines on a single frame and compare against FK ground truth."""
        # 1. FK ground truth
        fk_pixels = self.fk_engine_func(joint_angles) if self.fk_engine_func else [(0, 0)] * 5

        # 2. CV pipeline
        t0 = time.monotonic()
        cv_dets = self.cv_detector.detect_joints(segmentation, list(fk_pixels))
        cv_latency = (time.monotonic() - t0) * 1000

        # 3. LLM pipeline
        llm_pixels: list[Optional[tuple[float, float]]] = [None] * 5
        llm_confidences: list[Optional[str]] = [None] * 5
        llm_latency = 0.0
        llm_tokens = 0

        if self.llm_detector:
            llm_result = self._run_llm(frame_bytes, camera_id)
            llm_latency = llm_result.get("latency_ms", 0.0)
            llm_tokens = llm_result.get("input_tokens", 0) + llm_result.get("output_tokens", 0)
            w, h = self.camera_resolution
            for j in llm_result.get("joints", []):
                idx = _joint_name_to_index(j.get("name", ""))
                if idx is not None and j.get("x") is not None and j.get("y") is not None:
                    llm_pixels[idx] = (j["x"] * w, j["y"] * h)
                    llm_confidences[idx] = j.get("confidence", "medium")

        # 4. Build per-joint comparisons
        joints: list[JointComparison] = []
        cv_map = {d.joint_index: d for d in cv_dets}
        for i, name in enumerate(JOINT_NAMES):
            fk_px = fk_pixels[i] if i < len(fk_pixels) else None
            cv_det = cv_map.get(i)

            # Don't count FK_ONLY as real CV detection
            cv_pixel = None
            cv_source = None
            if cv_det:
                cv_source = cv_det.source.value
                if cv_det.source != DetectionSource.FK_ONLY:
                    cv_pixel = cv_det.pixel_pos

            joints.append(JointComparison(
                name=name,
                fk_pixel=fk_px,
                cv_pixel=cv_pixel,
                llm_pixel=llm_pixels[i],
                cv_error_px=_pixel_dist(cv_pixel, fk_px),
                llm_error_px=_pixel_dist(llm_pixels[i], fk_px),
                agreement_px=_pixel_dist(cv_pixel, llm_pixels[i]),
                cv_source=cv_source,
                llm_confidence=llm_confidences[i],
            ))

        return ComparisonResult(
            pose_index=pose_index,
            camera_id=camera_id,
            joint_angles=joint_angles,
            joints=joints,
            cv_latency_ms=cv_latency,
            llm_latency_ms=llm_latency,
            llm_tokens=llm_tokens,
        )

    def _run_llm(self, frame_bytes: bytes, camera_id: int) -> dict:
        """Run LLM detector, handling async interface."""
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(
                            asyncio.run,
                            self.llm_detector.detect_joints(frame_bytes, camera_id)
                        ).result(timeout=30)
                else:
                    return loop.run_until_complete(
                        self.llm_detector.detect_joints(frame_bytes, camera_id)
                    )
            except RuntimeError:
                return asyncio.run(
                    self.llm_detector.detect_joints(frame_bytes, camera_id)
                )
        except Exception as e:
            logger.warning("LLM detection failed: %s", e)
            return {"joints": [], "input_tokens": 0, "output_tokens": 0}

    def compare_batch(self, poses: list[PoseCapture]) -> ComparisonReport:
        """Run comparison across all calibration poses, aggregate stats."""
        results = []
        for pose in poses:
            result = self.compare_single(
                frame_bytes=pose.frame_bytes,
                camera_id=pose.camera_id,
                joint_angles=pose.joint_angles,
                segmentation=pose.segmentation,
                pose_index=pose.pose_index,
            )
            results.append(result)
        return self._aggregate(results)

    def _aggregate(self, results: list[ComparisonResult]) -> ComparisonReport:
        """Compute aggregate metrics."""
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
        agreement_eligible = 0
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
                if jc.cv_pixel is not None and jc.llm_pixel is not None:
                    agreement_eligible += 1
                    if jc.agreement_px is not None and jc.agreement_px < self.AGREE_THRESHOLD_PX:
                        agreements += 1

        cv_det_rate = cv_detected / total_joints if total_joints > 0 else 0.0
        llm_det_rate = llm_detected / total_joints if total_joints > 0 else 0.0
        cv_mean = float(np.mean(cv_errors)) if cv_errors else 0.0
        llm_mean = float(np.mean(llm_errors)) if llm_errors else 0.0
        agree_rate = agreements / agreement_eligible if agreement_eligible > 0 else 0.0

        # Estimate cost
        input_tok = int(total_tokens * 0.75)
        output_tok = total_tokens - input_tok
        cost = input_tok * self.COST_PER_INPUT_TOKEN + output_tok * self.COST_PER_OUTPUT_TOKEN

        recommendation = self._recommend(llm_det_rate, llm_mean, agree_rate)

        return ComparisonReport(
            results=results,
            cv_detection_rate=cv_det_rate,
            llm_detection_rate=llm_det_rate,
            cv_mean_error_px=cv_mean,
            llm_mean_error_px=llm_mean,
            agreement_rate=agree_rate,
            total_llm_tokens=total_tokens,
            total_llm_cost_usd=cost,
            recommendation=recommendation,
        )

    @staticmethod
    def _recommend(det_rate: float, mean_error: float, agreement_rate: float) -> str:
        """Kill/continue decision based on thresholds."""
        # Kill criteria
        if det_rate < 0.30:
            return "archive"
        if mean_error > 100.0:
            return "archive"
        # Success criteria
        if det_rate > 0.50 and mean_error < 50.0 and agreement_rate > 0.60:
            return "continue"
        return "archive"

    def summary(self, report: ComparisonReport) -> dict:
        """Generate summary metrics dict."""
        return {
            "num_poses": len(report.results),
            "cv_detection_rate": round(report.cv_detection_rate, 3),
            "llm_detection_rate": round(report.llm_detection_rate, 3),
            "cv_mean_error_px": round(report.cv_mean_error_px, 1),
            "llm_mean_error_px": round(report.llm_mean_error_px, 1),
            "agreement_rate": round(report.agreement_rate, 3),
            "total_llm_tokens": report.total_llm_tokens,
            "total_llm_cost_usd": round(report.total_llm_cost_usd, 6),
            "recommendation": report.recommendation,
        }
