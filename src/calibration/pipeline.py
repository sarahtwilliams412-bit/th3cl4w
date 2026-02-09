"""End-to-end LLM calibration pipeline."""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from src.calibration.calibration_runner import CalibrationRunner, CalibrationSession
from src.calibration.results_reporter import (
    CalibrationReporter,
    CalibrationSession as ReporterSession,
    ComparisonReport,
    ComparisonResult,
    JointComparison,
)
from src.vision.joint_detector import JointDetector, DetectionSource, JOINT_NAMES
from src.vision.arm_segmenter import ArmSegmenter
from src.vision.fk_engine import fk_positions, project_to_camera_pinhole

try:
    from src.vision.llm_detector import LLMJointDetector
except Exception:
    LLMJointDetector = None

logger = logging.getLogger("th3cl4w.calibration.pipeline")


def _pixel_dist(a, b):
    if a is None or b is None:
        return None
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LLMCalibrationPipeline:
    """End-to-end pipeline: calibration runner + dual detection + comparison + reporting."""

    AGREE_THRESHOLD_PX = 50.0
    CAMERA_RESOLUTION = (1920, 1080)

    def __init__(
        self,
        arm_host: str = 'localhost',
        arm_port: int = 8080,
        cam_host: str = 'localhost',
        cam_port: int = 8081,
        gemini_api_key: str | None = None,
        settle_time: float = 2.5,
    ):
        self.runner = CalibrationRunner(arm_host, arm_port, cam_host, cam_port, settle_time=settle_time)
        self.segmenter = ArmSegmenter()
        self.cv_detector = JointDetector()

        self.llm_detector = None
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if LLMJointDetector is not None and api_key:
            try:
                self.llm_detector = LLMJointDetector(api_key=api_key)
                logger.info("LLM detector initialized successfully")
            except Exception as e:
                logger.warning(f"LLM detector unavailable: {e}")
        elif not api_key:
            logger.warning("No GEMINI_API_KEY — LLM detector disabled")

        self.reporter = CalibrationReporter()

    async def _detect_pose(self, capture, pose_index: int) -> ComparisonResult:
        """Run CV + LLM detection on a single captured pose, return ComparisonResult."""
        import cv2
        import numpy as np

        w, h = self.CAMERA_RESOLUTION
        joint_angles = capture.actual_angles

        # FK ground truth
        try:
            fk_3d = fk_positions(joint_angles)
            fk_pixels = [project_to_camera_pinhole(p, camera_id=0) for p in fk_3d]
        except Exception:
            fk_pixels = [(0, 0)] * 5

        # CV pipeline
        import time
        cv_latency = 0.0
        cv_dets = []
        if capture.cam0_jpeg:
            frame = cv2.imdecode(np.frombuffer(capture.cam0_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                try:
                    t0 = time.monotonic()
                    seg = self.segmenter.segment_arm(frame)
                    cv_dets = self.cv_detector.detect_joints(seg, list(fk_pixels))
                    cv_latency = (time.monotonic() - t0) * 1000
                except Exception as e:
                    logger.warning(f"CV detection failed pose {pose_index}: {e}")

        # LLM pipeline
        llm_pixels = [None] * 5
        llm_confidences = [None] * 5
        llm_latency = 0.0
        llm_tokens = 0

        if self.llm_detector and capture.cam0_jpeg:
            try:
                llm_result = await self.llm_detector.detect_joints(
                    capture.cam0_jpeg, camera_id=0,
                    joint_angles=joint_angles,
                )
                llm_latency = llm_result.latency_ms
                llm_tokens = llm_result.tokens_used
                if llm_result.success:
                    for j in llm_result.joints:
                        try:
                            idx = JOINT_NAMES.index(j.name)
                        except ValueError:
                            continue
                        llm_pixels[idx] = (j.norm_x * w, j.norm_y * h)
                        llm_confidences[idx] = j.confidence
            except Exception as e:
                logger.warning(f"LLM detection failed pose {pose_index}: {e}")

        # Build per-joint comparisons
        cv_map = {d.joint_index: d for d in cv_dets}
        joints = []
        for i, name in enumerate(JOINT_NAMES):
            fk_px = fk_pixels[i] if i < len(fk_pixels) else None
            cv_det = cv_map.get(i)
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
            camera_id=0,
            joint_angles=joint_angles,
            joints=joints,
            cv_latency_ms=cv_latency,
            llm_latency_ms=llm_latency,
            llm_tokens=llm_tokens,
        )

    def _aggregate(self, results: list[ComparisonResult]) -> ComparisonReport:
        """Aggregate comparison results into a report."""
        import numpy as np

        if not results:
            return ComparisonReport(
                results=[], cv_detection_rate=0.0, llm_detection_rate=0.0,
                cv_mean_error_px=0.0, llm_mean_error_px=0.0, agreement_rate=0.0,
                total_llm_tokens=0, total_llm_cost_usd=0.0, recommendation="inconclusive",
            )

        total_joints = 0
        cv_detected = 0
        llm_detected = 0
        cv_errors = []
        llm_errors = []
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

        cv_det_rate = cv_detected / total_joints if total_joints else 0.0
        llm_det_rate = llm_detected / total_joints if total_joints else 0.0
        cv_mean = float(np.mean(cv_errors)) if cv_errors else 0.0
        llm_mean = float(np.mean(llm_errors)) if llm_errors else 0.0
        agree_rate = agreements / agreement_eligible if agreement_eligible else 0.0

        # Cost estimate (Gemini 2.0 Flash)
        input_tok = int(total_tokens * 0.75)
        output_tok = total_tokens - input_tok
        cost = input_tok * 0.075 / 1_000_000 + output_tok * 0.30 / 1_000_000

        recommendation = "inconclusive"
        if llm_det_rate < 0.30:
            recommendation = "archive"
        elif llm_mean > 100.0:
            recommendation = "archive"
        elif llm_det_rate > 0.50 and llm_mean < 50.0 and agree_rate > 0.60:
            recommendation = "continue"

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

    async def run(self, output_dir: str = 'calibration_results') -> str:
        """Run full calibration pipeline, return path to report directory."""
        # Run calibration (arm movement + frame capture)
        session = await self.runner.run_full_calibration()

        # Run CV + LLM detection on all captured frames
        comparison_results = []
        for capture in session.captures:
            result = await self._detect_pose(capture, capture.pose_index)
            comparison_results.append(result)

        report = self._aggregate(comparison_results)

        # Save results
        session_id = self.runner._session_id or "calibration"
        report_session = ReporterSession(
            session_id=session_id,
            start_time=session.start_time,
            end_time=session.end_time,
        )

        out_path = Path(output_dir) / f"cal_{session_id}" if not session_id.startswith("cal_") else Path(output_dir) / session_id
        out_path.mkdir(parents=True, exist_ok=True)

        # Save report files via reporter
        self.reporter.save_report(report, report_session, output_dir)

        # Save frames
        frames_dir = out_path / "frames"
        frames_dir.mkdir(exist_ok=True)
        for cap in session.captures:
            if cap.cam0_jpeg:
                (frames_dir / f"pose_{cap.pose_index:02d}_cam0.jpg").write_bytes(cap.cam0_jpeg)
            if cap.cam1_jpeg:
                (frames_dir / f"pose_{cap.pose_index:02d}_cam1.jpg").write_bytes(cap.cam1_jpeg)

        # Save raw calibration session data
        raw_session = {
            "session_id": session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "total_poses": session.total_poses,
            "captures": [
                {
                    "pose_index": c.pose_index,
                    "commanded_angles": list(c.commanded_angles),
                    "actual_angles": c.actual_angles,
                    "timestamp": c.timestamp,
                    "has_cam0": len(c.cam0_jpeg) > 0,
                    "has_cam1": len(c.cam1_jpeg) > 0,
                }
                for c in session.captures
            ],
        }
        (out_path / "calibration_session.json").write_text(json.dumps(raw_session, indent=2))

        logger.info(f"Calibration results saved to {out_path}")
        return str(out_path)

    async def run_detection_only(self, session: CalibrationSession) -> ComparisonReport:
        """Run CV + LLM detection on an existing calibration session (no arm movement)."""
        comparison_results = []
        for capture in session.captures:
            result = await self._detect_pose(capture, capture.pose_index)
            comparison_results.append(result)
        return self._aggregate(comparison_results)

    def save_results(
        self,
        session: CalibrationSession,
        report: ComparisonReport,
        output_dir: str = 'calibration_results',
    ) -> str:
        """Save calibration results to disk. Returns output path."""
        session_id = self.runner._session_id or "calibration"
        out_path = Path(output_dir) / session_id
        out_path.mkdir(parents=True, exist_ok=True)

        # Reporter saves raw_results.json, summary.json, report.md
        report_session = ReporterSession(
            session_id=session_id,
            start_time=session.start_time,
            end_time=session.end_time,
        )
        self.reporter.save_report(report, report_session, output_dir)

        # Save frames
        frames_dir = out_path / "frames"
        frames_dir.mkdir(exist_ok=True)
        for cap in session.captures:
            if cap.cam0_jpeg:
                (frames_dir / f"pose_{cap.pose_index:02d}_cam0.jpg").write_bytes(cap.cam0_jpeg)
            if cap.cam1_jpeg:
                (frames_dir / f"pose_{cap.pose_index:02d}_cam1.jpg").write_bytes(cap.cam1_jpeg)

        # Rename report.md -> comparison_report.md for consistency
        report_md = out_path / "report.md"
        comparison_md = out_path / "comparison_report.md"
        if report_md.exists() and not comparison_md.exists():
            report_md.rename(comparison_md)

        logger.info(f"Results saved to {out_path}")
        return str(out_path)
