"""End-to-end LLM calibration pipeline."""

import asyncio
import logging
import os
from pathlib import Path

from src.calibration.calibration_runner import CalibrationRunner, CalibrationSession
from src.calibration.results_reporter import CalibrationReporter, CalibrationSession as ReporterSession
from src.vision.detection_comparator import DetectionComparator, ComparisonReport
from src.vision.joint_detector import JointDetector
from src.vision.arm_segmenter import ArmSegmenter

try:
    from src.vision.llm_detector import LLMJointDetector
except Exception:
    LLMJointDetector = None

logger = logging.getLogger("th3cl4w.calibration.pipeline")


class LLMCalibrationPipeline:
    """End-to-end pipeline: calibration runner + dual detection + comparison + reporting."""

    def __init__(
        self,
        arm_host: str = 'localhost',
        arm_port: int = 8080,
        cam_host: str = 'localhost',
        cam_port: int = 8081,
        gemini_api_key: str | None = None,
    ):
        self.runner = CalibrationRunner(arm_host, arm_port, cam_host, cam_port)
        self.segmenter = ArmSegmenter()
        self.cv_detector = JointDetector()

        self.llm_detector = None
        if LLMJointDetector is not None:
            try:
                self.llm_detector = LLMJointDetector(api_key=gemini_api_key)
            except Exception as e:
                logger.warning(f"LLM detector unavailable: {e}")

        self.comparator = DetectionComparator(
            segmenter=self.segmenter,
            joint_detector=self.cv_detector,
            llm_detector=self.llm_detector,
        )
        self.reporter = CalibrationReporter()

    async def run(self, output_dir: str = 'calibration_results') -> str:
        """Run full calibration pipeline, return path to report directory."""
        session = await self.runner.run_full_calibration()

        # Run comparisons on captured frames
        for capture in session.captures:
            llm_result = None
            if self.llm_detector and capture.cam0_jpeg:
                try:
                    llm_result_obj = await self.llm_detector.detect_joints(
                        capture.cam0_jpeg, camera_id=0,
                        joint_angles=capture.actual_angles,
                    )
                    if llm_result_obj.success:
                        llm_result = {
                            "joints": [
                                {"name": j.name, "x": j.norm_x, "y": j.norm_y, "confidence": j.confidence}
                                for j in llm_result_obj.joints
                            ],
                            "input_tokens": llm_result_obj.tokens_used,
                            "output_tokens": 0,
                            "latency_ms": llm_result_obj.latency_ms,
                            "cost_usd": llm_result_obj.tokens_used * 0.00000012,
                        }
                except Exception as e:
                    logger.warning(f"LLM detection failed for pose {capture.pose_index}: {e}")

            # Run CV + comparison
            import cv2
            import numpy as np
            if capture.cam0_jpeg:
                frame = cv2.imdecode(np.frombuffer(capture.cam0_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.comparator.compare_single(
                        frame=frame,
                        joint_angles=capture.actual_angles,
                        camera_id=0,
                        pose_index=capture.pose_index,
                        llm_result=llm_result,
                    )

        report = self.comparator.generate_report()
        report_session = ReporterSession(
            session_id=self.runner._session_id or "calibration",
            start_time=session.start_time,
            end_time=session.end_time,
        )
        report_path = self.reporter.save_report(report, report_session, output_dir)
        return report_path
