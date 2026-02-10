#!/usr/bin/env python3
"""
Run CV vs LLM comparison pipeline on calibration data.

Fetches calibration results, captures frames from cameras,
runs both CV and LLM pipelines, compares against FK ground truth,
and generates a markdown report.

Usage:
    cd th3cl4w && python -m scripts.run_comparison [--session cal_1770588119]
    # or directly:
    cd th3cl4w && python scripts/run_comparison.py [--session cal_1770588119]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.camera_config import CAM_SIDE, CAM_OVERHEAD
from src.vision.fk_engine import fk_positions
from src.vision.arm_segmenter import ArmSegmenter
from src.vision.joint_detector import JointDetector, JOINT_NAMES
from src.vision.detection_comparator import (
    DetectionComparator, ComparisonResult, ComparisonReport,
    JointComparison, PoseCapture, _pixel_dist,
)
from src.calibration.results_reporter import (
    CalibrationReporter, CalibrationSession,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-5s %(message)s",
)
logger = logging.getLogger("run_comparison")

ARM_API = "http://localhost:8080"
CAM_API = "http://localhost:8081"
DEFAULT_SESSION = "cal_1770588119"
OUTPUT_DIR = PROJECT_ROOT / "calibration_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_calibration(session_id: str) -> dict:
    """Fetch calibration results from arm server API."""
    url = f"{ARM_API}/api/calibration/results/{session_id}"
    logger.info("Fetching calibration: %s", url)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    logger.info("Got %d captures", len(data.get("captures", [])))
    return data


def capture_frame(camera_id: int) -> bytes:
    """Capture a JPEG frame from the camera server."""
    url = f"{CAM_API}/snap/{camera_id}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.content


def jpeg_to_cv(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG")
    return frame


def fk_to_dummy_pixels(joint_angles: list[float], cam_id: int) -> list[tuple[float, float]]:
    """
    Convert FK 3D positions to approximate pixel coordinates.
    
    Since we don't have calibrated camera extrinsics, we use a simple
    projection based on camera perspective:
    - cam0 (side): projects X→u, Z→v (inverted)
    - cam1 (overhead/arm): projects X→u, Y→v
    
    This is approximate but gives us FK ground truth in pixel space
    for error measurement.
    """
    positions = fk_positions(joint_angles)
    
    # Image dimensions (assumed 1920x1080)
    W, H = 1920, 1080
    
    # Simple projection parameters (rough estimates)
    # These would normally come from camera calibration
    if cam_id == CAM_SIDE:
        # Side camera: looking along Y axis
        # X -> horizontal, Z -> vertical (inverted)
        cx, cy = W / 2, H * 0.7  # arm base roughly at 70% down
        scale = 1500.0  # pixels per meter (rough)
        pixels = []
        for p in positions:
            u = cx + p[0] * scale  # X -> right
            v = cy - p[2] * scale  # Z -> up (invert for pixel coords)
            pixels.append((u, v))
    else:
        # Overhead camera: looking down along Z axis
        cx, cy = W / 2, H / 2
        scale = 1500.0
        pixels = []
        for p in positions:
            u = cx + p[0] * scale  # X -> right
            v = cy - p[1] * scale  # Y -> up
            pixels.append((u, v))
    
    return pixels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cv_pipeline(
    frame: np.ndarray,
    segmenter: ArmSegmenter,
    detector: JointDetector,
    fk_pixels: list[tuple[float, float]],
) -> tuple[list, float]:
    """Run CV detection pipeline, return (detections, latency_ms)."""
    t0 = time.monotonic()
    seg = segmenter.segment_arm(frame)
    dets = detector.detect_joints(seg, fk_pixels)
    latency = (time.monotonic() - t0) * 1000
    return dets, latency, seg


async def run_llm_pipeline(
    llm_detector,
    jpeg_bytes: bytes,
    camera_id: int,
    joint_angles: list[float],
) -> dict:
    """Run LLM detection, return result dict."""
    if llm_detector is None:
        return {"joints": [], "latency_ms": 0, "tokens": 0, "success": False, "error": "No LLM detector"}
    
    try:
        result = await llm_detector.detect_joints(
            jpeg_bytes=jpeg_bytes,
            camera_id=camera_id,
            joint_angles=joint_angles,
        )
        return {
            "joints": [
                {"name": j.name, "x": j.norm_x, "y": j.norm_y, 
                 "pixel_x": j.pixel_x, "pixel_y": j.pixel_y,
                 "confidence": j.confidence}
                for j in result.joints
            ],
            "latency_ms": result.latency_ms,
            "tokens": result.tokens_used,
            "success": result.success,
            "error": result.error,
            "raw": result.raw_response[:200] if result.raw_response else "",
        }
    except Exception as e:
        logger.error("LLM detection failed: %s", e)
        return {"joints": [], "latency_ms": 0, "tokens": 0, "success": False, "error": str(e)}


def build_comparison_result(
    pose_index: int,
    camera_id: int,
    joint_angles: list[float],
    fk_pixels: list[tuple[float, float]],
    cv_dets: list,
    cv_latency_ms: float,
    llm_result: dict,
) -> ComparisonResult:
    """Build a ComparisonResult from CV and LLM outputs."""
    from src.vision.joint_detector import DetectionSource
    
    # Map LLM results by joint name
    llm_map = {}
    for j in llm_result.get("joints", []):
        llm_map[j["name"]] = j
    
    # Map CV results by joint index
    cv_map = {d.joint_index: d for d in cv_dets}
    
    joints = []
    for i, name in enumerate(JOINT_NAMES):
        fk_px = fk_pixels[i] if i < len(fk_pixels) else None
        
        # CV
        cv_det = cv_map.get(i)
        cv_pixel = None
        cv_source = None
        if cv_det:
            cv_source = cv_det.source.value
            if cv_det.source != DetectionSource.FK_ONLY:
                cv_pixel = cv_det.pixel_pos
        
        # LLM
        llm_pixel = None
        llm_conf = None
        if name in llm_map:
            lj = llm_map[name]
            llm_pixel = (lj["pixel_x"], lj["pixel_y"])
            llm_conf = lj["confidence"]
        
        joints.append(JointComparison(
            name=name,
            fk_pixel=fk_px,
            cv_pixel=cv_pixel,
            llm_pixel=llm_pixel,
            cv_error_px=_pixel_dist(cv_pixel, fk_px),
            llm_error_px=_pixel_dist(llm_pixel, fk_px),
            agreement_px=_pixel_dist(cv_pixel, llm_pixel),
            cv_source=cv_source,
            llm_confidence=llm_conf,
        ))
    
    return ComparisonResult(
        pose_index=pose_index,
        camera_id=camera_id,
        joint_angles=joint_angles,
        joints=joints,
        cv_latency_ms=cv_latency_ms,
        llm_latency_ms=llm_result.get("latency_ms", 0),
        llm_tokens=llm_result.get("tokens", 0),
    )


async def main():
    parser = argparse.ArgumentParser(description="Run CV vs LLM comparison")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--max-poses", type=int, default=20)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM pipeline")
    args = parser.parse_args()

    # 1. Fetch calibration data
    cal_data = fetch_calibration(args.session)
    captures = cal_data["captures"][:args.max_poses]
    
    # 2. Initialize pipelines
    logger.info("Initializing CV pipeline...")
    segmenters = {0: ArmSegmenter(), 1: ArmSegmenter()}
    detector = JointDetector()
    
    # Capture background frames (arm should be mostly stationary now)
    logger.info("Capturing background frames for segmentation...")
    for cam_id in [0, 1]:
        bg_frames = []
        for _ in range(3):
            try:
                jpeg = capture_frame(cam_id)
                bg_frames.append(jpeg_to_cv(jpeg))
                time.sleep(0.1)
            except Exception as e:
                logger.warning("Failed to capture bg frame cam%d: %s", cam_id, e)
        if bg_frames:
            segmenters[cam_id].capture_background(bg_frames)
            logger.info("Background captured for cam%d (%d frames)", cam_id, len(bg_frames))
    
    # 3. Initialize LLM detector
    llm_detector = None
    if not args.skip_llm:
        try:
            from src.vision.llm_detector import LLMJointDetector
            llm_detector = LLMJointDetector()
            logger.info("LLM detector initialized (model: %s)", llm_detector.model_name)
        except Exception as e:
            logger.warning("Could not initialize LLM detector: %s", e)
            logger.warning("Continuing with CV-only comparison")
    
    # 4. Run comparison for each pose × camera
    all_results: list[ComparisonResult] = []
    total_llm_tokens = 0
    
    for capture in captures:
        pose_idx = capture["pose_index"]
        angles = capture["actual_angles"]
        
        for cam_id in [0, 1]:
            has_cam = capture.get(f"has_cam{cam_id}", False)
            if not has_cam:
                continue
            
            logger.info("Processing pose %d, cam %d (angles: %s)",
                       pose_idx, cam_id, [f"{a:.1f}" for a in angles])
            
            # Capture fresh frame
            try:
                jpeg = capture_frame(cam_id)
                frame = jpeg_to_cv(jpeg)
            except Exception as e:
                logger.error("Failed to capture frame cam%d: %s", cam_id, e)
                continue
            
            # FK ground truth in pixel space
            fk_pixels = fk_to_dummy_pixels(angles, cam_id)
            
            # CV pipeline
            cv_dets, cv_latency, seg = run_cv_pipeline(
                frame, segmenters[cam_id], detector, fk_pixels
            )
            cv_real_count = sum(1 for d in cv_dets 
                              if d.source.value != "fk_only")
            logger.info("  CV: %d/%d real detections, %.1fms",
                       cv_real_count, len(cv_dets), cv_latency)
            
            # LLM pipeline
            llm_result = await run_llm_pipeline(
                llm_detector, jpeg, cam_id, angles
            )
            if llm_result["success"]:
                logger.info("  LLM: %d joints, %d tokens, %.1fms",
                           len(llm_result["joints"]),
                           llm_result["tokens"],
                           llm_result["latency_ms"])
                total_llm_tokens += llm_result["tokens"]
            elif llm_result.get("error"):
                logger.info("  LLM: failed — %s", llm_result["error"][:80])
            
            # Build comparison
            result = build_comparison_result(
                pose_idx, cam_id, angles, fk_pixels,
                cv_dets, cv_latency, llm_result,
            )
            all_results.append(result)
    
    # 5. Aggregate into report
    logger.info("Aggregating %d results...", len(all_results))
    comparator = DetectionComparator(
        cv_detector=detector,
        camera_resolution=(1920, 1080),
    )
    report = comparator._aggregate(all_results)
    
    # 6. Generate reports
    session = CalibrationSession(
        session_id=args.session,
        start_time=cal_data.get("start_time", 0),
        end_time=cal_data.get("end_time", 0),
        num_poses=len(captures),
        status="complete",
    )
    
    reporter = CalibrationReporter()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = OUTPUT_DIR / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    
    md_report = reporter.generate_markdown(report, session)
    md_path = session_dir / "comparison_report.md"
    md_path.write_text(md_report)
    logger.info("Markdown report: %s", md_path)
    
    json_report = reporter.generate_json(report)
    json_report["session_id"] = args.session
    json_path = session_dir / "summary.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    logger.info("JSON report: %s", json_path)
    
    # Raw results
    raw_data = []
    for r in all_results:
        raw_data.append({
            "pose_index": r.pose_index,
            "camera_id": r.camera_id,
            "joint_angles": r.joint_angles,
            "cv_latency_ms": r.cv_latency_ms,
            "llm_latency_ms": r.llm_latency_ms,
            "llm_tokens": r.llm_tokens,
            "joints": [
                {
                    "name": j.name,
                    "fk_pixel": j.fk_pixel,
                    "cv_pixel": j.cv_pixel,
                    "llm_pixel": j.llm_pixel,
                    "cv_error_px": j.cv_error_px,
                    "llm_error_px": j.llm_error_px,
                    "agreement_px": j.agreement_px,
                    "cv_source": j.cv_source,
                    "llm_confidence": j.llm_confidence,
                }
                for j in r.joints
            ],
        })
    raw_path = session_dir / "raw_results.json"
    raw_path.write_text(json.dumps(raw_data, indent=2))
    
    # Also copy report to top-level for easy access
    top_report = OUTPUT_DIR / "comparison_report.md"
    top_report.write_text(md_report)
    
    # Print summary
    summary = comparator.summary(report)
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  total_results: {len(all_results)}")
    logger.info(f"  output_dir: {session_dir}")
    logger.info("=" * 60)
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
