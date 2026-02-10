#!/usr/bin/env python3
"""Offline LLM vs CV comparison using pre-captured calibration frames."""

import asyncio
import json
import time
import os
import sys
import logging
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()  # Load .env file
# GEMINI_API_KEY must be set in environment (never hardcode API keys)
if not os.environ.get("GEMINI_API_KEY"):
    logger.info("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
    sys.exit(1)

from src.vision.llm_detector import LLMJointDetector
from src.vision.fk_engine import fk_positions
from src.vision.arm_segmenter import ArmSegmenter
from src.vision.joint_detector import JointDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("offline_comparison")

SESSION_ID = "cal_1770590048"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration_results", SESSION_ID)
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
CALIB_DATA = os.path.join(BASE_DIR, "calibration_data.json")


async def run():
    session_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.info("=== Offline LLM vs CV Comparison for %s ===", SESSION_ID)

    with open(CALIB_DATA) as f:
        calib = json.load(f)

    captures = calib["captures"]
    log.info("Loaded %d captures", len(captures))

    # Init detectors
    llm = LLMJointDetector(ascii_width=80, ascii_height=35)
    segmenter = ArmSegmenter()
    jd = JointDetector()

    # Build background model from first frame (home position)
    first_frame_path = os.path.join(FRAMES_DIR, "calib_0001.jpg")
    bg_frame = cv2.imread(first_frame_path)
    if bg_frame is None:
        log.error("Cannot read first frame for background")
        return

    # Use first frame as background (pose 0 is home, arm present but close enough)
    segmenter.capture_background([bg_frame] * 5)

    results = {
        "session_id": SESSION_ID,
        "session_start": session_start,
        "note": "Checkerboard pattern present in scene (covered by user)",
        "poses": [],
        "summary": {},
    }

    total_llm_calls = 0
    total_tokens = 0
    llm_successes = 0
    cv_successes = 0
    total_detections = 0

    for i, cap in enumerate(captures):
        pose_idx = cap["pose_index"]
        actual = cap["actual_angles"]
        commanded = cap["commanded_angles"]
        frame_path = os.path.join(FRAMES_DIR, f"calib_{pose_idx+1:04d}.jpg")

        if not os.path.exists(frame_path):
            log.warning("Frame missing: %s", frame_path)
            continue

        log.info("--- Pose %d/%d: commanded=%s actual=%s ---",
                 pose_idx + 1, len(captures), commanded, [round(a, 1) for a in actual])

        jpeg = open(frame_path, "rb").read()
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        fk = fk_positions(actual)

        pose_result = {
            "index": pose_idx,
            "commanded": commanded,
            "actual": [round(j, 2) for j in actual],
            "fk_positions": [[round(c, 4) for c in p] for p in fk],
        }

        # We only have overhead (cam1) frames
        cam_id = 1

        # --- LLM detection ---
        t0 = time.monotonic()
        try:
            llm_result = await llm.detect_joints(jpeg, cam_id, actual)
            llm_latency = (time.monotonic() - t0) * 1000
            llm_joints = [
                {"name": j.name, "nx": round(j.norm_x, 3), "ny": round(j.norm_y, 3),
                 "px": j.pixel_x, "py": j.pixel_y, "conf": j.confidence}
                for j in llm_result.joints
            ]
            llm_tokens = llm_result.tokens_used
            llm_success = llm_result.success
            llm_error = llm_result.error
        except Exception as e:
            llm_latency = (time.monotonic() - t0) * 1000
            llm_joints = []
            llm_tokens = 0
            llm_success = False
            llm_error = str(e)

        total_llm_calls += 1
        total_tokens += llm_tokens
        if llm_success and len(llm_joints) >= 3:
            llm_successes += 1
        total_detections += 1

        # --- CV detection ---
        t0 = time.monotonic()
        try:
            seg_result = segmenter.segment_arm(frame)
            # Project FK to pixels (overhead cam1: Y->horizontal, X->vertical)
            fk_px = []
            for p in fk:
                px = int(w / 2 + p[1] * w * 2)
                py = int(h / 2 - p[0] * h * 2)
                fk_px.append((float(px), float(py)))

            cv_detections = jd.detect_joints(seg_result, fk_px)
            cv_latency = (time.monotonic() - t0) * 1000
            cv_joints = [
                {"name": d.name, "px": round(d.pixel_pos[0], 1), "py": round(d.pixel_pos[1], 1),
                 "conf": round(d.confidence, 3), "source": d.source.value}
                for d in cv_detections
            ]
            cv_success = any(d.confidence > 0.3 for d in cv_detections)
        except Exception as e:
            cv_latency = (time.monotonic() - t0) * 1000
            cv_joints = []
            cv_success = False
            log.warning("CV detection failed pose%d: %s", pose_idx, e)

        if cv_success:
            cv_successes += 1

        pose_result["cam1"] = {
            "llm_joints": llm_joints,
            "llm_tokens": llm_tokens,
            "llm_latency_ms": round(llm_latency, 1),
            "llm_success": llm_success,
            "llm_error": llm_error if not llm_success else None,
            "cv_joints": cv_joints,
            "cv_latency_ms": round(cv_latency, 1),
            "cv_success": cv_success,
        }

        log.info("  LLM=%s (%d joints, %d tok, %.0fms) CV=%s (%d joints, %.0fms)",
                 "OK" if llm_success else "FAIL", len(llm_joints), llm_tokens, llm_latency,
                 "OK" if cv_success else "FAIL", len(cv_joints), cv_latency)

        results["poses"].append(pose_result)

        # Rate limit for Gemini
        await asyncio.sleep(1)

    # Summary
    results["summary"] = {
        "total_poses": len(captures),
        "frames_per_pose": 1,
        "camera": "cam1 (overhead)",
        "total_llm_calls": total_llm_calls,
        "total_tokens": total_tokens,
        "avg_tokens_per_call": round(total_tokens / max(1, total_llm_calls)),
        "llm_detection_rate": round(llm_successes / max(1, total_detections), 3),
        "cv_detection_rate": round(cv_successes / max(1, total_detections), 3),
        "llm_successes": llm_successes,
        "cv_successes": cv_successes,
        "total_detections": total_detections,
    }

    # Save JSON
    json_path = os.path.join(BASE_DIR, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", json_path)

    # Generate markdown report
    s = results["summary"]
    md = f"""# LLM vs CV Joint Detection — Offline Comparison Report

**Session:** {SESSION_ID}
**Date:** {session_start}
**Poses:** {s['total_poses']} (overhead camera only)
**Note:** Checkerboard calibration pattern present in scene

## Summary

| Metric | LLM (Gemini) | CV Pipeline |
|--------|-------------|-------------|
| Detection rate | {s['llm_detection_rate']*100:.1f}% | {s['cv_detection_rate']*100:.1f}% |
| Successful detections | {s['llm_successes']}/{s['total_detections']} | {s['cv_successes']}/{s['total_detections']} |
| Total API calls | {s['total_llm_calls']} | N/A |
| Total tokens | {s['total_tokens']} | N/A |
| Avg tokens/call | {s['avg_tokens_per_call']} | N/A |

## Per-Pose Results

"""
    for p in results["poses"]:
        c = p["cam1"]
        md += f"### Pose {p['index']}: commanded={p['commanded']}, actual={p['actual']}\n\n"
        md += f"LLM={'✅' if c['llm_success'] else '❌'} "
        md += f"({len(c['llm_joints'])} joints, {c['llm_tokens']} tok, {c['llm_latency_ms']:.0f}ms) | "
        md += f"CV={'✅' if c['cv_success'] else '❌'} "
        md += f"({len(c['cv_joints'])} joints, {c['cv_latency_ms']:.0f}ms)\n\n"

    # Latencies
    llm_lats = [p["cam1"]["llm_latency_ms"] for p in results["poses"]]
    cv_lats = [p["cam1"]["cv_latency_ms"] for p in results["poses"]]
    md += f"""## Latency

| | LLM | CV |
|---|---|---|
| Avg | {sum(llm_lats)/len(llm_lats):.0f}ms | {sum(cv_lats)/len(cv_lats):.0f}ms |
| Min | {min(llm_lats):.0f}ms | {min(cv_lats):.0f}ms |
| Max | {max(llm_lats):.0f}ms | {max(cv_lats):.0f}ms |

## vs Previous Run (cal_1770588119)

The previous run had NO checkerboard pattern:
- LLM: 100% detection rate
- CV: 10% detection rate

This run includes checkerboard calibration patterns which should improve CV detection
through better visual contrast and reference features.
"""

    md_path = os.path.join(BASE_DIR, "comparison_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    log.info("Report saved to %s", md_path)
    log.info("=== Comparison complete! ===")
    log.info("LLM: %d/%d (%.0f%%) | CV: %d/%d (%.0f%%)",
             llm_successes, total_detections, llm_successes/max(1,total_detections)*100,
             cv_successes, total_detections, cv_successes/max(1,total_detections)*100)


if __name__ == "__main__":
    asyncio.run(run())
