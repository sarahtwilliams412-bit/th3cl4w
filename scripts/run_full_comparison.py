#!/usr/bin/env python3
"""Full LLM vs CV calibration comparison across 20 poses × 2 cameras."""

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

import httpx
from src.vision.llm_detector import LLMJointDetector
from src.vision.fk_engine import fk_positions
from src.vision.arm_segmenter import ArmSegmenter
from src.vision.joint_detector import JointDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("comparison")

ARM = "http://localhost:8080"
CAM = "http://localhost:8081"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration_results")
FRAMES_DIR = os.path.join(RESULTS_DIR, "frames")

POSES = [
    (0, 0, 0, 0, 0, 0), (30, 0, 0, 0, 0, 0), (-30, 0, 0, 0, 0, 0),
    (0, -30, 0, 0, 0, 0), (0, -60, 0, 0, 0, 0), (0, 0, 45, 0, 0, 0),
    (0, 0, -45, 0, 0, 0), (0, -30, 30, 0, 0, 0), (0, -45, 45, 0, -30, 0),
    (30, -30, 30, 0, 0, 0), (-30, -30, 30, 0, 0, 0), (0, 30, 0, 0, 0, 0),
    (0, -30, 45, 0, 45, 0), (45, -45, 30, 0, 0, 0), (-45, -45, 30, 0, 0, 0),
    (0, 0, 0, 0, -45, 0), (0, -30, 0, 0, 45, 0), (60, 0, 30, 0, 0, 0),
    (-60, 0, 30, 0, 0, 0), (0, -60, 60, 0, -45, 0),
]

JOINT_IDS = [0, 1, 2, 3, 4, 5]


async def read_state(client: httpx.AsyncClient) -> dict:
    """Read arm state with retries for stale DDS."""
    for attempt in range(5):
        r = await client.get(f"{ARM}/api/state", timeout=5)
        data = r.json()
        if data.get("connected"):
            return data
        await asyncio.sleep(0.3)
    return data


async def move_joint_incrementally(client: httpx.AsyncClient, joint_id: int, target: float, current: float):
    """Move a single joint in ≤10° increments."""
    diff = target - current
    if abs(diff) < 0.5:
        return
    steps = max(1, int(abs(diff) / 10.0 + 0.99))  # ceil
    for s in range(1, steps + 1):
        angle = current + diff * s / steps
        for attempt in range(5):
            r = await client.post(
                f"{ARM}/api/command/set-joint",
                json={"id": joint_id, "angle": round(angle, 1)},
                timeout=5,
            )
            if r.status_code == 200:
                break
            await asyncio.sleep(0.3)
        await asyncio.sleep(0.5)


async def move_to_pose(client: httpx.AsyncClient, target: tuple, current_joints: list[float]):
    """Move all joints to target pose safely."""
    for jid in JOINT_IDS:
        tgt = target[jid] if jid < len(target) else 0.0
        cur = current_joints[jid] if jid < len(current_joints) else 0.0
        await move_joint_incrementally(client, jid, tgt, cur)


async def snap_camera(client: httpx.AsyncClient, cam_id: int) -> bytes:
    """Capture JPEG from camera."""
    r = await client.get(f"{CAM}/snap/{cam_id}", timeout=10)
    return r.content


async def run():
    session_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.info("=== Full LLM vs CV Comparison ===")
    log.info("Session start: %s", session_start)

    # Init detectors
    llm = LLMJointDetector(ascii_width=80, ascii_height=35)
    segmenter = ArmSegmenter()
    jd = JointDetector()

    results = {
        "session_start": session_start,
        "poses": [],
        "summary": {},
    }

    total_llm_calls = 0
    total_tokens = 0
    llm_successes = 0
    cv_successes = 0
    total_detections = 0

    async with httpx.AsyncClient() as client:
        # Enable arm
        log.info("Enabling arm...")
        await client.post(f"{ARM}/api/command/enable", timeout=5)
        await asyncio.sleep(1)

        # Read initial state
        state = await read_state(client)
        current_joints = state.get("joints", [0]*6)
        log.info("Initial joints: %s", [round(j,1) for j in current_joints])

        # Capture background frames (arm at home position first)
        log.info("Capturing background frames for CV segmenter...")
        bg_frames = {0: [], 1: []}
        for _ in range(5):
            for cam_id in [0, 1]:
                jpeg = await snap_camera(client, cam_id)
                arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                bg_frames[cam_id].append(arr)
            await asyncio.sleep(0.2)

        # We'll use per-camera segmenters
        segmenters = {}
        for cam_id in [0, 1]:
            seg = ArmSegmenter()
            seg.capture_background(bg_frames[cam_id])
            segmenters[cam_id] = seg

        for pose_idx, pose in enumerate(POSES):
            log.info("--- Pose %d/%d: %s ---", pose_idx + 1, len(POSES), pose)

            # Move arm
            await move_to_pose(client, pose, current_joints)
            await asyncio.sleep(3)  # settle

            # Read actual angles
            state = await read_state(client)
            actual = state.get("joints", [0]*6)
            current_joints = actual
            log.info("Actual joints: %s", [round(j,1) for j in actual])

            # FK ground truth
            fk = fk_positions(actual)

            pose_result = {
                "index": pose_idx,
                "commanded": list(pose),
                "actual": [round(j, 2) for j in actual],
                "fk_positions": [[round(c, 4) for c in p] for p in fk],
            }

            for cam_id in [0, 1]:
                cam_key = f"cam{cam_id}"
                jpeg = await snap_camera(client, cam_id)

                # Save frame
                fname = f"pose{pose_idx:02d}_cam{cam_id}.jpg"
                with open(os.path.join(FRAMES_DIR, fname), "wb") as f:
                    f.write(jpeg)

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
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    seg_result = segmenters[cam_id].segment_arm(frame)
                    # For CV we need FK pixel positions - use dummy projection for now
                    # (The CV pipeline needs calibrated camera params to project FK to pixels)
                    h, w = frame.shape[:2]
                    # Simple projection: use normalized FK coords scaled to frame
                    # This is approximate - real pipeline uses calibrated intrinsics
                    fk_px = []
                    for p in fk:
                        # Simple front-view projection (cam0) or top-down (cam1)
                        if cam_id == 0:
                            px = int(w/2 + p[1] * w * 2)  # Y->horizontal
                            py = int(h - p[2] * h * 1.5)   # Z->vertical (flip)
                        else:
                            px = int(w/2 + p[1] * w * 2)  # Y->horizontal
                            py = int(h/2 - p[0] * h * 2)  # X->vertical
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
                    log.warning("CV detection failed cam%d pose%d: %s", cam_id, pose_idx, e)

                if cv_success:
                    cv_successes += 1

                pose_result[cam_key] = {
                    "llm_joints": llm_joints,
                    "llm_tokens": llm_tokens,
                    "llm_latency_ms": round(llm_latency, 1),
                    "llm_success": llm_success,
                    "llm_error": llm_error if not llm_success else None,
                    "cv_joints": cv_joints,
                    "cv_latency_ms": round(cv_latency, 1),
                    "cv_success": cv_success,
                }

                log.info("  cam%d: LLM=%s (%d joints, %d tok, %.0fms) CV=%s (%d joints, %.0fms)",
                         cam_id,
                         "OK" if llm_success else "FAIL", len(llm_joints), llm_tokens, llm_latency,
                         "OK" if cv_success else "FAIL", len(cv_joints), cv_latency)

            results["poses"].append(pose_result)

        # Return arm to home
        log.info("Returning to home position...")
        await move_to_pose(client, (0, 0, 0, 0, 0, 0), current_joints)

    # Summary
    results["summary"] = {
        "total_poses": len(POSES),
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
    json_path = os.path.join(RESULTS_DIR, "full_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", json_path)

    # Generate markdown report
    s = results["summary"]
    md = f"""# LLM vs CV Joint Detection — Full Comparison Report

**Date:** {session_start}
**Poses tested:** {s['total_poses']}
**Cameras:** 2 (front + overhead)

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
        md += f"### Pose {p['index']}: commanded={p['commanded']}, actual={p['actual']}\n\n"
        for cam_key in ["cam0", "cam1"]:
            c = p[cam_key]
            md += f"**{cam_key}:** LLM={'✅' if c['llm_success'] else '❌'} "
            md += f"({len(c['llm_joints'])} joints, {c['llm_tokens']} tok, {c['llm_latency_ms']:.0f}ms) | "
            md += f"CV={'✅' if c['cv_success'] else '❌'} "
            md += f"({len(c['cv_joints'])} joints, {c['cv_latency_ms']:.0f}ms)\n\n"

    # Avg latencies
    llm_lats = [p[ck]["llm_latency_ms"] for p in results["poses"] for ck in ["cam0","cam1"]]
    cv_lats = [p[ck]["cv_latency_ms"] for p in results["poses"] for ck in ["cam0","cam1"]]
    md += f"""## Latency

| | LLM | CV |
|---|---|---|
| Avg | {sum(llm_lats)/len(llm_lats):.0f}ms | {sum(cv_lats)/len(cv_lats):.0f}ms |
| Min | {min(llm_lats):.0f}ms | {min(cv_lats):.0f}ms |
| Max | {max(llm_lats):.0f}ms | {max(cv_lats):.0f}ms |
"""

    md_path = os.path.join(RESULTS_DIR, "full_comparison_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    log.info("Report saved to %s", md_path)
    log.info("=== Comparison complete! ===")


if __name__ == "__main__":
    asyncio.run(run())
