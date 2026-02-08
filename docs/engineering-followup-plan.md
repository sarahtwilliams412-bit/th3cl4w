# Engineering Follow-Up Plan

**Date:** 2026-02-08  
**Author:** Engineering Follow-up Planning Lead  
**Goal:** Pick up the Red Bull can.

---

## Section 1: LLM vs CV Comparison Analysis

### Verdict: Both pipelines failed. Archive LLM approach.

**Data from session `cal_1770588119` (20 poses, 40 frames, 231s):**

| Pipeline | Detection Rate | Mean Error (px) | Cost |
|----------|---------------|-----------------|------|
| CV (frame differencing) | 1.5% | 17.3 px | $0 |
| LLM (Gemini ASCII) | 0.0% | N/A | $0 (never ran) |

**What happened:**
- **LLM pipeline never executed.** `GEMINI_API_KEY` was not set as an environment variable. Zero frames were sent to Gemini. This is an infrastructure failure, not an algorithmic one.
- **CV pipeline nearly completely failed.** 1.5% detection rate (3 detections out of 200 joint×frame opportunities). Background subtraction couldn't distinguish the matte-black arm from the background because the arm was present in the "background" reference frame. Only cam1 detected anything (end_effector at poses 9, 12, 13 via gold HSV — not frame differencing).

**Where LLM could have helped:** Unknown — experiment was inconclusive since LLM never ran.

**Where CV failed:** Background subtraction is fundamentally the wrong approach when you can't capture a clean background frame with the arm absent. The arm is always in frame.

**Cost-effectiveness:** The LLM approach costs ~$0.02 per full calibration run with Gemini Flash. If it had run and achieved even 30% detection rate, it would have been worth continuing at that price point. However, ASCII art spatial reasoning is a long shot — the plan doc's own author rated it as "LLMs are bad at this."

**Recommendation:** 
1. **Archive LLM ASCII approach.** Even if Gemini API key were set, ASCII-based joint detection is unlikely to beat direct image analysis. If LLM vision is revisited, send raw JPEG frames to multimodal models (Gemini Vision, Claude Vision) instead.
2. **Replace background subtraction** with gold HSV detection as primary CV method — the 3 successful detections all came from gold segment HSV matching, not frame differencing.
3. **The real blocker is calibration, not detection.** Even perfect detection is useless with ~100px reprojection error.

---

## Section 2: Camera Calibration Improvements

### The Critical Blocker: ~100px Reprojection Error

Current state from `calibration_results/camera_extrinsics.json`:
- **cam0:** mean reprojection error = **138.8 px**, max = 202.7 px (5 poses)
- **cam1:** mean reprojection error = **90.4 px**, max = 182.2 px (9 poses, 5 inliers)
- **Root cause:** Default intrinsics (`fx=fy=1000, cx=960, cy=540`) — no checkerboard calibration has been performed.

### Plan to Reduce Reprojection Error from ~100px to <10px

**Step 1: Checkerboard Intrinsic Calibration (CRITICAL, do first)**

- Print 7×5 checkerboard at 23.8mm squares (matches `DEFAULT_SQUARE_SIZE_MM` in `src/vision/calibration.py`)
- Capture 15+ images per camera from varied angles and distances
- Run existing `tools/calibrate_cameras.py --cam0 0 --cam1 2 --num-images 15`
- Uses `cv2.findChessboardCorners` + `cv2.cornerSubPix` → `cv2.calibrateCamera`
- Expected: reprojection error < 1.0px for intrinsics
- **Output:** `calibration/cam0_calibration.json`, `calibration/cam1_calibration.json` with `camera_matrix` and `dist_coeffs`

**Step 2: Re-solve Extrinsics with Proper Intrinsics**

- Rerun the bootstrap PnP solve from `docs/joint-mapping.md` annotated poses (9 for cam1, 5 for cam0)
- Use the new intrinsics + distortion coefficients instead of `fx=fy=1000`
- Expected: reprojection error drops to 10-30px range
- File: modify the extrinsics solver in the calibration pipeline to load intrinsics from JSON

**Step 3: Live Arm-Based Extrinsic Refinement**

- Run the 20-pose calibration sequence (already implemented) with proper intrinsics
- Use gold HSV detection (not frame differencing) to find end-effector
- `cv2.solvePnPRansac` with FK positions as 3D points, detected pixels as 2D points
- Refine with `cv2.solvePnPRefineLM`
- Target: reprojection error < 5px

**Step 4: DH Parameter Validation**

- `viz_calibrator.py` uses link lengths 80+170+170+60+60+50=590mm
- `kinematics.py` DH params give 121.5+208.5+208.5+113=651.5mm total height
- **60mm discrepancy must be resolved** before trusting FK as ground truth
- Validate: command arm to home, physically measure gripper height, compare with FK prediction
- If off > 10mm: fit link lengths during calibration (scipy optimizer)

### Recommended Calibration Procedure (User Steps)

1. Print checkerboard, hold in front of each camera, capture 15 images each (~10 min)
2. Run intrinsic calibration tool (~2 min compute)
3. Power arm, run 20-pose auto-calibration (~5 min)
4. Verify: reprojection error < 10px in report
5. Save calibration, test with 5 validation poses

---

## Section 3: Pick-and-Place Pipeline

### What's Needed to Pick Up the Red Bull Can

```
Object Detection → 3D Localization → Grasp Planning → Motion Planning → Execution
```

### Module Status

| Module | File | Status | Gap |
|--------|------|--------|-----|
| Object detection (HSV) | `src/vision/object_detection.py` (213 lines) | ✅ Exists | Need Red Bull can color profile (red/blue/silver) |
| Arm tracker | `src/vision/arm_tracker.py` (372 lines) | ✅ Exists | Cross-references cam0/cam1 detections |
| Grasp planner | `src/vision/grasp_planner.py` (381 lines) | ✅ Exists | Needs calibrated 3D positions to work |
| Pick executor | `src/planning/pick_executor.py` | ✅ Exists | Needs grasp planner output |
| Vision task planner | `src/planning/vision_task_planner.py` | ✅ Exists | Orchestrates detect→plan→execute |
| FK engine (Python) | `src/vision/fk_engine.py` (171 lines) | ✅ Exists | Ported from JS |
| Pose fusion | `src/vision/pose_fusion.py` (338 lines) | ✅ Exists | Blends FK + vision |
| Camera calibration | `src/vision/calibration.py` (312 lines) | ⚠️ Partial | Intrinsics not calibrated |
| IK solver | `src/kinematics/kinematics.py` | ✅ Exists | Damped least squares IK |
| Motion planner | `src/planning/motion_planner.py` | ✅ Exists | S-curve trajectories |
| Collision detection | `src/planning/collision_preview.py` | ✅ Exists | Memory-based collision avoidance |

### What's Missing (in priority order)

1. **Calibrated camera intrinsics** — without this, no 3D localization works (see Section 2)
2. **Red Bull can detection profile** — add HSV ranges for red (H:0-10, 170-180) and silver/blue to `object_detection.py`
3. **Reliable 3D localization** — triangulate can position from both cameras with calibrated extrinsics
4. **Pre-grasp approach pose** — compute IK for a pose 50mm above the can, then descend
5. **Gripper force feedback** — detect when can is grasped (torque on J5/J6)

### Execution Sequence for Pick-and-Place

```python
# Pseudocode for the happy path
1. detect_can(cam0, cam1) → can_pixel_cam0, can_pixel_cam1
2. triangulate(can_pixel_cam0, can_pixel_cam1, calibration) → can_xyz_3d
3. pre_grasp_pose = ik_solve(can_xyz_3d + [0, 0, 0.05])  # 50mm above
4. grasp_pose = ik_solve(can_xyz_3d)
5. move_to(pre_grasp_pose)  # approach from above
6. move_to(grasp_pose)      # descend to can
7. close_gripper()
8. lift_pose = grasp_pose + [0, 0, 0.1]  # lift 100mm
9. move_to(lift_pose)
```

### Estimated effort to first successful pick: 2-3 focused sessions
- Session 1: Checkerboard calibration + extrinsic refinement
- Session 2: Can detection + 3D localization testing
- Session 3: Grasp execution + iteration

---

## Section 4: Real-Time 3D Reconstruction Improvements

### Visual Hull Quality

- Current: `src/vision/arm_segmenter.py` (259 lines) uses frame differencing + gold HSV
- Frame differencing failed in calibration (1.5% detection rate)
- **Fix:** Switch primary detection to gold HSV segments + contour-based arm following
- **GPU acceleration:** Already using `cv2.UMat` on RX 580 (OpenCL 3.0) for segmentation and visual hull

### Factory 3D Arm Tracking Accuracy

- Current: Factory 3D tab uses animated arm, not real tracking
- Pipeline exists: `fk_engine.py` → `joint_detector.py` → `pose_fusion.py` → WebSocket `/ws/arm3d`
- **Blocker:** Same calibration issue — without good extrinsics, fusion produces garbage
- After calibration fix, target: <20mm end-effector position error (3.6% of 550mm reach)

### Frame Rate / Latency Targets

| Component | Current | Target |
|-----------|---------|--------|
| Camera capture | 15 fps | 15 fps (hardware limit) |
| CV detection | ~672ms (!) | <50ms |
| FK computation | <1ms | <1ms |
| Fusion + push | ~10ms | <10ms |
| **End-to-end** | **~700ms** | **<100ms (10Hz)** |

CV detection at 672ms is way too slow — likely due to background model computation on full 1920×1080 frames. Fix: ROI-based processing using FK predictions (only process the region where the arm should be).

---

## Section 5: Prioritized Task List

### Tier 1: Critical Path to Picking Up the Can

| # | Task | Size | Agents | Dependencies | Impact |
|---|------|------|--------|-------------|--------|
| 1 | **Checkerboard intrinsic calibration** | S | 1 | Physical checkerboard + cameras | Unlocks everything — current ~100px error makes all vision useless |
| 2 | **Re-solve extrinsics with proper intrinsics** | S | 1 | Task 1 | Drops reprojection error from ~100px to <10px |
| 3 | **DH parameter validation** | S | 1 | Physical measurement | Resolves 60mm link length discrepancy between `viz_calibrator.py` and `kinematics.py` |
| 4 | **Red Bull can HSV detection profile** | S | 1 | None | Add red/silver color ranges to `object_detection.py` |
| 5 | **3D can localization from dual cameras** | M | 1 | Tasks 1-2 | Triangulate can position with calibrated cameras |
| 6 | **Pick-and-place execution** | M | 1 | Tasks 3-5 | Wire detection → IK → motion → grasp |

### Tier 2: Quality & Reliability

| # | Task | Size | Agents | Dependencies | Impact |
|---|------|------|--------|-------------|--------|
| 7 | **Replace frame differencing with HSV+contour detection** | M | 1 | None | Fix 1.5% detection rate |
| 8 | **CV detection latency optimization** | S | 1 | None | Cut from 672ms to <50ms via ROI processing |
| 9 | **Store JPEG frames during calibration** | S | 1 | None | Enable offline replay and debugging |
| 10 | **DDS feedback freshness hardening** | S | 1 | None | Stop false positives from stale 0.0° readings |

### Tier 3: Nice to Have

| # | Task | Size | Agents | Dependencies | Impact |
|---|------|------|--------|-------------|--------|
| 11 | **Factory 3D live arm tracking** | M | 1 | Tasks 1-2 | Cool visualization, validates pipeline |
| 12 | **Multimodal LLM vision** (send JPEGs to Gemini Vision) | M | 1 | None | Better than ASCII approach, if LLM detection is ever revisited |
| 13 | **Automated calibration validation** | S | 1 | Tasks 1-2 | Move to N test poses, report accuracy |

### Dependency Graph

```
[1: Intrinsics] → [2: Extrinsics] → [5: 3D Localization] → [6: Pick-and-Place]
                                   ↗
[3: DH Validation] ──────────────┘
[4: Can Detection] ──────────────→ [5: 3D Localization]
[7: Better CV] ──────────────────→ [8: Latency Fix]
```

**Critical path: Tasks 1 → 2 → 5 → 6 (with 3 and 4 in parallel)**

---

## Section 6: Architecture Improvements

### Code Quality / Tech Debt

| Issue | File(s) | Severity | Fix |
|-------|---------|----------|-----|
| `viz_calibrator.py` link lengths don't match `kinematics.py` DH params (590mm vs 651.5mm) | `src/vision/viz_calibrator.py`, `src/kinematics/kinematics.py` | **High** — two conflicting FK models | Reconcile, single source of truth |
| `detect_arm_joints()` returns `[None]*6` (stub) | `src/vision/viz_calibrator.py` | Medium | Implement or remove |
| Default intrinsics hardcoded (`fx=fy=1000`) used when no calibration file exists | Extrinsics solver | **High** | Require calibration file, fail explicitly if missing |
| `app.mount("/", StaticFiles(html=True))` catch-all blocks API routes | `web/server.py` | Fixed (moved to `/ui/`) | Already resolved |
| `smoother._arm_enabled` lost on server restart | `web/server.py` | Fixed (auto-sync from DDS) | Already resolved |
| LLM detector, comparison engine, calibration runner, results reporter — dead code if LLM approach archived | `src/vision/llm_detector.py`, `detection_comparator.py`, etc. | Low | Move to `src/vision/experimental/` or delete |

### Testing Gaps

| Gap | Current | Needed |
|-----|---------|--------|
| Integration tests with real camera frames | 0 | 5+ tests with saved JPEG fixtures |
| Calibration roundtrip test | 0 | Save calibration → load → reproject → verify error |
| End-to-end pick-and-place test (mocked hardware) | 0 | Detect → localize → plan → execute with mock arm |
| FK Python vs FK JS consistency | Exists in plan but unclear if tested | Verify for all poses in `joint-mapping.md` |
| Total test count | 460 passing | Good coverage, but vision integration tests are missing |

### Performance Bottlenecks

| Bottleneck | Current | Target | Fix |
|-----------|---------|--------|-----|
| CV detection latency | 672ms | <50ms | ROI processing, skip full-frame background model |
| Full-frame processing at 1920×1080 | Every frame | Only when needed | Downsample to 960×540 for detection, full-res for calibration only |
| Background model update | Runs every frame | Unnecessary | Remove background subtraction, use HSV + FK-guided ROI |
| eGPU (RX 580) utilization | `cv2.UMat` for some ops | All heavy CV ops | Ensure `gpu_preprocess.py` is used consistently |

---

## Summary

**The path to picking up the Red Bull can is shorter than it looks.** Most of the pipeline exists. The single biggest blocker is **camera calibration** — specifically, nobody has done a checkerboard intrinsic calibration yet. Everything downstream (3D localization, grasp planning, pick execution) is built but can't work with ~100px reprojection error.

**Minimum viable path:** 
1. Print checkerboard, calibrate intrinsics (30 min)
2. Re-solve extrinsics (10 min compute)
3. Add Red Bull can color profile (15 min)
4. Test 3D localization (30 min)
5. Run pick-and-place (30 min iteration)

**Total: ~2 hours of focused work to attempt first pick.**

The LLM ASCII experiment was inconclusive (never ran) but the approach is theoretically weak. Archive it. If LLM vision is revisited, send raw images to multimodal models.
