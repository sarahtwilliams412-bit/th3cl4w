# 3D Reconstruction Plan — Lead Agent B (Calibration & Testing)

**Date:** 2026-02-08  
**Author:** Lead Agent B  
**Scope:** Calibration strategy, testing plan, ASCII vs raw decision, error metrics

---

## 1. Key Finding: Reconstruct from RAW Frames, Not ASCII

**Decision: Use raw camera frames for reconstruction. ASCII is display-only.**

Reasoning:
- ASCII conversion loses spatial resolution catastrophically. A 640×480 frame becomes ~80×24 ASCII characters — a 160× reduction in horizontal resolution and 20× in vertical. Sub-pixel accuracy (needed for calibration) is impossible.
- The existing `calibration.py` already works with raw OpenCV frames (`cv2.findChessboardCorners`, `cv2.cornerSubPix` — sub-pixel refinement).
- The camera server at `localhost:8081` serves raw JPEG snapshots (`/snapshot?cam=0`). ASCII conversion happens downstream.
- The `IndependentCalibrator` and `CameraCalibration` classes are already designed for raw pixel coordinates.
- ASCII can still be used for real-time monitoring/display of the reconstruction results.

## 2. Existing Infrastructure Assessment

### What already exists and works:
| Component | File | Status |
|-----------|------|--------|
| Per-camera intrinsic calibration | `src/vision/calibration.py` | ✅ Complete — checkerboard-based, OpenCV |
| Extrinsic (cam→workspace) | `calibration.py:compute_extrinsic()` | ✅ Complete — solvePnP + board_to_workspace |
| Pixel-to-ray | `CameraCalibration.pixel_to_ray()` | ✅ Complete |
| Pixel-to-workspace-plane | `CameraCalibration.pixel_to_workspace()` | ✅ Complete — ray-plane intersection |
| CLI calibration tool | `tools/calibrate_cameras.py` | ✅ Complete — interactive capture + save |
| 7-DOF DH FK/IK | `src/kinematics/kinematics.py` | ✅ Complete — D1Kinematics class |
| Joint positions 3D | `D1Kinematics.get_joint_positions_3d()` | ✅ Returns all frame origins |
| Dual-camera tracker | `src/vision/arm_tracker.py` | ✅ Cross-references cam0/cam1 detections |
| Viz calibrator (2D) | `src/vision/viz_calibrator.py` | ✅ Side-view FK→pixel optimization |
| Empirical joint mapping | `docs/joint-mapping.md` | ✅ Pixel coords for known poses |
| Joint state API | `/api/state` | ✅ Returns 6 joint angles in degrees |
| Safety monitor | `src/safety/safety_monitor.py` | ✅ Joint/velocity/torque limits |

### What's missing for 3D reconstruction:
1. **Camera-to-arm-base calibration** — extrinsics exist but need to be computed relative to the arm base frame specifically (not just a generic "workspace")
2. **Multi-view 3D triangulation** — `pixel_to_workspace()` projects onto a single plane (Z=known). Need ray-ray closest-point for arbitrary 3D
3. **Joint detection in images** — `detect_arm_joints()` in viz_calibrator returns `[None]*6` (stub)
4. **Reconstruction pipeline** — no pipeline that: detect joints in both cameras → triangulate → compare with FK
5. **Test infrastructure for 3D accuracy** — no tests comparing vision-estimated pose with FK ground truth

## 3. Calibration Strategy

### 3.1 Camera Intrinsics (already implemented)
Use existing `IndependentCalibrator` with checkerboard. Need 10+ images per camera from varied angles.

### 3.2 Camera-to-Arm-Base Extrinsics (the hard part)

**Approach: Hand-eye calibration using known arm poses as ground truth.**

The key insight: we have FK and joint encoders. If we move the arm to N known poses and detect the end-effector (or any joint) in both cameras, we can solve for the camera extrinsics relative to the arm base.

**Method:**
1. Place checkerboard at a known position relative to arm base (measure manually)
2. Use `compute_extrinsic()` to get cam→checkerboard transform
3. Compose with measured checkerboard→arm-base transform to get cam→arm-base

**Alternative (better, no checkerboard needed):**
1. Move arm to N known poses (using `/api/command/set-all-joints`)
2. For each pose, compute FK to get ground-truth 3D joint positions
3. Detect end-effector (or gripper tip) in both camera frames
4. Solve for each camera's extrinsic [R|t] using PnP with the FK positions as 3D points and detected pixel locations as 2D points
5. Minimum 4 poses needed; 10+ recommended for robustness

**Why the FK-based approach is better:**
- No need for a physical calibration target near the arm
- Uses the arm itself as the calibration object
- Automatically accounts for arm mounting position
- Joint mapping data in `docs/joint-mapping.md` already gives us pixel coords for known poses — this is literally calibration data waiting to be used!

### 3.3 Using Existing Joint Mapping Data as Bootstrap

The `joint-mapping.md` already contains gripper pixel coordinates for 8 poses in both cameras. This is enough for an initial calibration:

```
Home:      FK→(x,y,z) known,  cam0→(1130,220), cam1→(760,135)
J0+45:     FK→computed,        cam0→(1430,105), cam1→(680,520)
J0-45:     FK→computed,        cam1→(660,290)
J1+45:     FK→computed,        cam0→(1050,220), cam1→(760,270)
J1-45:     FK→computed,        cam0→(540,230),  cam1→(760,470)
J2+45:     FK→computed,        cam1→(660,330)
J2-45:     FK→computed,        cam1→(830,490)
J4+45:     FK→computed,        cam1→(760,290)
J4-45:     FK→computed,        cam1→(760,200)
```

We can compute FK for each of these poses, then use `cv2.solvePnP` with the 3D→2D correspondences to get extrinsics immediately — no arm movement needed.

## 4. 3D Reconstruction Pipeline

### Ray-ray triangulation (not plane intersection):

Since cam0 (front) and cam1 (overhead) look at the arm from different directions, we can triangulate any point visible in both views:

1. Detect feature point in cam0 → pixel (u0, v0) → ray R0 in world frame
2. Detect feature point in cam1 → pixel (u1, v1) → ray R1 in world frame
3. Find closest point between rays R0 and R1 → 3D position
4. Distance between rays at closest point = reconstruction error metric

The existing `pixel_to_ray()` already converts pixels to unit rays in camera frame. With extrinsics, transform to world frame, then standard closest-point-on-two-lines formula.

### What to detect:
- **Gripper tip** — most distinctive, gold colored, at the end of the chain
- **Joint positions** — harder but possible via arm segmentation + skeleton fitting
- **Arm silhouette** — matte black body against background, use edge detection

## 5. Error Metrics and Acceptance Criteria

### Primary metric: FK comparison error
For each pose, compare:
- `P_vision` = 3D position from dual-camera triangulation
- `P_fk` = 3D position from FK using joint angles from `/api/state`
- Error = `||P_vision - P_fk||` in mm

### Acceptance criteria:
| Metric | Target | Stretch |
|--------|--------|---------|
| EE position error (static) | < 20mm | < 10mm |
| EE position error (moving) | < 40mm | < 20mm |
| Joint position error | < 30mm | < 15mm |
| Ray intersection distance | < 15mm | < 8mm |
| Reprojection error (intrinsics) | < 1.0px | < 0.5px |
| Extrinsic reprojection error | < 5.0px | < 3.0px |
| Update rate | > 5 Hz | > 10 Hz |

### Why these numbers:
- D1 arm reach is 550mm. 20mm error = 3.6% of reach — acceptable for obstacle avoidance and visualization
- 10mm gets us into useful territory for guided grasping
- Ray intersection distance > 15mm means cameras disagree — detection problem

### Ground truth validation:
- Move arm to 20+ test poses spanning the workspace
- At each pose: read joint angles from `/api/state`, compute FK, detect in cameras, triangulate
- Compare FK positions with triangulated positions
- Report mean, median, max, and 95th percentile errors

## 6. ASCII vs Raw: Architecture Decision

```
Raw frames (640×480 JPEG) ──→ Detection/Triangulation ──→ 3D Model
                               │
                               ├──→ ASCII rendering (for display)
                               └──→ Joint angle estimation (for validation)
```

The ASCII pipeline is a CONSUMER of reconstruction results, not an input.

For the "ASCII stereo" concept: we could potentially detect bright/structured ASCII characters as features, but this is a novelty approach with terrible accuracy. The practical path is raw frames → reconstruct → render result as ASCII.

## 7. Proposed Calibration Procedure (User Steps)

### Phase A: Intrinsic calibration (one-time, ~10 min)
1. Print/display a 7×5 checkerboard (23.8mm squares — matches `DEFAULT_SQUARE_SIZE_MM`)
2. Run `python3.12 tools/calibrate_cameras.py --cam0 0 --cam1 2 --num-images 15`
3. Hold checkerboard in front of each camera, capture 15 images from varied angles
4. Results saved to `calibration/cam0_calibration.json`, `calibration/cam1_calibration.json`

### Phase B: Extrinsic calibration (arm-based, ~15 min)
1. Power on arm, enable motors
2. Run new calibration script: `python3.12 tools/calibrate_arm_cameras.py`
3. Script moves arm through 15-20 calibration poses automatically
4. At each pose: captures both cameras, detects gripper, records joint angles
5. Solves for cam0→arm_base and cam1→arm_base transforms
6. Saves to `calibration/cam0_extrinsic.json`, `calibration/cam1_extrinsic.json`

### Phase C: Bootstrap from existing data (no arm needed)
1. Use joint mapping pixel data + FK to compute initial extrinsic estimate
2. Refine when arm is available with Phase B

### Phase D: Validation
1. Move arm to 20 random poses
2. At each: compare triangulated positions with FK
3. Report accuracy metrics
4. If error > threshold, recalibrate

## 8. DH Parameter Concern

The `kinematics.py` DH params may not match the actual arm perfectly. The DH table shows:
- J1→J3 offset: 208.5mm  
- J3→J5 offset: 208.5mm
- J5→EE: 113mm
- Base to J1: 121.5mm

The `viz_calibrator.py` uses different link lengths (170mm shoulder/elbow). These should be reconciled before using FK as ground truth. The DH params in `kinematics.py` are more likely correct (standard convention) but should be validated.

**Recommendation:** During calibration Phase B, also fit link lengths to minimize reprojection error. This gives us corrected DH params as a bonus.
