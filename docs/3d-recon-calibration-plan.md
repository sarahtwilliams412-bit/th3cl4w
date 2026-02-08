# 3D Reconstruction — Calibration & Testing Plan

**Date:** 2026-02-08  
**Author:** Lead Agent B  
**Purpose:** Detailed calibration sequence, test cases, and sub-agent testing plan

---

## 1. Calibration Pose Sequence

### 1.1 Bootstrap Poses (from existing joint-mapping.md data — no arm needed)

These poses have known pixel coordinates in both cameras. Use FK to compute 3D positions, then solve extrinsics.

| # | Pose (J0,J1,J2,J3,J4,J5) | cam0 pixel | cam1 pixel | Notes |
|---|---------------------------|------------|------------|-------|
| 1 | (0,0,0,0,0,0) | (1130,220) | (760,135) | Home — vertical |
| 2 | (45,0,0,0,0,0) | (1430,105) | (680,520) | Base yaw +45 |
| 3 | (-45,0,0,0,0,0) | (920,175) | (660,290) | Base yaw -45 |
| 4 | (0,45,0,0,0,0) | (1050,220) | (760,270) | Shoulder pitch +45 |
| 5 | (0,-45,0,0,0,0) | (540,230) | (760,470) | Shoulder pitch -45 |
| 6 | (0,0,45,0,0,0) | — | (660,330) | Elbow pitch +45 |
| 7 | (0,0,-45,0,0,0) | — | (830,490) | Elbow pitch -45 |
| 8 | (0,0,0,0,45,0) | — | (760,290) | Wrist pitch +45 |
| 9 | (0,0,0,0,-45,0) | — | (760,200) | Wrist pitch -45 |

cam1 has 9 correspondences, cam0 has 5. Enough for PnP (need ≥4).

### 1.2 Live Calibration Poses (arm powered — 20 poses)

Move arm through diverse configurations spanning the workspace. Each pose should:
- Be within safe limits (J1: ±85°, J4: ±85°, others: ±135°)
- Produce gripper positions spread across the camera FOV
- Include combinations (not just single-joint moves)

**Sequence:**
```
 1. (0, 0, 0, 0, 0, 0)         — home
 2. (30, 0, 0, 0, 0, 0)        — yaw right
 3. (-30, 0, 0, 0, 0, 0)       — yaw left
 4. (0, -30, 0, 0, 0, 0)       — lean forward
 5. (0, -60, 0, 0, 0, 0)       — lean far forward
 6. (0, 0, 45, 0, 0, 0)        — elbow out
 7. (0, 0, -45, 0, 0, 0)       — elbow in
 8. (0, -30, 30, 0, 0, 0)      — forward + elbow
 9. (0, -45, 45, 0, -30, 0)    — reaching forward-down
10. (30, -30, 30, 0, 0, 0)     — right + forward + elbow
11. (-30, -30, 30, 0, 0, 0)    — left + forward + elbow
12. (0, 30, 0, 0, 0, 0)        — lean back
13. (0, -30, 45, 0, 45, 0)     — forward + elbow + wrist down
14. (45, -45, 30, 0, 0, 0)     — diagonal reach
15. (-45, -45, 30, 0, 0, 0)    — opposite diagonal
16. (0, 0, 0, 0, -45, 0)       — wrist up only
17. (0, -30, 0, 0, 45, 0)      — forward + wrist down
18. (60, 0, 30, 0, 0, 0)       — wide yaw + elbow
19. (-60, 0, 30, 0, 0, 0)      — opposite wide yaw
20. (0, -60, 60, 0, -45, 0)    — maximum forward reach
```

At each pose:
1. Command pose via `/api/command/set-all-joints`
2. Wait 2.5s for settling
3. Read actual angles from `/api/state`
4. Capture raw frames from both cameras
5. Detect gripper tip in each frame
6. Record: {commanded, actual_angles, cam0_pixel, cam1_pixel, timestamp}

### 1.3 Calibration Solve

**For each camera independently:**
1. Compute FK 3D positions for all poses using actual joint angles from `/api/state`
2. Use `cv2.solvePnP` (or `solvePnPRansac` for robustness) with:
   - objectPoints: FK end-effector positions in arm-base frame (Nx3)
   - imagePoints: detected pixel positions (Nx2)
   - cameraMatrix: from intrinsic calibration
   - distCoeffs: from intrinsic calibration
3. Output: rvec, tvec → compose into 4×4 cam_to_arm_base transform
4. Compute reprojection error on held-out poses

**Refinement:**
- If reprojection error > 5px, try `cv2.solvePnPRefineLM` 
- If still bad, check for detection outliers (RANSAC)
- Optionally co-optimize DH link lengths (see Plan B §8)

## 2. Test Cases for Accuracy Validation

### 2.1 Static Pose Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| T1: Home position | Triangulate at (0,0,0,0,0,0) → compare FK | Error < 15mm |
| T2: Single-joint sweep | 10 poses per joint, single joint varied | Mean error < 20mm |
| T3: Multi-joint poses | 20 random feasible poses | Mean error < 25mm, max < 50mm |
| T4: Extreme poses | Poses near joint limits | Error < 40mm |
| T5: Reprojection | Project FK positions into cameras | Mean < 5px, max < 10px |
| T6: Ray intersection | Distance between closest points on rays | Mean < 12mm |
| T7: Consistency | Same pose measured 10× | Std dev < 5mm |

### 2.2 Dynamic Tests

| Test | Method | Pass Criteria |
|------|--------|---------------|
| T8: Slow motion | Move one joint at 10°/s, track continuously | Lag < 200ms, error < 30mm |
| T9: Normal motion | Execute wave task, track | Lag < 300ms, error < 50mm |
| T10: Update rate | Measure pipeline throughput | > 5 Hz sustained |

### 2.3 Regression Tests (offline, no hardware)

| Test | Method | Pass Criteria |
|------|--------|---------------|
| T11: FK consistency | Compare Python FK with JS FK for 100 random poses | Max difference < 0.1mm |
| T12: Ray math | Unit test ray-ray intersection with known geometry | Exact to float precision |
| T13: PnP synthetic | Synthetic 3D→2D projection + PnP solve → recover known extrinsic | Rotation < 0.1°, translation < 1mm |
| T14: Calibration save/load | Save + load calibration files | Roundtrip exact |
| T15: Degenerate inputs | All points collinear, duplicate points, noise | Graceful failure, not crash |

## 3. Testing Plan for Lead A's Sub-Agents

Lead A is expected to create ~5 sub-agents for the reconstruction pipeline. Here's how to test each:

### Sub-Agent 1: Joint/Feature Detection
- **Input:** Raw camera frame + known pose
- **Expected:** Pixel coordinates of gripper tip (and ideally joints)
- **Test:** Compare detected pixels against manual annotations for 20 frames
- **Metric:** Detection rate > 90%, position error < 10px

### Sub-Agent 2: Camera Calibration Integration  
- **Input:** Calibration data files
- **Expected:** Loaded CameraCalibration objects with valid intrinsics + extrinsics
- **Test:** Load saved calibration, reproject known 3D points, check error
- **Metric:** Reprojection error < 5px

### Sub-Agent 3: Triangulation
- **Input:** Two pixel coordinates (one per camera) + calibration
- **Expected:** 3D point in arm-base frame
- **Test:** Synthetic test data (project known 3D point → 2D → triangulate → compare)
- **Metric:** Error < 2mm on synthetic data, < 20mm on real data

### Sub-Agent 4: FK Comparison / Validation
- **Input:** Estimated 3D joint positions + actual joint angles
- **Expected:** Per-joint error metrics
- **Test:** Compare against FK at 50+ poses
- **Metric:** Reports correct error values (verified against manual calculation)

### Sub-Agent 5: Real-time Pipeline / WebSocket Integration
- **Input:** Continuous camera stream + joint state stream
- **Expected:** Merged 3D arm model at ≥5 Hz
- **Test:** Run for 30s during arm motion, verify no crashes, measure latency
- **Metric:** No frame drops, latency < 300ms

### Integration Tests

| Test | Components | Method |
|------|-----------|--------|
| I1: Full pipeline offline | All 5 sub-agents | Replay saved frames + joint data → verify output |
| I2: Full pipeline live | All 5 + hardware | Move arm through test sequence, record all metrics |
| I3: Degraded mode | Missing one camera | Should fallback to single-camera estimation |
| I4: Stale calibration | Use old calibration with moved camera | Should detect and warn |

## 4. Data Collection Protocol

For both calibration and testing, save:
```
data/calibration/
  session_YYYYMMDD_HHMMSS/
    poses.json          — [{pose_idx, commanded, actual, timestamp}, ...]
    cam0/
      pose_001.jpg      — raw frame
      pose_001_det.json — detected features
    cam1/
      pose_001.jpg
      pose_001_det.json
    calibration_result.json  — final extrinsics
    metrics.json             — reprojection errors, etc.
```

This enables:
- Offline re-calibration with different algorithms
- Regression testing without hardware
- Debugging detection failures

## 5. DH Parameter Validation

Before trusting FK as ground truth, validate DH params:

1. Command arm to home → measure physical height with ruler
2. FK says home EE is at Z = 0.1215 + 0.2085 + 0.2085 + 0.1130 = **0.6515m** above base
3. Measure actual height — if off by > 10mm, DH params need correction
4. Repeat for a few extended poses

The `viz_calibrator.py` link lengths (80+170+170+60+60+50 = 590mm) don't match the DH params (121.5+208.5+208.5+113 = 651.5mm). This 60mm discrepancy MUST be resolved before using FK as ground truth.

## 6. Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| DH params wrong | All FK comparisons invalid | Validate physically first |
| Gripper detection fails | Can't calibrate extrinsics | Use colored marker on gripper, or ArUco tag |
| Cameras moved after calibration | Extrinsics stale | Check reprojection error periodically |
| Matte-black arm hard to detect | Low detection rate | Use IR camera, or add visual markers |
| ASCII-only access to cameras | Can't get raw frames | Camera server already serves raw JPEG — verified |
| Both cameras can't see gripper simultaneously | Can't triangulate some poses | Plan poses where both cameras have visibility |
