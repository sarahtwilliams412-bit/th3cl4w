# Calibration Data Review — cal_1770588119
**Date:** 2026-02-08 | **Reviewer:** Claw (subagent) | **Verdict:** Run failed, not usable

---

## 1. Joint Movement Accuracy (Commanded vs Actual)

The arm executed 20 unique poses across 6 joints. Motor control was **excellent** — commanded vs actual angles are consistently within ±0.5° for non-primary joints (the ones not being intentionally moved):

| Pose | Primary Motion | Key Angle Accuracy |
|------|---------------|-------------------|
| 0 | Home position | All joints ~0° (within 0.6°) |
| 1 | Base +30° | Base actual: 29.5° (Δ=-0.5°) |
| 2 | Base -30° | Base actual: -29.4° (Δ=+0.6°) |
| 3 | Shoulder -30° | Shoulder actual: -29.7° (Δ=+0.3°) |
| 4 | Shoulder -60° | Shoulder actual: -60.3° (Δ=-0.3°) |
| 5 | Elbow +45° | Elbow actual: 45.3° (Δ=+0.3°) |
| 6 | Elbow -45° | Elbow actual: -44.5° (Δ=+0.5°) |
| 13 | Multi-joint | Base: 44.7°, Shoulder: -44.9°, Elbow: 30.7° |
| 19 | Extreme multi | Shoulder: -60.2°, Elbow: 60.1°, Wrist: -44.1° |

**Assessment:** Joint control is accurate to ~0.5° across all poses. The D1 arm's servos are performing well. This is NOT the problem.

Non-primary joints show residual drift of 0.1-0.8° from zero, which is normal for this class of arm. Cross-coupling between joints is minimal.

---

## 2. CV Detection Analysis — Why 1.5% Detection Rate?

### The Numbers
- **Total observations:** 200 (20 poses × 2 cameras × 5 joints)
- **Total detections:** 3 (1.5%)
- **Camera 0 detections:** 0/100 (0%)
- **Camera 1 detections:** 3/100 (3%)
- **LLM detections:** 0/200 (LLM was never invoked at all — 0 tokens, 0 latency)

### The 3 Successful Detections

| Pose | Cam | Joint | FK Pixel | CV Pixel | Error (px) | CV Source |
|------|-----|-------|----------|----------|-----------|-----------|
| 9 | 1 | end_effector | (1242, 380) | (1245, 377) | 4.6 | "width" |
| 12 | 1 | end_effector | (1191, 540) | (1184, 522) | 19.6 | "width" |
| 13 | 1 | elbow | (803, 695) | (827, 709) | 27.6 | "gold" |

Key observations:
- All 3 detections came from **camera 1 only** (the overhead/top-down camera)
- Only **end_effector** (2 detections) and **elbow** (1 detection) were ever found
- **Base, shoulder, and wrist were never detected** across any pose or camera
- CV source was "width" (2×) and "gold" (1×) — suggests width-based blob detection found the end effector tip, while the "gold" detection may have used fiducial markers
- When detected, accuracy ranged from excellent (4.6px) to poor (27.6px)

### Why So Low?

Based on visual inspection of the frames:

1. **Camera 0 (front-facing):** The checkerboard calibration pattern fills the background directly behind the arm. The arm is black and positioned against alternating black/white squares, which **destroys edge contrast** and confuses feature extraction. The high-frequency repeating pattern is essentially adversarial to the detector.

2. **Camera 1 (overhead/top-down):** Better perspective — the arm is visible against the gray base plate and white desk. But the checkerboard is still present in frame (left side), and the arm's dark color provides limited texture differentiation between joints.

3. **Joint geometry:** Base and shoulder joints are at the arm's mounting point — from overhead they overlap with the base plate. Wrist joints are small and lack distinctive visual features. End effector (gripper) has the most distinctive shape, which explains why it was the most-detected joint (5%).

4. **No color markers or fiducials on joints:** The arm appears to be bare black — no ArUco markers, no colored tape, no LED markers on the joints. The CV system has nothing distinctive to latch onto.

---

## 3. Visual Frame Inspection

I examined 6 frames across different poses and cameras:

### Camera 0 (Front View)
- **Scene:** Garage/workshop with the arm on a table, large checkerboard calibration board propped up behind it
- **Arm visibility:** Clearly visible to human eyes — black multi-jointed arm (Unitree D1) on cylindrical base
- **CV challenge:** The checkerboard completely dominates the background. The arm's black links blend into the black squares. Cluttered environment (Red Bull can, Unitree box, DeWalt tools, racing wheel, whiteboard)
- **Assessment:** This camera angle is nearly unusable for automated joint detection without markers

### Camera 1 (Overhead/Top-Down View)  
- **Scene:** Bird's-eye view of the workspace. Arm visible on gray mounting plate, checkerboard to the left, measuring tape for scale reference, circular targets on paper
- **Arm visibility:** Better — arm stands out against the lighter mounting plate
- **CV challenge:** Still cluttered. The checkerboard is partially in frame. Arm is still uniformly black.
- **Assessment:** This is the better of the two cameras (3 detections vs 0), but still insufficient without markers

### Key Visual Findings
- The arm IS clearly visible and identifiable to human vision in all frames
- The arm is **entirely black** with no visual markers on any joints
- The checkerboard calibration board is counterproductive — it helps with camera intrinsics but actively hurts joint detection
- The workspace is cluttered with distracting objects

---

## 4. Camera Extrinsics Analysis

### Camera 0 (Front-facing)
- **Reprojection error mean: 138.8 px** ← This is **terrible**
- **Reprojection error max: 202.7 px**
- **Poses used: 5, Inliers: 5**
- Camera matrix: f=1000, cx=960, cy=540 (default/assumed values)

### Camera 1 (Overhead)
- **Reprojection error mean: 90.4 px** ← Still very bad
- **Reprojection error max: 182.2 px**
- **Poses used: 9, Inliers: 5**
- Camera matrix: f=1000, cx=960, cy=540 (default/assumed values)

### Assessment
Both cameras have **absurdly high reprojection errors.** For reference:
- Good calibration: <1 px reprojection error
- Acceptable: <5 px
- Poor: 5-20 px
- **This run: 90-139 px** — off by 1-2 orders of magnitude

The camera intrinsics appear to be **default/placeholder values** (focal length 1000, principal point at image center). These were not properly calibrated. The extrinsics were computed from only 5 inlier poses with these bad intrinsics, producing garbage transforms.

**This alone invalidates the entire calibration run.** Even if CV detected every joint perfectly, the FK→pixel projection would be wildly wrong, making error measurements meaningless.

---

## 5. What Needs to Change for the Next Run

### Critical (Must Fix)

1. **Calibrate camera intrinsics FIRST**
   - Use the checkerboard to run proper camera calibration (OpenCV `calibrateCamera`)
   - Get actual focal lengths, principal points, and distortion coefficients
   - Target: <1 px reprojection error on the intrinsics calibration
   - Do this BEFORE the arm calibration run

2. **Add visual markers to the arm joints**
   - Stick colored tape, ArUco markers, or small LEDs at each joint
   - Each joint should have a unique, easily-detectable marker
   - Even simple colored dots (red, green, blue, yellow, orange) would be a massive improvement
   - This is the single biggest improvement for detection rate

3. **Remove or reposition the checkerboard during the arm calibration run**
   - Use the checkerboard to calibrate the cameras, then remove it
   - Or move it completely out of the arm's background
   - It's actively degrading detection by confusing the CV pipeline

### Important (Should Fix)

4. **Enable the LLM detection path**
   - Zero LLM calls were made (0 tokens, 0 latency) — it appears completely disabled/unwired
   - The whole point of a CV vs LLM comparison is moot without LLM data
   - Debug why `llm_tokens` = 0 and `llm_latency_ms` = 0 across all 40 observations

5. **Improve camera 0 positioning**
   - The front view against the checkerboard is nearly useless
   - Consider: side view, 45° angle, or a second overhead at a different height
   - Ensure the arm has a clean, contrasting background

6. **Declutter the workspace**
   - Remove the Red Bull can, Unitree box, tools, and other objects from the arm's vicinity
   - A clean background dramatically improves segmentation

### Nice to Have

7. **Add more poses** — 20 poses is reasonable but more data points (30-50) would give better coverage
8. **Increase CV detection timeout** — current ~670ms average latency is fine but check if the pipeline is timing out before completing
9. **Consider controlled lighting** — current garage lighting seems adequate but could be more uniform

---

## Summary

| Metric | Value | Target |
|--------|-------|--------|
| Arm accuracy | ±0.5° | ✅ Good |
| CV detection rate | 1.5% | ≥80% needed |
| LLM detection rate | 0% (never called) | Should be tested |
| Camera 0 reproj error | 138.8 px | <1 px |
| Camera 1 reproj error | 90.4 px | <1 px |
| Usable data points | 3 of 200 | ≥160 of 200 |

**Bottom line:** The arm works great. Everything else failed. The cameras aren't calibrated (placeholder intrinsics), the CV pipeline can't see unmarked black joints against a checkerboard background, and the LLM path was never invoked. Fix the camera intrinsics, add joint markers, remove the checkerboard from the background, and enable the LLM pipeline. Then re-run.
