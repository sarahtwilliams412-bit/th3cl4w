# Calibration Assessment v2 — D1 Arm, 20-Frame Review
**Date:** 2026-02-08 | **Analyst:** Lead Analyst (synthesized from 4 batch reports + prior data_review)

---

## Executive Summary

The arm hardware is solid (±0.5° accuracy). Everything else needs work. Across all 20 frames, the arm is **clearly visible to humans** but **nearly impossible for automated CV** — the prior run achieved only 1.5% detection rate (3/200 observations). This second capture set (with checkerboard covered) shows the same fundamental problems: black arm on cluttered background, no markers, minimal pose diversity, and the checkerboard is still very much in frame despite being "covered." The 20 frames contain at most **3-4 unique poses** — the rest are near-duplicates from burst captures.

---

## 1. Overall Visibility

**Human visibility: Excellent.** The arm is clearly identifiable in all 20 frames from the overhead camera. The dark black multi-DOF arm (5-6 joints) extends from its circular base across the OSB plywood workspace. Base, shoulder, elbow, wrist, and end-effector are all discernible to a human observer.

**Machine visibility: Poor.** The arm is:
- Uniformly black/dark gray — no distinctive color or texture
- Positioned against a mixed black/white/brown background
- Surrounded by objects with similar visual properties (shadows, dark monitors, cables)
- Lacking any markers, tags, or visual features at joints

**Segmentability score: 2/10** for classical CV without modifications.

---

## 2. Background Quality — Did Covering the Checkerboard Help?

**No.** The checkerboard was NOT effectively covered/removed. Both batch reports consistently describe:
- Two large checkerboard targets still visible — one propped at an angle (left side), one flat on the desk
- The arm still crosses over the checkerboard pattern
- Black squares still merge with the black arm silhouette
- The draped/angled checkerboard shows warping/curling at edges

**The checkerboard remains the single biggest obstacle to arm segmentation.** Its high-frequency B&W pattern is essentially adversarial to edge detection, contour extraction, and background subtraction. It also degrades its own usefulness for camera calibration when warped.

**Remaining background issues:**
- Severe workshop clutter (monitors, racing wheel, vacuum, boxes, bucket, tripod, tools)
- Metal L-square ruler and yellow measuring tape run parallel to arm links — confusing for edge detection
- Cables routed along arm links add false contours
- A pet (dog) appears in frames 7 and 10, breaking motion-based approaches
- Human hand/arm visible in frames 11-15

---

## 3. Pose Diversity

**Critically insufficient.** The 20 frames contain roughly **3-4 unique poses at most:**

| Batch | Frames | Unique Poses | Notes |
|-------|--------|-------------|-------|
| A (1-5) | 5 | ~1-2 | Extended diagonal, incremental micro-adjustments |
| B (6-10) | 5 | ~1 | Nearly identical extended pose throughout |
| C (11-15) | 5 | 1 | Static burst — zero movement detected |
| D (16-20) | 5 | 1 | Static burst — frames 16-17 are near-duplicates |

**All poses are "extended/outstretched"** — the arm reaches diagonally from base over the checkerboard in every frame. There are:
- ❌ No folded/tucked configurations
- ❌ No vertical/elevated poses
- ❌ No base rotation variations (all roughly same yaw)
- ❌ No wrist articulation variety
- ❌ No extreme joint angle poses

For calibration, you need **diverse joint configurations** that exercise each joint through its range. The prior run's commanded poses (±30° base, ±60° shoulder, ±45° elbow, multi-joint combos) were good — but the frames captured here don't reflect that diversity. Either the capture happened during a single static pose, or the camera captured burst frames without waiting for arm movement.

---

## 4. CV Feasibility

### What WON'T work:
- **HSV/color segmentation:** The arm is black. The background has black objects. No distinctive hue to isolate. **Dead on arrival.**
- **Edge/contour detection:** The checkerboard generates thousands of edges that overwhelm arm contours. The L-square, measuring tape, and cables add more false edges. **Unusable without major cleanup.**
- **Simple background subtraction:** Requires a clean reference frame AND a stable scene. The pet, human hand, and general clutter instability make this fragile. Could work in theory but unreliable in practice.

### What COULD work (with modifications):
- **Background subtraction + colored markers:** Capture a clean background frame, then use markers to validate/filter foreground detections. **Medium effort, medium reliability.**
- **HSV marker detection:** Add bright fluorescent markers at joints → detect markers via HSV thresholding. **Low effort, high reliability.** This is the fastest path to working detection.
- **ArUco/fiducial markers:** Place unique ArUco tags at each joint. Gives both position AND orientation per joint. **Low-medium effort, high reliability, rich data.**
- **Deep learning (fine-tuned segmentation):** Train a model on annotated frames of this specific arm. **High effort, high reliability, but overkill for calibration.**

### What WOULD work best:
- **Depth camera (RGB-D):** An Intel RealSense or similar would trivially separate the arm plane from the desk plane. **Medium cost, very high reliability.** Multiple batch analysts recommended this.

---

## 5. LLM Feasibility (Gemini)

**Promising but untested.** The prior run showed 0 LLM calls (pipeline was never invoked). Based on frame quality:

- **Joint identification:** A VLM like Gemini could likely identify the arm and approximate joint locations from these frames — the arm is clearly visible to human-level vision. Accuracy would be rough (±20-50px from overhead at this resolution).
- **Reliability concerns:**
  - The cluttered background could confuse even VLMs
  - Black-on-black overlap with checkerboard is problematic
  - Overhead perspective compresses depth, making joint articulation ambiguous
  - Without markers, distinguishing shoulder from elbow from wrist is genuinely hard even for humans at this angle
- **Recommendation:** Enable the LLM pipeline and test. It may achieve 30-60% detection with current frames, dramatically better than 1.5% CV. Adding markers would push LLM reliability to 80%+.

---

## 6. Marker Recommendations

**This is the highest-impact, lowest-effort improvement available.**

### Marker Placement (5 markers minimum):
1. **Base joint** — ring or cap on top of circular base housing
2. **Shoulder joint** — where lower link meets base assembly
3. **Elbow joint** — on the joint cluster mid-arm (most visually confusing area)
4. **Wrist joint** — on wrist housing near end-effector
5. **End-effector tip** — at or near the camera/sensor mount (TCP)

### Color Selection:
| Color | Why | Avoid? |
|-------|-----|--------|
| **Neon green** | ✅ Nothing green in scene, high saturation | Best choice |
| **Fluorescent orange** | ✅ High contrast, distinct from scene | Excellent |
| **Hot pink / magenta** | ✅ Absent from scene | Good |
| Red | ❌ Red accents already on arm, steering wheel | Avoid |
| Blue | ❌ Shop vac, cart, tape in scene | Avoid |
| Yellow | ❌ Measuring tape already yellow | Avoid |
| White/black | ❌ Checkerboard, arm itself | Absolutely not |

### Marker Types (ranked by utility):
1. **ArUco tags** — unique per joint, gives position + orientation, OpenCV native support. Print small (2-3cm) tags, attach to flat surfaces on each joint. **Best option if joints have flat surfaces.**
2. **Fluorescent tape rings** — wrap 1-2cm bands of neon tape around each joint. Simple, cheap, immediately effective for HSV detection. **Best option for speed.**
3. **3D-printed marker caps** — custom-fit colored caps for each joint. More work but durable and precisely positioned.
4. **LED markers** — small colored LEDs at each joint. Excellent for detection but requires wiring.

### Link marking (optional but helpful):
- Wrap arm links in white or bright tape to distinguish from parallel objects (ruler, measuring tape)
- Even partial wrapping would dramatically improve contour-based approaches

---

## 7. Lighting & Environment

### Current state:
- Overhead fluorescent/LED lighting — relatively even, adequate intensity
- Shadows under the arm are soft, not dramatic
- No severe reflections (matte surfaces throughout)
- Specular highlights on arm's metal/plastic surfaces are minor

### Recommended changes:

| Change | Impact | Effort |
|--------|--------|--------|
| **Remove checkerboard from frame during arm tracking** | 🔴 Critical | Low — just move it |
| **Clear L-square, measuring tape, loose tools from workspace** | 🟡 High | Very low |
| **Remove/tidy cables along arm** | 🟡 Medium | Low |
| **Secure pet during captures** | 🟡 Medium | Low |
| **Add a plain contrasting backdrop behind arm** | 🟢 Medium | Low — white sheet/foam board |
| **Add diffuse side lighting to reduce shadows** | 🟢 Low-medium | Medium |
| **Use controlled/uniform background (green screen)** | 🟢 Medium | Medium |

---

## 8. Prioritized Action Plan

### 🔴 Tier 1 — Do Before Next Run (Critical, Low Effort)

| # | Action | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 1 | **Add colored markers to all joints** | 30 min | ★★★★★ | Neon tape or ArUco tags. Transforms detection from impossible to trivial. |
| 2 | **Remove checkerboard from arm's background** | 5 min | ★★★★★ | Use it for camera intrinsic cal first, then physically remove before arm poses. |
| 3 | **Calibrate camera intrinsics properly** | 30 min | ★★★★★ | Run OpenCV `calibrateCamera` with the checkerboard. Get actual focal lengths + distortion. Current placeholder values (f=1000) produce 90-139px reproj error. |
| 4 | **Clear workspace clutter near arm** | 10 min | ★★★★ | Remove L-square, measuring tape, loose tools, cables from arm's reach zone. |
| 5 | **Enable LLM detection pipeline** | 15 min | ★★★★ | Debug why 0 LLM calls were made. The CV-vs-LLM comparison needs both working. |

### 🟡 Tier 2 — Important Improvements

| # | Action | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 6 | **Ensure pose diversity in capture** | Code fix | ★★★★★ | 20 frames should be 20 DIFFERENT poses. Current data has ~3-4 unique poses. Verify the capture script waits for arm movement completion before triggering camera. |
| 7 | **Secure pet during calibration** | 5 min | ★★★ | Dog appears in 2 frames, would break motion-based detection. |
| 8 | **Flatten or replace warped checkerboard** | 10 min | ★★★ | The draped board is curled — bad for intrinsic calibration. Use a rigid board. |
| 9 | **Add second camera angle** | 30 min | ★★★ | Current overhead-only view compresses depth. A 45° side view would help disambiguate joints. |

### 🟢 Tier 3 — Nice to Have

| # | Action | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 10 | **Wrap arm links in contrasting tape** | 20 min | ★★ | White tape on black links improves contour detection. |
| 11 | **Add depth camera (RealSense)** | $150 + setup | ★★★★ | Would make segmentation trivial regardless of colors/markers. |
| 12 | **Improve lighting (diffuse fill)** | 30 min | ★★ | Reduces shadow confusion, minor improvement. |
| 13 | **Green screen backdrop** | 20 min | ★★★ | Enables perfect background subtraction but overkill if markers work. |

---

## Bottom Line

The D1 arm's motors are accurate. The vision pipeline is broken. **Three changes would fix 90% of the problem:**

1. **Markers on joints** (neon tape = 30 min)
2. **Checkerboard out of frame** (5 min)
3. **Real camera calibration** (30 min)

Total: ~1 hour of prep work. After that, even simple HSV detection should achieve >80% joint detection rate. Add the LLM pipeline and you'll have a meaningful CV-vs-Gemini comparison.

The current 20 frames are also nearly useless for pose calibration because they contain only ~3-4 unique arm configurations instead of 20. Fix the capture script to verify arm movement between frames before the next run.
