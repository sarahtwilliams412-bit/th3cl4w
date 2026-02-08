# Frame Analysis — Batch B (Frames 6–10)

> Generated 2026-02-08 by subagent frame-review-B

## Common Scene Description

All five frames share the same overhead camera viewpoint looking down at a workshop/garage desk. The robot arm is a dark black multi-DOF articulated manipulator (likely 5-6 joints) mounted on a circular base on an OSB plywood board. Two large printed checkerboard calibration targets are present — one laid flat, one propped/draped at an angle. A yellow measuring tape provides scale reference. The environment is **extremely cluttered** with monitors, keyboards, a racing wheel, cardboard boxes, a CRT monitor, a shop vac, a paint bucket, a tripod, cables, tools, and other garage items surrounding the workspace.

---

## Frame 6 (`calib_0006.jpg`)

**Arm Visibility:** Yes — clearly visible. The arm extends from the circular base (center-right) upward toward the top of frame. At least 3 segments and 4 joints are discernible: base rotary joint, shoulder, elbow, and a wrist/end-effector cluster at the tip. A small colored element (possibly a camera module) is visible at the end effector.

**Distinguishability:** The arm is dark black against a mixed background. Over the white checkerboard paper and the light plywood, contrast is **good**. However, where the arm crosses over the black squares of the checkerboard or passes near the dark background clutter (monitors, boxes), it **blends in badly**. Edges are hard to trace in those regions.

**Occlusion/Issues:** No significant occlusion of the arm itself. The checkerboard behind the arm is partially occluded by the arm's shadow and body. The draped checkerboard is visibly warped/curved — not flat. Minor shadows under the arm on the plywood. No obvious reflections.

**Pose:** Near-fully extended, roughly straight from base toward top of frame. Appears to be a reference/home calibration pose — links mostly in-line.

**CV Segmentation Feasibility:** Difficult with naive approaches. Background subtraction could work if you have a clean background reference, but the extreme clutter means any camera bump would break it. HSV segmentation is **poor** — the arm is black/dark gray with no distinctive hue, and the background contains similar dark objects. Contour detection would struggle where arm overlaps dark checkerboard squares. **Reliability: Low-Medium** without additional markers.

**Colored Markers:** Would help significantly. Recommended placement:
- Base joint (where arm meets the circular platform)
- Shoulder joint (first major bend point)
- Elbow joint (mid-arm)
- Wrist joint (near end-effector cluster)
- End-effector tip

Bright, distinct colors (neon green, orange, magenta) would contrast well against both the dark arm and the black/white checkerboard.

---

## Frame 7 (`calib_0007.jpg`)

**Arm Visibility:** Yes — same general view as frame 6. The arm is in an almost identical pose. 3 segments, ~4 joints visible. Very similar to frame 6.

**Distinguishability:** Same characteristics as frame 6. Good contrast over light surfaces, poor over dark ones. No meaningful change in lighting or arm color.

**Occlusion/Issues:** Nearly identical to frame 6. One notable difference: **a dog (or pet) is visible in the lower-right corner**, partially blurred — this is a moving object that could confuse background subtraction or motion-based segmentation. Otherwise, same shadow/clutter situation.

**Pose:** Essentially the same extended pose as frame 6. If the arm moved between these two captures, the change is minimal/imperceptible from the overhead view.

**CV Segmentation Feasibility:** Same as frame 6, with the added complication that a **pet entering the frame** could generate false contours or motion artifacts. Background subtraction would flag the dog as foreground along with the arm. **Reliability: Low-Medium**, slightly worse than frame 6 due to the pet.

**Colored Markers:** Same recommendations as frame 6. The presence of a moving pet makes marker-based tracking even more important — it provides a way to distinguish arm joints from other moving objects.

---

## Frame 8 (`calib_0008.jpg`)

**Arm Visibility:** Yes — clearly visible. Same overhead perspective. The dark arm extends from the circular base outward. At least 2-3 intermediate joints visible as bulkier servo housings along the length, plus the base and end-effector. Total: ~4-5 discernible joints, 3 link segments.

**Distinguishability:** Dark arm against mixed background. Moderate-to-good contrast over the flat checkerboard and plywood. Poor contrast where it overlaps dark clutter. The arm's servo housings create recognizable bulges that help delineate joints visually, but they're all the same dark color.

**Occlusion/Issues:** No arm occlusion. The larger draped checkerboard is still visibly warped/curled — this would degrade any checkerboard corner detection for camera calibration. Arm shadow visible on the plywood surface. The cluttered right side of the frame (overturned equipment, cables) remains problematic for any whole-frame segmentation.

**Pose:** Arm is in a somewhat stretched/extended configuration, roughly in-line from base toward the top of the frame. Similar to frames 6-7 but possibly with very slight joint angle differences (hard to confirm from overhead).

**CV Segmentation Feasibility:** Same challenges as previous frames. The black arm on a mixed black/white/brown background is not HSV-separable. Background subtraction requires a stable, arm-free reference frame and would be sensitive to lighting changes and the cluttered periphery. Contour-based approaches would pick up too many edges from the checkerboard pattern itself. **Reliability: Low-Medium.**

**Colored Markers:** Same placement recommendations. Bright markers at each joint would be the most reliable segmentation approach given this environment.

---

## Frame 9 (`calib_0009.jpg`)

**Arm Visibility:** Yes — same setup. The arm is clearly present, extending from the base. Joint/segment count consistent with previous frames: circular base, 2-3 link segments, 4-5 joints including wrist cluster.

**Distinguishability:** Same as other frames. The arm is uniformly dark, blending with dark background elements. Reasonable contrast only over light surfaces (white paper, light plywood).

**Occlusion/Issues:** No occlusion of the arm. The flat checkerboard shows some warping/rippling on the paper surface. The paint bucket and various tools are still in frame. No new reflections or shadows compared to previous frames.

**Pose:** Near-horizontal, outstretched, arm reaching over the checkerboard — appears to be positioning the end effector (or mounted camera) over the calibration target. Essentially the same extended pose seen throughout this batch.

**CV Segmentation Feasibility:** Same assessment. Without colored markers or a controlled background, reliable segmentation is **unlikely** using basic CV methods. The checkerboard's own high-contrast pattern creates enormous numbers of edges/contours that would overwhelm arm detection. **Reliability: Low-Medium.**

**Colored Markers:** Same recommendations as above. Additionally, consider placing a distinctly-colored marker or tag at the end-effector tip for precise tip-tracking during calibration.

---

## Frame 10 (`calib_0010.jpg`)

**Arm Visibility:** Yes — same overhead view. Arm visible from base through end effector. Same joint/segment count as other frames.

**Distinguishability:** Consistent with all previous frames in this batch. Dark arm, mixed background, moderate contrast over light areas only.

**Occlusion/Issues:** **A dog/pet is again partially visible** in the lower-right (blurred, moving into frame) — same as frame 7. This is a recurring issue that would disrupt motion-based or background-subtraction approaches. Otherwise, same clutter and shadow situation.

**Pose:** Extended, similar to all other frames in this batch. The arm appears to be in roughly the same configuration throughout frames 6-10, suggesting either: (a) these are near-static calibration captures at the same pose, or (b) arm movements between frames are very small.

**CV Segmentation Feasibility:** Same challenges, compounded by the pet. Background subtraction would flag the dog. HSV is useless for the dark arm. Contour detection overwhelmed by checkerboard edges. **Reliability: Low**, worst in this batch due to the pet.

**Colored Markers:** Critical here. Recommendations:
- **4-5 bright colored markers** at each joint (base, shoulder, elbow, wrist, tip)
- Use **distinct hues** (e.g., neon green, hot pink, bright orange, cyan, yellow) — one per joint for unambiguous identification
- Markers should be **visible from the overhead camera** — top-facing, not just side-facing
- Size: at least 1-2 cm diameter to be reliably detected at this camera distance

---

## Batch B Summary

| Aspect | Assessment |
|--------|-----------|
| **Arm visible in all frames** | ✅ Yes — consistently visible |
| **Pose variation across batch** | ❌ Minimal — arm appears nearly static in all 5 frames |
| **Background complexity** | 🔴 Severe — cluttered garage with many dark objects |
| **Arm-background contrast** | 🟡 Moderate — good over light surfaces, poor over dark areas |
| **Naive CV segmentation** | 🔴 Unreliable — dark arm, checkerboard edges, clutter, pet |
| **Pet intrusion** | ⚠️ Frames 7 and 10 — moving animal in lower-right |
| **Colored markers needed** | ✅ Strongly recommended — only reliable path for joint tracking |
| **Checkerboard quality** | 🟡 Flat board OK; draped board is warped and should be flattened or removed |

### Key Recommendations
1. **Add colored joint markers** — this is essentially required for reliable CV-based joint tracking in this environment
2. **Flatten or remove the draped checkerboard** — the warped sheet will degrade camera calibration
3. **Clear clutter from the arm's workspace** if possible, or use a controlled backdrop
4. **Capture more pose variety** — all 5 frames show nearly identical arm configuration; need diverse poses for meaningful calibration
5. **Secure the pet** during calibration captures to avoid motion artifacts
