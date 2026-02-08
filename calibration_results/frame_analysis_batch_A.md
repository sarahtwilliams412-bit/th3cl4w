# Calibration Frame Analysis — Batch A (Frames 1–5)

**Date:** 2026-02-08  
**Analyst:** Subagent Batch A  
**Camera angle:** Overhead / bird's-eye (all frames)

---

## Common Scene Description

All five frames share the same environment: a **garage/workshop workspace** shot from a ceiling-mounted overhead camera. The robotic arm is a **multi-DOF (likely 5-6 joint) black arm** bolted to an OSB plywood board sitting on a white desk surface. Two large **black-and-white checkerboard calibration targets** dominate the left side — one propped up at an angle, one lying flat. Surrounding clutter includes: a monitor (left edge), keyboard, racing wheel/sim rig, cardboard boxes, a Dyson-like vacuum, a bucket, a blue shop vac, wire shelving, various tools, and a metal L-square ruler on the work surface.

---

## Frame 1 (`calib_0001.jpg`)

### Arm Visibility
- **Yes, clearly visible.** The arm extends from its dark cylindrical base (center-right on the OSB board) diagonally toward the upper-center of the frame.
- **Segments:** 2–3 link segments visible. The longest link runs from base to elbow area; a shorter segment continues to the wrist/end-effector.
- **Joints:** Base revolute joint (yaw), elbow joint (visible as a bulkier cluster mid-arm), and wrist area with what appears to be a small camera/sensor mount at the tip. ~3–4 joints discernible.

### Contrast & Distinguishability
- The arm is **black against a mixed background** — it contrasts well against the white desk and OSB board, but **blends into the checkerboard pattern** where black squares overlap with the arm's silhouette. The arm also overlaps with the metal L-square ruler, creating edge confusion.
- Where the arm crosses the white portions of the desk/board, edges are crisp and high-contrast.

### Occlusion / Reflections / Shadows / Clutter
- **Shadows:** Visible beneath the arm on the OSB board — soft shadows from overhead lighting. Not severe but could confuse contour detection.
- **Clutter:** The L-square ruler runs nearly parallel to the arm segment and could be misidentified as part of the arm. Cables drape along the arm and onto the board.
- **Reflections:** Minimal — matte surfaces throughout.
- **Checkerboard interference:** The high-frequency B&W pattern is a major issue for any edge-based segmentation; the arm crosses over it.

### Pose
- **Extended, roughly diagonal.** Arm reaches from base toward upper-left area of workspace. Slight elbow bend. End-effector is positioned over or near the flat checkerboard — consistent with a hand-eye calibration capture pose.

### CV Segmentation Assessment
- **Background subtraction:** Could work IF a clean background frame (no arm) is available. The arm moved between frames so differential would isolate it. However, the checkerboard is static high-contrast clutter that could produce noise.
- **HSV/color segmentation:** Difficult — the arm is black, and the checkerboard has black squares. Not enough color differentiation. Would need the arm to be a distinctive color.
- **Contour detection:** Risky due to the checkerboard edges, ruler edges, and cable clutter all producing strong contours.
- **Overall:** Moderately difficult. Background subtraction is the most viable approach.

### Colored Markers Recommendation
- **Yes, highly recommended.** Place bright colored markers (e.g., neon green or orange tape rings) at:
  - Base joint (J1)
  - Elbow joint
  - Wrist joint(s)
  - End-effector tip
- Colors should be chosen to avoid black, white, blue (shop vac), and brown (OSB board). **Neon green or fluorescent orange** would stand out against everything in this scene.

---

## Frame 2 (`calib_0002.jpg`)

### Arm Visibility
- **Yes, clearly visible.** Very similar to Frame 1 — the arm appears to be in a nearly identical or only very slightly shifted pose.
- **Segments/joints:** Same count as Frame 1 — 2–3 links, 3–4 joints visible.

### Contrast & Distinguishability
- Essentially identical to Frame 1. Black arm against mixed background. Good contrast on white/OSB areas, poor where it overlaps the checkerboard.

### Occlusion / Reflections / Shadows / Clutter
- Same issues as Frame 1. The L-square ruler is still present and parallel to the arm. Cables visible. Soft shadows beneath arm.
- Very slight difference in arm position (if any) — the frames may represent micro-adjustments or the arm holding position during a multi-capture sequence.

### Pose
- **Extended, diagonal** — effectively the same as Frame 1. If the arm moved, the displacement is very small (sub-centimeter from this viewing angle).

### CV Segmentation Assessment
- Same as Frame 1. The near-identical framing means all the same challenges apply.
- **Frame differencing between 1 and 2** would yield almost nothing, suggesting the arm barely moved. This is actually useful — it confirms background stability.

### Colored Markers Recommendation
- Same as Frame 1. Identical scene conditions.

---

## Frame 3 (`calib_0003.jpg`)

### Arm Visibility
- **Yes, clearly visible.** The arm is again extended from its base across the workspace.
- **Segments:** 2–3 major link segments. The long lower link from base to elbow is the most prominent feature.
- **Joints:** Base, elbow, and wrist/end-effector cluster — 3–4 joints. A small camera module with a red/colored cable is visible at the distal end, positioned over the checkerboard.

### Contrast & Distinguishability
- Similar to previous frames. The overhead perspective compresses depth, making joint articulation harder to distinguish. The arm's black color blends with checkerboard black squares.
- The red cable at the end-effector is actually a useful visual landmark — it stands out.

### Occlusion / Reflections / Shadows / Clutter
- Same clutter field. The arm is now potentially in a slightly different position over the checkerboard, but the same interference issues apply.
- Cable routing along the arm is visible and could confuse width-based segmentation.

### Pose
- **Extended, near-horizontal, slight elbow bend.** The end-effector/camera appears positioned directly over the flat checkerboard — a calibration capture pose. Very similar to Frames 1–2 with possibly a slight angular adjustment.

### CV Segmentation Assessment
- Same challenges. The red cable could actually serve as a crude marker for the end-effector in an HSV pipeline (isolate red channel).
- Background subtraction remains the best bet.

### Colored Markers Recommendation
- Same recommendation. The existing red cable is a good sign — more deliberate colored markers at each joint would make this much easier.

---

## Frame 4 (`calib_0004.jpg`)

### Arm Visibility
- **Yes, clearly visible.** Same overhead view. The arm extends diagonally from base toward upper-left.
- **Segments:** 2–3 links visible. The longest link (base to elbow) is prominent.
- **Joints:** Base revolute, elbow cluster (servo housings visible), wrist with compact joints. ~3–4 joints. More wrist detail seems visible in this frame — possibly wrist pitch/roll joints and a gripper or tool mount.

### Contrast & Distinguishability
- Consistent with other frames. Black arm on mixed background. The elbow joint area has slightly more visible mechanical detail (actuator housings), suggesting a marginally different arm angle exposing more of the joint structure.

### Occlusion / Reflections / Shadows / Clutter
- Same clutter. Cables, L-square, checkerboard interference all present.
- The arm's cable bundle appears slightly different in routing, suggesting the arm has rotated somewhat.

### Pose
- **Extended, mostly horizontal, slight elbow bend.** Very similar to other frames — this appears to be part of a systematic calibration sweep with small incremental pose changes. The arm is reaching over the checkerboard pattern.

### CV Segmentation Assessment
- No change from previous assessments. The incremental pose differences between frames are subtle enough that frame-to-frame differencing would primarily highlight joint rotations.
- A robust approach would combine background subtraction + color markers.

### Colored Markers Recommendation
- Same as all others. Fluorescent markers at base, elbow, wrist, and end-effector.

---

## Frame 5 (`calib_0005.jpg`)

### Arm Visibility
- **Yes, clearly visible.** Same scene, same overhead angle.
- **Segments:** 2–3 links. The longest segment (base-to-elbow) is clearly the dominant visual feature.
- **Joints:** Base, elbow, wrist cluster. 3–4 joints visible. The end-effector area shows what may be a camera/sensor mount positioned over the checkerboard.

### Contrast & Distinguishability
- Same as all previous frames. Black arm against busy, mixed-contrast background. The checkerboard's high-frequency B&W pattern remains the primary challenge for edge/contour-based methods.
- The arm at ~45° diagonal relative to desk edges provides a consistent geometry across all frames.

### Occlusion / Reflections / Shadows / Clutter
- Same issues throughout. The scene is **very cluttered** around the periphery (monitor, sim wheel, vacuum, boxes, tools, bucket, shelving). However, the immediate workspace around the arm is relatively controlled — the main confounds are the checkerboard, ruler, and cables.
- Slight elbow bend change compared to earlier frames — the arm has a marginally different configuration.

### Pose
- **Extended, slight elbow bend, ~45° diagonal.** The pose variation across all 5 frames is subtle — small joint angle changes consistent with a calibration sweep. This is not a dramatic pose change (no folded/tucked configurations in this batch).

### CV Segmentation Assessment
- Consistent with all frames. **Background subtraction is the most viable single method** given the black-on-mixed-background situation.
- HSV alone will not work — the arm has no distinctive color.
- Contour detection will fire on checkerboard edges, ruler edges, cables, and desk edges — too noisy without pre-filtering.

### Colored Markers Recommendation
- Same. Bright fluorescent markers (green, orange, or pink) at each joint. Would transform this from a hard segmentation problem to a trivial one.

---

## Summary & Cross-Frame Observations

| Aspect | Assessment |
|---|---|
| **Arm visible in all frames?** | Yes — clearly visible in all 5 |
| **Joint count** | 3–4 visible from overhead (base, elbow, wrist cluster); depth compression hides some |
| **Pose variation across batch** | Very small — incremental angle changes, all in extended/diagonal configuration |
| **Primary segmentation challenge** | Black arm on B&W checkerboard = poor color contrast; busy clutter field |
| **Best CV approach** | Background subtraction (with clean reference frame) + colored joint markers |
| **Worst CV approach** | Pure contour/edge detection (checkerboard will dominate) |
| **Colored markers needed?** | **Absolutely yes** — at base, elbow, wrist, end-effector minimum |
| **Recommended marker colors** | Fluorescent green or orange (avoid black/white/blue/brown present in scene) |
| **Scene cleanup helpful?** | Yes — removing the L-square ruler and tidying cables would reduce false positives |
| **Checkerboard conflict** | The calibration target itself is the biggest obstacle to arm segmentation; consider moving checkerboard off-surface during arm tracking, or use it only for camera calibration then remove |
