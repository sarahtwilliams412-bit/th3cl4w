# Batch D — Calibration Frame Analysis (Frames 16–20)

**Analyzed:** 2026-02-08 | **Camera:** Overhead (bird's-eye view)

---

## General Notes

All five frames in this batch are captured from the same overhead camera looking down at the robot arm workspace. The scene is remarkably consistent across frames 16–20 — the arm appears to be in the same or very nearly the same pose in all of them, suggesting these were captured in rapid succession or the arm was stationary. The environment is a **cluttered workshop/garage**.

---

## Frame 16 (`calib_0016.jpg`)

### Arm Visibility
- **Yes, clearly visible.** The arm extends from a dark circular base (center-right of frame) diagonally up-left across the workspace.
- **Segments:** 2–3 dark/black link segments visible, slender profiles (aluminum or carbon fiber).
- **Joints:** At least 3 joints identifiable — base rotary joint, shoulder/elbow cluster mid-arm, and a wrist/end-effector assembly at the far end (upper-center). The end-effector appears to have a small camera or sensor mounted on it.

### Background Distinguishability
- **Moderate difficulty.** The arm is predominantly **black** against a mixed background: plywood (tan/brown OSB), white table surface, and dark workshop clutter. The arm blends somewhat with shadows and dark objects in the background.
- The arm crosses over the checkerboard patterns, creating **high contrast against the white squares** but **low contrast against the black squares**.

### Occlusion / Clutter Issues
- **Severe background clutter:** CRT monitors, a racing steering wheel, cardboard boxes, a Dyson vacuum, blue hand cart, paint bucket, tripod, exercise equipment, tools — all visible in the periphery.
- **Checkerboard overlap:** Two large checkerboard calibration boards are present — one propped at an angle (upper-left) and one flat on the table. The arm passes over/near both.
- A **yellow measuring tape** runs along the arm's reach axis on the plywood base.
- **Shadows** from overhead lighting are visible but not severe.

### Arm Pose
- **Extended/outstretched** — the arm reaches diagonally from base toward the calibration boards. Roughly horizontal configuration, not folded. Consistent with a hand-eye calibration pose.

### CV Segmentation Feasibility
- **Background subtraction:** Would work if a clean background reference frame (no arm) is available. The arm's dark color against lighter surfaces gives decent signal, but the checkerboard patterns would confuse simple differencing.
- **HSV/color-based:** Difficult — the arm is generic black, same as many background objects (monitors, boxes, shadows).
- **Contour detection:** The arm's linear geometry (long, thin) could be exploited, but clutter edges would produce many false contours.
- **Overall: Challenging without markers.** The black-on-mixed-background scenario is not ideal.

### Colored Joint Markers
- **Would significantly help.** Recommended placement:
  - **Base rotary joint** (circular housing, center-right)
  - **Shoulder/elbow joint** (mid-arm cluster)
  - **Wrist joint** (near end-effector, upper-center)
  - **End-effector tip** (camera/tool mount)
- Use **bright, distinct colors** (neon green, orange, magenta) that don't appear elsewhere in the scene. Avoid red (present on steering wheel/cables) and blue (tape, cart already in scene).

---

## Frame 17 (`calib_0017.jpg`)

### Arm Visibility
- **Virtually identical to Frame 16.** The arm is in the same pose, same position. All the same segments and joints are visible.

### Background Distinguishability
- Same as Frame 16. No detectable change in lighting or arm position.

### Occlusion / Clutter Issues
- Identical clutter. No new occlusions or changes.

### Arm Pose
- **Same extended pose** as Frame 16. The arm has not moved between these captures.

### CV Segmentation Feasibility
- Same assessment as Frame 16. No improvement or degradation.

### Colored Joint Markers
- Same recommendations as Frame 16.

### Note
Frames 16 and 17 appear to be **duplicate or near-duplicate captures** — likely sequential frames with no arm movement. For calibration diversity, this provides no additional information.

---

## Frame 18 (`calib_0018.jpg`)

### Arm Visibility
- **Yes, visible from overhead.** Same bird's-eye perspective. The arm extends from the dark circular base across the workspace toward the upper-left.
- **Segments:** 2–3 link segments, dark/black, slender profiles.
- **Joints:** Base rotary, shoulder/elbow cluster, wrist assembly all visible. A small camera/sensor at the end-effector tip.

### Background Distinguishability
- Same mixed background. The arm's dark color **blends with shadows** at certain points. Against the plywood and white table surface, edges are discernible but not high-contrast.
- The arm crossing over checkerboard squares creates alternating contrast — visible over white squares, harder to see over black squares.

### Occlusion / Clutter Issues
- Same severe workshop clutter as frames 16–17.
- The large angled checkerboard shows **slight warping/curling** at edges — this could affect calibration corner detection but doesn't affect arm segmentation.
- Measuring tape and blue/green tape pieces on the workspace add small color noise.

### Arm Pose
- **Extended**, same general configuration. May have very slight positional change from 16–17 but hard to confirm — appears nearly identical.

### CV Segmentation Feasibility
- Same challenges. Background subtraction remains the most viable approach if a reference frame exists.
- The **warped checkerboard** adds edge noise that could confuse contour-based methods.

### Colored Joint Markers
- Same recommendations. Neon green/orange/magenta at base, shoulder, elbow, wrist, and end-effector.

---

## Frame 19 (`calib_0019.jpg`)

### Arm Visibility
- **Yes, visible.** Overhead view, same setup. The arm's dark links and joints are identifiable extending from the base.
- **Segments:** 2–3 segments. Slender, dark, somewhat hard to distinguish from shadows at this resolution.
- **Joints:** Same joint locations visible — base, mid-arm, wrist. A small green LED or indicator is visible on the base housing.

### Background Distinguishability
- Arm is dark against mixed surfaces. **Shadow-arm confusion** is the main issue — overhead lighting creates shadows from the arm and from the angled checkerboard that have similar darkness to the arm itself.
- Where the arm crosses over the flat white table or lighter plywood areas, edges are clearer.

### Occlusion / Clutter Issues
- Identical clutter environment. The yellow measuring tape partially parallels the arm, which could confuse edge detection.
- Objects near the arm (tape, small tools on the plywood base) are close enough to potentially merge with arm contours.

### Arm Pose
- **Extended/outstretched.** Same or very similar to all previous frames in this batch.

### CV Segmentation Feasibility
- Same challenges as prior frames. The measuring tape running parallel to the arm is a specific concern — it could merge with arm edges in contour detection.
- **Recommendation:** If possible, clear the workspace of loose objects (tape, tools) before capture for cleaner segmentation.

### Colored Joint Markers
- Same recommendations. Additionally, a **strip of colored tape along the arm links** (not just at joints) could help distinguish the arm body from parallel objects like the measuring tape.

---

## Frame 20 (`calib_0020.jpg`)

### Arm Visibility
- **Yes, visible.** Same overhead perspective. Full kinematic chain from base to end-effector is in view.
- **Segments:** 2–3 link segments, dark/black.
- **Joints:** Base, shoulder/elbow, wrist all identifiable. End-effector at far end with possible camera mount.

### Background Distinguishability
- Consistent with all prior frames. Dark arm against mixed background. The **dual checkerboard setup** (angled + flat) dominates the visual field and creates busy, high-contrast patterns that could interfere with arm detection.

### Occlusion / Clutter Issues
- Same clutter. The angled checkerboard has slight **edge curl/warp** visible.
- Blue painter's tape spots on the flat checkerboard are visible.
- No new occlusions compared to prior frames.

### Arm Pose
- **Extended.** Same pose as all frames in this batch. The arm has not moved appreciably across frames 16–20.

### CV Segmentation Feasibility
- Same overall assessment: **challenging without additional aids.**
- Background subtraction is the best bet but requires a clean reference.
- HSV alone won't work — too many dark objects.
- Deep learning (e.g., fine-tuned segmentation model) would handle this better than classical CV.

### Colored Joint Markers
- Same recommendations as prior frames.

---

## Batch D Summary

| Aspect | Assessment |
|--------|-----------|
| **Arm visible in all frames** | Yes — 2–3 segments, 3+ joints, extended pose |
| **Pose diversity** | **None** — all 5 frames show the same pose. Frames 16–17 appear to be near-duplicates. |
| **Background clutter** | Severe — busy workshop with many dark objects |
| **Arm-background contrast** | Low-to-moderate. Black arm against mixed surfaces. Shadow confusion likely. |
| **CV segmentation (classical)** | Difficult without markers. Background subtraction is most viable. |
| **Colored markers needed?** | **Yes, strongly recommended.** Neon green/orange/magenta at each joint + along links. |
| **Key recommendation** | Clear workspace of loose objects (measuring tape, tools), add bright joint markers, and capture with more pose diversity. |
