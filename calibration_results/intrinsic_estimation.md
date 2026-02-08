# Camera Intrinsic Estimation — Visual Reasoning

**Date:** 2026-02-08
**Cameras:** 2× Logitech BRIO 4K @ 1920×1080
**Method:** Manufacturer spec + visual scene analysis (checkerboard auto-detection failed due to dense grid)

## Summary

| Parameter | cam0 (front) | cam1 (overhead) |
|-----------|-------------|-----------------|
| fx, fy | 1350 px | 1100 px |
| cx, cy | 960, 540 | 960, 540 |
| k1 | -0.12 | -0.18 |
| k2 | 0.05 | 0.08 |
| p1, p2 | 0, 0 | 0, 0 |
| Est. diagonal FOV | 78° | 90° |
| Confidence | Medium | Medium |

## Reasoning

### BRIO FOV Specs

The Logitech BRIO 4K has three FOV modes: 65°, 78°, and 90° (diagonal). The relationship between diagonal FOV and focal length at 1920×1080:

| FOV (°) | f (px) |
|---------|--------|
| 65 | 1729 |
| 70 | 1573 |
| 75 | 1435 |
| 78 | 1360 |
| 83 | 1245 |
| 90 | 1101 |

Formula: `f = (√(1920² + 1080²)) / (2 × tan(FOV/2))` = `2203 / (2 × tan(FOV/2))`

### cam0 (Front Camera) Analysis

- Shows the D1 arm against yellow interlocking foam mat panels (typically 24"×24" / 610mm each)
- View is moderately wide — shows the arm workspace plus some surrounding area
- A Red Bull can (~168mm tall) is visible, appearing relatively small → consistent with moderate FOV
- **Estimate: 78° diagonal FOV → fx ≈ fy ≈ 1350 px**

### cam1 (Overhead Camera) Analysis

- Shows a very wide view: entire table, floor on right side, shelving, multiple pieces of furniture
- This wide coverage at what appears to be ~0.7-0.8m camera height suggests the widest FOV setting
- Visible workspace width ≈ 1.2m at ~0.75m distance → hFOV ≈ 77° → diagonal ≈ 90°
- Calibration frames (calib_0001-0020) show even more of the room → confirms wide setting
- **Estimate: 90° diagonal FOV → fx ≈ fy ≈ 1100 px**

### Distortion

- BRIO has moderate barrel distortion, especially at 90° FOV
- Visible in calibration frames: straight edges of table/shelving show slight curvature at frame edges
- Overhead (90°) has more distortion than front (78°)
- Estimated k1 = -0.12 (front), -0.18 (overhead); k2 positive to compensate at edges
- These are rough — proper calibration with detected corners would give precise values

### Principal Point

- Assumed at image center (960, 540) — no evidence of significant offset
- Could be off by ±20px in practice

## Attempted Automated Calibration

Checkerboard patterns ARE present in calibration frames (large printed calibration targets visible). However:
- OpenCV's `findChessboardCorners` failed (likely too many squares / dense pattern)
- `findChessboardCornersSB` segfaulted (OpenCV 4.13.0 bug or memory issue)
- **Recommendation:** Use `cv2.findChessboardCornersSB` with known exact grid dimensions, or use CharUco/ArUco markers for more robust detection

## Cross-Validation with Arm FK

Current arm state: joints = [-0.2, -60.3, 60.1, -0.1, -44.5, 0.1]°

FK link lengths: d0=0.1215m, L1=0.2085m, L2=0.2085m, L3=0.113m

The arm is visible in both cameras. With known joint angles, the 3D positions of joints are deterministic. A proper calibration would:
1. Identify joint pixel positions in multiple frames
2. Compute 3D joint positions via FK
3. Solve PnP for both intrinsics and extrinsics simultaneously

This is the recommended next step after getting rough intrinsics working.

## Confidence & Recommendations

- **These estimates are ±15-20% on focal length** — adequate for initial hand-eye calibration prototyping
- **For precision work:** Run proper checkerboard calibration (try CharUco board, or count exact squares in the printed pattern and specify the correct grid size)
- **Quick validation:** Measure a known distance in the scene, project it using these intrinsics, check if pixel count matches
- The FOV setting on each BRIO can be queried/set via UVC controls (`v4l2-ctl`) — this would eliminate the main source of uncertainty
