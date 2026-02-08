# 3D Reconstruction Plan — Lead Agent A

## 1. Codebase Assessment

### What exists and is reusable

| Module | Status | Reuse |
|--------|--------|-------|
| `calibration.py` — `IndependentCalibrator`, `CameraCalibration` | **Excellent** — already designed for independent (non-stereo) cameras, has intrinsics, extrinsics, `pixel_to_workspace()`, `pixel_to_ray()`, save/load | **Direct reuse** |
| `ascii_converter.py` — `AsciiConverter` | Good — converts frames to ASCII with per-char RGB color data | Reuse `frame_to_color_data()` for visualization; raw frames needed for detection |
| `object_detection.py` — `ObjectDetector` | Good — HSV segmentation, contour finding, morphological cleanup | Reuse for gold segment detection (H:20-40). Need new detector for arm body (frame differencing) |
| `arm_tracker.py` — `DualCameraArmTracker` | **Excellent** — already cross-references cam0 (front→height) and cam1 (overhead→XY) detections by label. Has `TrackedObject` with 3D position | **Direct reuse** as foundation; extend for multi-joint tracking |
| `viz_calibrator.py` | Exists — calibrates camera projection for visualization | Reuse calibration data |
| `web/server.py` — WebSocket ASCII endpoint, `/api/state` | Good — already streams ASCII frames and joint state at 10Hz | Reuse endpoints |
| `web/static/index.html` — `fkPositions()` JS function | **Key asset** — empirically verified geometric FK returning 5 joint positions in 3D (meters) | **Port to Python** for server-side FK |
| `camera_server.py` — MJPEG + snapshot endpoints | Good — 1920×1080 at 15fps, JPEG snapshots at `/snap/0` and `/snap/1` | Direct reuse |
| `docs/joint-mapping.md` — empirical pixel coordinates per joint pose | **Valuable** — ground truth for validation | Reference data |

### What needs building

1. **Arm body segmentation** — frame differencing + gold HSV hybrid detector
2. **Multi-joint visual tracker** — detect individual joint positions (not just objects), tracking arm links across frames  
3. **FK-to-pixel projector** — Python FK + camera projection to predict where joints should appear in each camera
4. **Visual-FK fusion** — combine joint-angle FK predictions with visual detections to correct the 3D model
5. **Real-time pipeline** — async frame grab → segment → detect joints → fuse with FK → update 3D model → push to Factory 3D tab
6. **Factory 3D integration** — update the voxel world's robotic arm to match real arm pose

## 2. Proposed Approach: FK-Guided Visual Pose Refinement

### Why NOT pure multi-view reconstruction

Pure vision-based 3D reconstruction from two non-overlapping cameras is extremely hard:
- No stereo overlap means no triangulation
- Matte-black arm has minimal visual features
- Only gold accents are reliably detectable via HSV

### Why FK + Vision fusion is the right approach

We already have **joint angles from `/api/state`** at 10Hz. The FK function gives us predicted 3D positions. The cameras then **validate and refine** those predictions:

1. **FK predicts** where each joint/link should appear in each camera view
2. **Vision detects** where the arm actually appears (silhouette, gold segments)
3. **Fusion corrects** any FK errors (gear backlash, calibration drift, unexpected collisions)

This is essentially **model-based tracking** — the FK model is the prior, vision provides the measurement update.

### Algorithm: Per-frame pipeline

```
1. GET /api/state → joint angles [j0..j5] (radians)
2. FK(joints) → predicted 3D positions [base, shoulder, elbow, wrist, EE]
3. Project predicted 3D → predicted 2D pixels in cam0 and cam1
4. Grab camera frames (cam0 front, cam1 overhead)
5. Segment arm from background:
   a. Frame differencing (background model vs current) → arm silhouette mask
   b. HSV filter H:20-40 → gold segment mask
   c. Combine masks
6. In each camera view:
   a. Find arm contour from silhouette mask
   b. Find gold segment centroids
   c. Match detected features to predicted joint positions (nearest-neighbor with constraints)
7. Compute correction:
   a. If visual joint position deviates from FK prediction by > threshold → flag
   b. Use weighted average: 3D_final = α * FK_prediction + (1-α) * visual_estimate
   c. α starts high (trust FK) and decreases as visual tracking confidence grows
8. Push corrected 3D model to visualization
```

## 3. Arm Segmentation Strategy

### Background: matte-black arm on varied background

**Primary: Frame differencing**
- Maintain a background model (running average when arm is at home/known pose)
- Difference current frame against background → binary mask
- Works for the entire arm body regardless of color
- Weakness: requires initial background capture; fails if background changes

**Secondary: HSV gold detection**
- Gold segments at H:20-40, S:100-255, V:100-255 (from existing `COLOR_PRESETS`)
- Reliable markers for specific joint locations
- The gold accents are on specific links → can be mapped to specific joints

**Tertiary: Edge/contour analysis**
- Canny edges on the arm region help define link boundaries
- Skeleton extraction from silhouette → link directions

### Per-camera segmentation

| Camera | View | What's visible | Segmentation approach |
|--------|------|---------------|----------------------|
| cam0 (front) | Side view | Full arm profile, height info, link angles | Frame diff → silhouette → skeleton → joint angles |
| cam1 (overhead) | Top-down | Base rotation (J0), forward reach, X/Y position | Frame diff → centroid chain → base angle + reach |

## 4. Correlating Arm Pose Across Views

Since cameras have **no overlapping FOV**, we can't triangulate. Instead:

### Strategy: FK as the bridge

1. FK gives us the 3D model. Each camera independently validates part of it:
   - **cam0 (front)**: Validates pitch angles (J1, J2, J4) and height (Z)
   - **cam1 (overhead)**: Validates yaw (J0) and XY reach

2. Each camera produces a **partial pose correction**:
   - cam0 → corrections to [J1, J2, J4] (pitch joints visible in side view)
   - cam1 → corrections to [J0] (base rotation visible from above), and forward reach (validates J1+J2 extension)

3. Corrections are merged since they affect different DOFs (minimal conflict).

### Consistency check
- FK predicts end-effector position P_ee
- cam0 gives P_ee.z (height) and P_ee.x_proj (horizontal position in side view)
- cam1 gives P_ee.x, P_ee.y (overhead position)
- If these don't agree → one camera or the FK is wrong → flag for correction

## 5. Calibration Requirements

### Already available
- `IndependentCalibrator` handles checkerboard-based intrinsic calibration
- `CameraCalibration.cam_to_workspace` stores camera-to-workspace extrinsic
- `pixel_to_workspace()` projects pixels onto workspace planes
- `viz_calibrator.py` has camera projection parameters

### Needed calibration data
1. **Camera intrinsics** (both cameras) — `IndependentCalibrator.calibrate_camera()` with ≥3 checkerboard images each
2. **Camera extrinsics** — where each camera is relative to the arm base:
   - cam0: position + orientation relative to arm base (front/side view)
   - cam1: position + orientation relative to arm base (overhead view)
   - Can use `compute_extrinsic()` with a checkerboard at a known position relative to arm base
3. **Arm base position in workspace** — the origin for FK coordinates
4. **FK link lengths verification** — current values in JS: d0=0.1215m, L1=0.2085m, L2=0.2085m, L3=0.1130m

### Calibration procedure (one-time)
1. Place checkerboard at arm base (known position)
2. Capture 5+ images from each camera with checkerboard at different angles
3. Run `IndependentCalibrator` for each camera
4. Compute extrinsics with `compute_extrinsic()`
5. Move arm through known poses, compare FK predictions to visual detections → refine

## 6. Data Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Camera Server│     │  Arm Server  │     │  Calibration │
│  :8081       │     │  :8080       │     │  Data (JSON) │
│ /snap/0, /1  │     │ /api/state   │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    │
┌──────────────────────────────────────────────────────────┐
│                   3D Reconstruction Engine                │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Frame   │  │ Arm      │  │ FK       │  │ Fusion   │ │
│  │ Grabber │→ │ Segmenter│→ │ Projector│→ │ Engine   │ │
│  │ (async) │  │ (CV2)    │  │ (Python) │  │          │ │
│  └─────────┘  └──────────┘  └──────────┘  └────┬─────┘ │
│                                                  │       │
│  Input: 2x camera frames + joint angles          │       │
│  Output: corrected 3D joint positions            │       │
└──────────────────────────────────────────────────┼───────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────┐
                                    │  WebSocket Push      │
                                    │  → Factory 3D Tab    │
                                    │  → ASCII overlay     │
                                    │  (voxel arm updates) │
                                    └──────────────────────┘
```

### Target performance
- **Input rate**: 10-15 fps from cameras, 10Hz joint state
- **Processing target**: <100ms per frame pair (10Hz output)
- **Output**: Corrected 3D positions for 5 key points (base, shoulder, elbow, wrist, EE)

## 7. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Matte-black arm invisible to cameras | Can't segment arm | Frame differencing + IR illumination if needed; gold segments as fallback landmarks |
| Camera calibration drift | Wrong projections | Periodic auto-recalibration using known poses |
| FK model inaccuracy | Bad predictions | Use vision to continuously refine FK parameters |
| Processing too slow for real-time | Laggy visualization | Downsample frames, ROI-based processing, async pipeline |
| Cameras may be offline | No visual data | Graceful fallback to FK-only mode (current behavior) |

## 8. Summary

The approach is **FK-guided visual pose refinement**:
- FK from joint angles provides 80% of the answer
- Camera vision provides validation and correction
- Each camera validates different DOFs (front→pitch/height, overhead→yaw/XY)
- Gold accents serve as reliable visual landmarks for joint identification
- Frame differencing handles the matte-black arm body
- Existing `calibration.py`, `arm_tracker.py`, and `object_detection.py` provide strong foundations
- The `fkPositions()` function needs porting from JS to Python
