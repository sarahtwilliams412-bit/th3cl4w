# 3D Reconstruction — Work Breakdown Plan

## Architecture Overview

```
Sub-Agent 1: FK Engine (Python)
Sub-Agent 2: Arm Segmentation Pipeline  
Sub-Agent 3: Visual Joint Detector
Sub-Agent 4: Fusion Engine + WebSocket API
Sub-Agent 5: Factory 3D Integration + Calibration Tooling
```

Lead A oversees Sub-Agents 1-3 (core pipeline).  
Lead B oversees Sub-Agents 4-5 (integration + visualization).  
Both leads co-own testing and calibration procedures.

---

## Sub-Agent 1: FK Engine (Python Port)

**Goal**: Port `fkPositions()` from JS to Python, add camera projection.

### Tasks
1. Port `fkPositions(jointsDeg)` from `index.html` to `src/vision/fk_engine.py`
   - Input: 6 joint angles (degrees)
   - Output: 5 positions [base, shoulder, elbow, wrist, EE] in meters (3D)
   - Link lengths: d0=0.1215, L1=0.2085, L2=0.2085, L3=0.1130
   - Must match JS version exactly (test against known poses from `joint-mapping.md`)

2. Add `project_to_camera(positions_3d, calibration) → pixels_2d` function
   - Uses `CameraCalibration` extrinsics to project 3D points into each camera's pixel space
   - Returns predicted pixel coordinates for each joint in each camera

3. Add `joints_from_api() → list[float]` helper
   - Fetches `GET /api/state` and returns current joint angles

4. Unit tests: verify FK output matches JS for poses in `joint-mapping.md`

**Dependencies**: None (standalone)  
**Estimated effort**: Small  
**Output file**: `src/vision/fk_engine.py`

---

## Sub-Agent 2: Arm Segmentation Pipeline

**Goal**: Detect the arm in each camera frame, produce binary masks and contours.

### Tasks
1. **Background modeler** (`src/vision/arm_segmenter.py`)
   - Running average background model (updateable)
   - `capture_background(n_frames=30)` — grab frames while arm is at known pose
   - `subtract_background(frame) → foreground_mask` — frame differencing with adaptive threshold
   - Morphological cleanup (open/close, same approach as `ObjectDetector`)

2. **Gold segment detector**
   - Extend existing `ObjectDetector` with a gold color range: H:20-40, S:100-255, V:100-255
   - Return centroids of gold segments as potential joint markers

3. **Combined segmentation**
   - `segment_arm(frame, bg_model) → ArmSegmentation` dataclass with:
     - `silhouette_mask`: binary mask of entire arm
     - `gold_centroids`: list of (x,y) gold segment centers
     - `contour`: largest contour of the arm
     - `skeleton`: medial axis / skeleton of the silhouette (optional, stretch goal)
   - Per-camera: cam0 produces side-view segmentation, cam1 produces top-view segmentation

4. **ROI optimization**
   - Use FK-predicted joint positions to define ROIs → only process relevant image regions
   - Reduces per-frame processing time significantly

**Dependencies**: Sub-Agent 1 (for ROI from FK predictions, but can start without)  
**Estimated effort**: Medium  
**Output file**: `src/vision/arm_segmenter.py`

---

## Sub-Agent 3: Visual Joint Detector

**Goal**: From segmented arm images, identify individual joint positions in pixel space.

### Tasks
1. **Joint position estimator** (`src/vision/joint_detector.py`)
   - Input: `ArmSegmentation` from Sub-Agent 2 + FK-predicted pixel positions from Sub-Agent 1
   - For each predicted joint location:
     - Search nearby region for gold segment centroid (if joint has gold accent)
     - Search for contour inflection points (joint = direction change in arm silhouette)
     - Search for local width minima in silhouette (joints are narrower than links)
   - Output: detected pixel positions with confidence scores

2. **Per-camera joint mapping**
   - cam0 (front): Can see J1 (shoulder), J2 (elbow), J4 (wrist pitch) angles from link directions
   - cam1 (overhead): Can see J0 (base yaw) from arm direction, reach distance
   - Map detected features to specific joints based on camera view geometry

3. **Temporal smoothing**
   - Kalman filter or exponential moving average on detected joint positions
   - Reject outliers (sudden jumps > threshold)
   - Maintain detection confidence per joint

4. **Validation against empirical data**
   - Use `joint-mapping.md` pixel coordinates as ground truth
   - Move arm to known poses, compare detected positions to documented positions

**Dependencies**: Sub-Agent 1 (FK predictions), Sub-Agent 2 (segmentation)  
**Estimated effort**: Medium-Large  
**Output file**: `src/vision/joint_detector.py`

---

## Sub-Agent 4: Fusion Engine + WebSocket API

**Goal**: Combine FK predictions with visual detections, serve corrected 3D model via WebSocket.

### Tasks
1. **Fusion algorithm** (`src/vision/pose_fusion.py`)
   - Input: FK 3D positions + visual detections (pixel positions + confidence) from both cameras
   - Back-project visual detections to 3D using camera calibration:
     - cam0 pixel → ray in 3D → intersect with FK-predicted depth plane
     - cam1 pixel → project onto workspace XY plane at FK-predicted height
   - Weighted fusion: `P_final = α * P_fk + (1-α) * P_visual` where α depends on:
     - Visual detection confidence
     - FK model trust (decreases if repeated visual disagreement)
   - Output: corrected 3D positions for all 5 key points

2. **Disagreement detection**
   - If visual position deviates from FK by > threshold (e.g., 20mm):
     - Log warning
     - Could indicate collision, gear slip, or calibration drift
   - Provide `get_tracking_quality() → dict` with per-joint agreement metrics

3. **WebSocket endpoint** — `/ws/arm3d`
   - Push corrected 3D arm positions at 10Hz
   - Message format: `{"type": "arm3d", "positions": [[x,y,z], ...], "confidence": [...], "source": "fused"|"fk_only"|"vision_only"}`
   - Graceful degradation: if cameras offline → FK-only mode

4. **REST endpoint** — `GET /api/arm3d/status`
   - Pipeline status, per-camera health, fusion quality metrics

**Dependencies**: Sub-Agents 1, 2, 3  
**Estimated effort**: Medium  
**Output files**: `src/vision/pose_fusion.py`, additions to `web/server.py`

---

## Sub-Agent 5: Factory 3D Integration + Calibration Tooling

**Goal**: Make the Factory 3D voxel arm track the real arm; build calibration tools.

### Tasks
1. **Factory 3D arm update**
   - Modify `index.html` Factory 3D tab to consume `/ws/arm3d` WebSocket
   - Replace static `getAnimatedHeight()` with real-time arm voxel placement
   - The factory already uses `d1Joints` for animation — extend to use corrected 3D positions
   - Dynamically generate arm voxels along the FK chain (base→shoulder→elbow→wrist→EE)

2. **Calibration UI** — `/calibrate` page or modal in existing UI
   - Step-by-step calibration wizard:
     1. "Place checkerboard at arm base" → capture images
     2. "Move arm to home position" → capture background
     3. "Arm will move through calibration poses" → collect data
   - Show calibration quality metrics (reprojection error, etc.)
   - Save/load calibration data

3. **Calibration procedure automation**
   - `src/vision/auto_calibrator.py`
   - Move arm through N known poses (reuse existing task system)
   - At each pose: capture frames, detect arm, compare to FK
   - Compute camera extrinsics and refine FK parameters
   - Uses existing `IndependentCalibrator` infrastructure

4. **Debug overlay**
   - Option to overlay FK predictions and visual detections on camera feeds
   - Show in ASCII Video tab: predicted joint positions as colored markers
   - Helps diagnose calibration and detection issues

**Dependencies**: Sub-Agent 4 (WebSocket API), existing calibration.py  
**Estimated effort**: Medium  
**Output files**: modifications to `index.html`, `src/vision/auto_calibrator.py`

---

## Testing Plan (Lead A + Lead B co-owned)

### Phase 1: Unit tests
- FK Python matches FK JS for all poses in joint-mapping.md
- Segmenter produces valid masks on test images
- Gold detector finds centroids on sample frames

### Phase 2: Integration tests
- Full pipeline with recorded camera frames + known joint angles
- Compare output 3D positions against hand-measured ground truth
- Measure latency end-to-end

### Phase 3: Live testing
- Run with real cameras and arm
- Move arm through standard poses, verify Factory 3D tracks correctly
- Test degradation: cover one camera, disconnect camera, etc.

### Calibration validation
- Calibrate, then move arm to 10 test poses not used in calibration
- Measure reprojection error (FK-projected pixels vs actual arm position in image)
- Target: <10px reprojection error at 1920×1080

---

## Execution Order

```
Phase 1 (parallel):
  Sub-Agent 1: FK Engine          ← START IMMEDIATELY
  Sub-Agent 2: Arm Segmentation   ← START IMMEDIATELY

Phase 2 (after Phase 1):
  Sub-Agent 3: Joint Detector     ← needs 1 + 2

Phase 3 (after Phase 2):
  Sub-Agent 4: Fusion + API       ← needs 1 + 2 + 3
  Sub-Agent 5: Integration        ← needs 4 (can start calibration tooling in parallel)
```

Estimated total: Sub-Agents 1+2 can start in parallel. Full pipeline operational after all 5 complete.
