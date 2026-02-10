# 3D Workspace Reconstruction Plan — Unitree D1 Arm-Mounted Camera

> **Date:** 2026-02-09
> **Status:** Draft
> **Key insight:** Known FK + joint feedback = known camera poses at every frame. This is known-pose SfM — no SLAM or pose estimation needed.

---

## 1. Camera-to-End-Effector Calibration

### The Problem

The MX Brio is bolted behind the gripper, but we don't know the exact transform from the last joint frame (joint 7 / flange) to the camera's optical center. We need to find this fixed 4×4 matrix **T_cam_ee** (the "hand-eye" transform).

### Approach: Checkerboard Hand-Eye Calibration

1. **Print a large checkerboard** (8×6, 30mm squares) and fix it rigidly to the desk
2. **Move the arm to 20-30 diverse poses** where the checkerboard is fully visible
   - Vary position AND orientation — tilts, rotations, distances
   - Avoid near-singular configurations (camera nearly parallel to board)
3. **At each pose, record:**
   - Joint angles → compute **T_ee_base** via FK
   - Camera image → compute **T_cam_board** via `cv2.solvePnP`
4. **Solve AX = XB** (classic hand-eye calibration)
   - Use `cv2.calibrateHandEye()` with method `CALIB_HAND_EYE_TSAI` or `PARK`
   - A = relative EE motion, B = relative camera motion, X = **T_cam_ee**

### Alternative: ArUco Board

If checkerboard detection is flaky at steep angles, use a ChArUco board — combines checkerboard accuracy with ArUco robustness to partial occlusion.

### Refinement

- **Reprojection validation:** After calibration, move arm to new poses, project board corners using the full chain (FK + T_cam_ee + intrinsics). Reprojection error should be **< 2px**.
- **FK tuning:** If reprojection error is high, the DH parameters may need adjustment. Can add FK offsets as optimization variables in a joint hand-eye + FK refinement.
- **Expected accuracy:** With good FK and 25+ poses: **1-3mm translational, 0.5-1° rotational** error in the camera pose.

### Implementation

```python
# src/vision/hand_eye_calibration.py
# Inputs: list of (joint_angles, camera_image) pairs
# Outputs: T_cam_ee (4x4 numpy array), saved to config
```

---

## 2. Scanning Strategy

### Workspace Geometry

The D1 has 550mm reach from its base (mounted on desk). The useful scanning volume is roughly a hemisphere in front of/around the base, minus the table surface below.

### Scan Trajectories

#### Trajectory 1: Hemisphere Sweep (Primary)
Move the camera over a set of viewpoints on concentric hemispheres around the workspace center:

```
Elevation angles: [-10°, 15°, 40°, 65°] (from horizontal)
Azimuth angles: [-150° to +150°] in 15° steps
Radii: [300mm, 450mm] (two shells)
Camera always pointing inward toward workspace center
```

This gives ~160 viewpoints with good coverage and baseline diversity.

#### Trajectory 2: Tabletop Detail Pass
For the desk surface (where objects sit):
- Lower the camera to ~200mm above table
- Sweep in a grid pattern (raster scan)
- Camera pointing straight down or 30° tilt
- ~5cm spacing between adjacent viewpoints

#### Trajectory 3: Perimeter Look-Outward
To capture surroundings beyond the workspace:
- Move arm to extended positions
- Point camera outward (away from base)
- Sweep azimuth at 2-3 elevation angles

### Self-Occlusion Avoidance

The arm itself will occlude parts of the scene. Strategies:
1. **Redundancy:** 7-DOF means we can reach the same camera pose with different arm configurations — pick the one that minimizes occlusion
2. **Arm mask:** We know the arm geometry. For each pose, render the arm in the image and mask those pixels before reconstruction
3. **Opposing viewpoints:** Any region occluded from one direction will be visible from the opposite side

### Motion Planning

- Use joint-space interpolation between scan poses (smooth, predictable)
- Pause briefly at each viewpoint (100-200ms) to avoid motion blur at 15fps
- Verify each target pose is reachable and collision-free before executing
- Stay within conservative joint limits (80% of hardware limits)

### Coverage Estimation

Before scanning, simulate the trajectory and compute a coverage map:
- Voxelize the workspace
- For each viewpoint, ray-cast the camera frustum
- Mark voxels as "observed" if seen from ≥3 viewpoints with sufficient baseline
- Identify gaps and add supplementary viewpoints

---

## 3. 3D Reconstruction Pipeline

### Overview

```
Joint angles + FK + T_cam_ee → Camera pose (known)
Camera image + intrinsics → Undistorted frame
Known poses + frames → Multi-view stereo → Depth maps
Depth maps + poses → TSDF fusion → Mesh
```

### 3.1 Frame Selection

Not every frame is useful. Select frames that are:
- **Sharp:** Laplacian variance > threshold (reject motion blur)
- **Non-redundant:** Minimum 2cm translation OR 3° rotation from last selected frame
- **Well-exposed:** Reject over/underexposed frames (histogram check)

Expected: ~100-200 selected frames from a full scan.

### 3.2 Depth Estimation

Three approaches, in order of preference:

#### Option A: Multi-View Stereo (MVS) — Recommended

Since we have known, accurate poses, classical MVS works extremely well:

1. **PatchMatch stereo** (OpenCV `cv::StereoSGBM` or COLMAP's MVS)
   - For each reference image, pick 4-8 neighboring views with good baseline (5-15cm, 10-30° angle)
   - Compute per-pixel depth via plane-sweep or PatchMatch
   - Geometric consistency filtering across views

2. **Using COLMAP with known poses:**
   - Skip feature matching and pose estimation
   - Import known camera poses directly
   - Run only the dense reconstruction step (`colmap patch_match_stereo` + `colmap stereo_fusion`)
   - This is the easiest path to high-quality results

#### Option B: Monocular Depth (Depth Anything v2)

- Run per-frame, no multi-view needed
- Produces relative/metric depth maps
- **Pros:** Fast, works with single frames, good for textureless regions
- **Cons:** Scale ambiguity (though we know metric poses, so we can align), less precise than MVS
- **Use case:** Quick preview, fill gaps where MVS fails (textureless surfaces)

#### Option C: Hybrid

- MVS for well-textured regions
- Monocular depth for textureless areas (white walls, flat surfaces)
- Confidence-weighted blending

### 3.3 TSDF Fusion

The core integration step. Takes depth maps + known poses → volumetric representation.

```python
# Using Open3D's TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.002,  # 2mm resolution
    sdf_trunc=0.01,      # 10mm truncation
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

for frame, pose in zip(selected_frames, camera_poses):
    depth = estimate_depth(frame)  # from MVS or mono
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=1.5, convert_rgb_to_intensity=False
    )
    volume.integrate(rgbd, intrinsics, np.linalg.inv(pose))

mesh = volume.extract_triangle_mesh()
```

**Parameters:**
- Voxel size: 2mm (good detail, manageable memory for desk-scale)
- Truncation distance: 5× voxel size
- Depth range: 0.1m - 1.5m (arm workspace)

### 3.4 Point Cloud Generation

If mesh isn't needed, extract point cloud directly:
```python
pcd = volume.extract_point_cloud()
```

Or merge per-frame point clouds:
1. Back-project each depth map to 3D using known pose + intrinsics
2. Concatenate all point clouds
3. Voxel downsample (2mm grid) to remove redundancy
4. Statistical outlier removal

### 3.5 Mesh Reconstruction

From TSDF: marching cubes (built into Open3D's `extract_triangle_mesh`)

From point cloud (if not using TSDF):
1. Estimate normals (`pcd.estimate_normals()`)
2. Poisson reconstruction (`o3d.geometry.TriangleMesh.create_from_point_cloud_poisson`)
3. Trim low-density regions

### 3.6 Texture Mapping

- TSDF fusion naturally gives per-vertex color
- For higher-quality textures: UV-unwrap the mesh, project best-view images onto faces
- Select the image where each face is most frontal and closest

---

## 4. Integration with Existing System

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Camera       │     │ Arm Server   │     │ Reconstruction   │
│ Server :8081 │────▶│ :8080        │────▶│ Service :8082    │
│ (frames)     │     │ (joint state)│     │ (3D pipeline)    │
└─────────────┘     └──────────────┘     └──────────────────┘
```

### New API Endpoints (on reconstruction service or added to :8080)

```
POST /scan/start              — Begin a scan trajectory
GET  /scan/status             — Progress (frames captured, coverage %)
POST /scan/stop               — Abort scan
POST /reconstruct             — Run reconstruction on captured data
GET  /reconstruct/status      — Pipeline progress
GET  /model/latest            — Download latest mesh/pointcloud
GET  /model/query             — Query the model (e.g., height at point, surface normal)
POST /model/crop              — Extract sub-region
```

### Feeding Back into Pick-and-Place

The 3D model enables:
1. **Collision avoidance:** Check planned trajectories against the workspace model
2. **Object localization:** Segment new objects by differencing against the background model
3. **Grasp planning:** Surface normals and geometry inform grasp approach vectors
4. **Place planning:** Find flat, unoccupied surfaces for placing objects

### Real-Time vs Offline

| Mode | Use Case | Approach |
|------|----------|----------|
| **Offline full scan** | Initial workspace model, periodic updates | Full trajectory, COLMAP MVS, high quality |
| **Incremental update** | After objects move | Capture 10-20 new views of changed region, update TSDF |
| **Live depth** | During manipulation | Monocular depth on current frame, not full 3D |

### Storage Formats

- **Primary:** PLY (point cloud + mesh, widely supported, Open3D native)
- **Voxel grid:** `.npz` (numpy compressed) — for collision checking
- **Mesh for visualization:** GLB/glTF (web-viewable)
- **Octree:** Open3D Octree for efficient spatial queries
- **Config:** JSON sidecar with scan metadata (date, poses used, calibration params)

Storage location: `th3cl4w/data/models/workspace_YYYYMMDD_HHMMSS/`

---

## 5. Implementation Phases

### Phase 1: MVP (1-2 weeks)

**Goal:** Get a colored point cloud of the desk surface.

1. ✅ Implement hand-eye calibration (`src/vision/hand_eye_calibration.py`)
2. ✅ Define 5-10 scan poses manually (hardcoded joint angles)
3. ✅ Capture frames + record joint angles at each pose
4. ✅ Use Depth Anything v2 for per-frame monocular depth
5. ✅ Back-project to 3D, merge point clouds with known poses
6. ✅ Visualize in Open3D
7. ✅ Save as PLY

**Success metric:** Recognizable point cloud of desk + objects, ~5-10mm accuracy.

### Phase 2: Refinement (2-4 weeks)

1. Automated scan trajectories with coverage optimization
2. Switch to COLMAP MVS or PatchMatch stereo for better depth
3. TSDF fusion for clean mesh output
4. Self-occlusion masking (render arm, mask in images)
5. FK calibration refinement (minimize reprojection error)
6. API endpoints for scan/reconstruct
7. Background model for change detection

**Success metric:** Clean textured mesh, <3mm accuracy, automated scanning.

### Phase 3: Real-Time Updates (4-8 weeks)

1. Incremental TSDF updates from new frames during normal operation
2. Change detection — flag when workspace model is stale
3. Integration with motion planner for collision avoidance
4. Live depth overlay during manipulation
5. Web viewer for 3D model (three.js or similar)

**Success metric:** Model stays current within 30s of workspace changes.

---

## 6. Hardware/Software Requirements

### Software

| Library | Purpose | Version |
|---------|---------|---------|
| **Open3D** | TSDF, point clouds, mesh, visualization | ≥0.18 |
| **OpenCV** | Hand-eye calibration, image processing | ≥4.8 |
| **NumPy/SciPy** | Transforms, optimization | latest |
| **COLMAP** | MVS dense reconstruction (Phase 2) | ≥3.8 |
| **Depth Anything v2** | Monocular depth (Phase 1) | latest |
| **PyTorch** | Depth model inference | ≥2.0 |
| **trimesh** | Mesh I/O, analysis | latest |

```bash
pip install open3d opencv-python-headless trimesh torch torchvision
# COLMAP: install from source or apt (needs CUDA — see GPU note)
```

### GPU: RX 580 (AMD)

This is the tricky part:
- **COLMAP** requires CUDA for GPU-accelerated MVS → **won't work on RX 580**
  - Workaround: Use COLMAP CPU mode (slower, ~10-30min for 200 images)
  - Alternative: Use OpenMVS which has OpenCL support
- **Depth Anything v2** runs on PyTorch → works on CPU, or via ROCm on RX 580
  - ROCm support for RX 580 (Polaris) is limited/unofficial
  - Realistic plan: Run on CPU, ~0.5-1s per frame (acceptable for offline)
- **Open3D TSDF fusion** is CPU-based → works fine, ~1-2 min for 200 frames

### Computation Time Estimates (Phase 1, CPU)

| Step | Time |
|------|------|
| Scan (capture 50 frames) | ~2 min |
| Monocular depth (50 frames) | ~30-50s |
| Point cloud merging | ~5s |
| Voxel downsampling + cleanup | ~2s |
| **Total** | **~3 min** |

### Computation Time Estimates (Phase 2, CPU)

| Step | Time |
|------|------|
| Scan (capture 200 frames) | ~8 min |
| COLMAP MVS (CPU) | ~15-30 min |
| TSDF fusion | ~1-2 min |
| Mesh extraction + cleanup | ~1 min |
| **Total** | **~25-40 min** |

### Storage

- Raw scan data (200 frames + metadata): ~200MB
- Point cloud (PLY): ~10-50MB
- Mesh (PLY): ~20-100MB
- TSDF volume (in memory): ~500MB-1GB for 2mm voxels over 1m³

---

## Appendix: Key Transforms

```
T_base_world    — arm base in world frame (fixed, measure once)
T_ee_base(q)    — end-effector in base frame (FK from joint angles q)
T_cam_ee        — camera in EE frame (hand-eye calibration, fixed)

Camera world pose = T_base_world @ T_ee_base(q) @ T_cam_ee
```

## Appendix: Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| FK inaccuracy | Bad alignment, ghosting | Joint refinement calibration |
| Camera sync lag | Pose-image mismatch | Pause at each pose, timestamp matching |
| Textureless surfaces | MVS fails | Fall back to monocular depth |
| RX 580 too slow | Long reconstruction | CPU is fine for offline; consider cloud GPU for real-time |
| Motion blur at 15fps | Blurry frames | Pause at viewpoints, blur detection + rejection |
| Arm self-occlusion | Missing geometry | Multi-config coverage, arm masking |
