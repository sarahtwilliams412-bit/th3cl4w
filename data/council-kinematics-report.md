# Council Kinematics Report — Robotics/Kinematics Specialist

**Date:** 2026-02-07  
**Subject:** Calibration solver failure analysis and improvement plan

---

## 1. Why the Solver Produces Degenerate Results

The solver output (base=297mm, shoulder=50mm, offsets≈180°) is a **textbook degenerate local minimum**. Here's why:

### 1a. Fundamental coupling between link lengths and camera scale

The model has 13 parameters: 6 links, 3 offsets, 4 camera params (sx, sy, tx, ty). The camera affine maps FK-space → pixel-space:

```
px_x = sx * fk_x + tx
px_y = -sy * fk_y + ty
```

This means **link lengths and scale are multiplicatively coupled**: doubling all links while halving sx/sy produces identical pixel predictions. The solver has a degenerate manifold of equivalent solutions. L-BFGS-B will slide along this manifold to whatever the bounds allow.

The result confirms this: sx=sy=0.1 (at the lower bound) with base=297mm. The solver shrunk the scale to minimum and inflated the base to compensate. This is a **gauge freedom**, not a true optimum.

### 1b. Offsets at ~180° — a second equivalent configuration

With only 3 pitch joints and a cumulative-angle FK model, adding 180° to an offset while negating the link direction produces the same end-effector position. The solver found the "flipped" configuration where the arm is folded back on itself with offsets near 180° and tiny shoulder/elbow links — a valid minimum in the unconstrained landscape.

### 1c. Too few constraints

With 20 observations × 2 coordinates = 40 data points and 13 parameters, the system is technically over-determined. But:
- **40 collision events** means many observations likely have corrupted end-effector detections (arm hit obstacles → gripper wasn't where expected)
- End-effector-only observations provide no information about intermediate joints — link lengths are highly correlated
- Without observing multiple joints simultaneously, the chain is practically under-constrained

### 1d. Scalar residual function

The solver minimizes a single scalar (sum of squared distances). L-BFGS-B sees a flat valley with many equivalent minima. Using `least_squares` with a vector residual would give much better Jacobian information and avoid this.

**Diagnosis: The solver fails due to (1) gauge freedom between scale and link lengths, (2) configuration ambiguity from offset flipping, and (3) noisy/corrupted observations from collisions.**

---

## 2. DH 7-DOF → Viz 6-DOF Joint Mapping

### The DH model (kinematics.py)

7 joints with DH parameters using **d-only** links (all `a=0`):

| DH Joint | Name            | Type  | d (mm)  |
|----------|-----------------|-------|---------|
| J1       | shoulder_yaw    | yaw   | 121.5   |
| J2       | shoulder_pitch  | pitch | 0       |
| J3       | shoulder_roll   | roll  | 208.5   |
| J4       | elbow_pitch     | pitch | 0       |
| J5       | wrist_yaw       | yaw   | 208.5   |
| J6       | wrist_pitch     | pitch | 0       |
| J7       | wrist_roll      | roll  | 113.0   |

The DH chain has alternating ±π/2 alpha twists — this is a **spherical wrist** architecture with two S-R-S (spherical-revolute-spherical) segments.

### The viz model (drawArm / fk_2d)

6 joints indexed [0..5], with 6 link segments:

| Viz Idx | Viz Segment | Pitch? | Maps to DH Joint |
|---------|-------------|--------|-------------------|
| 0       | base        | fixed  | J1 (d=121.5)      |
| 1       | shoulder    | J1→yes | **J2** (pitch)     |
| 2       | elbow       | J2→yes | **J4** (pitch)     |
| 3       | wrist1      | no     | J3/J5 roll region  |
| 4       | wrist2      | J4→yes | **J6** (pitch)     |
| 5       | end         | no     | J7 roll region     |

**The mapping is:**
- Viz `joints[1]` (shoulder pitch) = DH J2 (shoulder_pitch)
- Viz `joints[2]` (elbow pitch) = DH J4 (elbow_pitch)
- Viz `joints[4]` (wrist pitch) = DH J6 (wrist_pitch)
- Viz `joints[0]` = DH J1 (shoulder_yaw / base rotation) — rendered as indicator only
- Viz `joints[3]` = DH J3 or J5 (roll) — rendered as indicator only
- Viz `joints[5]` = DH J7 (wrist_roll) — rendered as indicator only

### Link length mapping

The DH `d` parameters map to viz links approximately:
- DH J1 d=121.5mm → viz base=80mm (includes mounting offset?)
- DH J3 d=208.5mm → viz shoulder=170mm (the link between shoulder and elbow pivot)
- DH J5 d=208.5mm → viz elbow=170mm (the link between elbow and wrist)
- DH J7 d=113.0mm → viz wrist1+wrist2+end=170mm total (the wrist assembly)

**Key mismatch:** The DH model has links encoded as `d` offsets on joints with zero `a`. The actual physical link between J2 and J4 corresponds to J3's d=208.5mm, which is close to the viz shoulder=170mm but not exact. The 6-link viz model is an approximation that collapses certain DH frames.

---

## 3. Why J0/J3/J5/Gripper Should Be Calibrated and How

### J0 (Base Yaw) — Critical for 2D accuracy

J0 rotates the entire arm around the vertical axis. In a side-view camera:

**Effect:** At J0=0° (arm in the camera's sagittal plane), you see full link lengths. At J0=θ, every horizontal component is scaled by cos(θ) — **foreshortening**.

The current viz completely ignores this. If J0≠0° during calibration poses, the detected end-effector position won't match the planar FK prediction, **directly corrupting the calibration**.

**How to calibrate:**
```python
# In fk_2d, scale horizontal components by cos(J0):
cos_j0 = math.cos(math.radians(joint_angles[0]))
# For each segment, the x-component is foreshortened:
x = pts[i][0] + math.cos(cum_angle) * links[i] * cos_j0
y = pts[i][1] + math.sin(cum_angle) * links[i]  # vertical unaffected
```

Actually this is an oversimplification — the correct approach is to do the full 3D FK and project. But as a first-order correction: multiply each link's projected horizontal extent by cos(J0).

**For calibration:** Either (a) lock J0=0° during all calibration poses, or (b) include J0 in the FK model with proper foreshortening.

### J3/J5 (Wrist Rolls) — End-effector appearance

Roll joints don't move link positions in side view, but they **rotate the gripper/end-effector about the link axis**. This affects:
- The apparent shape of the gripper in camera images
- Whether the end-effector detection algorithm correctly identifies the tip
- The visual appearance used for overlay matching

**How to calibrate:** These don't need geometric calibration. Instead:
1. During calibration, lock J3=J5=0 so the gripper presents a consistent profile
2. For runtime viz, render the gripper rotation as a rotation of the end-effector sprite/shape

### Gripper Aperture

The gripper opening changes the visual footprint of the end-effector:
- Closed gripper: narrow profile, easy to detect as a point
- Open gripper: wider, detection may find the centroid between fingers rather than the tip

**How to calibrate:**
1. Record gripper state with each observation
2. For closed gripper: end-effector = tip position
3. For open gripper: end-effector = midpoint between fingers, offset from wrist by known gripper geometry
4. Or simply: **always calibrate with gripper closed** for consistency

---

## 4. Is the 2D Side-View Approach Fundamentally Flawed?

**Yes, partially.** Here's the breakdown:

### What works about 2D side-view
- Simple, fast, easy to debug visually
- For a planar subset of motions (J0=0, rolls=0), it's a valid projection
- The Unitree D1 has a natural sagittal working plane for many tasks

### What's fundamentally broken
1. **J0 destroys planarity.** Any base rotation takes the arm out of the camera plane. The 2D model can't represent this without the foreshortening correction, and even that's approximate.

2. **Camera is not a pure side view.** A real camera has perspective projection, lens distortion, and is not perfectly aligned to the arm's sagittal plane. The affine model (sx, sy, tx, ty) assumes orthographic side view — no perspective, no rotation, no distortion.

3. **Roll joints cause out-of-plane motion of the end-effector.** With nonzero J3/J5, the gripper tip traces a circle in 3D that projects to an ellipse in the camera. The 2D model can't capture this.

### Recommendation: Hybrid approach

**Don't abandon 2D viz** — it's great for the UI overlay. But **replace the calibration model** with:

1. **Full 3D FK** using the DH parameters (already implemented in `kinematics.py`)
2. **Pinhole camera projection** (3D → 2D): `[u, v] = K @ [R|t] @ P_3d`
3. Calibrate the **camera extrinsics** (6 DOF: position + orientation relative to arm base) and **intrinsics** (fx, fy, cx, cy + distortion)

This gives you:
- Correct foreshortening for any J0
- Proper handling of all 7 joints
- A physically meaningful camera model
- No gauge freedom (link lengths are fixed from DH; only camera params are calibrated)

For the viz overlay, project the 3D FK chain through the calibrated camera model to get 2D points, then draw those.

---

## 5. Concrete Solver Improvements

### 5.1 Fix the gauge freedom — FIRST PRIORITY

**Option A (quick fix):** Fix link lengths to known values from DH/CAD and only calibrate offsets + camera params. This reduces from 13 to 7 parameters and eliminates the scale-length degeneracy.

```python
# Fixed from DH model
FIXED_LINKS = [121.5, 208.5, 208.5, 60, 60, 113.0]
# Only optimize: 3 offsets + 4 camera params = 7 params
```

**Option B (better):** Fix camera scale from a known reference (e.g., measured distance between two points in the image) and only calibrate link lengths + offsets.

### 5.2 Use least_squares instead of minimize

```python
from scipy.optimize import least_squares

def residual_vector(params):
    """Return vector of (pred_x - obs_x, pred_y - obs_y) for each observation."""
    residuals = []
    for obs in valid_obs:
        fk_pts = fk_2d(obs.joint_angles, links, offsets)
        pred_px = sx * fk_pts[-1][0] + tx
        pred_py = -sy * fk_pts[-1][1] + ty
        residuals.extend([pred_px - obs.end_effector_px[0],
                          pred_py - obs.end_effector_px[1]])
    return np.array(residuals)

result = least_squares(residual_vector, x0, bounds=(lb, ub), method='trf')
```

This gives the Jacobian structure, enabling much better convergence.

### 5.3 Add regularization toward physical priors

```python
# Add penalty terms pulling toward known DH values
PRIOR_LINKS = [121.5, 208.5, 208.5, 60, 60, 113.0]
PRIOR_OFFSETS = [90, 90, 0]  # expected from DH analysis
LAMBDA_LINK = 0.1  # weight relative to pixel error
LAMBDA_OFFSET = 0.05

for i in range(6):
    residuals.append(LAMBDA_LINK * (params[i] - PRIOR_LINKS[i]))
for i, prior in enumerate(PRIOR_OFFSETS):
    residuals.append(LAMBDA_OFFSET * (params[6+i] - prior))
```

### 5.4 Multi-stage optimization

**Stage 1:** Fix links to DH values, calibrate only camera params (4 DOF) using a robust solver. This is a near-linear problem.

**Stage 2:** With camera params warm-started, add offset calibration (7 DOF total).

**Stage 3 (optional):** Release link lengths with strong regularization (13 DOF) for fine-tuning.

### 5.5 Better bounds

Current bounds allow physically impossible solutions. Tighten:

```python
bounds = [
    (60, 150),   # base: DH says 121.5mm
    (150, 250),  # shoulder: DH says 208.5mm
    (150, 250),  # elbow: DH says 208.5mm
    (30, 90),    # wrist1
    (30, 90),    # wrist2
    (80, 150),   # end: DH says 113mm
    (45, 135),   # offset1: expect ~90°
    (45, 135),   # offset2: expect ~90°
    (-45, 45),   # offset4: expect ~0°
    (0.5, 5),    # sx: reasonable scale
    (0.5, 5),    # sy
    (w*0.1, w*0.6),  # tx
    (h*0.5, h*0.95), # ty
]
```

### 5.6 Observation quality filtering

Before solving:
1. Reject observations where collision was detected
2. Reject observations where end-effector detection confidence is below threshold
3. Require J0 ≈ 0° (within ±5°) for all calibration poses, or implement foreshortening
4. Verify detections by checking that detected points change when joints change (static detections = false positives)

### 5.7 Multi-joint observation (if possible)

Instead of only detecting the end-effector, detect intermediate joints (elbow, wrist) too. Each additional joint observation:
- Adds 2 constraints per pose
- Breaks the correlation between link lengths
- Makes the system much more robust

---

## Summary of Recommendations (Priority Order)

| Priority | Action | Impact |
|----------|--------|--------|
| **P0** | Fix gauge freedom: freeze link lengths to DH values | Eliminates degenerate solutions |
| **P0** | Filter out collision-corrupted observations | Removes garbage data |
| **P1** | Switch to `least_squares` with vector residuals | Much better convergence |
| **P1** | Lock J0≈0° during calibration | Removes out-of-plane error |
| **P1** | Tighten bounds to physically plausible ranges | Prevents solver wandering |
| **P2** | Add regularization toward DH priors | Guides solver to physical solution |
| **P2** | Multi-stage optimization | Robust convergence |
| **P3** | Replace affine with pinhole camera model | Correct projection for all poses |
| **P3** | Full 3D FK projection for viz overlay | Accurate for any J0/roll config |
| **P4** | Multi-joint detection | Fundamentally better constraints |

The **single highest-impact fix** is P0: freeze link lengths and filter observations. This alone should reduce the residual from 122px to under 10px if the end-effector detection is reasonable.
