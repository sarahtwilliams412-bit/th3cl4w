# Council Vision Report — Computer Vision Analysis
**Date:** 2026-02-07  
**Author:** Vision Specialist 🦾

---

## Executive Summary

The calibration is failing for three compounding reasons: (1) `detect_end_effector()` is finding random dark blobs, not the gripper; (2) only end-effector position is used, leaving a massively underconstrained 13-parameter optimization; (3) J0 foreshortening is completely ignored. The solver has no chance.

---

## 1. Why End-Effector Detection Is Failing

### What the Arm Actually Looks Like

From the camera images, this is a **Unitree Z1** (or similar). The arm is:
- **Entirely matte black** — links, joints, gripper, everything
- **Black cylindrical base** on an OSB board
- **One golden/yellow segment** on the shoulder/upper-arm link (anodized aluminum or label)
- **Black parallel-jaw gripper** with fine tines — small, dark, low contrast
- The **checkerboard background** creates a nightmare of high-frequency B&W edges

### Why HSV Thresholding Fails

The current `detect_end_effector()` uses two masks:
1. **Blue mask** (H:100-130, S:50+, V:50+) — The arm is NOT blue. It's black. This mask catches nothing on the arm itself; it might catch the blue tape marker on the checkerboard.
2. **Dark mask** (H:0-180, S:0-80, V:20-120) — This catches *everything* dark: the arm, the base, the OSB board, shadows, the person's clothing, the Unitree controller box, cables, clamps...

The algorithm then finds the point **furthest from image center-bottom** among all dark contours. In a cluttered garage with a person standing right there in a dark jacket, this will almost certainly return a point on the person, the ceiling, or some random shadow — NOT the gripper.

### The Checkerboard Makes It Worse

The checkerboard behind the arm creates massive false contours. The black squares match the "dark mask" perfectly. The `MORPH_OPEN`/`MORPH_CLOSE` with a 7×7 kernel can't remove checkerboard squares that are similar in size to the gripper.

### Fundamental Problem

**HSV thresholding cannot distinguish a matte-black gripper from a matte-black background in a cluttered environment.** The approach is fundamentally wrong for this arm.

---

## 2. Detecting Intermediate Joints (Not Just End-Effector)

### Why This Matters

The current solver optimizes **13 parameters** (6 link lengths + 3 offsets + 4 camera params) from **~20 end-effector-only observations**. Each observation provides only 2 constraints (x, y pixels). That's 40 constraints for 13 unknowns — sounds okay mathematically, but the observations are noisy/wrong AND the problem is highly nonlinear with many local minima.

If we could detect **3-4 joint positions per frame**, we'd get 6-8 constraints per observation. With 20 poses, that's 120-160 constraints — massively overconstrained. More importantly, intermediate joints **break symmetries** in the optimization landscape (e.g., a short shoulder + long elbow looks like a long shoulder + short elbow from the end-effector alone).

### What's Detectable

From the images, there are clear visual landmarks:

| Joint | Visual Feature | Detectability |
|-------|---------------|---------------|
| **Base (J0)** | Black cylinder, top center | Easy — fixed position, large dark circle |
| **Shoulder (J1)** | Silver/chrome circular actuator housing | Medium — shiny metallic ring, distinct from matte black |
| **Elbow (J2)** | Junction where golden segment meets black upper forearm | Good — **color boundary** yellow↔black is highly detectable |
| **Wrist (J4)** | Small actuator housing | Hard — small, same color as links |
| **Gripper tip** | Fork tines at end | Hard — thin, dark, small |

### The Golden Segment is the Key

The **golden/yellow anodized segment** on the upper arm is the single most distinctive visual feature. It spans roughly from the shoulder actuator to the elbow joint. We should:

1. Detect the yellow segment via HSV (H:15-35, S:100+, V:120+) — this is highly selective
2. Find its **endpoints** — the top connects to the shoulder, the bottom connects to the elbow
3. These two points alone massively constrain the shoulder link length and angle

### Proposed Joint Detection Strategy

```python
def detect_arm_landmarks(frame):
    """Detect multiple arm landmarks using targeted strategies."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. GOLDEN SEGMENT — highest confidence detection
    # Unitree Z1 has distinctive gold/yellow anodized upper arm
    gold_mask = cv2.inRange(hsv, 
        np.array([15, 100, 120]),   # lower gold
        np.array([35, 255, 255]))   # upper gold
    # Find endpoints of the gold region → shoulder & elbow positions
    
    # 2. BASE — fixed position, find once
    # Large black cylinder. Detect via Hough circles or template.
    # Or just manually specify — it doesn't move.
    
    # 3. GRIPPER — track via frame differencing
    # Take two frames: one at pose, one at home.
    # The area that changed most and is furthest along the kinematic chain
    # is the end-effector. This is FAR more robust than color.
    
    return {
        'base': (x, y),        # fixed
        'shoulder': (x, y),    # top of gold segment
        'elbow': (x, y),       # bottom of gold segment
        'end_effector': (x, y) # from frame differencing
    }
```

### Frame Differencing for End-Effector

Instead of trying to find the gripper by color, use **motion**:

1. Capture a frame at home position → `home_frame`
2. Move to calibration pose, capture → `pose_frame`
3. `diff = cv2.absdiff(pose_frame, home_frame)`
4. Threshold the diff → shows exactly what moved
5. The **extremal point** of the moved region (furthest from base) is the end-effector

This is inherently robust to background clutter, checkerboard, people, etc. Only the arm moves.

---

## 3. Handling J0 (Base Yaw) Foreshortening

### The Problem

The viz is a 2D side view. When J0 ≠ 0°, the arm rotates around the vertical axis. From a fixed side camera:
- At J0=0°: full arm visible in profile, apparent link lengths = true lengths
- At J0=45°: links foreshortened by cos(45°) ≈ 0.71× in the horizontal axis
- At J0=90°: arm points at/away from camera, appears as just the base column

The current FK model **completely ignores J0**, treating it as always 0°.

### The Fix

The 2D side-view FK must project through J0:

```python
def fk_2d_with_yaw(joint_angles, links, offsets):
    """2D FK that accounts for J0 yaw via foreshortening."""
    # First compute the full 2D side-view FK as before (in the arm's own plane)
    pts_arm_plane = fk_2d_no_yaw(joint_angles, links, offsets)
    
    # Then project: J0 rotation foreshortens the horizontal component
    yaw = math.radians(joint_angles[0])
    cos_yaw = math.cos(yaw)
    
    # For a side camera looking along the X axis:
    # x_pixel ∝ x_arm * cos(yaw)  (foreshortened)
    # y_pixel ∝ y_arm              (unaffected by yaw)
    pts_projected = []
    for (x, y) in pts_arm_plane:
        pts_projected.append((x * cos_yaw, y))
    
    return pts_projected
```

This is a simplification (assumes camera is perfectly aligned with J0=0° plane), but it's far better than ignoring J0 entirely.

### Calibration Implications

- During calibration, **vary J0** too (not just J1, J2, J4)
- This provides observations at different foreshortening levels, which helps constrain the camera-to-base transform
- Add J0 to the pose generation: e.g., test at J0 = {-30°, 0°, 30°}

---

## 4. Camera Selection: cam0 (Overhead) vs cam1 (Front/Side)

### What Each Camera Sees

From the images:
- **cam0 (overhead):** Top-down bird's eye view. Shows the arm from above. You can clearly see J0 rotation and the horizontal reach. The arm's planar extent is visible. The golden segment is clearly visible from above.
- **cam1 (front/side):** Roughly front-facing, slightly elevated. Shows the arm in profile-ish view against the checkerboard. This is what the current calibrator uses.

### Analysis

**cam1 is reasonable for side-view calibration BUT:**
- The camera isn't perfectly perpendicular to the arm's J0=0° plane — it's slightly oblique
- The checkerboard dominates the background, wrecking color-based detection
- The person is often in frame, adding noise

**cam0 (overhead) is VERY useful and currently ignored:**
- Overhead view directly shows J0 yaw — the arm sweeps left/right visually
- X-axis in cam0 ≈ the arm's horizontal reach at the J0 angle
- Combined with cam1, you get a pseudo-stereo setup: cam1 gives (side_x, y), cam0 gives (top_x, top_y)
- Two cameras = 4 constraints per observation instead of 2

### Recommendation: Use BOTH Cameras

```python
@dataclass
class PoseObservation:
    joint_angles: List[float]
    cam0_landmarks: Dict[str, Optional[Tuple[int, int]]]  # overhead
    cam1_landmarks: Dict[str, Optional[Tuple[int, int]]]  # side
```

The solver should project FK points into both camera planes and minimize combined residuals. This effectively doubles the constraint count AND resolves the J0 ambiguity.

For the overhead cam, the FK projection is:
```python
# Overhead: project XZ plane (yaw rotates in this plane)
yaw = math.radians(joint_angles[0])
# After shoulder, each link endpoint has (horiz_dist, height) in arm plane
# In overhead view: x = horiz_dist * sin(yaw), z = horiz_dist * cos(yaw)
```

---

## 5. Specific Code Changes

### A. Replace `detect_end_effector()` entirely

```python
class ArmDetector:
    """Multi-landmark arm detector using frame differencing + color."""
    
    def __init__(self):
        self.home_frame_cam0 = None
        self.home_frame_cam1 = None
        self.base_pos_cam1 = None  # manually set or detected once
        self.base_pos_cam0 = None
    
    def set_home_frames(self, cam0_frame, cam1_frame):
        """Capture reference frames at home position."""
        self.home_frame_cam0 = cv2.GaussianBlur(cam0_frame, (5, 5), 0)
        self.home_frame_cam1 = cv2.GaussianBlur(cam1_frame, (5, 5), 0)
    
    def detect_gold_segment(self, frame):
        """Find the golden upper-arm segment. Returns (top_pt, bottom_pt) or None."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gold = cv2.morphologyEx(gold, cv2.MORPH_CLOSE, kernel)
        gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Largest gold contour = the arm segment
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        
        # Fit a line to get the segment axis, find extrema
        pts = c.reshape(-1, 2)
        top_pt = tuple(pts[pts[:, 1].argmin()])   # highest point (min y)
        bot_pt = tuple(pts[pts[:, 1].argmax()])    # lowest point (max y)
        return top_pt, bot_pt
    
    def detect_via_differencing(self, frame, home_frame):
        """Find moved regions by frame differencing."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        diff = cv2.absdiff(blurred, home_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def detect_landmarks(self, cam1_frame, cam0_frame=None):
        """Detect all available arm landmarks."""
        result = {'base': self.base_pos_cam1}  # fixed, set once
        
        # Gold segment → shoulder + elbow
        gold = self.detect_gold_segment(cam1_frame)
        if gold:
            top, bot = gold
            result['shoulder'] = top  # top of gold = shoulder joint
            result['elbow'] = bot     # bottom of gold = elbow joint
        
        # Frame differencing → end-effector
        if self.home_frame_cam1 is not None:
            contours = self.detect_via_differencing(cam1_frame, self.home_frame_cam1)
            if contours:
                # Find point furthest from base
                base = self.base_pos_cam1 or (cam1_frame.shape[1]//2, cam1_frame.shape[0])
                best_pt, best_dist = None, 0
                for c in contours:
                    if cv2.contourArea(c) < 200:
                        continue
                    for pt in c.reshape(-1, 2):
                        d = math.hypot(pt[0] - base[0], pt[1] - base[1])
                        if d > best_dist:
                            best_dist = d
                            best_pt = (int(pt[0]), int(pt[1]))
                result['end_effector'] = best_pt
        
        return result
```

### B. Update the FK model for J0

In `fk_2d()`, add yaw foreshortening:

```python
def fk_2d(joint_angles, links, offsets, camera='side'):
    """2D FK with J0 yaw projection."""
    # Compute arm-plane FK as before (ignoring J0)
    pts_arm = _fk_arm_plane(joint_angles, links, offsets)
    
    yaw = math.radians(joint_angles[0] + offsets[0])
    
    if camera == 'side':
        # Side view: horizontal foreshortened by cos(yaw)
        return [(x * math.cos(yaw), y) for x, y in pts_arm]
    elif camera == 'overhead':
        # Overhead: rotate horizontal component by yaw
        return [(x * math.sin(yaw), x * math.cos(yaw)) for x, y in pts_arm]
    # Note: overhead projection discards y (height) — need separate handling
```

### C. Restructure the solver for multi-landmark, dual-camera

```python
def solve_calibration(observations, image_shapes):
    """Multi-landmark, dual-camera solver."""
    # Parameters: 6 links + 6 offsets + 4 cam1 params + 4 cam0 params = 20
    # But with 4 landmarks × 2 cameras × 20 poses = up to 320 constraints
    
    def residuals(params):
        links = params[:6]
        offsets = [params[6], params[7], params[8], 0, params[9], 0]
        cam1 = params[10:14]  # sx, sy, tx, ty
        cam0 = params[14:18]  # sx, sy, tx, ty
        
        total = 0.0
        for obs in valid_obs:
            fk_side = fk_2d(obs.joint_angles, links, offsets, 'side')
            fk_over = fk_2d(obs.joint_angles, links, offsets, 'overhead')
            
            # Match each detected landmark to its FK joint
            for name, idx in [('base', 0), ('shoulder', 2), ('elbow', 3), ('end_effector', -1)]:
                if obs.cam1_landmarks.get(name):
                    pred = project(fk_side[idx], cam1)
                    obs_px = obs.cam1_landmarks[name]
                    total += (pred[0]-obs_px[0])**2 + (pred[1]-obs_px[1])**2
                    
                if obs.cam0_landmarks and obs.cam0_landmarks.get(name):
                    pred = project(fk_over[idx], cam0)
                    obs_px = obs.cam0_landmarks[name]
                    total += (pred[0]-obs_px[0])**2 + (pred[1]-obs_px[1])**2
        
        return total
```

### D. Add J0 to calibration poses

```python
CALIBRATION_JOINTS = [0, 1, 2, 4]  # Include J0

JOINT_LIMITS = {
    0: (-45.0, 45.0),  # yaw — limit range for side camera visibility
    1: (-85.0, 85.0),
    2: (-85.0, 85.0),
    4: (-85.0, 85.0),
}
```

### E. One-time base position detection

The base doesn't move. Detect it once at startup and fix it:

```python
async def detect_base_position(cam1_frame):
    """Find the base cylinder — large dark circle near bottom of frame."""
    gray = cv2.cvtColor(cam1_frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=100, param2=30,
                                minRadius=30, maxRadius=80)
    if circles is not None:
        # Pick the one closest to expected base position (center-bottom)
        h, w = cam1_frame.shape[:2]
        best = min(circles[0], key=lambda c: abs(c[0]-w*0.4) + abs(c[1]-h*0.7))
        return (int(best[0]), int(best[1]))
    return None
```

---

## 6. Priority Order of Changes

1. **🔴 Frame differencing for end-effector** — Immediate win, replaces broken HSV approach. Requires capturing home frame first.
2. **🔴 Gold segment detection** — Easy to implement, provides shoulder+elbow constraints that break solver degeneracy.
3. **🟡 Dual-camera solver** — Use cam0 overhead view for J0 observability. Moderate refactor.
4. **🟡 J0 foreshortening in FK** — Required for J0 calibration. Simple math change.
5. **🟢 Add J0 to calibration poses** — Once the above are in place.
6. **🟢 Constrain solver with known priors** — The Unitree Z1 link lengths are published (~170mm shoulder, ~170mm elbow). Use tight bounds, not 50-400mm.

---

## 7. Quick Wins Before Code Changes

- **Clear the workspace** — Remove checkerboard from the arm's movement plane, clamps, tools, person. The checkerboard should be a backdrop only, not draped over the work area.
- **Add a colored marker to the gripper** — A small piece of bright red or green tape on the gripper tip would make detection trivial. One piece of tape = 10× better detection.
- **Fix solver bounds** — Even without vision changes, tightening link bounds to ±30% of known Z1 dimensions would prevent the degenerate 297mm base / 50mm shoulder result.

---

## Summary Table

| Problem | Root Cause | Fix | Effort |
|---------|-----------|-----|--------|
| Gripper not detected | HSV matches everything dark | Frame differencing | Low |
| Solver degenerate | Underconstrained (EE only) | Multi-landmark detection | Medium |
| J0 ignored | 2D FK has no yaw | Cosine foreshortening | Low |
| cam0 unused | Only cam1 wired up | Dual-camera solver | Medium |
| Wild solver bounds | 50-400mm for 170mm links | Use Z1 spec priors | Trivial |
| Cluttered workspace | Garage environment | Physical cleanup | Zero code |
