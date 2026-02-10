# Real-Time Vision Positioning Plan — th3cl4w D1

> Using 3 camera feeds to position the arm in real-time for autonomous pick-and-place.

## Hardware Context

- **Arm**: Unitree D1, 6-DOF + gripper, controlled via DDS
- **Overhead cam** (cam0): Logitech BRIO @ `/dev/video0` — top-down view of workspace
- **Arm-mounted cam** (cam1): MX Brio @ `/dev/video4` — behind gripper, moves with arm
- **Side cam** (cam2): Logitech BRIO @ `/dev/video6` — lateral view for height/depth
- **Compute**: Intel NUC10i7FNH (6C/12T), RX 580 eGPU (OpenCL/ROCm)
- **Capture**: 1920×1080 @ 15fps, JPEG quality 92, V4L2 MJPG backend

---

## 1. Multi-View Arm Tracking

### Camera Roles

| Camera | Primary Axis | Field of View | Role |
|--------|-------------|---------------|------|
| **Overhead** (cam0) | XY plane | Full workspace (~600×400mm) | Object detection, coarse gripper XY, workspace mapping |
| **Side** (cam2) | XZ plane | Arm profile view | Height (Z) estimation, arm pose verification, collision clearance |
| **Arm-mounted** (cam1) | Local Z (approach axis) | ~120° FOV, close range | Fine alignment, grasp verification, obstacle detection near gripper |

### What Each Camera Sees

**Overhead (top-down):**
- Gripper position projected onto workspace plane → XY coordinates
- All objects on the workspace surface → XY positions + rough size
- Arm links (partial occlusion when arm is above objects)
- Workspace boundaries and obstacles

**Side (lateral):**
- Gripper height above workspace → Z coordinate
- Object heights (profile silhouette) → Z extent
- Arm pose (joint angles visually verifiable)
- Clearance between gripper and objects during approach/retract

**Arm-mounted (eye-in-hand):**
- Target object relative to gripper center → pixel offset = alignment error
- Grasp readiness (is object centered? is gripper oriented correctly?)
- Close-range obstacle detection (things the overhead cam can't resolve)
- Surface texture/features for grasp quality estimation

### Multi-View Fusion

The 3 cameras provide complementary information that fuses into full 3D:

```
Overhead cam → (X, Y) in world frame     ─┐
                                           ├──► (X, Y, Z) world position
Side cam → (X, Z) in world frame          ─┘
                                           
Arm cam → (Δx, Δy) error in gripper frame ──► Fine correction vector
```

**Fusion strategy: Not a single 3D reconstruction.** Instead, use each camera for what it's best at:

1. Overhead provides **XY target position** (top-down projection, most reliable for XY)
2. Side provides **Z height** (profile view, most reliable for height)
3. Arm cam provides **alignment error** (relative positioning, most reliable for final approach)

This avoids complex multi-view stereo reconstruction. Each camera contributes its strongest axis. Cross-validation (overhead XY should agree with side X) catches errors.

**When views disagree:**
- If overhead and side disagree on X position by >10mm → flag uncertainty, re-detect
- If arm cam shows no object when overhead says it should be there → object may have moved
- Weighted average: cameras with higher confidence get more weight

---

## 2. Object Detection & Tracking

### Detection Approaches

#### Option A: YOLOv8-nano (Recommended for general objects)

- **Model**: `yolov8n` (3.2M params, ~6MB)
- **Input**: 640×640 (auto-letterboxed from 1080p)
- **CPU inference** (i7-10710U): ~40–80ms per frame
- **GPU inference** (RX 580 via ONNX + OpenCL): ~15–30ms per frame (if ROCm/ONNX works)
- **Detects**: 80 COCO classes out of the box, or fine-tune on workspace objects
- **Pros**: General-purpose, robust to lighting/angle changes, returns bounding boxes + class + confidence
- **Cons**: Requires fine-tuning for custom objects, highest latency of options

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detect_objects(frame: np.ndarray) -> list[Detection]:
    results = model.predict(frame, conf=0.5, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections.append(Detection(
            label=model.names[int(box.cls)],
            confidence=float(box.conf),
            bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
        ))
    return detections
```

#### Option B: Color Segmentation (Fast, for known objects)

- **Latency**: 2–5ms per frame on CPU
- **Method**: HSV thresholding → contour detection → centroid
- **Best for**: Brightly colored objects (red cube, blue ball, etc.)
- **Pros**: Extremely fast, no model needed, deterministic
- **Cons**: Brittle to lighting changes, only works for color-distinct objects

```python
def detect_by_color(frame: np.ndarray, hsv_lower, hsv_upper) -> Optional[Detection]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:  # min area filter
        return None
    x, y, w, h = cv2.boundingRect(c)
    return Detection(label='target', confidence=0.9, bbox=(x, y, w, h))
```

#### Option C: ArUco Markers (For calibration + known object tracking)

- **Latency**: 3–8ms per frame
- **Method**: `cv2.aruco.detectMarkers()` → corner positions → pose estimation
- **Best for**: Calibration targets, objects you can stick markers on
- **Pros**: Sub-pixel accuracy, gives full 6-DOF pose, fast
- **Cons**: Requires physical markers on objects

#### Option D: Template Matching (For specific known objects)

- **Latency**: 5–15ms per frame
- **Method**: `cv2.matchTemplate()` with normalized cross-correlation
- **Best for**: Objects with distinctive visual features, consistent viewing angle
- **Pros**: No training needed, just a reference image
- **Cons**: Scale/rotation sensitive, single object at a time

### Recommended Stack

| Scenario | Method | Expected Latency |
|----------|--------|-----------------|
| General pick-and-place | YOLOv8n | 40–80ms CPU / 15–30ms GPU |
| Color-coded objects | HSV segmentation | 2–5ms |
| Calibration & testing | ArUco markers | 3–8ms |
| Gripper detection (overhead) | Color segmentation (gripper is distinctive) | 2–5ms |
| Fine alignment (arm cam) | Template match or feature match | 5–15ms |

**Use multiple methods simultaneously**: YOLO for initial detection, color segmentation for frame-to-frame tracking (much faster).

### Target Tracking Across Frames

Once an object is detected, track it across frames without re-running detection every frame:

#### Centroid Tracker (Simple)

```python
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.objects = {}  # id → centroid
        self.disappeared = {}  # id → frame count since last seen
        self.next_id = 0
        self.max_disappeared = max_disappeared
    
    def update(self, detections: list[Detection]) -> dict[int, Detection]:
        centroids = [(d.bbox[0]+d.bbox[2]//2, d.bbox[1]+d.bbox[3]//2) for d in detections]
        # Hungarian algorithm or simple nearest-neighbor matching
        # ... (standard centroid tracking implementation)
```

#### Kalman Filter (Predictive, handles occlusion)

```python
class KalmanTracker:
    """Per-object Kalman filter for position prediction."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # state: [x, y, vx, vy], measurement: [x, y]
        self.kf.transitionMatrix = np.array([
            [1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    
    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])
    
    def update(self, x: float, y: float):
        self.kf.correct(np.array([[x], [y]], np.float32))
```

**Strategy**: Run YOLO every 5th frame (200–400ms interval), run Kalman prediction + color tracking on intermediate frames. This gives smooth tracking at 15fps with YOLO-grade accuracy.

### GPU Acceleration on RX 580

**Current state of AMD GPU compute for ML:**

| Framework | RX 580 Support | Status |
|-----------|---------------|--------|
| ROCm (PyTorch) | GFX803 — officially dropped in ROCm 5+ | Broken. Don't count on it. |
| OpenCL (OpenCV) | Yes, works | Image processing ops (resize, threshold, morphology) |
| ONNX Runtime + OpenCL | Partial | Can work for YOLO inference, needs testing |
| DirectML (via ONNX) | Linux: No | Windows only |

**Practical plan:**
1. Use **OpenCV OpenCL** for image preprocessing (resize, color convert, blur) — this works and saves CPU
2. Run **YOLO on CPU** with `yolov8n` (smallest model) — 40–80ms is acceptable at 15fps
3. Try **ONNX Runtime with OpenCL EP** for YOLO — may get 2–3× speedup, but test thoroughly
4. **Color segmentation on GPU** via `cv2.UMat` — trivial speedup for already-fast ops

**Don't waste time trying to get ROCm working on GFX803.** The RX 580 is best used for OpenCV UMat operations and potentially ONNX inference. CPU is fine for YOLO-nano.

---

## 3. Visual Servoing Pipeline

### IBVS vs PBVS

| Aspect | IBVS (Image-Based) | PBVS (Position-Based) |
|--------|--------------------|-----------------------|
| **Input** | Pixel coordinates of features | 3D pose of target |
| **Requires** | Feature tracking in image | Camera calibration + depth |
| **Robustness** | More robust to calibration errors | Sensitive to calibration |
| **Path** | Straight in image space (can be weird in 3D) | Straight in 3D space |
| **Singularities** | Possible image jacobian issues | Possible joint singularities |
| **Best for** | Fine alignment (arm cam) | Coarse positioning (overhead/side) |

### Recommended: Hybrid PBVS (coarse) + IBVS (fine)

1. **Coarse positioning (PBVS)**: Use overhead + side cameras with calibrated pixel-to-world mapping. Compute target position in world frame → solve IK → move joints.
2. **Fine alignment (IBVS)**: Use arm camera. Compute pixel error (object center vs frame center) → map to gripper velocity via image Jacobian → servo until error < threshold.

### Control Loop Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VISUAL SERVO LOOP                     │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Camera   │───►│ Detector │───►│ Error Computer   │  │
│  │  Capture  │    │ (YOLO/   │    │ (pixel error or  │  │
│  │  ~15fps   │    │  color)  │    │  world error)    │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                            │            │
│                                            ▼            │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │   Arm     │◄───│   IK     │◄───│ Controller       │  │
│  │  Driver   │    │  Solver  │    │ (P/PI/PID)       │  │
│  │  (DDS)    │    │          │    │                  │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│                                                         │
│  Loop time budget: <100ms total                         │
│    Camera capture:  ~10ms (from buffer)                 │
│    Detection:       ~5-50ms (method dependent)          │
│    Error compute:   ~1ms                                │
│    IK solve:        ~2ms                                │
│    Arm command:     ~10ms (DDS publish + ack)           │
│    Margin:          ~27-72ms                            │
└─────────────────────────────────────────────────────────┘
```

### Per-Camera Servo Strategy

#### Overhead Camera → Coarse XY Positioning

```python
class OverheadServo:
    """PBVS using overhead camera for XY positioning."""
    
    def __init__(self, pixel_to_world: callable, arm_controller):
        self.p2w = pixel_to_world  # (px, py) → (world_x, world_y)
        self.arm = arm_controller
        self.kp = 0.5  # proportional gain
        self.tolerance_mm = 5.0
    
    def step(self, target_px: tuple[int,int], gripper_px: tuple[int,int]) -> bool:
        """One servo step. Returns True if converged."""
        # Convert to world coordinates
        target_world = self.p2w(*target_px)
        gripper_world = self.p2w(*gripper_px)
        
        # Compute error
        error_x = target_world[0] - gripper_world[0]
        error_y = target_world[1] - gripper_world[1]
        error_mag = np.sqrt(error_x**2 + error_y**2)
        
        if error_mag < self.tolerance_mm:
            return True  # Converged
        
        # P controller with velocity limit
        vx = np.clip(self.kp * error_x, -50, 50)  # mm/step, capped
        vy = np.clip(self.kp * error_y, -50, 50)
        
        # Command arm (cartesian velocity or incremental position)
        self.arm.move_cartesian_delta(dx=vx, dy=vy, dz=0)
        return False
```

#### Side Camera → Z Height Control

```python
class SideServo:
    """PBVS using side camera for Z (height) control."""
    
    def __init__(self, pixel_to_world_z: callable, arm_controller):
        self.p2wz = pixel_to_world_z  # (px, py) → (world_x, world_z)
        self.arm = arm_controller
        self.kp = 0.3  # lower gain for Z (gravity!)
        self.tolerance_mm = 3.0
    
    def step(self, target_z_px: int, gripper_z_px: int) -> bool:
        """Control gripper height to match target."""
        target_z = self.p2wz(0, target_z_px)[1]  # Z component
        gripper_z = self.p2wz(0, gripper_z_px)[1]
        
        error_z = target_z - gripper_z
        
        if abs(error_z) < self.tolerance_mm:
            return True
        
        vz = np.clip(self.kp * error_z, -30, 30)  # slower Z moves
        self.arm.move_cartesian_delta(dx=0, dy=0, dz=vz)
        return False
```

#### Arm Camera → Fine IBVS Alignment

```python
class ArmCamServo:
    """IBVS using arm-mounted camera for fine alignment."""
    
    def __init__(self, arm_controller, image_width=1920, image_height=1080):
        self.arm = arm_controller
        self.cx = image_width // 2   # image center x
        self.cy = image_height // 2  # image center y
        self.kp = 0.02  # pixel-to-mm gain (needs calibration)
        self.tolerance_px = 20  # ~2mm at typical working distance
    
    def step(self, object_px: tuple[int,int]) -> bool:
        """Servo to center object in arm camera view."""
        error_x = object_px[0] - self.cx  # positive = object right of center
        error_y = object_px[1] - self.cy  # positive = object below center
        error_mag = np.sqrt(error_x**2 + error_y**2)
        
        if error_mag < self.tolerance_px:
            return True
        
        # Map image error to gripper motion
        # Note: arm camera axes may be rotated relative to gripper frame
        # This mapping depends on camera mounting orientation
        dx = -self.kp * error_x  # image right → gripper left (mirror)
        dy = -self.kp * error_y  # image down → gripper forward
        
        self.arm.move_cartesian_delta(dx=dx, dy=dy, dz=0)
        return False
```

### Latency Budget (100ms target)

| Stage | Time | Notes |
|-------|------|-------|
| Frame grab from buffer | 1ms | Already captured by CameraThread |
| Detection (color seg) | 3ms | For tracking mode between YOLO runs |
| Detection (YOLO) | 50ms | Every 5th frame only |
| Error computation | 1ms | Simple arithmetic |
| IK solve | 2ms | Analytical for 6-DOF |
| DDS command publish | 5ms | UDP, local |
| Arm response | 20ms | Mechanical + controller lag |
| **Total (tracking mode)** | **~32ms** | **✓ Well under budget** |
| **Total (YOLO frame)** | **~79ms** | **✓ Under budget** |

At 15fps capture, we get a new frame every 66ms. The control loop runs at frame rate when tracking (32ms compute + 34ms idle), or slightly behind on YOLO frames.

---

## 4. Coordinate Frame Transforms

### Reference Frames

```
World Frame (workspace)
├── Origin: front-left corner of workspace surface
├── X: left → right (width)
├── Y: front → back (depth)  
├── Z: up (height)
└── Units: millimeters

Overhead Camera Frame
├── Mounted above workspace, looking down (-Z)
├── Image X ≈ World X (with rotation correction)
├── Image Y ≈ World Y (with rotation correction)
└── Calibrated via checkerboard or ArUco grid

Side Camera Frame  
├── Mounted to the side, looking across (+Y or -X)
├── Image X ≈ World X or Y (depending on mounting)
├── Image Y ≈ World Z (inverted: image Y increases downward)
└── Calibrated via checkerboard or ArUco markers at known heights

Arm Base Frame
├── Origin: arm base mounting point
├── Defined by Unitree D1 DH parameters
└── Transform to world frame: fixed rigid transform (measure once)

Arm Camera Frame
├── Mounted on gripper, moves with end-effector
├── Extrinsics relative to end-effector: fixed (hand-eye calibration)
├── World pose: FK(joint_angles) × T_ee_to_cam
└── Changes every frame as arm moves
```

### Camera-to-World Calibration (Fixed Cameras)

For overhead and side cameras, calibrate once at setup:

#### Step 1: Camera Intrinsics

```python
# Standard checkerboard calibration
def calibrate_intrinsics(images: list[np.ndarray], board_size=(9,6), square_mm=25.0):
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2) * square_mm
    
    obj_points, img_points = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return K, dist  # intrinsic matrix, distortion coefficients
```

#### Step 2: Pixel-to-World Mapping (Overhead Camera)

For the overhead camera looking down at a flat workspace, use a homography:

```python
def calibrate_overhead_homography(cam_points: np.ndarray, world_points: np.ndarray):
    """
    cam_points: Nx2 pixel coordinates of known points
    world_points: Nx2 world XY coordinates (mm) of same points
    Returns homography H such that world_xy = H @ [px, py, 1]
    """
    H, mask = cv2.findHomography(cam_points, world_points)
    return H

def pixel_to_world_xy(H, px, py) -> tuple[float, float]:
    """Convert overhead camera pixel to world XY (mm)."""
    pt = H @ np.array([px, py, 1.0])
    return float(pt[0]/pt[2]), float(pt[1]/pt[2])
```

Place 4+ ArUco markers at known positions on the workspace surface. Detect them, build the homography. Accuracy: typically <2mm across the workspace.

#### Step 3: Pixel-to-World Mapping (Side Camera)

Side camera sees the XZ or YZ plane. Similar homography approach but for a vertical plane:

```python
def calibrate_side_homography(cam_points: np.ndarray, world_xz_points: np.ndarray):
    """
    world_xz_points: Nx2 array of (world_x, world_z) in mm
    Use ArUco markers at known heights (stacked vertically)
    """
    H, _ = cv2.findHomography(cam_points, world_xz_points)
    return H

def pixel_to_world_xz(H, px, py) -> tuple[float, float]:
    pt = H @ np.array([px, py, 1.0])
    return float(pt[0]/pt[2]), float(pt[1]/pt[2])  # (world_x, world_z)
```

### Arm Camera Extrinsics (Hand-Eye Calibration)

The arm-mounted camera's world pose changes as the arm moves. To know where the camera is pointing:

```
T_cam_to_world = T_base_to_world × FK(joints) × T_ee_to_cam
```

Where:
- `T_base_to_world`: Fixed transform from arm base to world frame (measured at setup)
- `FK(joints)`: Forward kinematics giving end-effector pose from joint angles
- `T_ee_to_cam`: Fixed transform from end-effector to camera (hand-eye calibration)

#### Hand-Eye Calibration Procedure

1. Mount an ArUco marker at a known fixed position in the workspace
2. Move the arm to 15–20 different poses where the arm camera can see the marker
3. At each pose, record:
   - Joint angles → compute `T_base_to_ee` via FK
   - Camera image → detect marker → compute `T_cam_to_marker` via `cv2.aruco.estimatePoseSingleMarkers`
4. Solve the `AX = XB` problem:

```python
def hand_eye_calibrate(T_base_to_ee_list, T_cam_to_marker_list):
    """
    Solve for T_ee_to_cam (the hand-eye transform).
    Uses OpenCV's calibrateHandEye.
    """
    R_gripper2base = [T[:3,:3] for T in T_base_to_ee_list]
    t_gripper2base = [T[:3,3] for T in T_base_to_ee_list]
    R_target2cam = [T[:3,:3] for T in T_cam_to_marker_list]
    t_target2cam = [T[:3,3] for T in T_cam_to_marker_list]
    
    R, t = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    T_ee_to_cam = np.eye(4)
    T_ee_to_cam[:3,:3] = R
    T_ee_to_cam[:3,3] = t.flatten()
    return T_ee_to_cam
```

### Handling the Arm Camera's Changing Pose

Every time we read a frame from the arm camera, we also read the current joint angles:

```python
def get_arm_cam_world_pose(joint_angles, T_base_to_world, T_ee_to_cam, fk_solver):
    T_base_to_ee = fk_solver.compute(joint_angles)
    T_cam_to_world = T_base_to_world @ T_base_to_ee @ T_ee_to_cam
    return T_cam_to_world
```

This gives us the camera's world pose at the moment of frame capture. Important: **synchronize joint angle reading with frame capture** — the arm may have moved between frame grab and angle read. At 15fps with typical arm speeds, the error from this lag is <1mm, acceptable for our purposes.

---

## 5. Practical Pick Sequence Using Vision

### Full Autonomous Pick-and-Place

```
IDLE
  │
  ▼
STEP 1: DETECT TARGET (Overhead cam)
  ├─ Run YOLO or color detection on overhead frame
  ├─ Identify target object → bounding box → centroid
  ├─ Convert pixel centroid to world XY via overhead homography
  ├─ Output: target_xy = (X_mm, Y_mm)
  ├─ Confidence check: detection confidence > 0.7
  └─ If no target found → SEARCH pattern (scan workspace)
  │
  ▼
STEP 2: ESTIMATE HEIGHT (Side cam)
  ├─ Detect target object in side view (same color/YOLO class)
  ├─ Get bottom edge of bounding box → pixel Y
  ├─ Convert to world Z via side homography
  ├─ Output: target_z = Z_mm (height of object top surface)
  ├─ Cross-check: side cam X should ≈ overhead cam X (±10mm)
  └─ If height unclear → use default approach height (safe, high)
  │
  ▼
STEP 3: PLAN APPROACH
  ├─ Target pose: (target_xy.x, target_xy.y, target_z + approach_clearance)
  ├─ approach_clearance = 50mm (above object)
  ├─ Check IK feasibility for approach pose
  ├─ Plan joint trajectory (current → approach pose)
  └─ If IK fails → try alternative approach angle
  │
  ▼
STEP 4: MOVE TO APPROACH (Overhead + Side monitoring)
  ├─ Execute trajectory
  ├─ Overhead cam: track gripper, verify it's approaching target XY
  ├─ Side cam: track gripper height, verify correct altitude
  ├─ Visual servo corrections if drifting (overhead servo for XY)
  └─ Stop when overhead shows gripper over target (within 10mm XY)
  │
  ▼
STEP 5: ARM CAM TAKES OVER (Fine alignment)
  ├─ Switch to arm cam feed
  ├─ Detect target object in arm cam (should be visible, roughly centered)
  ├─ If not visible → search micro-pattern (small XY oscillation)
  ├─ Compute pixel error: (object_center - image_center)
  ├─ Begin IBVS: servo to center object in arm cam view
  └─ Converged when error < 20px (~2mm at working distance)
  │
  ▼
STEP 6: DESCEND (Side cam monitoring Z)
  ├─ Lower gripper toward object, Z velocity = -10mm/step
  ├─ Side cam: track gripper Z, track gap between gripper and object
  ├─ Arm cam: maintain fine XY alignment during descent (may need correction)
  ├─ Stop conditions:
  │   ├─ Side cam shows gripper at object height (gap < 5mm)
  │   ├─ Force/current sensor detects contact
  │   └─ Safety: max descent limit reached
  └─ Output: gripper is at grasp height
  │
  ▼
STEP 7: GRASP VERIFICATION (Arm cam)
  ├─ Arm cam: verify object is between gripper fingers
  ├─ Check: object centered, correct size, no obstruction
  ├─ Close gripper
  ├─ Monitor gripper current for grasp confirmation
  ├─ Arm cam: check if object is still visible (held) or disappeared (dropped/missed)
  └─ If grasp failed → open, re-detect (STEP 1), retry (max 3 attempts)
  │
  ▼
STEP 8: LIFT & VERIFY (Side cam)
  ├─ Lift gripper Z += 50mm
  ├─ Side cam: verify gripper is rising, object is attached
  ├─ Overhead cam: verify no collisions during lift
  ├─ If object fell during lift → side cam detects separation → re-attempt
  └─ Output: object secured, ready for transport
  │
  ▼
STEP 9: TRANSPORT TO DESTINATION
  ├─ Similar to STEPS 3-4 but for destination position
  ├─ Overhead cam: track gripper to destination XY
  ├─ Side cam: maintain safe height during transit
  └─ Lower + release at destination
  │
  ▼
DONE → return to IDLE
```

### Timing for Complete Pick Sequence

| Step | Duration | Camera |
|------|----------|--------|
| 1. Detect target | 50–100ms | Overhead |
| 2. Estimate height | 50–100ms | Side |
| 3. Plan approach | 10ms | Compute |
| 4. Move to approach | 2–4s | Overhead + Side |
| 5. Fine alignment | 0.5–2s (5–20 servo steps) | Arm |
| 6. Descend | 1–3s | Side + Arm |
| 7. Grasp | 0.5–1s | Arm |
| 8. Lift | 1–2s | Side |
| **Total** | **~5–12s** | |

---

## 6. Failure Detection & Recovery

### Failure Modes & Responses

| Failure | Detection Method | Response |
|---------|-----------------|----------|
| **Object lost from overhead** | No detection for 5 consecutive frames | Pause, widen search area, spiral scan pattern |
| **Object lost from arm cam** | No detection during fine align | Back up 30mm Z, retry alignment from overhead |
| **Grasp failed** | No current spike on close / object missing from arm cam | Open gripper, rise 50mm, re-detect from overhead |
| **Arm stalled** | Joint velocity = 0 but position error > 0 for >500ms | Back off 20mm in reverse direction, re-plan path |
| **Overcurrent** | Current exceeds safe threshold on any joint | Immediately stop, retract to safe home position |
| **Collision suspected** | Unexpected force/current, or visual: gripper near obstacle | Stop, retract 50mm Z, re-plan with obstacle avoidance |
| **Cameras disagree** | Overhead X vs Side X differ by >15mm | Stop, re-calibrate or use higher-confidence camera only |
| **Camera feed lost** | CameraThread.connected = False, no-signal frame | Fall back to remaining cameras, or pause and alert |

### Search Pattern (Object Lost)

```python
def spiral_search(arm, center_xy, max_radius=100, step=20):
    """Move arm in expanding spiral to find lost object."""
    for r in range(step, max_radius, step):
        for angle in np.linspace(0, 2*np.pi, max(8, int(r/5))):
            x = center_xy[0] + r * np.cos(angle)
            y = center_xy[1] + r * np.sin(angle)
            arm.move_to_xy(x, y)
            time.sleep(0.2)  # Wait for camera frame
            detection = detect_in_overhead()
            if detection:
                return detection
    return None  # Object truly lost
```

### Retry Policy

```python
MAX_PICK_ATTEMPTS = 3
GRASP_RETRY_OFFSET_MM = 5  # Shift position slightly on retry

for attempt in range(MAX_PICK_ATTEMPTS):
    result = execute_pick(target)
    if result.success:
        break
    
    # Adjust approach on retry
    if result.failure == 'grasp_missed':
        target.x += random.uniform(-GRASP_RETRY_OFFSET_MM, GRASP_RETRY_OFFSET_MM)
        target.y += random.uniform(-GRASP_RETRY_OFFSET_MM, GRASP_RETRY_OFFSET_MM)
    elif result.failure == 'object_lost':
        target = search_for_object()  # Re-detect from scratch
    elif result.failure == 'collision':
        plan_alternative_approach(target)  # Different angle
```

---

## 7. Implementation Phases

### Phase 1: Single-Camera Visual Servo (Overhead Only, XY) — 1–2 weeks

**Goal:** Move gripper to any XY position on the workspace using overhead camera feedback.

1. Calibrate overhead camera (checkerboard → intrinsics, ArUco grid → homography)
2. Implement gripper detection in overhead view (color segmentation — the gripper has distinctive color)
3. Implement object detection in overhead view (color or YOLO)
4. Build `OverheadServo` class with P controller
5. Test: place colored object → arm moves gripper to hover above it
6. Tune gains, measure accuracy

**Success criteria:** Gripper positions within 5mm of target XY, measured with ruler.

### Phase 2: Dual-Camera (Overhead + Side for XYZ) — 1–2 weeks

**Goal:** Add Z-axis control using side camera.

1. Calibrate side camera (intrinsics + homography for XZ plane)
2. Implement height detection: gripper Z and object Z from side view
3. Build `SideServo` class for Z control
4. Integrate overhead XY servo + side Z servo into coordinated controller
5. Test: move gripper to specific 3D positions
6. Implement approach sequence (move XY → descend Z)

**Success criteria:** Gripper reaches 3D target within 5mm in all axes.

### Phase 3: Tri-Camera with Arm Cam for Fine Alignment — 2–3 weeks

**Goal:** Use arm-mounted camera for precision grasping.

1. Perform hand-eye calibration for arm camera
2. Implement object detection in arm cam view (close-range, high detail)
3. Build `ArmCamServo` for IBVS fine alignment
4. Integrate full 3-camera pipeline: overhead→side→arm cam handoff
5. Implement grasp verification (arm cam checks gripper-object alignment)
6. Test: pick up objects of various sizes

**Success criteria:** Successfully pick known objects >60% of the time.

### Phase 4: Autonomous Pick-and-Place Loop — 2–3 weeks

**Goal:** Full autonomous operation with error recovery.

1. Implement failure detection for all modes
2. Implement retry logic and search patterns
3. Build state machine for complete pick-place sequence
4. Add destination placement (reverse of pick)
5. Multi-object handling (detect all objects, pick sequentially)
6. Endurance testing (run for 30+ minutes continuously)

**Success criteria:** >80% pick success rate, handles failures gracefully, runs unattended.

---

## 8. Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Detection latency (color) | <10ms | Timer around detection function |
| Detection latency (YOLO) | <50ms | Timer around model.predict() |
| Control loop period | <100ms | Timestamp delta between servo steps |
| XY positioning accuracy | <5mm | ArUco marker as target, measure offset |
| Z positioning accuracy | <5mm | Side camera measurement |
| Fine alignment accuracy | <3mm | Arm cam pixel error → mm conversion |
| Pick success (known objects) | >80% | 50-trial test, count successes |
| Pick success (novel objects) | >50% | 20-trial test with unseen objects |
| End-to-end pick time | <15s | Timer from detection to confirmed grasp |
| System uptime | >95% | Run for 1h, measure downtime |

### Latency Breakdown Budget

```
100ms total budget:
┌─────────────────────────────────┐
│ Frame from buffer:     1ms      │
│ Preprocessing:         2ms      │
│ Detection:            5-50ms    │
│ Error computation:     1ms      │
│ IK solve:              2ms      │
│ DDS command:           5ms      │
│ Network/overhead:      5ms      │
│ ────────────────────────────    │
│ Total:               21-66ms    │
│ Margin:              34-79ms    │
└─────────────────────────────────┘
```

---

## 9. Libraries & Dependencies

### Core

| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.9+ | Camera capture, image processing, calibration, ArUco |
| `numpy` | 1.26+ | Array operations, transforms |
| `ultralytics` | 8.1+ | YOLOv8 object detection |
| `scipy` | 1.12+ | Optimization, spatial transforms |

### Optional / Phase-Dependent

| Library | Purpose | Phase |
|---------|---------|-------|
| `open3d` | Point cloud visualization, 3D reconstruction | Future |
| `onnxruntime` | YOLO inference with OpenCL acceleration | Phase 3+ |
| `filterpy` | Kalman filter implementations | Phase 2+ |
| `transforms3d` | Rotation/transform utilities | Phase 2+ |

### Already Available in Project

Based on existing `camera_server.py` and `server.py`:
- `cv2` (OpenCV) — already used for capture and encoding
- `numpy` — already imported
- `threading` — already used for camera threads
- FastAPI + WebSocket — already in server.py

### GPU Acceleration Feasibility (RX 580)

| Capability | Status | Notes |
|------------|--------|-------|
| OpenCV OpenCL (UMat) | ✅ Works | `OPENCV_OPENCL_DEVICE=:GPU:0` already set in server.py |
| YOLO via PyTorch + ROCm | ❌ Not viable | GFX803 dropped from ROCm 5+ |
| YOLO via ONNX + OpenCL | ⚠️ Experimental | May work, needs testing with onnxruntime OpenCL EP |
| Image preprocessing on GPU | ✅ Works | resize, cvtColor, threshold, morphology via UMat |
| JPEG encoding on GPU | ❌ No path | AMD has no GPU JPEG encoder API on Linux |
| H.264 encoding (VA-API) | ✅ Works | Via ffmpeg, not needed for current plan |

**Bottom line:** Use the RX 580 for OpenCV image processing operations (UMat). Run YOLO on CPU with the nano model. CPU at ~50ms per YOLO inference is fine given the 100ms loop budget and the tracking-between-detections strategy.

### Installation

```bash
pip install opencv-python-headless numpy ultralytics scipy
# Optional:
pip install open3d filterpy transforms3d onnxruntime
```

---

## Appendix A: Calibration Checklist

Before first autonomous operation:

- [ ] Print checkerboard pattern (9×6 inner corners, 25mm squares)
- [ ] Print 4+ ArUco markers (DICT_4X4_50, IDs 0–3, 50mm size)
- [ ] Place ArUco markers at known positions on workspace (measure with ruler)
- [ ] Calibrate overhead camera intrinsics (20+ checkerboard images)
- [ ] Calibrate overhead homography (ArUco grid on workspace surface)
- [ ] Calibrate side camera intrinsics
- [ ] Calibrate side homography (ArUco markers at known heights)
- [ ] Measure arm base position relative to world frame origin
- [ ] Perform hand-eye calibration for arm camera (15+ poses)
- [ ] Validate: command gripper to known world position, measure actual position
- [ ] Save all calibration data to `config/calibration.json`

## Appendix B: Quick Accuracy Test

```python
def test_overhead_accuracy(arm, overhead_servo, test_points_world):
    """Move gripper to known world positions, measure visual error."""
    errors = []
    for target_xy in test_points_world:
        # Move to target using FK (no vision)
        arm.move_to_cartesian(target_xy[0], target_xy[1], z=100)
        time.sleep(1.0)
        
        # Detect gripper in overhead cam
        gripper_px = detect_gripper_overhead()
        gripper_world = pixel_to_world_xy(H_overhead, *gripper_px)
        
        error = np.linalg.norm(np.array(target_xy) - np.array(gripper_world))
        errors.append(error)
        print(f"Target: {target_xy}, Measured: {gripper_world}, Error: {error:.1f}mm")
    
    print(f"Mean error: {np.mean(errors):.1f}mm, Max: {np.max(errors):.1f}mm")
    return errors
```
