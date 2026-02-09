# Contact Detection Plan for Unitree D1 Arm

**Date:** 2026-02-08  
**Status:** Draft

## 1. What Data the D1 Actually Provides

### Current DDS Interface (confirmed from codebase)

The D1 communicates via CycloneDDS using a JSON-string IDL type (`ArmString_`) on two topics:

| Topic | Direction | Content |
|-------|-----------|---------|
| `rt/arm_Feedback` | Arm → Host | JSON with `funcode`, `seq`, `data` |
| `rt/arm_Command` | Host → Arm | JSON with `funcode`, `seq`, `data`, `address` |

**Feedback funcodes observed:**
- **funcode 1** — Joint angles: `{angle0..angle6}` (degrees). This is the primary state feedback.
- **funcode 3** — Status: `{enable_status, power_status, error_status, recv_status, exec_status}`

**What is NOT available in feedback:**
- Motor torques/currents — **not returned**
- Joint velocities — **not returned** (must be computed from angle differences)
- Motor temperature — **not returned**
- Per-motor error/overcurrent codes — **not returned** (only aggregate `error_status`)

### Unitree SDK2 Architecture

The `unitree_sdk2_python` package uses CycloneDDS with structured IDL types for the quadruped robots (Go2, B2, etc.), where `LowState_` includes per-motor `q` (position), `dq` (velocity), `tauEst` (estimated torque), `temperature`, and fault flags. However, the **D1 arm uses a simplified JSON-string protocol** (`ArmString_`), not the structured motor-level IDL. This is a deliberate design choice by Unitree — the D1's onboard controller handles motor-level control internally and exposes only a high-level position interface.

### Possible Additional DDS Topics

The D1 firmware *may* publish additional topics not currently subscribed to. Worth investigating empirically:
- `rt/arm_State` — possible extended state topic
- `rt/arm_MotorState` — possible per-motor data  
- `rt/arm_Error` — possible error/fault topic

**Recommendation:** Use `ddsspy` or CycloneDDS discovery to enumerate all topics the D1 publishes. This is the single highest-value investigation item.

```bash
# Install and run ddsspy to discover all DDS topics on the arm's network
# (requires cyclonedds-tools or ddsperf)
pip install cyclonedds-tools
ddsspy --interface eno1
```

### The `error_status` Field

The `error_status` from funcode 3 likely encodes bitfield flags for overcurrent, stall, communication errors, etc. **This needs reverse engineering** — send the arm into known stall conditions and observe the error_status value. Even a binary "error vs no error" signal would be valuable for contact detection.

## 2. ContactDetector Architecture

```
┌─────────────────────────────────────────────┐
│              ContactDetector                 │
│                                              │
│  ┌───────────────┐  ┌────────────────────┐  │
│  │ AngleBuffer    │  │ ErrorStatusMonitor │  │
│  │ (ring buffer   │  │ (watches funcode 3)│  │
│  │  per joint,    │  │                    │  │
│  │  ~100 samples) │  └────────────────────┘  │
│  └───────┬───────┘                           │
│          │                                   │
│  ┌───────▼───────────────────────────────┐  │
│  │         Detection Methods              │  │
│  │  1. TrackingErrorDetector              │  │
│  │  2. VelocityStallDetector              │  │
│  │  3. ErrorStatusDetector                │  │
│  │  4. OscillationDetector                │  │
│  └───────┬───────────────────────────────┘  │
│          │                                   │
│  ┌───────▼───────┐  ┌────────────────────┐  │
│  │ ContactEvent   │  │ Per-Joint State    │  │
│  │ (joint, type,  │→ │ Machine: FREE →   │  │
│  │  confidence,   │  │ SUSPECTED →        │  │
│  │  timestamp)    │  │ CONFIRMED →        │  │
│  └───────────────┘  │ RELEASED           │  │
│                      └────────────────────┘  │
└──────────────────────┬──────────────────────┘
                       │ callbacks
          ┌────────────▼────────────┐
          │  SafetyMonitor / Servo  │
          └─────────────────────────┘
```

### Core Class

```python
class ContactDetector:
    """Detects contact/resistance on D1 arm joints using position-only feedback."""
    
    def __init__(self, config: ContactConfig):
        self._buffers: Dict[int, RingBuffer]  # per-joint angle history
        self._targets: Dict[int, float]        # current commanded targets
        self._state: Dict[int, ContactState]   # per-joint state machine
        self._callbacks: List[Callable]         # on_contact, on_release
        self._config = config
    
    def update_feedback(self, angles: np.ndarray, timestamp: float):
        """Called every feedback cycle (~10Hz). Core detection loop."""
        
    def update_target(self, joint_id: int, target_deg: float):
        """Called when a new command is sent to a joint."""
        
    def update_status(self, status: Dict[str, int]):
        """Called on funcode 3 status feedback."""
        
    def get_contact_state(self, joint_id: int) -> ContactState:
        """Query current contact state for a joint."""
        
    def on_contact(self, callback: Callable[[ContactEvent], None]):
        """Register callback for contact events."""

@dataclass
class ContactConfig:
    tracking_error_threshold_deg: float = 3.0   # error to trigger suspicion
    tracking_error_confirm_deg: float = 5.0     # error to confirm contact
    stall_velocity_threshold: float = 0.1       # deg/s — below this = stalled
    confirm_duration_s: float = 0.3             # how long error must persist
    release_threshold_deg: float = 1.0          # error below this = released
    buffer_size: int = 100                       # ~10s at 10Hz

class ContactState(Enum):
    FREE = "free"           # No contact detected
    SUSPECTED = "suspected" # Error rising, not yet confirmed
    CONFIRMED = "confirmed" # Definite contact/stall
    RELEASED = "released"   # Was in contact, now free (transient)
```

## 3. Detection Methods (Ranked by Reliability)

### Method 1: Tracking Error with Temporal Confirmation (★★★★★)
**Most reliable. Implement first.**

The core signal: `|target_angle - actual_angle|` persists above threshold for N consecutive cycles.

```
Detection logic:
1. error = |commanded[j] - actual[j]|
2. If error > threshold for > confirm_duration → CONFIRMED
3. If error < release_threshold → FREE
4. Hysteresis prevents oscillation between states
```

**Why it works:** When a joint hits resistance, the motor controller saturates — it commands maximum effort but the joint can't move. The position error grows and plateaus. This is fundamentally different from normal motion where the error is transient (ramping toward target).

**Tuning:** The threshold must be above normal tracking lag (~1-2° during smooth motion at our speeds) but below the error that indicates real obstruction (~5°+). Start conservative at 5° confirmation threshold.

**False positive sources:**
- Large commanded jumps (error is large during normal ramp) — mitigate by only triggering when velocity also drops to ~0
- Gripper closing on empty air vs object — gripper (joint 6) needs separate thresholds

### Method 2: Velocity Stall Detection (★★★★☆)
**Second most reliable. Combine with Method 1.**

Compute velocity from consecutive angle readings: `v[j] = (angle[j][t] - angle[j][t-1]) / dt`

A stalled joint shows: `error > threshold AND |velocity| < stall_threshold`

This eliminates false positives from large commanded steps (during normal motion, error is high but velocity is also high).

```python
def _check_stall(self, joint_id: int) -> bool:
    buf = self._buffers[joint_id]
    if len(buf) < 3:
        return False
    velocity = (buf[-1] - buf[-3]) / (2 * self._dt)  # central difference
    error = abs(self._targets[joint_id] - buf[-1])
    return error > self._config.tracking_error_threshold_deg and abs(velocity) < self._config.stall_velocity_threshold
```

### Method 3: Error Status Monitoring (★★★☆☆)
**Potentially very reliable, but needs reverse engineering.**

The `error_status` field from funcode 3 may contain overcurrent or stall flags. This would be the gold standard if available.

**Action items:**
1. Log `error_status` continuously during normal operation → establish baseline (likely 0)
2. Deliberately stall a joint against a known obstacle → observe error_status changes
3. Document the bitfield mapping

### Method 4: Position Oscillation Detection (★★☆☆☆)
**Supplementary. Detects servo hunting near obstacles.**

When a joint is near an obstacle, the position controller may oscillate (overshoot, hit obstacle, reverse, approach again). Detect this via frequency analysis of the position signal.

```python
def _check_oscillation(self, joint_id: int) -> bool:
    buf = self._buffers[joint_id]
    if len(buf) < 20:
        return False
    # Count zero-crossings of the velocity signal
    velocities = np.diff(buf[-20:])
    sign_changes = np.sum(np.abs(np.diff(np.sign(velocities))) > 0)
    return sign_changes > 6  # High frequency = hunting
```

### Method 5: Expected vs Actual Trajectory Comparison (★★☆☆☆)
**More sophisticated but requires trajectory model.**

Given the commanded smoother trajectory, predict where the joint *should* be at each timestep (exponential decay model from `CommandSmoother`). Compare predicted vs actual. Divergence beyond model error indicates external resistance.

This is more work to implement but handles the "normal motion" false positive problem elegantly.

## 4. Integration with Visual Servo and Pick Pipeline

### Integration Points

#### 4.1 CommandSmoother Integration
The `CommandSmoother` already runs at 10Hz and knows both target and current positions. It's the natural place to feed the `ContactDetector`:

```python
# In CommandSmoother._tick():
if self._contact_detector:
    angles = [self._current[i] for i in range(self._num_joints)]
    self._contact_detector.update_feedback(np.array(angles), time.monotonic())
    for jid in self._dirty_joints:
        if self._target[jid] is not None:
            self._contact_detector.update_target(jid, self._target[jid])
```

#### 4.2 Visual Servo Integration
During visual servoing toward a grasp pose:
- **On SUSPECTED contact:** Reduce servo gain (slow down approach)
- **On CONFIRMED contact:** Stop servo, report "object contacted" or "unexpected obstacle"
- **During grasp:** CONFIRMED contact on gripper (joint 6) = successful grip detection

```python
class VisualServo:
    def step(self):
        contact = self._contact_detector.get_contact_state(self._active_joint)
        if contact == ContactState.CONFIRMED:
            if self._phase == ServoPhase.APPROACH:
                self._handle_obstacle_contact()
            elif self._phase == ServoPhase.GRASP:
                self._handle_grasp_success()
```

#### 4.3 Pick Pipeline Integration
```
APPROACH → (contact on arm joints?) → STOP / REPLAN
         → (near target, no contact) → GRASP
GRASP    → close gripper → (contact on J6?) → GRIP CONFIRMED → LIFT
         → (no contact on J6 after timeout) → GRIP FAILED → RETRY/ABORT
LIFT     → (contact on arm joints during lift?) → SNAGGED → E-STOP
```

#### 4.4 Collision Memory Integration
Replace the current `CollisionMemory` (which was too aggressive because it permanently banned angles) with a softer approach:

- **Transient obstacles:** Contact detected → stop → back off 5° → retry once → if contact again, mark as semi-permanent (expires after N minutes)
- **Permanent obstacles:** Only ban angles after 3+ contacts at the same location within a session
- **Workspace change:** Clear on user command (keep existing `clear()` method)

## 5. Safety Implications

### E-Stop on Unexpected Contact

```python
class ContactSafetyPolicy:
    """Decides what to do when contact is detected."""
    
    def on_contact(self, event: ContactEvent):
        if event.joint_id <= 5:  # Arm joints (not gripper)
            if self._phase == Phase.IDLE:
                # Unexpected — something bumped the arm
                self._safety_monitor.trigger_estop("unexpected_contact")
            elif self._phase in (Phase.SERVO, Phase.APPROACH):
                # Could be the target object or an obstacle
                self._stop_motion()  # Stop but don't e-stop
                self._notify_planner(event)
            elif self._phase == Phase.LIFT:
                # Snagged on something — dangerous
                self._safety_monitor.trigger_estop("snag_during_lift")
        elif event.joint_id == 6:  # Gripper
            if self._phase == Phase.GRASP:
                # Expected — gripping object
                pass  # Normal operation
            else:
                self._stop_motion()
```

### Force Limiting via Position Control

Since we can't set torque limits directly, we implement "soft" force limiting:
1. When tracking error exceeds a "force limit" threshold (e.g., 8°), immediately retract the target to the current actual position
2. This prevents the motor from applying increasing force against an obstacle
3. The threshold maps roughly to a force level (requires empirical calibration)

```python
FORCE_LIMIT_ERROR_DEG = 8.0  # ~approximate force limit

def _enforce_force_limit(self, joint_id, target, actual):
    error = abs(target - actual)
    if error > FORCE_LIMIT_ERROR_DEG:
        # "Yield" — retract target to actual position
        self._smoother.set_joint_target(joint_id, actual)
        logger.warning("Force limit: J%d yielding (error=%.1f°)", joint_id, error)
```

### Immediate Action Items

1. **DDS Topic Discovery** (30 min) — Run `ddsspy` to find any hidden topics
2. **Error Status Reverse Engineering** (1 hr) — Log error_status during stall conditions
3. **Implement ContactDetector v1** (2 hr) — Methods 1+2 (tracking error + velocity stall)
4. **Integrate with CommandSmoother** (1 hr) — Feed detector from smoother tick loop
5. **Tune Thresholds** (ongoing) — Empirical testing with real arm
6. **Gripper Contact Detection** (1 hr) — Specialized thresholds for grip confirmation

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| False positive → unnecessary stop | Low (annoying) | Conservative thresholds, require temporal confirmation |
| False negative → missed contact | High (hardware damage) | Multiple detection methods, low threshold for e-stop-worthy events |
| Threshold too aggressive (like old CollisionMemory) | Medium | Hysteresis, temporal filtering, no permanent bans without multiple confirmations |
| Feedback latency (DDS at 10Hz) | Medium | At 10Hz, 100ms between samples. A fast collision may not be caught. Accept this limitation — the D1 is slow enough that this is usually OK. |
| Position data noise | Low | The D1 encoders are precise. Noise floor is <0.1°. |

## Appendix: DDS Topic Discovery Script

```python
"""Run on the arm's network to discover all DDS topics."""
import time
from cyclonedds.domain import DomainParticipant
from cyclonedds.builtin import DcpsParticipant, DcpsTopic, DcpsEndpoint
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

dp = DomainParticipant()

# Built-in topic readers
topic_reader = DataReader(dp, Topic(dp, "DCPSTopic", DcpsTopic))

print("Listening for DDS topics for 10 seconds...")
time.sleep(2)

for sample in topic_reader.take(N=100):
    print(f"Topic: {sample.topic_name}  Type: {sample.type_name}")
```
