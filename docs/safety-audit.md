# Safety & Operations Audit — th3cl4w / Unitree D1

**Date:** 2026-02-08  
**Auditor:** Safety & Operations Reviewer (automated)  
**System:** Unitree D1 7-DOF robotic arm, 550mm reach, 1kg payload  
**Control:** DDS over UDP at 500Hz (feedback), 10Hz command smoother, FastAPI web server  

---

## 1. Critical Safety Gaps

These issues could damage the arm, its surroundings, or injure a person.

### 1.1 Safety Monitor Is Not In the Command Path

**Severity: CRITICAL**

The `SafetyMonitor` class (`src/safety/safety_monitor.py`) exists and is well-designed, but **it is never instantiated or called in the web server** (`web/server.py`). Commands go:

```
HTTP API → server.py validation → CommandSmoother → D1DDSConnection.set_joint()
```

The `SafetyMonitor.validate_command()` and `SafetyMonitor.check_state()` methods are never invoked. The monitor's e-stop, torque limits, velocity limits, and workspace bounds are **completely bypassed** because the monitor isn't wired into the pipeline.

**Impact:** All safety checks in `safety_monitor.py` are dead code in production.

### 1.2 No Torque or Velocity Monitoring

**Severity: CRITICAL**

The DDS feedback (`d1_dds_connection.py`) only parses joint angles and status. Velocities and torques are **hardcoded to zero** in `get_state()`:

```python
joint_velocities=np.zeros(NUM_JOINTS, dtype=np.float64),
joint_torques=np.zeros(NUM_JOINTS, dtype=np.float64),
```

Even if the safety monitor were connected, it could never detect torque or velocity violations because that data is never read from the arm. The overcurrent trip that killed power would have been **undetectable** by the current system.

### 1.3 Emergency Stop Doesn't Guarantee Motor Stop

**Severity: CRITICAL**

The `/api/command/stop` endpoint sends two sequential DDS commands: `disable_motors()` then `power_off()`. These are fire-and-forget UDP publishes with **no acknowledgment verification**. If either packet is lost (UDP has no delivery guarantee), the arm continues moving.

There is no:
- Hardware e-stop integration
- Watchdog timer (arm should stop if no command received within N ms)
- Verification that the arm actually stopped after e-stop

### 1.4 Stale Feedback Can Mask Dangerous States

**Severity: CRITICAL**

Known issue: DDS feedback returns stale 0.0° readings. The system caches last-known-good values (`_cached_joint_angles`) when feedback is `None`, but there's no mechanism to:
- Halt commands when feedback is stale beyond a threshold
- Alert when feedback age exceeds tolerance
- Prevent the smoother from continuing to send commands into the void

The `is_feedback_fresh()` method exists in `D1DDSConnection` but is **never called** by the server or smoother.

### 1.5 No Multi-Joint Torque/Current Coordination

**Severity: HIGH**

The overcurrent trip occurred when commanding J1=30° with J2=70° + J4=80° simultaneously. There is **no combined torque budget** — each joint is validated independently. The system has no concept of:
- Total power draw across all motors
- Gravitational torque at extended configurations
- Configurations where multiple high-torque joints compound load

### 1.6 Collision Detector Disabled

**Severity: HIGH**

The collision detector is explicitly disabled in the server startup:

```python
collision_detector.enabled = False
logger.info("Collision detector initialized but DISABLED (too aggressive for real arm)")
```

This means the arm has **zero collision detection** in production. The 3° threshold was too aggressive, but instead of tuning it, it was turned off entirely.

---

## 2. Operational Risks

### 2.1 No Rate Limiting on API Endpoints

Any client can send unlimited rapid-fire commands to `/api/command/set-joint`. While the `CommandSmoother` buffers at 10Hz, the API itself has no rate limiting. A misbehaving client or script could:
- Overwhelm the server with requests
- Set conflicting targets faster than the smoother can track
- Cause unpredictable motion if targets oscillate rapidly

### 2.2 CORS Allows All Origins

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

Any webpage on the internet can send commands to the arm if the server is network-accessible. This is a remote code execution risk on a physical robot.

### 2.3 No Authentication

Zero authentication on any endpoint. Anyone on the network can power on, enable, and command the arm. Combined with CORS *, this is exploitable from any browser.

### 2.4 Server Crash During Motion

If the FastAPI server crashes while the arm is moving (e.g., unhandled exception, OOM), the smoother's `asyncio.Task` is killed. The arm's last-received DDS command continues executing. There is no watchdog on the arm side to detect loss of the control connection.

### 2.5 Task Cancellation Doesn't Stop Motion

`/api/task/stop` cancels the asyncio task but doesn't send a stop command to the arm. The smoother continues interpolating toward whatever target was last set. The arm keeps moving after "stop."

### 2.6 `set_all_joints()` Known to Cause Freezes

From lessons learned: "set_all_joints (funcode 2) causes freezes on large moves." The server still uses `set_all_joints` when ≥3 joints change in a single smoother tick. This codepath is actively dangerous.

### 2.7 No Log Rotation or Persistent Logging

The `ActionLog` is an in-memory deque of 200 entries. When the server restarts, all logs are lost. Telemetry goes to SQLite but there's no rotation, size limit, or alerting.

---

## 3. Missing Safeguards

| Safeguard | Status | Priority |
|-----------|--------|----------|
| Safety monitor in command pipeline | **Missing** | P0 |
| Feedback freshness gate (halt if stale) | **Missing** | P0 |
| Hardware e-stop integration | **Missing** | P0 |
| DDS command acknowledgment verification | **Missing** | P0 |
| Combined torque/current budget | **Missing** | P1 |
| API authentication | **Missing** | P1 |
| API rate limiting | **Missing** | P1 |
| Watchdog timer (arm-side) | **Missing** | P1 |
| Collision detector (tuned, not disabled) | **Disabled** | P1 |
| Startup self-test / calibration check | **Missing** | P2 |
| Graceful degradation on feedback loss | **Missing** | P2 |
| Pre-move workspace collision check | Available but opt-in | P2 |
| Power draw monitoring | **Missing** | P2 |
| Persistent structured logging | Partial (SQLite) | P3 |
| Documented operational procedures | **Missing** | P3 |

---

## 4. Recommended Safety Improvements (Prioritized)

### P0 — Must Fix Before Next Operation

#### 4.1 Wire SafetyMonitor Into Command Pipeline
Create a singleton `SafetyMonitor` in server lifespan. Validate every command through it before the smoother or DDS connection sends anything. Reject unsafe commands with HTTP 400/409.

```python
# In server.py lifespan:
from src.safety.safety_monitor import SafetyMonitor, d1_default_limits
safety = SafetyMonitor(d1_default_limits())

# In every command endpoint, before sending:
# Convert degrees to radians, validate, reject if unsafe
```

#### 4.2 Feedback Freshness Gate
In the smoother's `_tick()`, check `arm.is_feedback_fresh(max_age_s=1.0)`. If stale, stop sending commands and log an error. Resume only when fresh feedback returns.

#### 4.3 E-Stop Verification Loop
After sending e-stop commands, poll feedback for up to 2 seconds to verify motors are actually disabled. If not confirmed, retry and alert.

#### 4.4 Add Dangerous Configuration Detection
Before any multi-joint move, compute approximate gravitational torque for the target configuration. Reject configurations where estimated total torque exceeds 80% of the overcurrent threshold. Start with a simple heuristic: if `abs(J1) + abs(J2) > 120°` or `abs(J2) + abs(J4) > 130°`, reject.

### P1 — Fix Soon

#### 4.5 Tune and Re-enable Collision Detector
Increase `position_error_deg` from 3° to 8°, increase `stall_duration_s` to 1.5s. This avoids false positives from normal motion lag while still catching real stalls.

#### 4.6 Add Basic API Authentication
At minimum, a shared secret in a header (`X-API-Key`). Restrict CORS to known origins (localhost, the NUC's IP).

#### 4.7 Rate Limit Command Endpoints
Max 20 requests/second per endpoint using FastAPI middleware or `slowapi`.

#### 4.8 Stop Using `set_all_joints()` in Smoother
Always use individual `set_joint()` calls. The batch command is known to cause freezes.

### P2 — Important but Not Urgent

#### 4.9 Add Startup Self-Test
On server start, read all joint positions and verify they're within expected ranges. Check feedback freshness. Log warnings if any joint reads 0.0° (likely stale).

#### 4.10 Integrate Pre-Move Collision Preview
Make `CollisionPreview` mandatory before trajectory execution, not opt-in.

---

## 5. Operational Runbook

### 5.1 Startup Procedure

1. **Physical check:** Verify arm workspace is clear of people and obstacles within 60cm radius
2. **Power on controller:** Ensure D1 controller box has power, Ethernet cable connected to NUC on `eno1`
3. **Start camera server:** `cd th3cl4w && python web/camera_server.py`
4. **Start control server:** `cd th3cl4w && python web/server.py --interface eno1`
5. **Verify DDS connection:** Check server logs for "DDS connected on eno1"
6. **Verify feedback:** Open web UI, confirm joint angles are non-zero and updating
7. **Power on arm:** Click Power On in UI, wait for confirmation
8. **Enable motors:** Click Enable in UI, wait for confirmation
9. **Test with small move:** Command J0 to +5°, verify it moves and feedback updates
10. **Begin operations**

### 5.2 Shutdown Procedure

1. **Return to home:** Execute Go Home task or manually return all joints to ~0°
2. **Disable motors:** Click Disable in UI
3. **Power off arm:** Click Power Off in UI
4. **Verify power off:** Confirm status shows power=OFF in feedback
5. **Stop server:** Ctrl+C or kill the process
6. **Stop camera server:** Ctrl+C

### 5.3 Emergency Procedures

#### Software E-Stop
1. Click STOP button in web UI (or `POST /api/command/stop`)
2. **Verify** arm has stopped moving visually
3. If arm continues moving: **pull the power cable from the D1 controller box**

#### Overcurrent Trip
1. The arm will lose power automatically — this is the hardware protection working
2. **Do not** immediately re-power. Wait 30 seconds for capacitors to discharge
3. Check for mechanical obstructions or entanglement
4. Review telemetry logs for the joint configuration that caused the trip
5. Re-power and return to home position before attempting any new moves

#### Network Loss
1. The arm will continue executing its last received command
2. If the arm is moving and you lose the web UI: **pull the Ethernet cable from the D1 controller** to cut the DDS link, then pull power if needed
3. After reconnection, verify joint positions match expected values before re-enabling

#### Unexpected Motion
1. **Step back** — do not reach into the arm's workspace
2. Press E-Stop (software) or pull power
3. The D1's motors have enough force to injure fingers at close range

### 5.4 Operating Rules

1. **Never** command more than 10° per step on any joint
2. **Never** extend the elbow (J2 > 60°) while shoulder (J1) is above 30°
3. **Never** command J1, J2, or J4 beyond ±80° (5° margin from hardware limits)
4. **Always** verify DDS feedback is fresh before starting a sequence
5. **Always** have physical access to the power cable during operation
6. **Never** operate the arm with people within 60cm of the workspace
7. **Never** leave the arm unattended while motors are enabled

---

## 6. Incident Analysis: Overcurrent Trip

### What Happened
During a Red Bull can pickup attempt on 2026-02-07, commands were sent to position J1=30°, J2=70°, and J4=80° simultaneously. This caused an overcurrent trip that killed power to the arm.

### Root Cause
The D1 has a combined current limit across all motors. When multiple joints are under high gravitational load simultaneously (shoulder raised, elbow extended, wrist pitched), the total current draw exceeds the controller's protection threshold. Specifically:
- J1 at 30° requires significant holding torque against gravity
- J2 at 70° extends the arm nearly horizontal, creating maximum moment arm
- J4 at 80° adds wrist torque to an already-loaded kinematic chain
- The combined current draw tripped the hardware overcurrent protection

### Contributing Factors
1. **No software-side current/torque budget** — each joint validated independently
2. **Used `set_all_joints()`** — sent all positions simultaneously instead of sequentially
3. **Pushed past safe operating envelope** — J2=70° + J4=80° is near the limit of safe configurations
4. **No feedback monitoring** — the system couldn't detect rising current before the trip
5. **Stale feedback** — DDS was returning 0.0° readings, masking the actual arm state

### Prevention
1. **[Implemented]** Soft limits of ±80° on J1/J2/J4 in server.py
2. **[Implemented]** Max 10° step limit in CommandSmoother (15° currently — should reduce)
3. **[NOT implemented]** Combined torque budget / dangerous configuration detection
4. **[NOT implemented]** Feedback freshness gate to halt commands on stale data
5. **[NOT implemented]** Reduce smoother `max_step_deg` from 15° to 10° to match operating rules
6. **[Recommendation]** Add a "configuration danger score" based on gravitational torque estimates; reject moves that would exceed 70% of known overcurrent threshold

### Smoother max_step_deg Mismatch
The operating rule says ≤10° per command, but `CommandSmoother` is configured with `max_step_deg=15.0`. This should be reduced to 10.0 to match the documented rule.

---

## Summary

The th3cl4w system has a well-designed `SafetyMonitor` class that is **not connected to the actual command pipeline**. The most critical action is wiring it in. The second most critical action is gating commands on feedback freshness. The third is adding combined torque/configuration limits to prevent another overcurrent trip.

The system currently operates with **zero runtime safety checks** beyond basic angle range validation in the API endpoint handlers. For a real robot that has already tripped its overcurrent protection, this is unacceptable.

**Recommended action:** Do not operate the arm until items 4.1 and 4.2 are implemented.
