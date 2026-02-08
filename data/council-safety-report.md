# Council Safety Report — Systems/Safety Specialist
**Date:** 2026-02-07  
**Subject:** Calibration Collision Analysis & Safety Recommendations  
**Author:** Safety Specialist (Council Member)

---

## Executive Summary

59 collision events were logged during calibration, with 39 (66%) showing `actual=0.0°` — a telemetry bug, not real collisions. The collision detector is correctly designed but is being fed stale/zero feedback data, causing massive false-positive rates. The calibration routine also lacks any workspace pre-checking, running blind in a cluttered garage workshop.

---

## 1. Why 40+ Collisions Occurred During Calibration

### Root Cause: Stale/Zero Joint Feedback (False Positives)

The collision data tells a clear story. Of 59 events examined:

- **39 events (66%)** have `actual_deg = 0.0` across multiple joints (J1, J2, J4, J5). This is physically impossible — it would mean every joint simultaneously returned to exactly 0.0° at the moment of "collision."
- **Only ~20 events** show non-zero actual positions (e.g., `commanded=-40.3, actual=-48.1` or `commanded=-40.0, actual=-12.5`), which could be genuine stalls.

The `actual=0.0` pattern strongly indicates **the joint state feedback is returning zeros** — either:
1. The state query (`get_arm_state()`) is returning a stale/default-initialized response
2. The WebSocket state stream hasn't received an update yet
3. The API returns positions in a different frame/format than what the detector expects

When the detector sees `commanded=30.0°, actual=0.0°`, the error is 30° (>>3° threshold), it waits 0.5s, still sees 0.0°, and fires a stall event. **The arm may be moving perfectly fine.**

### Some Events Are Likely Real

Events with non-zero actuals suggest genuine physical obstructions:
- `commanded=-40.3°, actual=-48.1°, error=7.8°` → J1 overshot or was pushed past target (gravity/inertia?)
- `commanded=-40.0°, actual=-12.5°, error=27.5°` → J2 stopped at -12.5° when trying to reach -40° — likely hit the checkerboard/clamps mentioned in the briefing

### Verdict: ~70% false positives from telemetry bug, ~30% genuine obstructions from cluttered workspace.

---

## 2. The `actual=0.0°` Problem

**This does NOT mean the arm isn't moving.** It means feedback is broken.

Evidence:
- 0.0° appears across J1, J2, J4, and J5 — joints with very different physical positions
- The arm would need to be in a physically impossible "all-zeros" configuration
- The calibration routine successfully collects 20 observations (per briefing), meaning the arm IS reaching poses between these "collision" events
- Vision analysis is also `"unavailable"` for all events — the collision handler's camera capture path may also be racing against the state update

**Most likely cause:** The `check_for_stall()` function in `run_calibration` (line ~499) queries arm state with a 1.0s timeout. If the state query returns the initial/default state (all zeros) before the arm's actual position is reported, every commanded position >3° from zero triggers a false stall.

**Recommended fix:** 
- Add a state-validity check: reject any state where ALL joints read exactly 0.0°
- Require at least 2 consecutive non-zero state readings before evaluating stall
- Log raw state timestamps to detect stale data

---

## 3. Workspace Pre-Check (CRITICAL MISSING FEATURE)

The calibration routine (`run_calibration`) does **zero workspace verification**. It:
1. Captures home pose image → but only for calibration data, not safety
2. Immediately starts moving through poses
3. Has no awareness of obstacles, people, or clutter

The briefing documents: checkerboard draped across workspace, bar clamps in sweep path, ruler/pliers/tape on surface, a person standing close, controller box in reach, and cable snag points.

### Required: Camera-Based Workspace Pre-Check

```python
async def pre_check_workspace(camera_url: str, api_base: str) -> WorkspaceStatus:
    """Capture from all cameras. Use vision to detect:
    1. Objects in the arm's sweep volume
    2. People within reach radius  
    3. Known hazards (cables, clamps, tools)
    4. Whether the calibration target is properly positioned (not draped on arm)
    
    Returns CLEAR, OBSTRUCTED (with obstacle list), or UNSAFE (person detected).
    """
```

This should be **mandatory** before any calibration run. The function should:
- Capture from both cam0 and cam1
- Run obstacle detection (even simple contour/color analysis)
- Check for human presence (skin detection, motion, or ML-based)
- Return a structured result with obstacle locations
- **Block calibration if workspace is not clear**

---

## 4. Proposed Safe Calibration Sequence

### Phase 0: Environment Verification
```
1. Capture images from all cameras
2. Run workspace clearance check (see §3)
3. Verify arm communication (state query returns valid, non-zero data)
4. Verify state feedback is LIVE (command a 0.5° wiggle, confirm feedback changes)
5. If any check fails → ABORT with clear error message
```

### Phase 1: Feedback Validation ("Heartbeat Test")
```
For each joint j (0-5):
    1. Read current position P0
    2. Command P0 + 1.0° (tiny move)
    3. Wait 0.3s
    4. Read position P1
    5. Verify |P1 - (P0 + 1.0)| < 2.0°  (feedback is live and reasonable)
    6. Command P0 (return)
    7. If feedback doesn't change → flag joint as "feedback-dead", abort
```
This catches the `actual=0.0°` bug before it cascades into 40 false collisions.

### Phase 2: Single-Joint Baseline Mapping
```
For each joint j:
    1. Command small moves: ±5°, ±10°
    2. At each position, capture image + read state
    3. Record actual range achieved vs commanded
    4. Identify any joint that can't reach targets (mechanical limit or obstruction)
    5. Build a "safe range map" for each joint
```

### Phase 3: Obstacle Detection Sweep
```
1. Move J0 through ±30° in 10° steps (base yaw sweep)
2. At each J0 position, capture camera images
3. Compare images to detect objects entering/leaving frame
4. Build a rough obstacle map of the workspace
5. Flag any J0 positions where obstacles are detected
```

### Phase 4: Full Calibration (Current Logic, Improved)
```
Use current progressive round approach (±5°, ±10°, ... ±45°) BUT:
- Only move within the safe ranges established in Phase 2
- Skip poses that would enter obstacle zones from Phase 3
- Use validated feedback (Phase 1 confirmed it works)
- Pre-check camera before each pose (abort if person detected)
```

### Dry Run Mode
```python
async def run_calibration(dry_run: bool = False, ...):
    """
    If dry_run=True:
    - Compute all poses that WOULD be commanded
    - For each pose, compute FK to get expected end-effector position
    - Visualize planned trajectory on camera image (overlay)
    - Report which poses are in obstacle zones
    - Do NOT send any motor commands
    - Return a CalibrationPlan with all poses, expected images, risk flags
    """
```

This lets the operator review the calibration plan before the arm moves.

---

## 5. Collision Detector Threshold Analysis

### Current Thresholds
- **Position error:** 3.0° — triggers when |commanded - actual| > 3°
- **Stall duration:** 0.5s — must persist for 0.5s before firing
- **Cooldown:** 5.0s — suppresses re-triggers for 5s per joint

### Assessment

**The 3° threshold is too tight for calibration moves.** Here's why:

1. **Calibration commands large moves** (up to ±45°). During a 45° move, the arm is transiently >3° from target for the entire motion duration. If the move takes >0.5s (it will — the arm isn't instant), the detector fires before the arm arrives.

2. **The 0.5s duration is too short for slow, safe calibration moves.** A conservative calibration should move slowly. A 20° move at 1 rad/s ≈ 0.35s. At 0.5 rad/s (safer) ≈ 0.7s. The detector fires at 0.5s — before the arm finishes moving.

3. **The thresholds are designed for runtime collision detection** (arm is holding a position and gets bumped). During calibration, the arm is intentionally far from target while in transit.

### Recommendations

| Parameter | Current | Calibration Mode | Runtime Mode |
|-----------|---------|-------------------|--------------|
| `position_error_deg` | 3.0° | 5.0° | 3.0° |
| `stall_duration_s` | 0.5s | 2.0s | 0.5s |
| `cooldown_s` | 5.0s | 3.0s | 5.0s |

**Better approach:** Don't check for stalls DURING motion. Instead:
1. Send move command
2. Wait for move to complete (or timeout)
3. THEN check |commanded - actual| with a tight threshold (e.g., 2°)
4. This is a "post-move verification" rather than continuous stall detection

The `check_for_stall()` in the calibration routine should be replaced with:
```python
async def verify_pose_reached(api_base, target_pose, tolerance_deg=3.0, timeout_s=5.0):
    """Wait for arm to reach target, return True if reached, False if timed out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = await get_arm_state(api_base)
        if state and all(abs(t - a) < tolerance_deg for t, a in zip(target_pose, state["joints"])):
            return True
        await asyncio.sleep(0.1)
    return False  # Timed out = possible collision/obstruction
```

---

## 6. Additional Safety Concerns

### No Vision-Based Collision Detection
The collision detector is purely position-based. With two cameras available, we should also:
- Detect unexpected objects entering the frame during motion
- Track the arm visually and flag if visual position diverges from commanded
- This would catch real collisions that the position-based detector misses (e.g., arm pushes a light object aside without stalling)

### No Emergency Stop Integration
The `SafetyMonitor` has e-stop capability, but the calibration routine doesn't reference it. If 40 collisions fire, the arm keeps going. The calibration should:
- Have a max-collision-count threshold (e.g., 5) before auto-aborting
- Integrate with the SafetyMonitor's e-stop
- Alert the operator when collision rate is abnormal

### Person Detection
The briefing notes a person standing near the arm. This is the highest-risk item. Pre-check (§3) should include person detection, and calibration should pause if a person enters the workspace during operation.

---

## Summary of Recommendations (Priority Order)

1. **FIX THE TELEMETRY BUG** — `actual=0.0°` feedback is the root cause of ~70% of false collisions. Validate state before evaluating stalls.
2. **Add workspace pre-check** — Camera-based obstacle/person detection before calibration starts.
3. **Add feedback heartbeat test** — Verify each joint's feedback is live before calibration.
4. **Use post-move verification** instead of continuous stall detection during calibration.
5. **Increase stall thresholds for calibration mode** — 5°/2.0s instead of 3°/0.5s.
6. **Add dry-run mode** — Preview calibration poses without moving.
7. **Add auto-abort** — Stop calibration after N consecutive failures.
8. **Add person detection** — Highest safety priority for a garage workshop environment.

---

*Report prepared for the Claw 🦾 Calibration Council, 2026-02-07*
