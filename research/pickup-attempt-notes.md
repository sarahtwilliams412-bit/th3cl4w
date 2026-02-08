# Pickup Attempt Observation Notes

**Date:** 2026-02-07 22:00–22:06 PST  
**Observer:** Automated subagent  
**Result:** ❌ UNSUCCESSFUL — can remained on table  
**Grid:** Each checkerboard square = 15/16" = 23.8mm  
**Snapshots:** /tmp/observer_000.jpg through observer_006.jpg

---

## Baseline (22:00:00)

**Arm state:** Home position, NOT enabled  
- Joints: [0.2, -90.4, 92.1, 0.2, -90.0, 0.1], Gripper: 49.8mm (open)  
- Power: ON, Enabled: FALSE  

**Can position:** Red/gold can to the RIGHT of arm base, estimated ~20-30cm away on wooden plank. Slightly behind the base.

---

## Movement Log

### Move 1: Initial positioning (~22:01:45)
**Joints:** [0.2, -90.4, 92.1, 0.2, -90.0, 0.1] → [-29.9, 0.4, 0.4, 0.2, -44.5, 0.1]  
- **J0:** 0° → -30° (rotated RIGHT toward can ✓)
- **J1:** -90° → 0° (shoulder from vertical to horizontal — big move)
- **J2:** 90° → 0° (elbow from folded to perpendicular)
- **J4:** -90° → -44.5° (wrist pitch adjusted)
- **Gripper:** 50mm (open ✓)
- **Assessment:** Large initial move. Arm extended horizontally, pointing right. Good direction but gripper was way above the can and not close enough.

### Move 2: Further extension (~22:02:35)
**Joints:** [-29.9, 0.4, 0.4, ...] → [-39.6, 30.6, 20.4, -0.3, -49.4, 0.1]  
- **J0:** -30° → -40° (more rotation right ✓)
- **J1:** 0° → 31° (shoulder tilted forward/down ✓)
- **J2:** 0° → 20° (elbow folding slightly)
- **Assessment:** Arm reaching more toward table level but still above and to the left of the can. Camera showed ~15-25cm gap remaining.

### Move 3: Wrist adjustment (~22:03:20)
**Joints:** → [-39.5, 79.4, -0.4, -31.2, -39.6, 0.0]  
- **J1:** 31° → 79° (shoulder MUCH more forward/down — nearly at limit!)
- **J2:** 20° → -0.4° (elbow extending)
- **J3:** -0.3° → -31.2° (wrist pitched up)
- **Assessment:** Arm now reaching far down toward table. Getting closer to can height but lateral alignment still off.

### Move 4: Fine tuning (~22:03:50)
**Joints:** → [-39.1, 78.0, 8.2, -33.4, -60.2, 0.0]  
- **J2:** -0.4° → 8.2° (slight elbow fold)
- **J4:** -39.6° → -60.2° (wrist bending down more)
- **Assessment:** Small adjustments. Still ~15-25cm from can per camera.

### Move 5: Apparent loss of control (~22:04:35)
**Joints:** → [-2.6, 88.9, 15.8, -4.6, -89.4, 0.0]  
- **J0:** -39° → -3° (swung LEFT back to center!)
- **J1:** 78° → 89° (near max forward/down)
- **Power: OFF** ⚠️
- **Assessment:** Something went wrong. The arm swung away from the can and power was cut. Possible emergency stop or fault.

### Move 6: Gravity sag (22:04:56)
**Joints:** [-2.5, 99.1, 5.3, -5.5, -14.3, 0.0]  
- **J1:** 89° → 99° (PAST +90° limit — gravity pulling arm down)
- **Power: OFF, Enabled: TRUE**
- **Assessment:** Arm sagging under gravity with motors off. Dangerous configuration.

### Move 7: Recovery to home (22:05:26)
**Joints:** [0.9, -90.5, 92.1, -4.8, -85.0, 0.0]  
- Power: ON, Enabled: FALSE
- Arm returned to near-home position safely.
- **Can still on table — pickup failed.**

---

## Issues Observed

1. **Never reached the can.** The gripper was consistently 15-25cm short of the target throughout the attempt.

2. **J0 direction correct but insufficient.** Rotated to -40° (right) but the can may have required more rotation or the approach vector was wrong.

3. **J1 pushed to near-limit.** Went to 79-89° (max is 90°), meaning the shoulder was almost fully forward. This suggests the arm was trying to reach something at the edge of its workspace.

4. **Power loss / emergency stop.** Around 22:04:35, power dropped to false and the arm swung back toward center unpredictably. This could be:
   - Emergency stop triggered by operator
   - Over-torque protection (J1 near limit with arm extended = high torque)
   - Software fault

5. **Gravity sag when power off.** J1 reached 99° with power off — the extended arm sagged under gravity. This is a safety concern.

6. **Gripper never closed.** Stayed at ~50mm throughout — never attempted a grasp. The arm never got close enough to try.

7. **Camera went down** during the power-off incident (curl error code 7 at 22:04:40).

---

## Distance Analysis (using checkerboard grid)

Based on camera observations, the can appeared to be ~20-30cm to the right and slightly behind the arm base. With J0 at -40° and J1 at 79°, the arm was reaching forward-right but the effective reach in that direction may have been insufficient.

The arm's upper arm + forearm length is roughly 25cm each (~500mm total reach). At J0=-40°, J1=80°, J2=0°, the endpoint would be roughly:
- Forward: ~490mm × sin(80°) ≈ 480mm
- Right: offset by sin(40°) ≈ 310mm right
- But height above table and joint geometry reduce effective reach to the table plane

The can was likely just within or at the edge of the reachable workspace, making the grasp very difficult.

---

## Suggestions for Next Attempt

### Approach Strategy
1. **Position the can closer** — move it to ~15cm from base, directly in front or slightly right. The current distance (~25cm right) is at the workspace limit.

2. **Use a top-down approach** instead of reaching from the side:
   - J0: rotate to align with can (-20° to -30°)
   - J1: go to ~60-70° (not 80°+, which is near the torque limit)
   - J2: extend (-20° to -45°) to reach outward
   - J4: adjust to point gripper straight down

3. **Move joints sequentially, not simultaneously:**
   - First: J0 to aim at can
   - Second: J1+J2 to get rough position above can  
   - Third: J3+J4 to orient gripper downward
   - Fourth: Lower with J1 to approach
   - Fifth: Close gripper

### Safety
4. **Don't push J1 past 75°** with the arm extended — high torque risk.
5. **Add a software torque limit** or watchdog to prevent power-off sag.
6. **Keep power on when disabling** — use controlled return to home rather than cutting power.

### Gripper
7. **The can diameter matters.** If it's a standard energy drink (~53mm diameter), it's right at the gripper's 50mm effective limit. May need to grab from a narrower section or use a different object for testing.

### Joint Sequence
8. **Recommended joint sequence for pickup:**
   ```
   Home → Open gripper (50mm)
   → J0 to target azimuth (-25°?)
   → J1 to 45° (partial lean)
   → J2 to -30° (extend forearm)
   → J4 to -45° (gripper pointing more downward)
   → J1 to 65° (lower toward can)
   → Fine-tune J0, J2 as needed
   → J1 to 70° (final approach to can height)
   → Close gripper (0-10mm depending on can size)
   → J1 to 30° (lift)
   → J0 to 0° (center)
   → Open gripper (release)
   ```

---

## Raw State Snapshots

| Check | Time | J0 | J1 | J2 | J3 | J4 | J5 | Grip | Power | Enabled |
|-------|------|-----|------|------|------|------|------|------|-------|---------|
| 0 | 22:00 | 0.2 | -90.4 | 92.1 | 0.2 | -90.0 | 0.1 | 49.8 | ✓ | ✗ |
| 1 | 22:01:55 | -29.9 | 0.4 | 0.4 | 0.2 | -44.5 | 0.1 | 50.0 | ✓ | ✓ |
| 4 | 22:02:40 | -39.6 | 30.6 | 20.4 | -0.3 | -49.4 | 0.1 | 49.7 | ✓ | ✓ |
| 6 | 22:03:10 | -39.7 | 80.9 | -19.6 | -1.4 | -39.5 | 0.1 | 49.7 | ✓ | ✓ |
| 7 | 22:03:25 | -39.5 | 79.4 | -0.4 | -31.2 | -39.6 | 0.0 | 49.7 | ✓ | ✓ |
| 9 | 22:03:55 | -39.1 | 78.0 | 8.2 | -33.4 | -60.2 | 0.0 | 49.7 | ✓ | ✓ |
| 12 | 22:04:40 | -2.6 | 88.9 | 15.8 | -4.6 | -89.4 | 0.0 | 49.8 | ✗ | ✓ |
| 13 | 22:04:56 | -2.5 | 99.1 | 5.3 | -5.5 | -14.3 | 0.0 | 49.7 | ✗ | ✓ |
| 14 | 22:05:11 | -8.2 | 85.8 | 3.9 | -5.5 | -14.2 | 0.0 | 49.7 | ✓ | ✗ |
| 15 | 22:05:26 | 0.9 | -90.5 | 92.1 | -4.8 | -85.0 | 0.0 | 49.7 | ✓ | ✗ |
