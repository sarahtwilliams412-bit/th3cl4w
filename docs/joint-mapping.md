# D1 Joint Mapping — Empirical Results

Tested 2026-02-08. Each joint moved ±10° and ±25° from home, verified via both cameras.

## Home Position (all joints = 0°)
- Arm is **fully vertical**, pointing straight up from the base
- From overhead (cam1): arm points at **12 o'clock** (straight forward)
- From front (cam0): arm is centered, vertical

## Joint Map

| Joint | Type | +angle Direction | -angle Direction |
|-------|------|-----------------|-----------------|
| **J0** | Base yaw | CCW from above (swings left, ~1 o'clock overhead) | CW from above (swings right, ~10-11 o'clock overhead) |
| **J1** | Shoulder pitch | Arm pitches UP/backward (foreshortens from above) | Arm pitches DOWN/forward (extends from above) |
| **J2** | Elbow pitch | Forearm bends RIGHT (from front cam) | Forearm bends LEFT (from front cam) |
| **J3** | Forearm roll | CW twist looking down arm axis | CCW twist |
| **J4** | Wrist pitch | Wrist bends RIGHT/forward (same plane as J1/J2) | Wrist bends LEFT/backward |
| **J5** | Gripper roll | CW from above (gripper points right from front) | CCW from above (gripper points left) |

## Architecture (from base to gripper)
```
Base → J0 (yaw) → J1 (pitch) → J2 (pitch) → J3 (roll) → J4 (pitch) → J5 (roll) → Gripper
```

## Key Insights

### Pitch Plane
- J1, J2, J4 all pitch in the **same plane**
- That plane is perpendicular to J0's rotation axis
- When J0=0°, the pitch plane is the LEFT-RIGHT plane (from front camera view)
- To reach FORWARD from the base: set J0 to aim, then use J1 (negative = lean forward)

### Reaching a Target
1. **J0** aims the arm's pitch plane toward the target (left/right)
2. **J1** (negative) leans the shoulder forward/down to extend reach
3. **J2** bends the elbow to adjust reach distance
4. **J3** rolls the forearm to orient the gripper
5. **J4** adjusts wrist pitch for approach angle
6. **J5** rolls gripper to align with object

### Camera Orientation
- **cam0** (front): Looks at the arm from the front. Robot's left = camera's right
- **cam1** (overhead): Looks down from above. 12 o'clock = forward from base

## Safe Limits
- J0: ±135° (may be less physically)
- J1: ±85°
- J2: ±135°
- J3: ±135°
- J4: ±85°
- J5: ±135°

## DH Parameter Notes
- DH alpha signs may need flipping for J1/J2/J4 — calibration found ~180° offsets
- The "right from front cam" direction for +J1/+J2/+J4 needs to be reconciled with DH convention
- FK model may predict opposite pitch directions from reality
