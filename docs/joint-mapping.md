# D1 Joint Mapping — Empirical Results

Tested 2026-02-08. Each joint moved ±10°, ±25°, and ±45° from home, verified via both cameras.

## Home Position (all joints = 0°)
- Arm is **fully vertical**, pointing straight up from the base
- From overhead (cam1): arm points at **12 o'clock** (straight forward)
- From front (cam0): arm is centered, vertical

## Joint Map

| Joint | Type | +angle | -angle |
|-------|------|--------|--------|
| **J0** | Base yaw | CCW from above (swings left) | CW from above (swings right) |
| **J1** | Shoulder pitch | **Pitches UP/backward** (shorter from above) | **Pitches DOWN/forward** (longer from above, reaches toward table) |
| **J2** | Elbow pitch | **Extends forward/outward** (longer from above) | **Folds back** (shorter from above) |
| **J3** | Forearm roll | CW looking down arm axis | CCW looking down arm axis |
| **J4** | Wrist pitch | **Pitches forward/DOWN** (gripper toward floor) | **Pitches backward/UP** (gripper toward ceiling) |
| **J5** | Gripper roll | CW from above | CCW from above |

## Architecture
```
Base → J0 (yaw) → J1 (pitch) → J2 (pitch) → J3 (roll) → J4 (pitch) → J5 (roll) → Gripper
```

## Critical: How to Reach Objects on the Table

To reach something on the table in front of the arm:

1. **J0**: Aim the pitch plane toward the target (+ = left, - = right from above)
2. **J1**: Set NEGATIVE to lean forward/down toward the table
3. **J2**: Set POSITIVE to extend the forearm outward
4. **J4**: Set POSITIVE to angle the gripper down toward the object
5. **J3/J5**: Adjust roll to orient gripper for grasp

### Example: Reach forward and down
```
J0 = 0°     (straight ahead)
J1 = -45°   (lean forward toward table)
J2 = +30°   (extend forearm)
J4 = +30°   (angle gripper down)
```

## Camera Orientation
- **cam0** (front): Looks at the arm from the front
- **cam1** (overhead): Looks down from above. 12 o'clock = forward from base

## Safe Limits
- J0: ±135°
- J1: ±85°
- J2: ±135°
- J3: ±135°
- J4: ±85°
- J5: ±135°
