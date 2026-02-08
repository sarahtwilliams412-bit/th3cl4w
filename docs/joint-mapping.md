# D1 Joint Mapping — Empirical Results

Tested 2026-02-08 by moving each joint +10° from home (all zeros) and observing via cameras.

## Home Position (all joints = 0°)
- Arm is **fully vertical**, pointing straight up from the base
- From overhead (cam1): arm points at **12 o'clock** (straight forward from base)

## Joint Map

| Joint | Type | +10° Direction | Description |
|-------|------|---------------|-------------|
| J0 | Base yaw | **CCW from above** (left) | Rotates entire arm left/right. +angle = left, -angle = right |
| J1 | Shoulder pitch | **Right tilt** (from front view) | Tilts arm sideways. +angle = rightward lean |
| J2 | Elbow pitch | **Right fold** (from front view) | Bends forearm rightward at elbow. Same plane as J1 |
| J3 | Wrist roll | **CCW** (looking base→elbow) | Rolls forearm about its axis. Subtle when arm is vertical |
| J4 | Wrist pitch | **Forward tilt** | Tips gripper forward/down. Same pitch plane as J1/J2 |
| J5 | Wrist roll | **CW from above** | Rotates gripper about its axis |

## Key Observations
- J1 and J2 are BOTH pitch joints that tilt the arm in the SAME plane (sideways from front view)
- When arm is at home (vertical), "sideways" from front camera = the arm's pitch plane
- The arm's pitch plane rotates with J0
- At J0=0°, the pitch plane is perpendicular to the forward direction
- **To reach forward**: need J0 to aim, then J1/J2 to pitch in the arm's plane

## DH Convention Notes
- J0 +angle = CCW from above ✅ standard
- J1 +angle = rightward tilt (which direction depends on J0 orientation)
- J2 +angle = same direction as J1 (both pitch same way)
- J4 +angle = same pitch direction as J1/J2

## Safe Limits (from code, with 5° margin)
- J0: ±135° (software), physically limited
- J1: ±85°
- J2: ±135°
- J3: ±135°
- J4: ±85°
- J5: ±135°
