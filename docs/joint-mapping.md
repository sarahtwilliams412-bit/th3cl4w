# D1 Joint Mapping — Empirical Results

Tested 2026-02-08. Programmatic sweep: each joint ±45° from home, both cameras captured.
Arm base in overhead cam at approximately pixel (820, 590).

## Home Position (all joints ≈ 0°)
- Arm is **fully vertical**, pointing straight up from the base
- Overhead gripper tip: **(760, 135)** — near top of image
- Front gripper tip: **(1130, 220)** — upper center-right

## Gripper Pixel Coordinates (Overhead Camera, cam1)

| Pose | Gripper (x, y) | Δx from home | Δy from home | Movement in image |
|------|---------------|-------------|-------------|-------------------|
| Home (0°) | (760, 135) | 0 | 0 | — |
| **J0 +45°** | (680, 520) | -80 | +385 | LEFT, DOWN |
| **J0 -45°** | (660, 290) | -100 | +155 | LEFT, DOWN (less) |
| **J1 +45°** | (760, 270) | 0 | +135 | DOWN only (foreshortened = pitched UP) |
| **J1 -45°** | (760, 470) | 0 | +335 | DOWN more (extended = pitched DOWN/forward) |
| **J2 +45°** | (660, 330) | -100 | +195 | LEFT, DOWN (elbow extended forward) |
| **J2 -45°** | (830, 490) | +70 | +355 | RIGHT, DOWN (elbow folded back) |
| **J4 +45°** | (760, 290) | 0 | +155 | DOWN (wrist pitched forward/down) |
| **J4 -45°** | (760, 200) | 0 | +65 | Slight DOWN (wrist pitched up) |

## Gripper Pixel Coordinates (Front Camera, cam0)

| Pose | Gripper (x, y) | Δx from home | Δy from home | Movement in image |
|------|---------------|-------------|-------------|-------------------|
| Home (0°) | (1130, 220) | 0 | 0 | — |
| **J0 +45°** | (1430, 105) | +300 | -115 | RIGHT, UP |
| **J0 -45°** | (920, 175) | -210 | -45 | LEFT, UP |
| **J1 +45°** | (1050, 220) | -80 | 0 | Slight LEFT (pitching UP/back) |
| **J1 -45°** | (540, 230) | -590 | +10 | FAR LEFT (pitching DOWN/forward) |

## Joint Behavior Summary

| Joint | Type | +45° effect | -45° effect |
|-------|------|------------|------------|
| **J0** | Base yaw | Swings arm LEFT in overhead (CCW from above) | Swings arm RIGHT in overhead (CW from above) |
| **J1** | Shoulder pitch | Pitches UP/backward — arm shortens in overhead | Pitches DOWN/forward — arm extends in overhead |
| **J2** | Elbow pitch | Extends forearm forward/outward | Folds forearm back toward base |
| **J3** | Forearm roll | CW looking down arm axis | CCW looking down arm axis |
| **J4** | Wrist pitch | Gripper pitches DOWN (toward floor) | Gripper pitches UP (toward ceiling) |
| **J5** | Gripper roll | CW from above | CCW from above |

## How to Reach Objects on the Table

```
J0: Aim toward target    (+ = swing left, - = swing right from above)
J1: Lean forward/down    (NEGATIVE to reach forward, POSITIVE = pulls back/up)
J2: Extend forearm       (POSITIVE extends outward, NEGATIVE folds back)
J4: Angle gripper down   (POSITIVE = gripper faces floor)
J3/J5: Orient gripper    (roll to align with object)
```

## Architecture
```
Base → J0 (yaw) → J1 (pitch) → J2 (pitch) → J3 (roll) → J4 (pitch) → J5 (roll) → Gripper
```

## Safe Limits
- J0: ±135° | J1: ±85° | J2: ±135° | J3: ±135° | J4: ±85° | J5: ±135°
