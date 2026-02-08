# Unitree D1 Joint Calibration Results

**Date:** 2026-02-07  
**Method:** Systematic single-joint moves with dual camera snapshots  
**Grid:** Checkerboard behind arm, each square = 15/16" (23.8mm)  
**Images:** `/tmp/calib_*.jpg` (cam0 = right-side view, cam1 = front-left view)

## Home Position

**Commanded:** `[0, -90, 90, 0, -90, 0, 65]`  
**Actual:** `[-0.1, -90.3, 90.3, 0.2, -90.1, 0.2, 50.0]`

The home position has the arm reaching **straight up and slightly forward** toward the checkerboard. The upper arm extends vertically from the base, the forearm folds back (J2=90° = fully folded elbow), and the wrist points the gripper roughly toward the board.

**⚠️ Gripper anomaly:** Commanding 65mm consistently reads back ~50mm. The gripper may have a different scale or offset.

---

## Joint-by-Joint Results

### J0 — Base Rotation (Yaw)

**Type:** Rotates the entire arm around the vertical axis (base swivel)  
**Range:** ±135°  
**Home:** 0°

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -44.8° | +0.2° |
| 0° | 0.0° | 0.0° |
| +45° | 45.0° | 0.0° |

**Coordinate frame:**
- **J0 positive (+)** = arm rotates **left** (counterclockwise when viewed from above)
- **J0 negative (-)** = arm rotates **right** (clockwise when viewed from above)

**Accuracy:** Excellent. Sub-degree error.

---

### J1 — Shoulder Pitch

**Type:** Tilts the upper arm forward/backward (pitch from vertical)  
**Range:** ±90°  
**Home:** -90° (upper arm pointing straight up)

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -45.3° | -0.3° |
| 0° | 0.3° | +0.3° |
| +45° | 45.4° | +0.4° |

**Coordinate frame:**
- **J1 = -90° (home):** Upper arm points straight up (vertical)
- **J1 = 0°:** Upper arm is horizontal (forward)
- **J1 = +45°:** Upper arm tilts 45° past horizontal (forward and downward) — large torque, near limit
- **J1 negative** = arm tilts toward vertical/upward
- **J1 positive** = arm tilts forward/downward

**Accuracy:** Excellent. All moves reached target.

**Note:** In v1 (using `set_all_joints`), commanding J1=+45 caused the arm to freeze at 24.5° and stop responding. Using `set_joint` (single-joint command, funcode 1) worked perfectly. **Always use `set_joint` for individual joint moves.**

---

### J2 — Elbow

**Type:** Bends/extends the forearm relative to the upper arm  
**Range:** ±90°  
**Home:** 90° (forearm fully folded against upper arm)

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -45.3° | -0.3° |
| 0° | -0.3° | -0.3° |
| +45° | 45.1° | +0.1° |

**Coordinate frame:**
- **J2 = 90° (home):** Forearm fully folded/bent against upper arm
- **J2 = 0°:** Forearm perpendicular to upper arm (90° elbow bend)
- **J2 = -45°:** Forearm nearly extended outward (arm stretched out)
- **J2 positive** = more folded
- **J2 negative** = more extended/straightened

**Accuracy:** Excellent.

---

### J3 — Wrist Pitch (Forearm Rotation)

**Type:** Rotates the forearm/wrist about its long axis, tilting the gripper up/down  
**Range:** ±135°  
**Home:** 0°

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -44.5° | +0.5° |
| 0° | -0.2° | -0.2° |
| +45° | 44.6° | -0.4° |

**Coordinate frame:**
- **J3 = 0° (home):** Gripper aligned with forearm direction
- **J3 negative (-):** Gripper pitches/tilts **upward**
- **J3 positive (+):** Gripper pitches/tilts **downward**

**Accuracy:** Good. ~0.5° error.

---

### J4 — Wrist Pitch 2

**Type:** Second wrist pitch — tilts the gripper up/down relative to the forearm  
**Range:** ±90°  
**Home:** -90° (gripper bent perpendicular to forearm)

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -44.7° | +0.3° |
| 0° | 0.3° | +0.3° |
| +45° | **25.8°** | **-19.2°** ⚠️ |

**Coordinate frame:**
- **J4 = -90° (home):** Gripper bent 90° down from forearm
- **J4 = 0°:** Gripper in line with forearm
- **J4 positive** = gripper tilts upward/back
- **J4 negative** = gripper bends downward

**⚠️ J4=+45° only reached 25.8°.** This may be a physical limit (collision with forearm when the wrist bends too far back in this configuration), or the 2-second wait was insufficient for the large travel. The arm did return to home normally afterward, so no fault occurred.

**Accuracy:** Good for -45° and 0°. **Clamped/slow at +45°.**

---

### J5 — Wrist Roll

**Type:** Rotates the gripper around its approach axis (like turning a doorknob)  
**Range:** ±135°  
**Home:** 0°

| Commanded | Actual | Error |
|-----------|--------|-------|
| -45° | -44.7° | +0.3° |
| 0° | **-44.7°** | **-44.7°** ⚠️ |
| +45° | 44.9° | -0.1° |

**Coordinate frame:**
- **J5 = 0° (home):** Gripper fingers in default orientation
- **J5 negative** = gripper rolls clockwise (viewed from behind)
- **J5 positive** = gripper rolls counterclockwise

**⚠️ J5=0° did NOT execute.** The joint stayed at -44.7° (previous position). This appears to be a timing/command issue — the `set_joint` command to 0° was sent but the arm didn't move. The subsequent +45° command did work, jumping from -44.7° to 44.9°. This could be a DDS command delivery issue.

**Accuracy:** Good when commands execute. One missed command.

---

### Gripper (Joint 6)

**Type:** Parallel jaw gripper, opens/closes in mm  
**Range:** 0–65mm

| Commanded | Actual | Notes |
|-----------|--------|-------|
| 65mm (open) | 49.9mm | ⚠️ ~15mm offset |
| 32mm (half) | 32.1mm | ✓ Accurate |
| 0mm (closed) | 0.2mm | ✓ Accurate |

**⚠️ Gripper scale issue:** Commanding 65mm (full open) consistently reads back as ~50mm. However, 32mm and 0mm are accurate. This suggests either:
1. The gripper has a mechanical stop before the firmware thinks 65mm is
2. The feedback has a different scale at the upper range
3. The gripper physically maxes out around 50mm opening

**Practical implication:** Use 50mm as the effective max open, not 65mm.

---

## Summary Table

| Joint | Function | Home | Range | Accuracy | Issues |
|-------|----------|------|-------|----------|--------|
| J0 | Base yaw (left/right) | 0° | ±135° | <0.2° | None |
| J1 | Shoulder pitch (up/down) | -90° | ±90° | <0.4° | Freezes with `set_all_joints` |
| J2 | Elbow (fold/extend) | 90° | ±90° | <0.3° | None |
| J3 | Wrist pitch 1 (tilt) | 0° | ±135° | <0.5° | None |
| J4 | Wrist pitch 2 (tilt) | -90° | ±90° | <0.3° | Clamped at +45° (reached 25.8°) |
| J5 | Wrist roll (twist) | 0° | ±135° | <0.3° | One missed command |
| Grip | Gripper open/close | 65mm→50mm | 0–50mm | ~0.2mm | Effective max ~50mm, not 65mm |

## Key Findings

1. **Use `set_joint()` (funcode 1), not `set_all_joints()` (funcode 2)** for reliable single-joint moves. `set_all_joints` caused the arm to freeze after large J1 moves.

2. **Joint accuracy is excellent** — typically within 0.5° of commanded. The DDS interface is reliable.

3. **Gripper effective range is 0–50mm**, not 0–65mm. The firmware reports ~50mm when 65mm is commanded.

4. **J4 has a soft limit around +26°** in the home arm configuration (other joints at home values). This is likely a collision avoidance limit.

5. **Occasional missed commands on J5** — may need retry logic or longer delays for wrist roll.

6. **DH Convention mapping:**
   - J0: Base rotation (Z-axis, vertical)
   - J1: Shoulder pitch (Y-axis at base, -90°=up, +90°=forward-down)
   - J2: Elbow pitch (Y-axis at elbow, +90°=folded, -90°=extended)
   - J3: Forearm/wrist pitch (Y-axis, tilts gripper up/down)
   - J4: Wrist pitch 2 (Y-axis, -90°=perpendicular to forearm)
   - J5: Wrist roll (Z-axis of gripper, twists end effector)
