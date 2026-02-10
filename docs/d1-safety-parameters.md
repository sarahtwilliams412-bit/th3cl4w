# D1 Safety Parameters — Comprehensive Reference

## Overview

This document catalogs every safety parameter available in th3cl4w for the Unitree D1 robotic arm, organized by category. Parameters come from three layers:

1. **D1 Hardware/Firmware** — limits enforced by the arm's internal controllers
2. **DDS Protocol Layer** — parameters in the communication protocol
3. **th3cl4w Software Safety** — our application-level safety checks

---

## 1. Joint Position Limits

**Source:** `src/safety/limits.py`, `src/config/pick_config.py`  
**Layer:** Software (validated before sending DDS commands)

| Joint | Label | Hardware Limit | Software Default | Notes |
|-------|-------|----------------|-----------------|-------|
| J0 | Base Yaw | ±135° | ±135° | Full hardware range exposed |
| J1 | Shoulder Pitch | ±85° | ±90° | Widened from ±80° (5° margin removed) |
| J2 | Elbow Pitch | ±85° | ±90° | Widened from ±80° (5° margin removed) |
| J3 | Elbow Roll | ±135° | ±135° | Full hardware range |
| J4 | Wrist Pitch | ±85° | ±90° | Widened from ±80° |
| J5 | Wrist Roll | ±135° | ±135° | Full hardware range |
| J6 | Gripper | 0–65 mm | 0–65 mm | Treated as mm, not degrees |

**Runtime configurable:** Yes, via `pick_config.json` → `safety.joint_limits_deg`

---

## 2. Velocity Limits

**Source:** `src/safety/limits.py`  
**Layer:** Software (SafetyMonitor validates commands)

| Joint | Max Velocity (rad/s) | Max Velocity (°/s) |
|-------|---------------------|-------------------|
| J0 | 2.0 | ~114.6 |
| J1 | 2.0 | ~114.6 |
| J2 | 2.5 | ~143.2 |
| J3 | 2.5 | ~143.2 |
| J4 | 3.0 | ~171.9 |
| J5 | 3.0 | ~171.9 |
| J6 | 2.0 | ~114.6 |

**Motion planner velocity limits** (deg/s, 6 arm joints):
`[90, 90, 120, 120, 150, 150]`

**Runtime configurable:** Not currently exposed in UI

---

## 3. Acceleration Limits

**Source:** `src/safety/limits.py`  
**Layer:** Software (motion planner only)

| Joint | Max Accel (°/s²) |
|-------|-----------------|
| J0 | 180 |
| J1 | 180 |
| J2 | 240 |
| J3 | 240 |
| J4 | 300 |
| J5 | 300 |

**Runtime configurable:** Not currently exposed

---

## 4. Torque Limits

**Source:** `src/safety/limits.py`  
**Layer:** Software (SafetyMonitor validates commands)

| Joint | Max Torque (Nm) |
|-------|----------------|
| J0 | 20.0 |
| J1 | 20.0 |
| J2 | 15.0 |
| J3 | 10.0 |
| J4 | 5.0 |
| J5 | 5.0 |
| J6 | 5.0 |

**Runtime configurable:** Not currently exposed

---

## 5. Torque Proxy

**Source:** `src/config/pick_config.py`, `web/server.py`  
**Layer:** Software (command rejection in set-joint endpoint)

A simplified torque estimation that rejects commands likely to overload motors:

- **Formula:** `|J1_angle| + |J2_angle| × j2_factor`
- **torque_proxy_limit:** 150.0 (default) — reject if proxy exceeds this
- **torque_j2_factor:** 0.7 (default) — weight for elbow contribution
- **torque_proxy_enabled:** true (default)

**Runtime configurable:** Yes

---

## 6. Collision/Stall Detection

**Source:** `src/safety/collision_detector.py`, `src/config/pick_config.py`

### Stall Detection (server.py)
- **stall_check_delay_s:** 5.0 — seconds after command before checking
- **stall_threshold_deg:** 15.0 — degrees of error to declare stall
- **stall_detection_enabled:** true

### Collision Detector (CollisionDetector class)
- **position_error_deg:** 3.0 — error threshold to enter error state
- **stall_duration_s:** 0.5 — how long error must persist before stall
- **cooldown_s:** 5.0 — minimum time between stall events per joint
- **enabled:** false (disabled by default — too aggressive for real arm)

**Runtime configurable:** Stall detection: yes. Collision detector: partially (via pick_config)

---

## 7. Command Smoother

**Source:** `web/command_smoother.py`  
**Layer:** Software

| Parameter | Default | Description |
|-----------|---------|-------------|
| rate_hz | 10.0 | Smoothing loop frequency |
| smoothing_factor (alpha) | 0.35 | Exponential smoothing factor (0=no movement, 1=instant) |
| max_step_deg | 10.0 | Maximum angular change per tick |
| max_gripper_step_mm | 5.0 | Maximum gripper change per tick |
| feedback_staleness_s | 0.5 | Refuse commands if feedback older than this |

**Runtime configurable:** Not currently exposed in UI

---

## 8. Emergency Stop

**Source:** `src/safety/safety_monitor.py`, `web/server.py`, `web/command_smoother.py`

- **E-Stop trigger:** `POST /api/command/stop` → disables motors + powers off
- **E-Stop in SafetyMonitor:** blocks all commands until reset
- **E-Stop in smoother:** clears all targets, sets arm_enabled=false
- **Auto-triggers:** None currently (collision detector disabled)

**E-Stop reset:** Only via power-on + enable sequence

---

## 9. Control Modes

**Source:** `src/interface/d1_connection.py`

| Mode | Name | Description |
|------|------|-------------|
| 0 | Idle | No active control |
| 1 | Position | Joint position control (primary mode used) |
| 2 | Velocity | Joint velocity control |
| 3 | Torque | Direct torque control |

**Currently used:** Mode 0 via DDS (funcode-based, not mode-based)

---

## 10. DDS Protocol Funcodes

**Source:** `src/interface/d1_dds_connection.py`

| Funcode | Direction | Description |
|---------|-----------|-------------|
| 1 | Command | Set single joint angle (data: {id, angle, delay_ms}) |
| 1 | Feedback | Joint angle feedback (data: {angle0..angle6}) |
| 2 | Command | Set all joints (data: {mode, angle0..angle6}) |
| 3 | Feedback | Status (enable_status, power_status, error_status, recv_status, exec_status) |
| 5 | Command | Enable/disable motors (data: {mode: 0|1}) |
| 6 | Command | Power on/off (data: {power: 0|1}) |
| 7 | Command | Reset to zero position |

---

## 11. Workspace Bounds

**Source:** `src/safety/limits.py`, `src/safety/safety_monitor.py`

- **MAX_WORKSPACE_RADIUS_MM:** 550.0 mm
- **MAX_WORKSPACE_RADIUS_M:** 0.55 m
- **Reach check:** Simplified FK using J1, J2, J4 pitch joints
- **Link lengths (approx):** [0, 0.15, 0.22, 0, 0.18, 0, 0.05] m

---

## 12. Feedback Freshness

**Source:** `src/safety/limits.py`, `web/command_smoother.py`

- **FEEDBACK_MAX_AGE_S:** 0.5 seconds
- **Smoother staleness check:** Refuses commands if feedback > 0.5s old
- **FeedbackMonitor:** Filters zero-reads, tracks per-joint freshness

---

## 13. Motion Safety (Ramp Control)

**Source:** `web/server.py`

- **RAMP_THRESHOLD_DEG:** 20° — movements larger than this are ramped
- **RAMP_STEP_DEG:** 10° — size of each ramp increment
- **RAMP_DELAY_S:** 0.3s — delay between ramp steps

---

## 14. Power Loss Recovery

**Source:** `web/server.py`

- **Auto-recovery delay:** 3 seconds after power loss detected
- **Recovery action:** Enable motors + sync smoother to current position
- **No reset_to_zero** (avoids overcurrent)

---

## 15. Parameters NOT Currently Exposed

The following could be useful to expose:

1. **Velocity limits per joint** (currently hardcoded in limits.py)
2. **Acceleration limits per joint** (hardcoded)
3. **Torque limits per joint** (hardcoded)
4. **Smoother alpha/max_step** (hardcoded at startup)
5. **Smoother rate_hz** (hardcoded)
6. **Collision detector thresholds** (in pick_config but no dedicated UI)
7. **Workspace radius** (hardcoded)
8. **Feedback max age** (hardcoded)
9. **Ramp threshold/step/delay** (hardcoded in server.py)
10. **Auto-recovery enable/disable** (always on)
11. **Max gripper step** (hardcoded in smoother)
