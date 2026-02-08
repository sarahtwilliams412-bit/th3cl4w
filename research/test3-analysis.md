# Test 3 Analysis — 2026-02-07 17:22-17:25

## User Narration Summary
- Power on → Enable → rotate base → E-stop
- UI controls stopped working after E-stop
- E-stop button itself stopped working briefly
- Graphics still show wrong zero position
- Pressing Enable triggered Emergency Stop
- Had to physically unplug arm

## Critical Bugs

### BUG 1: Enable immediately disables (HIGHEST PRIORITY)
**Evidence:**
```
17:22:08 Arm enabled — smoother will accept commands
17:22:08 Arm disabled — smoother targets cleared
17:22:08 ENABLE | OK
```
Every single enable call in the session shows this pattern.

**Root cause:** `get_arm_state()` is called on every WebSocket state broadcast (~10Hz). It syncs `smoother.set_arm_enabled()` from the arm's `enable_status` feedback field. But when you send an enable command, the arm takes time to respond. The very next state poll reads `enable_status:0` (old value) and immediately disables the smoother.

**Fix:** Don't sync arm_enabled from feedback on every tick. Instead:
- Only set `arm_enabled=True` from explicit enable command response
- Only set `arm_enabled=False` from explicit disable/estop/power-off command response
- Use feedback `enable_status` only as a safety fallback (if feedback shows disabled for >1s after enable, THEN disable smoother)

### BUG 2: Commands bypass smoother when arm disabled
**Evidence:**
```
17:23:45 SET_JOINT | Request: J2 -> -15.0° (arm was disabled at this point)
```
**Root cause:** The `/api/command/set-joint` endpoint checks `hasattr(arm, 'set_joint')` but doesn't check if arm is enabled. It calls `smoother.set_joint_target()` which may silently accept targets even when disabled, or bypasses smoother entirely.

**Fix:** Command endpoints must check arm enabled state before accepting movement commands. Return 409 Conflict if arm not enabled.

### BUG 3: Reset causes race condition with E-stop
**Evidence:**
```
17:22:59 RESET | OK
17:22:59 Arm enabled — smoother will accept commands  (auto-enable from state sync)
17:23:00 EMERGENCY_STOP | ⚠ TRIGGERED  (0.3s later)
```
**Root cause:** Reset command causes the D1 to auto-enable motors. The state feedback picks this up and enables the smoother. But something (possibly the UI or a stale command) immediately triggers E-stop.

**Fix:** After reset, don't auto-enable smoother from feedback. Require explicit enable button press.

### BUG 4: DDS telemetry not recording
**Evidence:** `dds_commands: 0, dds_feedback: 0` despite commands clearly being sent
**Root cause:** The `d1_dds_connection.py` calls `get_collector()` which may create a DIFFERENT collector instance than the one started by server.py. The singleton may not work across module boundaries, or the `log_dds_command()` method isn't being called because the code path uses the old `emit()` API.

**Fix:** Pass the collector instance explicitly from server.py to D1DDSConnection instead of relying on get_collector() singleton.

### BUG 5: Telemetry timestamps all identical
**Evidence:** All 220 system events show timestamp 17:16:38 — this is from test suite runs, not live
**Root cause:** The test suite creates/destroys collectors rapidly, and the live collector may be sharing the same DB file with stale data. Or the timestamps are from a different session.

### BUG 6: UI shows wrong zero position
**User:** "saying that's a zero position but that should really be... graphics still aren't correct"
The 3D viz shows the arm at zero when it's actually at a different position. Likely the viz isn't using the actual feedback angles and is defaulting to zeros.

## Severity Ranking
1. **BUG 1** — Enable race condition (blocks all arm use)
2. **BUG 3** — Reset/enable/estop cascade (dangerous)
3. **BUG 2** — Commands bypass enable check (safety)
4. **BUG 6** — Wrong viz positions (usability)
5. **BUG 4** — No DDS telemetry (observability)
6. **BUG 5** — Timestamp issues (minor)
