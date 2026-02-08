# th3cl4w Work Plan — Post-Testing Issues

## Priority 1: SAFETY (Must fix before any arm testing)

### S1. Phantom commands on startup ⚠️ [ASSIGNED: safety-fix agent]
- Command smoother initializes `_current = [0.0]*6`, reads real arm at J1=-90°, J2=92° etc.
- On first user command, smoother interpolates from 0° → target, driving all joints toward zero
- **Fix:** Require sync from arm feedback before accepting any commands. `_synced` flag. No-op tick until synced.

### S2. No enable-state guard on smoother ⚠️ [ASSIGNED: safety-fix agent]  
- Smoother sends commands even when arm isn't enabled/powered
- **Fix:** `set_arm_enabled()` method, checked in `_tick()`

### S3. E-stop doesn't stop smoother ⚠️ [ASSIGNED: safety-fix agent]
- E-stop sends estop command but smoother keeps its targets and continues interpolating
- **Fix:** `emergency_stop()` on smoother clears all targets and dirty flags

### S4. UI slider init triggers commands before sync
- Sliders start at `value="0"`, `input` event can fire before real state arrives
- **Fix:** `armSynced` flag in JS, don't send commands until first WebSocket state received

---

## Priority 2: 3D Visualization Fixes

### V1. Joint mapping — some positions appear swapped
- User: "The visual jumps and does weird things and other parts of the visual appear to try to push and move"
- Current viz maps: J0=base rotation, J1=shoulder pitch, J2=elbow pitch, J3=roll (no viz), J4=wrist pitch, J5=roll (no viz)
- **Need:** Verify joint index mapping matches actual D1 hardware. The DDS feedback angle0-angle6 may not map to the joints in the order we assume.
- **Fix:** Use diagnostic tool to wiggle one joint at a time on real hardware and log which angle index changes.

### V2. Twist (roll) joints not visible
- J0 (base), J3 (forearm roll), J5 (wrist roll) are rotation/twist joints
- Current viz only shows rotation indicators (small arcs) — not very visible
- **Fix:** Improve rotation indicators: larger, clearer, maybe color-coded text showing degrees

### V3. Gripper not shown in viz
- No visual representation of gripper open/close state
- **Fix:** Add gripper visualization at end effector (two small lines that open/close)

### V4. UI flickering
- User: "The UI does a lot of weird flickering"
- Likely caused by rapid state updates repainting canvas + slider values fighting between state and user input
- **Fix:** RequestAnimationFrame throttle on canvas redraws. Double-buffer or dirty-flag canvas rendering.

---

## Priority 3: Debug & Telemetry Fixes

### T1. Debug endpoints return 501
- `/api/debug/telemetry`, `/api/debug/stats` etc. return 501 because `_HAS_TELEMETRY` is False at those code paths
- The debug panel polls these every 1s, flooding logs with 501 errors
- **Fix:** Wire debug endpoints to use the new `TelemetryCollector`/`TelemetryQuery` instead of the old system. Or remove old debug endpoints and have the debug panel use the `/ws/telemetry` WebSocket instead.

### T2. Telemetry DB empty — collector not writing
- DB exists but 0 rows in all tables despite server running and commands being sent
- Possible cause: `get_collector()` returns a different instance than the one `start()`ed in lifespan, or the writer thread isn't running
- **Fix:** Debug singleton pattern, ensure single instance, verify writer thread is alive

### T3. Debug panel log spam
- Even when debug panel is "closed", the JS may still poll. Each poll generates 2 HTTP requests/second
- **Fix:** Stop polling when panel is hidden. Use WebSocket for real-time data instead of HTTP polling.

---

## Priority 4: Code Quality & Tests

### Q1. Fix 39 test_web_server import errors
- Missing `scipy`, `cv2`, `cyclonedds` causing import failures
- **Fix:** Conditional imports or mock the missing deps in tests

### Q2. Fix `dds_discover.py` API bug
- `DataReader` constructor expects `Topic`, not builtin type directly
- **Fix:** Update to match installed CycloneDDS Python bindings API

### Q3. Import path issue in server.py
- Had to change `from web.command_smoother` → `from command_smoother`
- Fragile — depends on working directory
- **Fix:** Use relative imports or add web/ to sys.path properly

---

## Priority 5: Feature Completion

### F1. Wire planning module into web UI
- Motion planner, task planner, path optimizer exist but no UI
- Add buttons: Home, Ready, Wave, Pick & Place (with coordinate inputs)
- Route through smoother for safe execution

### F2. Stereo camera calibration
- Vision module built but uncalibrated
- Need physical checkerboard pattern
- Run `tools/calibrate_cameras.py` with both Brios

### F3. Continuous telemetry export to Observe
- Script runs but needs fresh data piped to ~/observelogs
- Set up periodic export from SQLite → JSON → Observe

---

## Execution Plan

| Agent | Tasks | Priority |
|-------|-------|----------|
| safety-fix (already running) | S1, S2, S3 | P1 |
| ui-safety | S4, V4 | P1/P2 |
| viz-overhaul | V1, V2, V3 | P2 |
| telemetry-fix | T1, T2, T3 | P3 |
| test-cleanup | Q1, Q2, Q3 | P4 |
