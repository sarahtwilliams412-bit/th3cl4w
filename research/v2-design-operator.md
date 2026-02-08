# V2 Design Proposal: Operator Workflow & Safety UX

**Author:** Operator Workflow & Safety Council Member  
**Date:** 2026-02-07  
**Scope:** How an operator actually uses this panel during real operations, and what's dangerous about V1.

---

## 1. V1 Safety Critique — What's Dangerous Right Now

### Critical Issues

1. **E-stop and raw command are on the same screen with no guard.** The raw command input accepts arbitrary JSON payloads with zero confirmation. An operator could accidentally send `{"action":"set-all-joints","angles":[90,90,90,90,90,90]}` while meaning to type something else. On a real arm this is a collision/damage risk.

2. **No confirmation on destructive actions.** Power On, Enable, and task execution are single-click with no "are you sure?" gate. Enable is the most dangerous transition — the arm becomes live. One mis-click and joints move.

3. **Joint sliders send commands on drag with no dead-man switch.** If an operator's mouse slips or a trackpad ghosts, the arm moves. There's no "hold to jog" or "press-and-drag" safety pattern. The 20Hz throttle helps but doesn't prevent unintended motion.

4. **E-stop button is at the bottom of a scrollable panel.** On mobile or small screens the e-stop may be below the fold. The single most critical safety control must NEVER be hidden.

5. **No visual/audible warning before motion starts.** When a task is executed (Home, Ready, Wave), the arm begins moving immediately. Industrial practice requires a 2-3 second warning with audible tone and visual countdown.

6. **Connection loss is cosmetic only.** The connection dot turns red, but there's no lockout. An operator could keep dragging sliders that queue up and fire when WS reconnects — causing a burst of motion commands.

7. **No workspace/collision envelope awareness.** The 2D viz shows joint angles but not whether the end-effector is about to hit the table, a person, or the arm itself. No self-collision detection, no workspace boundary visualization.

8. **Camera feeds are optional and hidden by default toggle.** For a real arm, the operator should always see what the arm is doing. Cameras hidden = operating blind.

9. **Error state is a tiny badge.** `ERR 7` in a 10px badge is not how you communicate a fault condition on hardware that can hurt someone. Errors need full-screen interruption with explanation and required acknowledgment.

10. **No operator authentication or session locking.** Anyone on the network can open the page and command the arm. No login, no lock screen, no audit trail of *who* sent commands.

### Moderate Issues

- No joint velocity or torque display — you can't tell if a joint is stalling or overloaded
- Gripper force feedback absent — operator can crush objects without knowing
- No indication of whether arm is currently moving vs stationary
- Task execution gives no progress indication (trajectory % complete)
- Log panel is tiny and mixes info/error without filtering

---

## 2. Operator Workflow Analysis

### Phase 1: Startup
**What operator needs:**
- System health check: DDS connected? Cameras online? Telemetry DB writing?
- Pre-flight checklist (displayed, not memorized): workspace clear, e-stop hardware accessible, cameras positioned
- Power-on sequence with explicit confirmation gates

**V2 design:** A startup wizard / checklist mode that must be completed before any motion commands are available. Each item requires operator acknowledgment.

### Phase 2: Calibration
**What operator needs:**
- Move joints to known positions (home/zero)
- Verify encoder feedback matches expected positions
- Check camera alignment
- Verify gripper open/close cycle

**V2 design:** Dedicated calibration mode with step-by-step guidance, comparison of commanded vs feedback angles, and pass/fail indicators per joint.

### Phase 3: Manual Jogging
**What operator needs:**
- Move one joint at a time with precise control
- Adjustable speed (crawl / normal / fast)
- Dead-man control: motion ONLY while button is held
- Real-time position + velocity readout
- Camera view of workspace
- Clear indication of joint limit proximity

**V2 design:** Jog mode with hold-to-move buttons (not sliders), speed selector, and joint limit proximity bars that turn yellow→red as limits approach.

### Phase 4: Task Execution
**What operator needs:**
- Select task, preview trajectory (on viz before execution)
- Pre-motion countdown with cancel option
- Progress bar during execution
- Pause/resume capability (not just stop)
- Camera feed prominent to watch arm

**V2 design:** Task mode with trajectory preview, 3-second countdown, progress overlay, and pause button.

### Phase 5: Monitoring (Autonomous Operation)
**What operator needs:**
- Hands-off monitoring with large status indicators
- Joint positions, velocities, torques in dashboard format
- Camera feeds large and center
- Alerts for anomalies (tracking error, stall, unexpected contact)
- Historical telemetry graphs

**V2 design:** Monitor mode — cameras and telemetry front-and-center, controls minimized.

### Phase 6: Emergency
**What operator needs:**
- E-stop accessible in <0.5 seconds from any screen state
- Clear post-stop status: what happened, what state is the arm in
- Recovery procedure guidance
- Incident logging (automatic)

---

## 3. Mode-Based UI Architecture

V2 should have **explicit operating modes** with distinct layouts:

### Mode Definitions

| Mode | Purpose | Controls Available | Layout Priority |
|------|---------|-------------------|-----------------|
| **Startup** | Pre-flight checks | Power on/off, system checks | Checklist + status |
| **Jog** | Manual positioning | Hold-to-jog per joint, speed select, gripper | Joint controls + cameras |
| **Task** | Execute trajectories | Task select, preview, run, pause, stop | Viz + progress + cameras |
| **Monitor** | Passive observation | None (read-only except e-stop) | Cameras + telemetry |
| **Debug** | Development only | Raw commands, full telemetry | Pipeline + events |
| **Lockout** | Post-error/post-estop | Error details, recovery steps, acknowledge | Error info only |

### Mode Transition Rules
- **Startup → Jog/Task:** Only after checklist complete AND power on AND enabled
- **Any → Lockout:** Automatic on error, e-stop, or connection loss
- **Lockout → Startup:** Only after error acknowledged and cleared
- **Any → Monitor:** Always available (read-only)
- **Jog ↔ Task:** Free transition when no task is running

### What Changes Per Mode
- Available controls (buttons, sliders) change — unavailable controls are **hidden**, not just disabled
- Layout reflows to prioritize what matters for that mode
- Camera feed size adjusts (small in startup, large in jog/task/monitor)
- Telemetry panel shows different metrics per mode

---

## 4. Always-Visible Safety Elements

These must be visible in **every mode, every screen size, at all times:**

### Top Safety Bar (fixed, full width, ~48px tall)
```
┌─────────────────────────────────────────────────────────────┐
│ [E-STOP]  ●CONNECTED  PWR:ON  MTR:ENABLED  ERR:NONE  MODE │
│  (large)   (dot+text)  (badge)  (badge)     (badge)  (tag) │
└─────────────────────────────────────────────────────────────┘
```

- **E-stop button:** Always top-left, minimum 48x48px touch target, requires NO scrolling to reach. Red, high contrast, works even if JS is partially broken (should also have a hardware e-stop — this is the software backup).
- **Connection status:** Dot + text + latency. When disconnected: entire bar turns red, all controls lock.
- **Power state:** Green/gray badge.
- **Motor state:** Green/gray badge. When enabled: subtle pulse to remind operator the arm is live.
- **Error state:** When non-zero: bar turns amber/red, error code + human-readable message, requires acknowledgment to dismiss.
- **Current mode:** So operator always knows what mode they're in.
- **Motion indicator:** Animated icon when arm is actively moving (any joint velocity > threshold). Critical so operator knows the arm is NOT stationary.

### Additional Always-Visible
- **Joint limit proximity:** Thin colored bar under each joint readout — green when centered, yellow within 15° of limit, red within 5°. Visible in all modes as a compact strip.
- **Stale data warning:** If no state update in >1 second, overlay a "STALE DATA" warning across the entire UI.

---

## 5. Camera Feed Integration

### Placement
- **Jog mode:** Cameras take 40% of the left panel, side-by-side, above the arm viz. Operator needs to see what the arm is doing while jogging.
- **Task mode:** Cameras take 50% of screen, with arm viz overlaid semi-transparently or in a small inset. The real view matters more than the schematic during execution.
- **Monitor mode:** Cameras are 70% of screen. This is a surveillance view.
- **Startup:** Small thumbnails to confirm cameras are online.

### Features to Add
- **Click-to-enlarge:** Click either camera to go full-panel.
- **Overlay toggle:** Overlay joint angle readouts on the camera feed (HUD-style).
- **Recording indicator:** If recording, show red dot + duration.
- **Frame rate display:** Per camera, always visible. If FPS drops below threshold, warn.
- **Snapshot with annotation:** Snap + add text note, saved with timestamp for incident review.

---

## 6. Telemetry: Real-Time vs Historical

### Real-Time (Always in sidebar or overlay)
| Metric | Why | Alert Threshold |
|--------|-----|-----------------|
| Joint positions (°) | Core feedback | Near limits |
| Joint velocities (°/s) | Detect stalls, unexpected motion | >threshold or unexpected |
| Gripper position (mm) | Object handling | — |
| Tracking error (cmd vs actual) | Detect mechanical issues | >2° sustained |
| Command latency (e2e) | System health | >100ms |
| Connection age | Freshness | >500ms |
| Motion state (moving/stopped) | Awareness | — |

### Historical (Telemetry page / graphs, not main panel)
| Metric | Why |
|--------|-----|
| Joint position traces over time | Post-task analysis, debugging |
| Tracking error over time | Detect degradation |
| Command rate history | Usage patterns |
| Error/event timeline | Incident investigation |
| Temperature (if available) | Motor thermal protection |
| Gripper force profile | Grasp analysis |

### V2 Telemetry Widget
A compact real-time telemetry strip at the bottom of the main panel:
```
J0: 45.2°  J1: -12.0°  J2: 30.5°  J3: 0.0°  J4: -5.2°  J5: 10.0°  G: 32.1mm
vel: 2.1    vel: 0.0     vel: 1.3    vel: 0.0   vel: 0.4    vel: 0.0   
err: 0.1    err: 0.0     err: 0.3    err: 0.0   err: 0.1    err: 0.0   [e2e: 23ms]
```
Color-coded: green=nominal, yellow=caution, red=alert.

---

## 7. Specific V2 UI Changes

### 7.1 Replace Sliders with Jog Buttons
Sliders are dangerous for real arm control. Replace with:
- **−/+ buttons per joint** with hold-to-move behavior (dead-man pattern)
- **Speed selector:** 0.5°/s (crawl), 5°/s (normal), 20°/s (fast)
- **Step mode toggle:** Single press = move exactly 1° (or 0.1° in fine mode)
- Keep sliders only in Debug mode for development convenience

### 7.2 Confirmation Dialogs
Required for:
- Power On (shows checklist)
- Enable Motors ("Arm will become live. Workspace clear?")  
- Any task execution (shows trajectory preview + countdown)
- Power Off while task is running

NOT required for:
- E-stop (must be instant)
- Jog motion (dead-man pattern is the safety mechanism)
- Disable motors (always safe to disable)

### 7.3 Connection Loss Lockout
When WebSocket disconnects:
1. Immediately disable all motion controls
2. Full-width red banner: "CONNECTION LOST — CONTROLS LOCKED"
3. On reconnect: require operator to acknowledge before re-enabling controls
4. If smoother has queued commands during disconnect, discard them

### 7.4 Error Handling
When `error != 0`:
1. Mode auto-transitions to **Lockout**
2. Full-screen overlay with error code, human-readable explanation, and timestamp
3. Operator must acknowledge error
4. Show recovery steps (e.g., "Power cycle required" or "Check joint 3 encoder")
5. Log incident automatically with full state snapshot

### 7.5 Post-E-Stop Recovery
After e-stop:
1. Show what triggered it (operator press vs error vs connection loss)
2. Show current joint positions (may have drifted under gravity)
3. Guided recovery: "1. Clear workspace → 2. Power On → 3. Enable → 4. Slowly jog to safe position"
4. Require full startup checklist before returning to normal operation

### 7.6 Remove Raw Command from Main UI
Move raw command input to Debug mode only. It has no place in an operator-facing interface. An accidental paste into that field could send arbitrary commands to the arm.

---

## 8. Keyboard Shortcuts

| Key | Action | Guard |
|-----|--------|-------|
| `Space` | E-stop | Always active |
| `Escape` | Cancel current task / close dialog | — |
| `1-6` | Select joint for jogging | Jog mode only |
| `←/→` | Jog selected joint −/+ | Jog mode, hold-to-move |
| `Shift+←/→` | Fine jog (0.1° steps) | Jog mode |
| `G` | Toggle gripper open/close | Jog mode |
| `H` | Go home task | Task mode, with confirmation |
| `M` | Cycle mode | — |
| `Tab` | Cycle camera focus | — |

**Space for e-stop must work regardless of focus state** — attach to `document`, not any specific element.

---

## 9. Mobile / Tablet Considerations

Since operators may use tablets near the arm:
- E-stop button: minimum 64x64px, positioned for thumb reach
- Jog controls: large touch targets, no hover-dependent interactions
- Camera feeds: swipe between left/right
- Mode switching: bottom tab bar, not top menu
- Landscape orientation preferred, but portrait must work for e-stop access

---

## 10. Implementation Priority

### P0 — Before any real-arm operation
1. E-stop always visible (fixed position, keyboard shortcut)
2. Connection loss lockout
3. Enable motors confirmation gate
4. Remove raw command from default view
5. Stale data warning overlay

### P1 — Before regular use
6. Mode-based UI (at minimum: Jog vs Task vs Monitor)
7. Dead-man jog controls replacing sliders
8. Pre-motion countdown for tasks
9. Error lockout with acknowledgment
10. Joint limit proximity indicators

### P2 — Quality of life
11. Camera feed integration improvements
12. Real-time telemetry strip
13. Task trajectory preview
14. Keyboard shortcuts
15. Startup checklist wizard

### P3 — Polish
16. Operator authentication
17. Incident logging/export
18. Mobile-optimized layout
19. Audible warnings
20. Historical telemetry dashboards

---

## Summary

V1 is a capable development tool but it is **not safe for routine operation**. The biggest risks are: (1) no motion lockout on connection loss, (2) e-stop can scroll off screen, (3) sliders send commands without dead-man intent verification, and (4) no confirmation before enabling a live arm. V2 must treat every command as potentially dangerous and design the UI around the assumption that operators will make mistakes.
