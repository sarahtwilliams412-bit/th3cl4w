# th3cl4w V2 — Layout & Information Architecture Proposal

**Author:** UX/UI Design Architect  
**Date:** 2026-02-07  
**Role:** Layout & Information Architecture

---

## Executive Summary

V1 works but has a classic "developer dashboard" problem: everything is visible at once, competing for attention. When you're operating a robotic arm, **safety and spatial awareness must dominate**. Joint sliders and debug telemetry should not share visual priority with the E-stop button.

V2 proposes a **command-center layout** with strict information hierarchy, zone-based grouping, and a responsive design that degrades gracefully to tablet (not phone — nobody should operate a robotic arm from a phone).

---

## V1 Problems

1. **E-stop buried** — it's mid-scroll in the right panel, below joint sliders. Unacceptable.
2. **No spatial separation between safety and motion controls** — Power On and joint sliders are visually adjacent.
3. **3D viz and cameras fight for left panel** — cameras collapse the viz space.
4. **Debug panel is a binary toggle** — either hidden or eating 340px of vertical space.
5. **Action log is tiny and fixed** — 150px strip at the bottom, hard to scan.
6. **No concept of operational modes** — setup vs. operation vs. diagnostics are all one view.
7. **Mobile breakpoint at 800px just stacks everything** — not actually usable.

---

## Design Principles

1. **Safety controls are always visible and always reachable** — E-stop, power state, and error indicators never scroll, never hide, never require a click to access.
2. **Spatial awareness is primary** — The 3D visualization is the largest element. An operator's eyes should be on the arm, not on buttons.
3. **Progressive disclosure** — Show what's needed for the current operational phase. Don't show joint control if the arm isn't powered and enabled.
4. **Dark industrial aesthetic** — High contrast for critical elements, low contrast for secondary info. Think aircraft cockpit, not SaaS dashboard.
5. **Minimum viable viewport: 1024×768** — This is an operator tool, not a consumer app. Tablet landscape is the floor.

---

## Information Hierarchy (Priority Order)

| Priority | Element | Rationale |
|----------|---------|-----------|
| P0 | E-Stop | Must be hittable in <300ms from any state |
| P0 | Error/Fault indicators | Operator must see faults instantly |
| P1 | Arm visualization (3D canvas) | Primary spatial feedback |
| P1 | Camera feeds | Secondary spatial feedback |
| P1 | Power/Enable state badges | Operational readiness at a glance |
| P2 | Joint control sliders | Primary manipulation interface |
| P2 | Gripper control | Part of manipulation |
| P3 | Task presets (Home/Ready/Wave) | Convenience, not critical |
| P3 | Connection status | Important but not moment-to-moment |
| P4 | Action log | Review, not real-time decision-making |
| P5 | Debug/Telemetry | Development only |
| P5 | Raw command input | Development only |

---

## Layout: Three-Zone Command Center

### Desktop (≥1024px)

```
┌─────────────────────────────────────────────────────────────────────┐
│ TOPBAR: logo │ conn │ badges [PWR|EN|ERR] │ spacer │ [ESTOP]██████ │
├────────────────────────────────────┬────────────────────────────────┤
│                                    │  CONTROL RAIL (right, 340px)   │
│                                    │ ┌──────────────────────────┐   │
│                                    │ │ ● SAFETY ZONE            │   │
│                                    │ │   [Power On] [Power Off] │   │
│     PRIMARY VIEWPORT               │ │   [Enable]  [Disable]   │   │
│     (3D arm visualization)         │ │   State: POWERED/READY  │   │
│                                    │ ├──────────────────────────┤   │
│     Canvas fills available space   │ │ ● JOINT CONTROL          │   │
│     Aspect ratio preserved         │ │   J0 ═══════●══════ 45° │   │
│                                    │ │   J1 ══●═══════════ -12° │   │
│                                    │ │   J2 ═════●════════  8°  │   │
│                                    │ │   J3 ════════●═════ 22°  │   │
│                                    │ │   J4 ═══════●══════ 15°  │   │
│                                    │ │   J5 ══════════●═══ 90°  │   │
│                                    │ │   G  ═══●══════════ 12mm │   │
│                                    │ ├──────────────────────────┤   │
│                                    │ │ ● TASKS                  │   │
│                                    │ │   [Home] [Ready] [Wave]  │   │
│                                    │ │   [Stop Task]            │   │
│                                    │ └──────────────────────────┘   │
├────────────────────────────────────┴────────────────────────────────┤
│ DRAWER BAR: [📷 Cameras ▼] [📋 Log ▼] [🔧 Debug ▼] [⌨ Raw ▼]     │
├─────────────────────────────────────────────────────────────────────┤
│ DRAWER CONTENT (expandable, 0–400px)                                │
│ Shows whichever drawer tab is active. Cameras show side-by-side.   │
│ Log shows scrollable entries. Debug shows pipeline + stats.         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Decisions

**E-Stop in the topbar, right-aligned, oversized.** It's always visible. Always. No scrolling to find it. It gets 120px+ of width and a distinct red background that contrasts with everything. On hover, the entire topbar border flashes red. This is non-negotiable.

**Control Rail is a fixed-width right sidebar (340px).** It doesn't scroll with the page. It has three visually distinct zones separated by horizontal rules with contrasting header backgrounds:
- **Safety Zone** (top, dark red-tinted background): Power/Enable toggles. These are gated — you can't enable without power, visually enforced with disabled states.
- **Joint Control** (middle, neutral): The six sliders plus gripper. These are greyed out / non-interactive when the arm isn't enabled. Visual lockout, not just `disabled` attribute.
- **Tasks** (bottom, neutral): Preset motions. Also locked when not enabled.

**Primary Viewport takes all remaining space.** The 3D canvas is the hero. It fills `calc(100vw - 340px)` width and `calc(100vh - topbar - drawer_bar)` height. No cameras competing for space here.

**Bottom Drawer replaces fixed panels.** Instead of a permanently visible 150px log or a 340px debug panel, V2 uses a tabbed drawer system at the bottom. Only one drawer open at a time. Drawers slide up and push the viewport smaller (not overlay). Default state: all closed. Tabs in the drawer bar show notification counts (e.g., "Log (3)" for new errors).

---

### Tablet Landscape (768–1023px)

```
┌─────────────────────────────────────────────────┐
│ TOPBAR: logo │ conn │ badges │ sp │ [ESTOP]████ │
├─────────────────────────────────────────────────┤
│                                                  │
│          PRIMARY VIEWPORT (full width)           │
│          3D arm visualization                    │
│          (reduced height: ~50vh)                 │
│                                                  │
├─────────────────────────────────────────────────┤
│ TAB BAR: [Controls] [Cameras] [Log] [Debug]     │
├─────────────────────────────────────────────────┤
│ TAB CONTENT (scrollable)                         │
│ Controls tab shows Safety + Joints + Tasks       │
│ Cameras tab shows feeds                          │
│ Log/Debug as before                              │
└─────────────────────────────────────────────────┘
```

On tablet, the control rail moves below the viewport as a tabbed panel. The viewport goes full-width but shorter (~50vh). Tabs replace the sidebar. E-stop remains in topbar, always visible.

---

### Portrait / Phone (<768px)

```
┌──────────────────────────┐
│  ⚠ ROTATE DEVICE ⚠      │
│  This application        │
│  requires landscape      │
│  orientation (≥768px)    │
│                          │
│  [Continue Anyway →]     │
└──────────────────────────┘
```

I'm serious. Don't design for portrait phone. Show a rotation prompt. If they insist, give them a read-only status view: badges + camera feeds + log. No joint control. No motion commands from a phone in portrait mode.

---

## Panel Detail Specs

### Topbar (48px fixed)

```
┌──────────────────────────────────────────────────────────────────┐
│  th3cl4w  ● CONNECTED  ● live    │   PWR ON │ ENABLED │ NO ERR │
│           ↑             ↑        │          │         │        │
│        conn dot      quality     │     status badges           │
│                                  │                   ┌────────┐│
│                                  │                   │ ESTOP  ││
│                                  │                   │  ⚠     ││
│                                  │                   └────────┘│
└──────────────────────────────────────────────────────────────────┘
```

- Logo: left-aligned, `font-family: mono`, `color: var(--danger)`, 18px
- Connection indicator: dot + label + latency quality
- Status badges: pill-shaped, color-coded (green=active, red=error, grey=off)
- E-Stop: **right-aligned, 120px wide, 40px tall, full red background, white text, always visible**. Not a small button. A landing strip.

### Safety Zone

Visual treatment: Subtle dark red background (`rgba(233,69,96,0.05)`), left border accent in red (`3px solid var(--danger)`).

Contains:
- Power On / Power Off (toggle pair, mutually exclusive visual state)
- Enable / Disable (toggle pair, gated on power)
- Current state summary: text like "ARM READY" or "POWERED — NOT ENABLED" or "DISABLED"
- Error display: if `error !== 0`, show error code prominently with description

### Joint Control Zone

Visual treatment: Neutral dark card background. Left border accent in blue (`3px solid var(--info)`).

Each joint row:
```
J0  Base Yaw     ═══════●══════  +45.0°  [-135, +135]
```

- Joint ID (monospace, dim)
- Joint name (small, dim) — show on hover or always if space permits
- Slider with custom thumb (12px circle, blue accent)
- Current value (monospace, bright)
- Range limits (tiny, very dim) — shown on hover tooltip

**New: Joint group headers.** Group joints semantically:
```
── BASE ──
J0  Base Yaw

── ARM ──
J1  Shoulder Pitch
J2  Elbow Pitch

── WRIST ──
J3  Wrist Roll
J4  Wrist Pitch
J5  Wrist Roll

── EFFECTOR ──
G   Gripper
```

**New: Lock toggles per joint.** Small 🔒 icon next to each slider to lock individual joints during manipulation. Prevents accidental bumps.

**Interaction gating:** When arm is not enabled, sliders render at 30% opacity with a subtle "ENABLE ARM TO CONTROL" overlay. They track incoming state but don't send commands.

### Task Zone

Simple button row. Same gating as joints.

Add: **active task indicator.** When a trajectory is executing, show:
```
▶ Executing: "wave" — 2.3s remaining ████████░░░░
[Stop Task]
```

### Camera Drawer

```
┌──────────────────────────────────────────────────┐
│ Camera 1 (Left)  ● LIVE     Camera 2 (Right) ● LIVE  │
│ ┌──────────────────┐       ┌──────────────────┐      │
│ │                  │       │                  │      │
│ │   MJPEG Feed     │       │   MJPEG Feed     │      │
│ │   4:3 aspect     │       │   4:3 aspect     │      │
│ │                  │       │                  │      │
│ └──────────────────┘       └──────────────────┘      │
│ [📷 Snap] [⛶ Fullscreen]  [📷 Snap] [⛶ Fullscreen] │
└──────────────────────────────────────────────────┘
```

- Feeds side-by-side when drawer is open
- Stream paused when drawer is closed (bandwidth saving, same as V1)
- New: fullscreen button per feed for close inspection
- New: PiP (picture-in-picture) button to float a feed over the 3D viz

### Log Drawer

- Scrollable, monospace, color-coded by level
- **New: level filter buttons** — [ALL] [INFO] [WARN] [ERROR]  
- **New: search/filter input** for scanning specific actions
- **New: clear button** with confirmation
- **New: unread count badge** on drawer tab when errors arrive while closed

### Debug Drawer

Same content as V1 (pipeline, stats, event stream) but better organized:

```
┌─────────────────────────────────────────────────────────────────┐
│ Pipeline:  CMD → DDS TX → ARM RX → EXEC → STATE → WS TX → UI │
│             ●      ●        ●       ●       ●       ●      ●  │
│            2ms    4ms      8ms    12ms     1ms     2ms    1ms  │
├──────────────────┬──────────────────┬───────────────────────────┤
│ DDS              │ Cameras          │ Latency                   │
│ TX: 10/s         │ C0: 15.0 fps    │ Cmd→Ack: 6ms              │
│ RX: 10/s         │ C1: 15.0 fps    │ Ack→Exec: 8ms             │
│ Last RX: 42ms    │ Motion: 0.12    │ End-to-end: 30ms          │
├──────────────────┴──────────────────┴───────────────────────────┤
│ Event Stream (filterable)                                       │
│ 19:04:32 CMD_SENT   a3f2b1c8 {"action":"set-joint","id":0...}  │
│ 19:04:32 DDS_PUBLISH a3f2b1c8 {"topic":"joint_cmd"...}         │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Color System

```
Token             Value       Usage
──────────────────────────────────────────────
--bg              #0f0f17     Page background (darker than V1)
--surface         #161625     Card/panel background
--surface-raised  #1c1c32     Elevated cards, drawer bg
--border          #2a2a44     Default borders
--border-focus    #4a90d9     Focused/active borders

--danger          #e94560     E-stop, errors, faults
--danger-bg       #2d0f15     Danger zone tint
--success         #53d769     Connected, enabled, OK
--success-bg      #0f2d15     Success zone tint
--warning         #f0ad4e     Warnings, caution
--info            #4a90d9     Links, pitch joints, interactive

--text            #e8e8f0     Primary text
--text-dim        #6b6b88     Secondary text
--text-muted      #3d3d55     Tertiary (timestamps, ranges)

--joint-pitch     #4a90d9     Pitch joint indicators (blue)
--joint-roll      #e8a838     Roll joint indicators (gold)
--gripper         #53d769     Gripper open state
```

### Typography

```
--font-mono: 'JetBrains Mono', 'Fira Code', monospace
--font-sans: 'Inter', -apple-system, sans-serif

Sizes:
  Topbar logo:      18px mono, weight 700, letter-spacing 2px
  Section headers:  11px mono, uppercase, letter-spacing 2px, --text-dim
  Body controls:    13px sans
  Values/readouts:  12px mono
  Timestamps:       10px mono, --text-muted
  Tiny labels:      9px mono, --text-muted
```

---

## Interaction Patterns

### Operational State Machine (Visual Gating)

The UI should reflect the arm's state machine visually:

```
DISCONNECTED → CONNECTED → POWERED → ENABLED → OPERATING
     │              │          │          │          │
     │              │          │          │          └─ Full control
     │              │          │          └─ Joints + tasks unlocked
     │              │          └─ Enable button unlocked
     │              └─ Power buttons unlocked
     └─ Read-only, reconnecting...
```

Each state gates the next level of controls. Not just `disabled` attributes — visual lockout with overlay text explaining what's needed: *"Power on the arm to continue"*.

### Slider Interaction Improvements

- **Dead-band on state updates:** V1 has this (0.25°). Keep it.
- **New: Double-click slider to type exact value.** Inline number input replaces slider temporarily.
- **New: Right-click joint → "Go to 0°"** quick reset.
- **New: Shift+drag for fine control** (0.1° steps instead of 0.5°).
- **Throttle stays at 50ms / 20Hz.** Server handles smoothing.

### Keyboard Shortcuts

```
Key          Action
─────────────────────────────
Space        E-Stop (global, always active)
1-6          Select joint (arrow keys to adjust)
G            Select gripper
H            Home task
Escape       Stop current task
D            Toggle debug drawer
C            Toggle camera drawer
L            Toggle log drawer
```

---

## 3D Visualization Enhancements (Viewport)

The canvas itself is the viz architect's domain, but the viewport container should support:

- **Fullscreen toggle** (F key or button)
- **Camera preset buttons** overlaid: [Side] [Top] [Front] [Iso]
- **Grid toggle**
- **Joint labels toggle**
- **Coordinate frame display** (XYZ arrows at base)
- **Ghost arm** — show target position as transparent overlay when a trajectory is executing

---

## Component Tree (for implementation)

```
<App>
  <Topbar>
    <Logo />
    <ConnectionIndicator />
    <StatusBadges />  (power, enabled, error)
    <EStopButton />
  </Topbar>
  
  <MainArea>
    <Viewport>
      <ArmCanvas />
      <ViewportOverlay>
        <CameraPresets />
        <FullscreenToggle />
      </ViewportOverlay>
    </Viewport>
    
    <ControlRail>        <!-- desktop only, becomes tab on tablet -->
      <SafetyZone>
        <PowerControls />
        <StateIndicator />
        <ErrorDisplay />
      </SafetyZone>
      <JointControl>
        <JointGroup label="Base">
          <JointSlider joint={0} />
        </JointGroup>
        <JointGroup label="Arm">
          <JointSlider joint={1} />
          <JointSlider joint={2} />
        </JointGroup>
        <JointGroup label="Wrist">
          <JointSlider joint={3} />
          <JointSlider joint={4} />
          <JointSlider joint={5} />
        </JointGroup>
        <GripperSlider />
      </JointControl>
      <TaskZone>
        <TaskButtons />
        <ActiveTaskIndicator />
      </TaskZone>
    </ControlRail>
  </MainArea>
  
  <DrawerBar>
    <DrawerTab icon="📷" label="Cameras" badge={0} />
    <DrawerTab icon="📋" label="Log" badge={errorCount} />
    <DrawerTab icon="🔧" label="Debug" badge={0} />
    <DrawerTab icon="⌨" label="Raw" badge={0} />
  </DrawerBar>
  
  <DrawerContent active={activeDrawer}>
    <CameraDrawer />
    <LogDrawer />
    <DebugDrawer />
    <RawCommandDrawer />
  </DrawerContent>
</App>
```

---

## What V2 Drops

- **Log panel as permanent fixture.** It becomes a drawer. Most of the time you don't need it.
- **Debug button in topbar.** Debug is a drawer tab now. Topbar is sacred — only status + safety.
- **Telemetry link in topbar.** Move to debug drawer or a settings menu.
- **Camera panel in left sidebar.** Cameras get their own drawer. The viewport is for the 3D viz only.

## What V2 Adds

- Keyboard shortcuts
- Operational state gating (visual lockout)
- Joint grouping with semantic labels
- Per-joint lock toggles
- Task progress indicator
- Drawer system (cameras, log, debug, raw command)
- Log filtering and search
- Camera PiP and fullscreen
- Double-click-to-type on sliders
- Portrait mode gate
- Error count badges on drawer tabs
- Ghost arm trajectory preview

---

## Implementation Notes

- **Stay vanilla.** V1 is zero-dependency. V2 should remain so. No React, no build step. This runs on a FastAPI static mount and needs to Just Work.
- **CSS Grid for main layout.** The three-zone layout (topbar / main / drawer) is a simple grid. The main area is a flexbox with viewport + rail.
- **Drawer animation:** CSS `max-height` transition on the drawer content div. ~200ms ease-out.
- **Keyboard handler:** Single global `keydown` listener with a `shortcutsEnabled` flag (disabled when typing in raw command input).
- **State machine gating:** Pure CSS via data attributes on `<body>`: `data-power="on|off"` `data-enabled="true|false"` `data-connected="true|false"`. Use CSS selectors like `body:not([data-enabled="true"]) .joint-slider { opacity: 0.3; pointer-events: none; }`.

---

*This layout prioritizes what matters when you're controlling a physical robot: safety first, spatial awareness second, fine control third, diagnostics last. Everything else is noise.*
