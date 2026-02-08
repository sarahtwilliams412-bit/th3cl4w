# V2 Interaction Design & Components Proposal

**Author:** Frontend Engineer (Interaction Design role)  
**Date:** 2026-02-07  
**Scope:** Joint controls, 3D viz, real-time feedback, animations, keyboard shortcuts, task UX, e-stop UX

---

## 1. Joint Control — Hybrid Slider/Knob System

### Problems with V1
- Native `<input type="range">` sliders have tiny 12px thumbs — unusable on touch, imprecise on desktop
- No visual distinction between pitch and roll joints
- No way to fine-tune: you either drag or you don't
- No indication of target vs actual position (tracking error invisible)

### V2 Design: Dual-Track Slider with Numeric Stepper

Each joint gets a custom component with three interaction modes:

```
┌─────────────────────────────────────────────────────────┐
│ J1  Shoulder Pitch                                      │
│ ──────────────●══════════════════○──────────── [  45.0°] │
│              actual            target          ↑↓ step  │
│ ◄ -90° ─────────────────────────────── +90° ►           │
│         [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░]  ← range bar   │
└─────────────────────────────────────────────────────────┘
```

**Component: `<joint-slider>`** (Custom Element, no framework)

- **Track:** 8px tall, rounded, with gradient showing safe (blue) → limit (orange → red) zones
- **Thumb (target):** 24×24px circle, draggable, shows commanded position — ring outline, hollow center
- **Indicator (actual):** 18×18px filled circle on same track, shows feedback position — NOT draggable
- **Gap between them** = tracking error, visually obvious as a colored arc/segment
- **Numeric input:** Editable `<input type="number">` at right, step=0.5° default. Click to type exact values. Arrow keys step ±0.5°, Shift+arrow ±5°
- **Touch:** Thumb hit-target expanded to 44×44px (Apple HIG minimum). Touch-drag with momentum disabled (immediate stop on release)
- **Double-tap thumb:** Opens a precision popover with ±0.1° step buttons and a rotary dial
- **Color coding:** Pitch joints = blue track, Roll joints = gold/amber track (matches V1 viz colors)

**Implementation:** Vanilla JS custom element `class JointSlider extends HTMLElement`. ~200 lines. Canvas-rendered track for smooth gradients and the dual-indicator overlay. Pointer events (not mouse events) for unified touch/mouse/pen.

### Gripper Control
Same component but horizontal with mm units, 0–65mm range. Add a squeeze/release icon that animates with position. Visual: two finger icons that open/close proportionally.

### Rotary Knob Alternative (for roll joints J0, J3, J5)
Since roll joints rotate ±135°, a **rotary knob** is more intuitive:

```
        ╭───────╮
       ╱    ●    ╲      ← needle shows angle
      │   ╱   ╲   │     ← arc shows range used
      │  ╱  ⊙  ╲  │     ← center dot
       ╲         ╱
        ╰───────╯
       J0  45.0°
```

**Component: `<rotary-knob>`** — Canvas-rendered circle, ~150 lines. Drag to rotate, scroll-wheel for fine adjustment. Same dual-indicator (target ring vs actual needle) pattern. Touch: drag anywhere in the knob area, angle follows pointer relative to center.

---

## 2. 3D Visualization

### V1 Analysis
V1 uses 2D Canvas with manual FK chain drawing — side view only. It works but:
- No depth perception for yaw (J0) — shown as separate mini compass
- Roll joints shown as arc indicators bolted onto the 2D view
- Can't rotate the view to inspect the arm from different angles
- No ground plane, no shadows, no spatial reference

### V2 Recommendation: **Three.js** (lightweight, no build step needed)

**Why Three.js over keeping Canvas 2D:**
- Three.js r160+ is 150KB gzipped (single file via CDN or local). No build tooling needed — `<script type="module">` import
- Gives us true 3D with orbit controls, so operators can rotate/zoom the arm freely
- Shadows and ground plane provide real spatial awareness
- The FK chain is the same math, just applied to 3D transforms instead of 2D points
- GPU-accelerated — actually lighter on CPU than the current Canvas 2D redraw

**Why NOT a heavier framework:**
- No React Three Fiber, no Babylon.js — Three.js is the sweet spot for vanilla JS
- Total viz module: ~400 lines of JS

### 3D Scene Design

```
Scene Layout:
┌────────────────────────────────────┐
│          Orbit Camera              │
│                                    │
│     ╱─────╲                        │
│    ╱   ARM  ╲   ← Arm model       │
│   ╱    ◆     ╲     (cylinders     │
│  ╱     │      ╲     + spheres)    │
│ ╱      │       ╲                   │
│ ───────┴────────── ← Ground grid   │
│                                    │
│  [Ghost arm]  ← transparent        │
│               overlay showing      │
│               target pose          │
└────────────────────────────────────┘
```

**Arm Model:**
- Each link = `CylinderGeometry` with rounded caps, colored by joint type (blue/amber)
- Each joint = `SphereGeometry` with emissive glow when active/moving
- Gripper = two thin box geometries that open/close
- No GLTF/OBJ models needed — procedural geometry keeps it dependency-free

**Ghost Arm (target overlay):**
- Semi-transparent (opacity 0.3) duplicate of the arm showing the *commanded* position
- When target ≈ actual (tracking error < 0.5°), ghost fades out completely
- When tracking error is large, ghost is clearly visible as a "where the arm is going"
- Color: white/cyan ghost vs solid blue/amber actual

**Ground Plane:**
- Infinite grid helper (Three.js `GridHelper`), subtle gray lines
- Shadow from a directional light gives depth cue
- Small coordinate axes (RGB = XYZ) in corner

**Camera:**
- `OrbitControls` — drag to rotate, scroll to zoom, right-drag to pan
- Preset views: Top, Front, Side, Iso — buttons or keyboard shortcuts (T/F/S/I)
- Camera smoothly animates between preset positions (GSAP-like lerp, no extra lib — just `requestAnimationFrame` with easing)

**Performance:**
- Only update joint transforms on state change (same `vizDirty` flag pattern from V1)
- 60fps render loop but geometry updates at 10Hz (matching WS rate)
- On mobile: reduce shadow resolution, disable anti-aliasing

---

## 3. Real-Time Feedback Patterns

### 3.1 Tracking Error Visualization

**On sliders:** Gap between target (hollow circle) and actual (filled circle) on the track. If error > 2°, the gap segment turns orange. If > 5°, red + pulse animation.

**On 3D viz:** Ghost arm separation. Plus a small HUD overlay:
```
┌─ TRACKING ──────────┐
│ J0  0.2°  ▓░░░░░░░  │  ← bar chart of per-joint error
│ J1  1.4°  ▓▓▓░░░░░  │
│ J2  0.1°  ▓░░░░░░░  │
│ J3  4.8°  ▓▓▓▓▓▓░░  │  ← orange when > 2°
│ J4  0.3°  ▓░░░░░░░  │
│ J5  0.0°  ░░░░░░░░  │
│ Σ   6.8°            │  ← total error
└─────────────────────┘
```

Rendered as a small overlay in the bottom-left of the 3D viewport. Toggle with `E` key.

### 3.2 Latency Indicator

**Topbar latency pill:**
```
[RTT 12ms] ← green when < 50ms, yellow < 200ms, red > 200ms
```

Measured via WebSocket ping/pong or timestamp delta. Replace the current `connQuality` text with this structured indicator.

**Pipeline latency** (keep from V1 debug panel but make it always-accessible via a collapsed drawer, not a separate mode).

### 3.3 Arm State Machine Display

Replace the three separate badges (PWR, ENABLED, ERR) with a **state machine strip**:

```
┌──────────────────────────────────────────────────┐
│ ○ OFF  →  ◉ POWERED  →  ○ ENABLED  →  ○ MOVING │
│                ▲                                  │
│           (current state)                         │
└──────────────────────────────────────────────────┘
```

- Visual breadcrumb/pipeline showing the arm's state progression
- Current state highlighted with glow + color
- Error state shown as a red overlay on the entire strip with error code
- States: `DISCONNECTED → CONNECTED → POWERED → ENABLED → IDLE → MOVING → TASK_RUNNING`
- Clicking a state shows what actions are available from there

### 3.4 Connection Health

- **Heartbeat ring:** Around the connection dot, a circular progress ring that fills up every 100ms (WS interval). If a message is missed, the ring turns yellow. Two misses = red. Three = "STALE" label.
- **Reconnection:** Show a progress bar during reconnect attempts with attempt count.

---

## 4. Animation and Transitions

### Principles
- **Duration:** 150ms for UI state changes (button press, badge update), 300ms for panel open/close, 0ms for real-time data (sliders, viz)
- **Easing:** `cubic-bezier(0.4, 0, 0.2, 1)` (Material ease-out) for entrances, `cubic-bezier(0.4, 0, 1, 1)` for exits
- **No animation on data:** Joint values, angles, positions update instantly or with physics-based smoothing (already in V1's `vizJoints` lerp)

### Specific Animations

| Element | Trigger | Animation |
|---------|---------|-----------|
| State badges | State change | Color fade 150ms + brief scale pulse (1.0 → 1.05 → 1.0) |
| E-stop button | Arm enabled | Breathing glow (keep V1's flash, but smoother sine wave) |
| Task progress | Task running | Progress bar fills along task duration |
| Panel collapse | Toggle | Height slide 200ms with content fade |
| Error banner | Error state | Slide down from top, red, shake animation (3px, 3 cycles) |
| Ghost arm | Tracking error | Opacity lerp proportional to error magnitude |
| Joint glow | Joint moving | Emissive intensity pulses on the 3D sphere |
| Camera transition | Preset view | Smooth quaternion slerp over 500ms |
| Log entry | New entry | Fade-in + slide-up, 100ms |

### CSS Transitions (not JS animations where possible)
```css
.state-badge {
  transition: background-color 150ms ease-out, 
              color 150ms ease-out,
              transform 150ms ease-out;
}
.state-badge.changed {
  transform: scale(1.05);
}
```

---

## 5. Keyboard Shortcuts

### Design: Vim-inspired operator mode

All shortcuts work without modifier keys when no input is focused. When a text input is focused, shortcuts are disabled.

| Key | Action | Notes |
|-----|--------|-------|
| `Space` | **E-STOP** | Always active, even in text inputs. Most critical shortcut. |
| `1`–`6` | Select joint J0–J5 | Highlights joint, subsequent arrow keys control it |
| `G` | Select gripper | |
| `←` / `→` | Adjust selected joint ±1° | |
| `Shift+←/→` | Adjust selected joint ±5° | Coarse mode |
| `Alt+←/→` | Adjust selected joint ±0.1° | Fine mode |
| `H` | Home task | |
| `R` | Ready position task | |
| `W` | Wave task | |
| `Escape` | Stop current task | |
| `P` | Toggle power on/off | |
| `E` | Toggle enable/disable | |
| `T` | 3D view: Top | |
| `F` | 3D view: Front | |
| `S` | 3D view: Side | |
| `I` | 3D view: Isometric | |
| `D` | Toggle debug panel | |
| `C` | Toggle cameras | |
| `?` | Show shortcut overlay | |
| `Tab` | Cycle through joints | |
| `Shift+Tab` | Cycle backwards | |

### Visual Feedback
- Show a brief toast in bottom-center when a shortcut is activated: `"J2 selected"`, `"E-STOP!"`, etc.
- Selected joint highlighted on both the slider panel and the 3D model (brighter glow)
- Shortcut overlay (`?`) shows a semi-transparent modal with all shortcuts, dismisses on any key

### Implementation
```js
document.addEventListener('keydown', (e) => {
  // E-stop on Space — ALWAYS, regardless of focus
  if (e.code === 'Space') {
    e.preventDefault();
    triggerEstop();
    return;
  }
  // Skip if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  // ... route other keys
});
```

---

## 6. Task Buttons — Sidebar Panel with Progress Timeline

### V1 Problem
Task buttons are inline with joint controls — no feedback on progress, no way to see what's happening, no parameter adjustment.

### V2 Design: Collapsible Task Sidebar

```
┌─ TASKS ─────────────────────┐
│                              │
│  ┌──────────────────────┐   │
│  │ 🏠 HOME              │   │
│  │ Return to zero pose  │   │
│  │ Speed: [====●===] 0.6│   │
│  │         [Execute]    │   │
│  └──────────────────────┘   │
│                              │
│  ┌──────────────────────┐   │
│  │ 👋 WAVE              │   │
│  │ Friendly wave gesture│   │
│  │ Speed: [======●=] 0.8│   │
│  │ Waves: [3] ← stepper│   │
│  │         [Execute]    │   │
│  └──────────────────────┘   │
│                              │
│  ┌──────────────────────┐   │
│  │ ✋ READY              │   │
│  │ Neutral position     │   │
│  │ Speed: [====●===] 0.6│   │
│  │         [Execute]    │   │
│  └──────────────────────┘   │
│                              │
│ ─── RUNNING ────────────── │
│ ┌──────────────────────┐   │
│ │ 👋 Wave (3x)         │   │
│ │ ████████░░░░ 65%     │   │
│ │ 2.1s / 3.2s          │   │
│ │ Point 8 / 12         │   │
│ │      [■ STOP TASK]   │   │
│ └──────────────────────┘   │
│                              │
└──────────────────────────────┘
```

**Not a modal.** Modals block interaction — you need to see the arm while a task runs. Instead:
- Task panel is a collapsible right sidebar section (below joint controls, above the raw command)
- Each task is an expandable card showing description + parameters
- When a task is running, the card expands to show a **progress timeline**:
  - Progress bar with percentage
  - Elapsed / total time
  - Current waypoint / total waypoints
  - Stop button prominently placed

**Progress tracking:** The server already returns `points` and `duration_s` from task endpoints. Add a WS message type for task progress:
```json
{"type": "task_progress", "task": "wave", "progress": 0.65, "point": 8, "total_points": 12, "elapsed_s": 2.1, "duration_s": 3.2}
```
(Requires server-side addition — propose to backend engineer.)

**Task history:** Below the active task, show last 3 completed tasks as collapsed entries with result (✓/✗) and duration.

---

## 7. Emergency Stop UX — Unmissable, Unreachable-proof

### V1 Analysis
V1's e-stop is good (big red button, flashing when enabled) but has gaps:
- Can scroll it out of view on mobile
- No keyboard shortcut
- No confirmation that it worked
- Looks like other buttons when arm is disabled (non-threatening)

### V2 Design: Multi-Layer E-Stop

#### Layer 1: Fixed-Position Button (always visible)
```css
.estop-fixed {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9999;
  width: 80px;
  height: 80px;
  border-radius: 50%;       /* Circular — looks like a real e-stop */
  background: radial-gradient(circle, #ff2244 0%, #cc0022 70%, #880011 100%);
  border: 4px solid #ff6b81;
  box-shadow: 0 0 0 4px #1a1a2e, 0 0 20px rgba(233,69,96,0.4);
  color: white;
  font-weight: 900;
  font-size: 11px;
  letter-spacing: 2px;
  cursor: pointer;
}
```

- **Always visible** — fixed position, can't scroll away
- **Circular** — mimics physical e-stop mushroom buttons operators are trained on
- **3D effect** — radial gradient + shadow makes it look pressable
- **Size:** 80×80px minimum (larger on touch devices via media query → 100×100px)
- **Label:** "STOP" in center, "⚠" icon above

#### Layer 2: Keyboard (Space bar)
- Space bar triggers e-stop regardless of focus state
- Even works when a text input is focused (override default behavior)

#### Layer 3: Visual Escalation States

| Arm State | E-Stop Appearance |
|-----------|-------------------|
| Disconnected / Off | Dark red, muted, no glow — present but calm |
| Powered (not enabled) | Medium red, subtle pulse every 3s |
| Enabled (idle) | Bright red, gentle breathing glow (2s cycle) |
| Moving / Task running | Bright red, faster pulse (1s cycle), enlarged shadow |
| Error state | FLASHING red/white, 0.3s cycle, outer ring pulses |

#### Layer 4: Activation Feedback
When e-stop is pressed:
1. **Immediate:** Button turns WHITE with red text "STOPPED" — visual confirmation
2. **Haptic:** `navigator.vibrate(200)` on mobile devices
3. **Audio:** Optional — short buzzer sound via Web Audio API (configurable, off by default)
4. **Full-screen flash:** Brief (100ms) red overlay on entire viewport at 30% opacity — unmistakable "something happened"
5. **Log entry:** Red banner slides down from top: "⚠ EMERGENCY STOP ACTIVATED" — stays until dismissed
6. **Recovery:** After e-stop, button shows "STOPPED ✓" for 3 seconds, then returns to normal state. Arm must be manually re-powered and re-enabled.

#### Layer 5: Mobile Consideration
On screens < 600px wide, the e-stop button moves to a **fixed bottom bar** spanning full width, 60px tall. Can't miss it, can't accidentally scroll past it.

```
┌─────────────────────────────────┐
│                                 │
│         (main UI)               │
│                                 │
├─────────────────────────────────┤
│  ⚠⚠⚠  EMERGENCY STOP  ⚠⚠⚠    │  ← fixed, full width
└─────────────────────────────────┘
```

---

## 8. Technology Summary

| Component | Technology | Size | Notes |
|-----------|-----------|------|-------|
| Joint sliders | Custom Elements (vanilla JS) | ~200 lines | Canvas-rendered tracks |
| Rotary knobs | Custom Elements (vanilla JS) | ~150 lines | For roll joints |
| 3D visualization | Three.js r160+ | ~150KB gz | Single ESM import, no build |
| Orbit controls | Three.js addon | included | `OrbitControls.js` |
| Animations | CSS transitions + rAF | 0 deps | No GSAP needed |
| Keyboard shortcuts | Vanilla JS | ~80 lines | Single event listener |
| State machine display | SVG + CSS | ~60 lines | Inline SVG pipeline |
| Numeric inputs | Native `<input type="number">` | 0 deps | With custom step logic |

**Total new dependencies:** Three.js only (~150KB gzipped, served locally).  
**No build step.** Everything is vanilla JS with ES modules.

---

## 9. Layout Restructure

```
┌─ Topbar: logo | connection | latency pill | state machine strip | debug toggle ─┐
├──────────────────────────────┬───────────────────────────────────────────────────────┤
│                              │  Joint Controls (sliders/knobs)                      │
│                              │  ┌─ J0 rotary ─┐ ┌─ J1 slider ──────────────────┐  │
│     3D Viewport              │  └──────────────┘ └──────────────────────────────┘  │
│     (Three.js)               │  ... (6 joints + gripper)                            │
│                              │                                                       │
│     [Top] [Front] [Side]     ├───────────────────────────────────────────────────────┤
│     [Iso] [Reset]            │  Tasks (collapsible cards)                            │
│                              │  [Home] [Ready] [Wave]                                │
│     Tracking error HUD       │  ── Running: Wave 65% ██████░░░ ──                   │
│                              ├───────────────────────────────────────────────────────┤
│                              │  Commands: [Power] [Enable] [Raw...]                  │
├──────────────────────────────┴───────────────────────────────────────────────────────┤
│  Log Panel (collapsible)                                                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  Camera Feeds (collapsible)                                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                                              ┌──────┐
                                                              │ STOP │  ← fixed position
                                                              └──────┘
```

---

## 10. Server-Side Requests (for backend engineer)

To support the interaction design above, the server API needs:

1. **Task progress WS messages:** Emit `{"type": "task_progress", ...}` during trajectory execution
2. **Ping/pong latency:** Add WS ping frame handling or a `{"type": "ping", "ts": ...}` message for RTT measurement
3. **Tracking error endpoint:** `GET /api/state/tracking-error` returning per-joint `{target, actual, error}` — or include `target_joints` in the WS state message alongside `joints` (actual)
4. **State machine:** Add an explicit `state` field to WS messages: `"DISCONNECTED" | "CONNECTED" | "POWERED" | "ENABLED" | "IDLE" | "MOVING" | "TASK_RUNNING"`
