# Engineering Plan: Dual-Mode Pick & Place Pipeline

## Overview

The pick & place pipeline should run in two modes:
- **PHYSICAL** — Real arm, real cameras, real objects
- **SIMULATION** — Simulated arm, real cameras detecting real objects, but execution is virtual. Generates identical telemetry so sim runs are indistinguishable from real runs in the data.

Goal: 1:1 testing parity. A sim pick attempt produces the same telemetry events, timing, and data as a real one.

---

## Current State

### What Exists
- `SimulatedArm` (`src/interface/simulated_arm.py`) — drop-in replacement for `D1DDSConnection`, interpolates joints toward targets in memory
- `_sim_mode` flag in `web/server.py` — togglable at runtime via `/api/sim-mode`
- `TelemetryCollector` (`src/telemetry/collector.py`) — records DDS events, commands, smoother state to SQLite
- `TelemetryWatcher` (`web/telemetry_watcher.py`) — captures snapshots, analysis, feedback sessions
- `AutoPick` (`src/planning/auto_pick.py`) — detect→plan→execute pipeline via HTTP calls to server
- 3D simulator with follow mode — tracks live arm position
- Object detection pipeline — overhead + side cameras, HSV + Gemini labeling

### What's Missing
1. **AutoPick doesn't know about sim mode** — always sends real HTTP commands
2. **No sim-specific object interaction** — detected objects don't react to sim arm proximity
3. **Telemetry doesn't distinguish sim vs real** — no `mode` field in events
4. **No virtual grip detection** — sim doesn't know when gripper "contacts" a detected object
5. **No pick episode recording** — no structured log of full pick attempts (phases, timing, success/fail)
6. **SimulatedArm doesn't emit telemetry events** — DDS feedback loop doesn't fire in sim

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AutoPick Pipeline                  │
│  detect() → plan() → execute() → verify()          │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
         ┌─────▼─────┐         ┌─────▼──────┐
         │  PHYSICAL  │         │ SIMULATION  │
         │  Mode      │         │ Mode        │
         ├───────────┤         ├────────────┤
         │ Real DDS   │         │ SimulatedArm│
         │ Real grip  │         │ Virtual grip│
         │ Cam verify │         │ Geom verify │
         └─────┬─────┘         └──────┬─────┘
               │                      │
               └──────────┬───────────┘
                          │
                   ┌──────▼──────┐
                   │  Telemetry   │
                   │  (identical) │
                   └─────────────┘
```

---

## Work Items

### Phase 1: Pick Episode Recorder (foundation)
**File:** `src/telemetry/pick_episode.py`

A structured record of each pick attempt:

```python
@dataclass
class PickEpisode:
    episode_id: str           # UUID
    mode: str                 # "physical" or "simulation"
    target: str               # "redbull", "blue_can", etc.
    start_time: float
    end_time: float
    
    # Detection
    detection_method: str     # "hsv", "llm", "manual"
    detected_position_px: tuple[int, int]  # pixel coords
    detected_position_mm: tuple[float, float, float]  # arm-frame XYZ
    detection_confidence: float
    detection_camera: int
    
    # Planning
    planned_joints: list[float]  # 6 target angles
    planned_gripper_mm: float
    approach_joints: list[float]
    
    # Execution (per-phase timing)
    phases: list[PhaseRecord]  # [{name, start_t, end_t, joints_at_start, joints_at_end}]
    
    # Result
    success: bool
    failure_reason: str
    grip_verified: bool
    
    # Telemetry refs
    telemetry_events: int     # count of events during episode
    peak_tracking_error_deg: float
    total_commands_sent: int
```

Store episodes in `data/pick_episodes.json` (append-only JSONL) and expose via API:
- `GET /api/pick/episodes` — list episodes with filters
- `GET /api/pick/episodes/{id}` — single episode detail

**Estimated effort:** 1 sub-agent, ~200 lines

---

### Phase 2: Sim Telemetry Bridge
**File:** `src/telemetry/sim_bridge.py`

Make `SimulatedArm` emit the same telemetry events as the real DDS connection:

```python
class SimTelemetryBridge:
    """Generates DDS-equivalent telemetry from SimulatedArm state changes."""
    
    def __init__(self, sim_arm: SimulatedArm, collector: TelemetryCollector):
        self._arm = sim_arm
        self._collector = collector
        self._rate_hz = 10.0  # match real DDS rate
    
    async def run(self):
        """Background loop: poll sim arm state, emit telemetry events."""
        while self._running:
            angles = self._arm.get_joint_angles()
            status = self._arm.get_status()
            
            # Emit dds_receive events (same format as real)
            self._collector.log_event("dds_receive", {
                "seq": self._seq,
                "funcode": 1,
                "angles": {f"angle{i}": angles[i] for i in range(6)},
            })
            self._collector.log_event("dds_receive", {
                "seq": self._seq,
                "funcode": 3,
                "status": status,
            })
            
            self._seq += 1
            await asyncio.sleep(1.0 / self._rate_hz)
```

Wire into server startup: when `_sim_mode`, start the bridge alongside the smoother.

**Estimated effort:** 1 sub-agent, ~150 lines

---

### Phase 3: Virtual Grip Detection
**File:** `src/planning/virtual_grip.py`

In simulation, detect when the gripper "contacts" a detected object using geometry:

```python
class VirtualGripDetector:
    """Determines if sim gripper has 'gripped' a detected object."""
    
    def check_grip(
        self,
        gripper_position_mm: np.ndarray,  # FK-computed gripper XYZ
        gripper_width_mm: float,
        detected_objects: list,  # from object detector
    ) -> Optional[str]:
        """Return label of gripped object, or None."""
        for obj in detected_objects:
            dist = np.linalg.norm(gripper_position_mm[:2] - obj.position_mm[:2])
            if dist < 50.0 and gripper_width_mm < obj.width_mm + 5:
                return obj.label
        return None
```

This replaces the camera verification step in sim mode:
- Physical: snap arm camera, check for object in gripper
- Sim: compute gripper FK position, check proximity to detected objects

**Estimated effort:** 1 sub-agent, ~100 lines

---

### Phase 4: AutoPick Dual-Mode Execution
**File:** Modify `src/planning/auto_pick.py`

Make `AutoPick` mode-aware:

```python
class AutoPick:
    def __init__(self, mode: str = "auto"):
        # "auto" = detect from server's _sim_mode
        # "physical" = force physical
        # "simulation" = force sim
        self.mode = mode
        self.episode_recorder = PickEpisodeRecorder()
        self.virtual_grip = VirtualGripDetector()
    
    async def execute(self, target="redbull"):
        episode = self.episode_recorder.start(mode=self.mode, target=target)
        
        try:
            # DETECT — same in both modes (real cameras, real detection)
            detection = await self._detect(target)
            episode.record_detection(detection)
            
            # PLAN — same in both modes
            joints = self._plan(detection)
            episode.record_plan(joints)
            
            # EXECUTE — same API calls (SimulatedArm handles them in sim)
            await self._execute_sequence(joints)
            
            # VERIFY — different per mode
            if self.mode == "simulation":
                gripped = self.virtual_grip.check_grip(...)
            else:
                gripped = await self._camera_verify()
            
            episode.record_result(success=gripped)
        finally:
            episode.finish()
```

Key insight: **detection and planning are identical** in both modes. Only execution backend (SimulatedArm vs DDS) and verification (geometric vs camera) differ.

**Estimated effort:** 1 sub-agent, ~200 lines modification

---

### Phase 5: UI Integration
**File:** Modify `web/static/index.html`

Add to the Auto Pick panel:
- **Mode toggle**: Physical / Simulation / Auto (auto-detects from server)
- **Episode history table**: shows past attempts with mode, target, result, timing
- **Episode detail view**: click an episode to see phases, timing, joint traces
- **Sim indicator**: when in sim mode, show a prominent "🔮 SIMULATION" badge
- **3D ghost playback**: replay a pick episode in the mini-sim by stepping through recorded joint states

**Estimated effort:** 1 sub-agent, ~300 lines

---

### Phase 6: Pick Analytics Dashboard
**File:** New panel in UI + `src/telemetry/pick_analytics.py`

Aggregate pick episode data:
- Success rate (overall, per-target, sim vs physical)
- Average pick duration per phase
- Detection accuracy (planned vs actual grip position)
- Common failure modes
- Sim→Physical correlation (do sim successes predict real successes?)

**Estimated effort:** 1 sub-agent, ~400 lines

---

## Execution Order

```
Phase 1: Pick Episode Recorder     ← foundation, everything depends on this
Phase 2: Sim Telemetry Bridge      ← parallel with Phase 1
Phase 3: Virtual Grip Detection    ← parallel with Phase 1
Phase 4: AutoPick Dual-Mode        ← depends on 1, 2, 3
Phase 5: UI Integration            ← depends on 4
Phase 6: Pick Analytics            ← depends on 1, 5
```

Phases 1-3 can run in parallel (3 sub-agents). Phase 4 integrates them. Phase 5-6 build on top.

**Total estimated effort:** 6 sub-agents, ~1400 lines of new code.

---

## Key Design Decisions

1. **Same API calls in both modes** — AutoPick always calls `/api/command/set-joint`. The server routes to real DDS or SimulatedArm based on `_sim_mode`. No branching in the pick logic.

2. **Real cameras in sim** — Sim mode still uses real camera feeds for detection. This means you need real objects on the table even in sim. The simulation is of the ARM, not the world.

3. **Telemetry parity** — SimTelemetryBridge generates identical event schemas. Analysis tools can't tell sim from real without checking the `mode` field.

4. **Episode recording is always on** — Every pick attempt (sim or physical) generates an episode. This builds a dataset for learning what works.

5. **Mode is per-attempt, not global** — You can run a sim pick followed by a physical pick without restarting. The mode flag is on the episode, not the server.

---

## Open Questions

- Should sim picks update the 3D sim's detected objects (remove "picked" objects from the scene)?
- Should we add configurable noise to SimulatedArm (joint angle noise, gripper slip) for more realistic testing?
- Do we want video recording of pick attempts (cam0/cam1 footage alongside telemetry)?
