# Sub-Agent Plan — Day 4 Afternoon

## Context
- Phases 1-4 of dual-mode pick pipeline: DONE
- `arm_operations.py` codified with 12 operational patterns: DONE
- Mini-sim panels: DONE (sub-agent completed)
- Next: wire arm_operations into AutoPick, build remaining phases, and new features

## Sub-Agents (5 parallel)

### 1. `wire-arm-ops` — Integrate arm_operations into AutoPick
**Goal:** Replace AutoPick's raw HTTP calls with ArmOps methods
**Files:** `src/planning/auto_pick.py`
**Tasks:**
- Import and use `ArmOps` instead of raw httpx calls in `_execute_pick`, `_move_to`
- Use `staged_reach` for approach, `staged_retract` for lift/abort
- Use `retreat_home` on any failure
- Use `full_recovery` when overcurrent detected
- Keep episode recording integration intact
- Run existing auto_pick tests + arm_operations tests

### 2. `pick-analytics` — Phase 6: Pick Analytics Dashboard
**Goal:** Aggregate pick episode data into useful stats
**Files:** NEW `src/telemetry/pick_analytics.py`, modify `web/server.py`, modify `web/static/index.html`
**Tasks:**
- Parse `data/pick_episodes.jsonl` 
- Compute: success rate (overall, per-target, sim vs physical), avg duration per phase, common failure modes
- API: GET `/api/pick/analytics` (summary), GET `/api/pick/analytics/phases` (per-phase breakdown)
- UI: "Pick Analytics" panel with success rate bar, phase timing chart (simple HTML/CSS bars, no charting lib)
- 8+ tests

### 3. `pick-ui-polish` — Phase 5: UI Integration
**Goal:** Make pick & place panels production-quality
**Files:** `web/static/index.html`, `web/static/js/mini-sim.js`
**Tasks:**
- Episode history table in Auto Pick panel (load from `/api/pick/episodes`)
- Episode detail view (click to expand: phases, timing, joint traces)
- Sim indicator badge ("🔮 SIMULATION" when mode=simulation)
- 3D ghost playback: replay episode joints in mini-sim (step through recorded states)
- Mode auto-detect from `/api/sim-mode`
- No external JS libs — vanilla only

### 4. `video-recording` — Camera recording during pick attempts
**Goal:** Record video from all cameras during pick attempts for review
**Files:** NEW `src/telemetry/pick_recorder.py`, modify `web/server.py`
**Tasks:**
- `PickVideoRecorder` class: grabs frames from cameras at ~5fps during pick
- Saves as MJPEG or individual frames to `data/pick_recordings/{episode_id}/`
- Auto-starts when AutoPick begins, auto-stops when done
- API: GET `/api/pick/recordings/{episode_id}` (list frames), GET `/api/pick/recordings/{episode_id}/{frame}` (serve frame)
- Lightweight — don't use OpenCV VideoWriter, just save JPEGs with timestamps
- Use `cam_snap()` or `/latest/` endpoint for frame capture
- 6+ tests

### 5. `place-pipeline` — Pick AND Place (not just pick)
**Goal:** Complete the pick-and-place loop — put objects down somewhere
**Files:** Modify `src/planning/auto_pick.py` (or new `src/planning/auto_place.py`), modify `web/server.py`, modify `web/static/index.html`
**Tasks:**
- Use `ArmOps.place_at()` for the place phase
- Define place targets: user-specified XY on overhead camera (click-to-place)
- Reuse geometric planner from AutoPick for place target joints
- Full sequence: detect → pick → transport → place → retract home
- API: POST `/api/autopick/start` with `place_target: {x_mm, y_mm}` optional field
- UI: "Place Target" click point on overhead camera feed (crosshair overlay)
- Episode recording covers both pick AND place phases
- 8+ tests

## Dependencies
- All agents can run in parallel (no blocking dependencies)
- Agent 1 (wire-arm-ops) should be reviewed first since agents 4 & 5 touch auto_pick.py
- Agents 2 & 3 are purely additive (new UI panels + analytics)

## Constraints for all agents
- Never use `set_all_joints` — individual `set_joint` only
- Use `src/config/camera_config.py` for camera URLs (never hardcode localhost:8081)
- Use `src/utils/gemini_limiter.py` for any Gemini calls
- Check FastAPI imports (Response, Request, JSONResponse) when adding endpoints
- Run `python3 -m pytest` on affected test files before declaring done
- Git author: TheClaw <claw@th3cl4w.dev>
