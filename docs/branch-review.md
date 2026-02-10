# Branch Review — Assessment for Council

## Hardware Context
- Unitree D1 7-DOF arm, 550mm reach, 1kg payload
- 3 cameras: overhead Logitech BRIO (video0), MX Brio (video4), arm-mounted BRIO (video6)
- RX 580 eGPU (no CUDA, limited ROCm), Intel NUC i7
- Current capabilities on main: DDS control, safety monitor, PID controller, FK/IK, motion planning, task planning, visual servo, multi-view fusion pick, 3D scanning, live video streaming, CV pipeline

## 8 Branches — ALREADY MERGED (0 unique commits, safe to delete)
1. `claude/3d-visual-hull-pipeline-l1def`
2. `claude/ascii-to-3d-converter-ZKoEs`
3. `claude/ascii-video-converter-pMnmJ`
4. `claude/bifocal-arm-mapping-eO4YV`
5. `claude/claw-position-prediction-m2Gye`
6. `claude/dual-camera-arm-tracking-4f8AH`
7. `claude/test-robot-base-movement-R0rYB`
8. `claude/vision-task-planning-RG8rd`

**Recommendation: DELETE all 8.** No code loss.

## 8 Branches — HAVE UNIQUE CODE (need evaluation)

### 1. `claude/add-simulator-mode-464fbb` (1 commit, +15 lines)
- Just a README safety disclaimer
- **Verdict: DELETE.** Trivial, already covered by existing docs.

### 2. `claude/add-text-command-interface-k84Y3` (1 commit, +776 lines)
- Natural language text command interface for arm control
- Parses commands like "move to position X" into joint commands
- **Verdict: INTEGRATE.** Useful for quick manual control and debugging. Would complement the web UI.

### 3. `claude/arm-introspection-replay-FEdTc` (3 commits, +3024 lines)
- Self-assessment, world model, replay/analysis system
- Arm evaluates its own performance and suggests improvements
- **Verdict: MAYBE INTEGRATE (partial).** The world model is useful for collision avoidance and planning. The "self-improvement" part is speculative. Extract the world model, skip the rest.

### 4. `claude/camera-ascii-conversion-Oeq0E` (2 commits, +4409 lines)
- Camera-to-ASCII pipeline, VLA (Vision-Language-Action) model, digital twin, waypoint navigation
- **Verdict: INTEGRATE (VLA + waypoint nav).** VLA model ties Gemini vision to arm actions — directly useful for pick tasks. Waypoint navigation useful for repeatable paths. ASCII pipeline is novelty, skip it.

### 5. `claude/code-review-improvements-et3pC` (1 commit, +242 lines)
- Bug fixes: planning module init, UI flickering, viz orientation, safe disable return
- **Verdict: INTEGRATE.** These are bug fixes that should be on main.

### 6. `claude/d1-arm-v3-viz-qEoNT` (1 commit, +1234 lines)
- Three.js 3D visualization with dual-camera calibration
- Full 3D arm model rendered in browser
- **Verdict: INTEGRATE.** We have a 3D scan viewer already; this adds live arm visualization. Very useful for debugging and monitoring.

### 7. `claude/object-dimension-analysis-TRtn6` (3 commits, +3363/-1385 lines)
- Startup object dimension analysis, world model, collision memory
- Large refactor touching many files (server.py, v2_server.py)
- **Verdict: PARTIAL INTEGRATE.** Object dimension analysis is useful for grasp planning. But the large refactor may conflict with current main. Cherry-pick the new modules, skip the refactor.

### 8. `claude/review-commit-13cbb3d-ho5mN` (1 commit, +2104 lines)
- Safety, control loop, kinematics, trajectory improvements + tests (watchdog test, trajectory test)
- **Verdict: INTEGRATE.** Core improvements and tests. Should be on main.

## Summary

| Action | Branches | Count |
|--------|----------|-------|
| DELETE (merged) | 3d-visual-hull, ascii-to-3d, ascii-video, bifocal, claw-position, dual-camera, test-robot, vision-task | 8 |
| DELETE (trivial) | simulator-mode | 1 |
| INTEGRATE fully | text-command, code-review-fixes, d1-arm-v3-viz, review-commit | 4 |
| INTEGRATE partially | arm-introspection (world model only), camera-ascii (VLA+waypoint only), object-dimension (analysis module only) | 3 |

**Total: 9 delete, 4 full integrate, 3 partial integrate**
