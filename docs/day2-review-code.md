# Day 2 Review — Code & Architecture

**Date:** 2026-02-08  
**Reviewer:** Code & Architecture Council Member  
**Scope:** All code, architecture, and audit artifacts produced on Day 2

---

## 1. What Went Well 🎉

**Massive output.** In a single day, the team shipped a VLA pipeline, 3D visualization, unified safety limits, three audit reports, and 145+ new tests. That's startup-level velocity on a robotics project.

**The VLA pipeline is well-architected.** `action_decoder.py` is a highlight — clean separation of concerns, projected state tracking through sequential actions, proper clamping, and the `_enforce_sequencing()` safety interlock that prevents simultaneous J1+J2 extension. This is the kind of defense-in-depth thinking a robot arm needs.

**`src/safety/limits.py` is exactly right.** A single source of truth with degrees, radians, velocity, torque, gripper range, and smoother params all in one place with clear comments. This should have existed from Day 1, but it's here now and it's solid.

**The prompts encode hard-won knowledge.** `prompts.py` captures the J1 inversion discovery, known-good positions, and sequencing rules. This is institutional memory baked into code — excellent.

**Audits were brutally honest.** The audit reports don't pull punches. Calling out SafetyMonitor as dead code, the three conflicting limit definitions, and the god object — this is healthy engineering culture.

**Physical contact with the can at 49.9mm.** The VLA system guided the gripper to within 0.1mm of the can diameter. That's remarkable for a Day 1 VLA system using an LLM backbone.

---

## 2. What Went Wrong 💥

**15+ failed pick attempts due to a documentation error.** J1 was documented backwards. This is the single most expensive bug of the day — hours of human time, arm wear, and an overcurrent trip all trace back to one wrong sentence about joint direction.

**SafetyMonitor was dead code for the entire project's life.** Someone built a comprehensive safety system and nobody ever wired it in. This suggests a disconnect between "writing safety code" and "making safety real."

**The can was contacted and then lost.** The system got 95% of the way to a successful pick and failed at the last 5% — tightening the grip knocked the can off. This is a controls precision problem, not a planning problem.

**server.py is 3,285 lines.** This file is load-bearing spaghetti. Every new feature was bolted onto this monolith. It has 17 `Any`-typed globals and ~30 endpoints. This is the #1 velocity killer going forward.

---

## 3. Root Cause Analysis — Why the Pick Kept Failing

The pick failures have a **causal chain**, not a single root cause:

1. **J1 inversion (primary):** With J1 documented backwards, every "move toward the can" command moved away from it. The operator had to fight the system's own model of reality. This caused the first ~10 failures.

2. **Vision noise (secondary):** Gemini/Claude pixel estimates have ±100px noise. At the D1's workspace scale, that's ±15-20mm — nearly half the can diameter. Fine for coarse approach, fatal for final alignment.

3. **No force/contact feedback (tertiary):** When the gripper contacted the can at 49.9mm, the system had no way to know it was touching something. It kept commanding, knocked the can sideways. A simple "gripper stopped closing early → contact detected" heuristic would have caught this.

4. **Stale DDS feedback (contributing):** Intermittent zero-readings meant the system sometimes didn't know where the arm actually was, making closed-loop control unreliable.

**The real lesson:** LLM vision is sufficient for approach (Phase 1-3) but insufficient for grasp (Phase 4-5). The last 20mm needs either force sensing, contact detection via gripper position monitoring, or a much tighter visual servo loop.

---

## 4. Code Quality Assessment

### VLA Pipeline — **B+**

**Strengths:**
- `action_decoder.py` is the best module in the codebase. Clean dataclasses, proper validation, projected state tracking, sequencing safety checks.
- Good separation: model ↔ decoder ↔ controller ↔ data collection.
- 52 tests for a Day 1 module is excellent coverage.

**Weaknesses:**
- `action_decoder.py` defines its own `JOINT_LIMITS` dict instead of importing from `src/safety/limits.py`. This is the same "three conflicting limits" problem that was just fixed — and a new instance was introduced in the same day.
- The prompts hardcode `±85°` limits while `limits.py` says `±80°` (with margin). Which is it?
- No timeout on Gemini API calls in the controller — a hung API call blocks the entire VLA loop.

### 3D Visualization — **Not reviewed** (JS, outside primary scope)

### Safety Integration — **A-**

**Strengths:**
- `limits.py` is clean, well-documented, and complete.
- Feedback freshness gating concept is sound.
- Step size properly set to 10°.

**Weaknesses:**
- The unified limits exist but aren't yet imported everywhere. `action_decoder.py`, `visual_servo.py`, and `safety_monitor.py` still have their own copies.
- The SafetyMonitor integration into `command_smoother.py` was described as done, but the audit report still flags it as missing. Unclear if this was actually shipped.

### server.py — **D**

3,285 lines, 17 globals, no DI, no routers, `Any` everywhere. This is the single biggest risk to the project. Every bug fix and feature add increases the probability of a regression.

---

## 5. Technical Debt

| Debt Item | Severity | Effort to Fix |
|-----------|----------|---------------|
| `server.py` god object (3,285 lines) | **Critical** | High (2-3 days) |
| `action_decoder.py` has its own joint limits (not importing from `limits.py`) | **High** | 10 min |
| `safety_monitor.py` still has its own limits (not importing from `limits.py`) | **High** | 10 min |
| No Gemini API timeout in VLA controller | **Medium** | 5 min |
| `visual_servo.py` hardcoded ±85° limits | **Medium** | 10 min |
| DDS `os.environ` mutation | **Medium** | 30 min |
| No integration tests (HTTP → DDS round-trip) | **High** | 1 day |
| Degrees vs radians convention undocumented per-function | **Medium** | 2 hours |
| `_estimate_reach()` always returns 0.6 (broken) | **Medium** | 1 hour |
| Collision detector disabled, not tuned | **Medium** | 2 hours |

---

## 6. Recommendations for Tomorrow (Prioritized)

### P0 — Before touching the arm

1. **Make `limits.py` the actual single source.** Find every file that defines joint limits and replace with imports from `src/safety/limits.py`. Grep for `[-]?80`, `[-]?85`, `[-]?135`, `[-]?130` in Python files. This is 30 minutes of work that prevents the next J1-inversion-class bug.

2. **Verify SafetyMonitor is actually in the command path.** The audit says it's dead code. The task list says it was fixed. One of these is wrong. `grep -r "SafetyMonitor" web/` and confirm it's instantiated and called in `_tick()`.

3. **Add feedback freshness gating.** The `FEEDBACK_MAX_AGE_S = 0.5` constant exists in `limits.py`. Wire it into the smoother: if feedback is stale, refuse to send commands.

### P1 — For the pick attempt

4. **Implement gripper contact detection.** When gripper is closing and position stabilizes above 20mm → contact detected → stop closing, hold position, declare grasp. This is the missing piece from yesterday's 49.9mm contact.

5. **Reduce VLA to approach-only.** Use VLA for coarse positioning (get within 30mm), then switch to a deterministic grasp sequence: open gripper → lower 10mm → close gripper → detect contact → lift. Don't let the LLM control the final grasp.

### P2 — Code health

6. **Start splitting `server.py`.** Extract the easiest routers first: telemetry endpoints, camera proxy, debug endpoints. Goal: get `server.py` under 2,000 lines by end of week.

7. **Add one integration test.** Start the server in simulated mode, send a full pick sequence via HTTP, verify the command pipeline end-to-end. This catches wiring bugs like the dead SafetyMonitor.

### P3 — Nice to have

8. **Fix `_estimate_reach()`.** It always returns 0.6m, making workspace checks useless. Either implement FK-based reach or remove the check.

9. **Add API authentication.** Even a simple `X-API-Key` header. The `allow_origins=["*"]` CORS policy with no auth on a robot arm is genuinely dangerous.

---

## Summary

Day 2 was a high-output day with strong new modules (VLA decoder, limits.py) built alongside persistent architectural debt (server.py, dead safety code). The pick failure is not a VLA failure — it's a sensing gap at the grasp phase. The VLA got the arm to within 0.1mm of the target diameter, which is impressive.

**The one-sentence takeaway:** The system can *find* the can and *reach* the can; it cannot yet *feel* the can. Tomorrow's win is contact detection, not better vision.
