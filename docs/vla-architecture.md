# VLA (Vision-Language-Action) Architecture for th3cl4w

## Overview

A closed-loop pipeline that takes camera images + natural language commands and produces
joint-level actions to control the Unitree D1 arm. The system observes, plans, acts, and
verifies in a continuous loop until the task is complete.

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Cameras     │───▶│  VLA Model   │───▶│ Action Decoder│───▶│  Arm Control │
│  (0: front)  │    │  (Gemini /   │    │ (JSON → joint │    │  (safety +   │
│  (1: overhead)│    │   Octo/OVL) │    │  commands)    │    │   smoother)  │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
       ▲                                                          │
       │                    ┌──────────────┐                      │
       └────────────────────│   Feedback    │◀─────────────────────┘
                            │   (verify)    │
                            └──────────────┘
```

## Model Backend Evaluation

### Tier 1: Works NOW — Gemini Flash Multimodal (Primary)

- **Model:** `gemini-2.0-flash` via API
- **Input:** 2 camera images (1920×1080) + language command + current joint state
- **Output:** Structured JSON with target joint deltas or absolute positions
- **Latency:** ~1-2s per inference (API call)
- **Cost:** Free tier / low cost
- **Pros:** Zero local compute, excellent vision understanding, can reason about tasks
- **Cons:** API latency (not real-time), rate limits, no fine-tuning on our embodiment

**This is our immediate approach.** The visual servo already uses Gemini for pixel detection,
but the VLA controller upgrades this to full action planning — instead of "where is the gripper?"
we ask "what joint moves should I make to pick up this object?"

### Tier 2: Near-term — Octo-Small (27M params)

- **Model:** Transformer diffusion policy, pretrained on Open X-Embodiment
- **Input:** Camera images + language instruction + proprioception
- **Output:** 7-DOF action chunks (position deltas)
- **VRAM:** ~2GB (27M params) — fits on RX 580
- **Latency:** ~50-200ms per chunk on GPU
- **Pros:** Real-time, designed for manipulation, action chunking
- **Cons:** Needs fine-tuning for D1 embodiment, JAX ecosystem, may need data collection

**Upgrade path:** Collect 50-200 demonstrations with data_collector → fine-tune Octo-Small →
deploy for real-time control.

### Tier 3: Long-term — OpenVLA-7B or π0

- **OpenVLA:** 7B params, needs ≥16GB VRAM (not feasible on RX 580)
- **π0:** Physical Intelligence's model, closed-source
- **Feasibility:** Would need a GPU upgrade (RTX 4090) or cloud inference
- **Why later:** Our current hardware can't run these, but the pipeline is designed to swap backends

### Decision: Hybrid Architecture

```
                    ┌─────────────────────────┐
                    │     VLA Controller       │
                    │  (vla_controller.py)     │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │    Model Backend         │
                    │  (vla_model.py)          │
                    │                          │
                    │  ┌───────────────────┐   │
                    │  │ GeminiVLABackend  │←── Works now
                    │  └───────────────────┘   │
                    │  ┌───────────────────┐   │
                    │  │ OctoVLABackend    │←── After fine-tuning
                    │  └───────────────────┘   │
                    │  ┌───────────────────┐   │
                    │  │ OpenVLABackend    │←── Future GPU
                    │  └───────────────────┘   │
                    └──────────────────────────┘
```

## Pipeline Design

### 1. Observation (cameras → features)

```python
observation = {
    "images": {
        "cam0_front": np.array(...),     # 1920×1080×3
        "cam1_overhead": np.array(...),  # 1920×1080×3
    },
    "joints": [j0, j1, j2, j3, j4, j5],  # degrees
    "gripper_mm": 0.0,                     # 0-65mm
    "task": "pick up the red bull can",    # natural language
}
```

### 2. Planning (model inference)

The model returns an **action plan** — a sequence of moves, not just one step:

```python
action_plan = {
    "reasoning": "The can is at overhead pixel (900, 600). Gripper is at (760, 135)...",
    "actions": [
        {"type": "joint", "id": 0, "angle": 15.0, "description": "rotate base toward can"},
        {"type": "joint", "id": 1, "angle": -30.0, "description": "lean forward"},
        {"type": "joint", "id": 2, "angle": 40.0, "description": "extend elbow"},
        {"type": "gripper", "position": 55.0, "description": "open gripper wide"},
        {"type": "joint", "id": 4, "angle": 60.0, "description": "angle wrist down"},
        {"type": "verify", "description": "check alignment before closing"},
        {"type": "gripper", "position": 10.0, "description": "close on can"},
    ],
    "confidence": 0.7,
    "needs_verification_after": 3,  # re-observe after 3 actions
}
```

### 3. Execution (action → arm)

The controller executes actions one at a time through the existing safety pipeline:
- Each joint move goes through `set-joint` API (respects limits, smoother)
- After N actions (or when plan says "verify"), re-observe and re-plan
- If a move worsens the situation, reverse it and re-plan
- Max 10° per command, 5° margin from limits

### 4. Verification (feedback loop)

After each action batch:
1. Wait for arm to settle (1-2s)
2. Snap both cameras
3. Ask model: "Did the last move help? What should we do next?"
4. Update plan or continue

### Safety Integration

```
User Command → VLA Controller → Action Decoder → Safety Check → Smoother → Arm
                    ↑                                    │
                    │              Safety Monitor ◄──────┘
                    │              (limits, e-stop)
                    └──────────────────────────────────────
```

- All actions pass through existing joint limits (±85° on J1/J2/J4, ±135° on J0/J3/J5)
- Max 10° per step enforced at the action decoder level
- NEVER extend elbow while lifting shoulder (sequencing enforced)
- Collision detector (when enabled) can abort moves
- Emergency stop always available

## Data Collection for Fine-Tuning

### Recording Format

Each demonstration is a sequence of (observation, action) pairs saved as:

```
data/demonstrations/
  demo_001/
    metadata.json          # task description, timestamps, success/fail
    frames/
      step_000_cam0.jpg    # front camera
      step_000_cam1.jpg    # overhead camera
      step_001_cam0.jpg
      ...
    trajectory.jsonl       # one JSON line per step
```

Each trajectory line:
```json
{
  "step": 0,
  "timestamp": 1707350400.0,
  "joints_before": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "joints_after": [15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "gripper_before": 0.0,
  "gripper_after": 0.0,
  "action": {"type": "joint", "id": 0, "angle": 15.0},
  "task": "pick up the red bull can"
}
```

### How to Collect Data

1. Start recording: `POST /api/vla/collect` with `{"action": "start", "task": "pick up the red bull can"}`
2. Teleoperate the arm through the task (web UI or API)
3. Stop recording: `POST /api/vla/collect` with `{"action": "stop"}`
4. Mark success/failure: metadata.json updated

### Fine-Tuning Path

1. Collect 50-200 demonstrations of common tasks
2. Convert to Octo training format (RLDS)
3. Fine-tune Octo-Small on our D1 embodiment data
4. Deploy locally on RX 580
5. Compare performance vs Gemini backend
6. Iterate

## Performance Expectations

| Backend | Latency/step | Steps to pick | Total time | Success rate (est) |
|---------|-------------|---------------|------------|-------------------|
| Gemini VLA | 2-3s | 8-15 | 20-45s | 60-80% |
| Octo-Small (fine-tuned) | 0.2s | 5-10 | 3-10s | 70-90% |
| OpenVLA (future) | 0.5s | 5-8 | 5-10s | 80-95% |
| Current visual servo | 3-5s | 15-25 | 60-120s | 30-50% |

The Gemini VLA should be a significant upgrade over the current visual servo because:
1. It plans multiple steps ahead instead of one-at-a-time
2. It uses both cameras simultaneously
3. It reasons about the task ("I need to approach from the side because the can is tall")
4. It uses joint state + pixel positions for better action mapping
5. It has the full joint mapping context baked into the prompt

## File Layout

```
src/vla/
  __init__.py
  vla_model.py           # Model backends (Gemini, Octo, OpenVLA)
  action_decoder.py      # Convert model output → safe joint commands
  vla_controller.py      # Closed-loop controller: observe → plan → act → verify
  data_collector.py       # Record demonstrations for fine-tuning
  prompts.py             # Gemini prompt templates
```
