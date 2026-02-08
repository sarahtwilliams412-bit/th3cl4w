# LLM-based ASCII Calibration Pipeline — Parallel to CV

**Date:** 2026-02-08  
**Status:** Experimental / Design  
**Purpose:** Use LLMs to analyze ASCII renderings of camera frames during calibration, running alongside the traditional CV pipeline as a second opinion.

---

## 1. Architecture

```
                    ┌─────────────────────────┐
                    │  Calibration Orchestrator │
                    │  (20 poses, 2 cameras)   │
                    └────────┬────────┬────────┘
                             │        │
                    ┌────────▼──┐  ┌──▼────────┐
                    │ CV Pipeline│  │LLM Pipeline│
                    │ (existing) │  │  (new)     │
                    └────────┬──┘  └──┬────────┘
                             │        │
                    ┌────────▼────────▼────────┐
                    │   Comparison / Fusion     │
                    │  agreement → confidence   │
                    │  disagreement → flag      │
                    └──────────────────────────┘
```

### Integration point

The LLM pipeline hooks into the existing calibration loop (Plan B §1.2). At each of the 20 poses, after the CV pipeline captures and processes frames:

1. The same raw frames are converted to ASCII via `AsciiConverter`
2. ASCII + metadata sent to LLM API
3. LLM response parsed → pixel coordinates
4. Results stored alongside CV detections for comparison

**Key principle:** The LLM pipeline is **read-only / advisory**. It never modifies the calibration result. It only produces comparison data.

### New module: `src/vision/llm_calibrator.py`

```python
class LLMCalibrator:
    """Parallel LLM-based joint detection during calibration."""
    
    def __init__(self, model: str = "gemini", ascii_width: int = 80, ascii_height: int = 35):
        self.converter = AsciiConverter(width=ascii_width, height=ascii_height, 
                                         charset=CHARSET_DETAILED, invert=True)
        self.model = model
        self.scale_x = 1920 / ascii_width   # 24.0 px per ASCII char
        self.scale_y = 1080 / ascii_height   # 30.9 px per ASCII char
    
    async def detect_joints(self, frame, camera_id: str, joint_angles: list[float],
                            fk_hints: Optional[list] = None) -> dict:
        """Send ASCII frame to LLM, get joint position estimates."""
        ascii_text = self.converter.frame_to_ascii(frame)
        prompt = self._build_prompt(ascii_text, camera_id, joint_angles, fk_hints)
        response = await self._call_llm(prompt)
        return self._parse_response(response)
    
    def _scale_to_pixels(self, ascii_x: float, ascii_y: float) -> tuple[float, float]:
        """Convert ASCII grid coords back to camera pixels."""
        return (ascii_x * self.scale_x, ascii_y * self.scale_y)
```

---

## 2. Prompt Engineering

### 2.1 Base prompt (structured output, blind detection)

```
You are analyzing an ASCII art rendering of a robotic arm captured by a camera.

Camera: {front | overhead}
Frame resolution: 80 columns × 35 rows
Original image: 1920×1080 pixels
Each ASCII character represents approximately 24×31 pixels.

The arm is a 6-DOF robotic arm (SO-ARM100 / D1). It has:
- A base (fixed, at the bottom/center of the scene)
- Shoulder joint
- Elbow joint  
- Wrist joint
- End-effector (gripper)

The arm is MATTE BLACK (appears as dim/sparse characters like . : - ).
Gold accents at joints may appear as brighter characters (* # % @).
The background may contain shelves, monitors, or other objects.

Current joint angles: J0={j0}° J1={j1}° J2={j2}° J3={j3}° J4={j4}° J5={j5}°

ASCII frame:
```
{ascii_text}
```

Identify the ASCII grid coordinates (column, row) of each visible joint.
Column 0 is the left edge, row 0 is the top.

Respond in JSON only:
{
  "base": {"col": N, "row": N, "confidence": 0.0-1.0} | null,
  "shoulder": {"col": N, "row": N, "confidence": 0.0-1.0} | null,
  "elbow": {"col": N, "row": N, "confidence": 0.0-1.0} | null,
  "wrist": {"col": N, "row": N, "confidence": 0.0-1.0} | null,
  "end_effector": {"col": N, "row": N, "confidence": 0.0-1.0} | null,
  "reasoning": "brief explanation of what you see"
}

Use null for joints you cannot identify.
```

### 2.2 FK-hinted prompt (add after joint angles)

```
Expected approximate positions (from forward kinematics model):
- Base: col≈{c}, row≈{r}
- Shoulder: col≈{c}, row≈{r}  
- Elbow: col≈{c}, row≈{r}
- Wrist: col≈{c}, row≈{r}
- End-effector: col≈{c}, row≈{r}

These are predictions — the actual arm may differ slightly. Use them as guidance
but report what you actually see in the ASCII rendering.
```

### 2.3 Prompt strategy decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Blind vs. FK-hinted** | Run BOTH modes | Blind tests true LLM spatial ability; hinted tests refinement ability. Compare results. |
| **Color ASCII vs. plain** | Plain ASCII only | Color data as JSON would add ~50K tokens per frame for minimal LLM benefit. LLMs can't process parallel color arrays spatially. The brightness→character mapping already encodes luminance. |
| **Charset** | `CHARSET_DETAILED` (68 chars) | More gradations = more spatial signal for the LLM. The matte-black arm needs subtle brightness differences. |
| **Resolution** | **80×35** (2,800 chars) for LLM | Good tradeoff: enough detail to see the arm, fits in ~1K tokens for the frame alone. Test 120×50 (6,000 chars) as a step-up if 80×35 is too coarse. |
| **Multi-frame** | Single frame per call | Sending frame pairs (before/after arm move) adds tokens for marginal benefit at this stage. |

### 2.4 Why NOT color ASCII for LLMs

The `frame_to_color_data()` returns per-character RGB as nested arrays. For 80×35:
- ASCII text: ~2,800 chars (~1K tokens)
- Color JSON: 80×35×3 ints = 8,400 numbers → ~25K tokens
- Total with color: ~26K tokens vs ~1K without

LLMs don't have spatial reasoning over parallel 2D arrays of RGB values. The ASCII characters already encode brightness. Gold accents (the key visual feature) map to bright characters (`*#%@`) which are visible in plain ASCII. **Color is not worth 25× the tokens.**

---

## 3. LLM Selection

### Available models

| Model | Access | Input cost | Output cost | Context | Spatial reasoning |
|-------|--------|-----------|-------------|---------|-------------------|
| **Gemini 2.0 Flash** | ✅ API key configured | $0.10/1M input | $0.40/1M output | 1M tokens | Moderate — good at structured tasks |
| **Gemini 2.0 Pro** | ✅ same key | $1.25/1M input | $5.00/1M output | 1M tokens | Better but 12× more expensive |
| **Claude Sonnet 4** | ✅ via OpenClaw | $3/1M input | $15/1M output | 200K | Good at following structured prompts |
| **Claude Opus 4** | ✅ via OpenClaw | $15/1M input | $75/1M output | 200K | Best reasoning but very expensive |
| **GPT-4o** | ❓ need API key | $2.50/1M input | $10/1M output | 128K | Decent |

### Recommendation: **Gemini 2.0 Flash** as primary

- Cheapest by far (~10-50× cheaper than Claude/GPT)
- Already configured
- 1M context window is irrelevant here (small prompts) but nice
- Run Gemini Flash for all 40 frames (20 poses × 2 cameras), then spot-check 5 frames with Claude Sonnet to compare quality

### Cost comparison for full calibration run (see §6)

---

## 4. Comparison Framework

### 4.1 Per-detection metrics

For each joint at each pose, compare:

```python
@dataclass
class DetectionComparison:
    pose_idx: int
    camera_id: str
    joint_name: str
    
    cv_pixel: Optional[tuple[float, float]]     # from ArmSegmenter
    llm_pixel: Optional[tuple[float, float]]     # from LLM (scaled from ASCII)
    fk_pixel: Optional[tuple[float, float]]      # from FK projection (ground truth-ish)
    
    cv_detected: bool
    llm_detected: bool
    
    pixel_agreement: Optional[float]   # ||cv - llm|| in pixels
    cv_fk_error: Optional[float]       # ||cv - fk|| in pixels
    llm_fk_error: Optional[float]      # ||llm - fk|| in pixels
    
    llm_confidence: float              # self-reported by LLM
```

### 4.2 Aggregate metrics

| Metric | Definition | What it tells us |
|--------|-----------|-----------------|
| **Detection rate** | % of joints where method returned a position | Reliability |
| **Agreement rate** | % of jointly-detected joints where \|\|cv - llm\|\| < 50px | Do they see the same thing? |
| **Mean FK error (CV)** | Average \|\|cv_pixel - fk_pixel\|\| | CV accuracy |
| **Mean FK error (LLM)** | Average \|\|llm_pixel - fk_pixel\|\| | LLM accuracy |
| **Complementary detections** | Cases where one detects and the other doesn't | Added value |
| **Both-agree FK error** | FK error when both pipelines agree (<50px) | Confidence boost |

### 4.3 Trust decision matrix

| CV detects | LLM detects | Agreement | Action |
|-----------|-------------|-----------|--------|
| ✅ | ✅ | < 50px | **High confidence** — use CV detection |
| ✅ | ✅ | > 50px | **Flag for review** — check both against FK |
| ✅ | ❌ | — | **Normal** — use CV (LLM unreliable expected) |
| ❌ | ✅ | — | **Interesting** — log LLM detection, don't use for calibration |
| ❌ | ❌ | — | **Pose failed** — skip or retry |

**Important:** The LLM pipeline NEVER overrides CV. It only adds confidence signals and catches potential issues. The 50px agreement threshold accounts for ASCII grid quantization (each char ≈ 24×31 px).

### 4.4 Success criteria for the experiment

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| LLM detection rate (end-effector) | > 50% | LLM can find the gripper tip at least half the time |
| LLM mean FK error (end-effector) | < 100px | Roughly correct quadrant (within ~2 ASCII chars) |
| Agreement rate (when both detect) | > 60% | More often agrees than disagrees with CV |
| Complementary value | ≥ 3 cases | LLM catches at least 3 CV failures across 40 frames |

**If these thresholds aren't met:** The LLM ASCII approach doesn't add enough value. Archive results and don't integrate further.

---

## 5. Data Flow

```
For each calibration pose (20 poses × 2 cameras = 40 frames):

1. CAPTURE
   raw_frame = camera_server.get_snapshot(cam_id)  # 1920×1080 JPEG
   joint_angles = arm_api.get_state()               # 6 floats (degrees)

2. CV PIPELINE (existing, runs first)
   segmentation = arm_segmenter.segment_arm(raw_frame)
   cv_detections = {joint: pixel for detected joints}

3. FK PROJECTION (reference)
   positions_3d = fk_positions(joint_angles)
   fk_pixels = project_to_camera(positions_3d, intrinsics, extrinsics)

4. ASCII CONVERSION
   ascii_text = ascii_converter.frame_to_ascii(raw_frame)  # 80×35

5. LLM CALL (async, doesn't block calibration)
   prompt = build_prompt(ascii_text, cam_id, joint_angles, fk_hints=None)
   response = await gemini_flash.generate(prompt)

6. PARSE & SCALE
   llm_grid_coords = parse_json(response)  # {joint: {col, row, confidence}}
   llm_pixels = {joint: (col * 24.0, row * 30.9) for each detection}

7. COMPARE
   comparison = compare(cv_detections, llm_pixels, fk_pixels)
   
8. STORE
   save to data/calibration/session_XXX/llm_results/pose_{N}_cam{id}.json:
   {
     "ascii_frame": ascii_text,
     "prompt": prompt,
     "raw_response": response,
     "llm_detections": llm_grid_coords,
     "llm_pixels": llm_pixels,
     "cv_detections": cv_detections,
     "fk_pixels": fk_pixels,
     "comparison": comparison,
     "model": "gemini-2.0-flash",
     "tokens_used": {input: N, output: N}
   }
```

### Async execution

The LLM calls should be **fire-and-forget** during calibration. The calibration loop doesn't wait for LLM results — it proceeds with CV detections as normal. LLM results are collected after all poses complete and analyzed in batch.

```python
# In calibration loop
llm_tasks = []
for pose_idx, pose in enumerate(calibration_poses):
    move_to_pose(pose)
    frame0, frame1 = capture_frames()
    
    # CV runs synchronously (fast)
    cv_result = cv_pipeline.detect(frame0, frame1)
    
    # LLM runs async (slow, ~1-3s per call)
    task = asyncio.create_task(llm_calibrator.detect_joints(frame0, "cam0", angles))
    llm_tasks.append(task)
    task = asyncio.create_task(llm_calibrator.detect_joints(frame1, "cam1", angles))
    llm_tasks.append(task)

# After all poses, collect LLM results
llm_results = await asyncio.gather(*llm_tasks)
comparison_report = generate_comparison(cv_results, llm_results, fk_predictions)
```

---

## 6. Cost Estimation

### Per-frame token budget (80×35 ASCII, Gemini Flash)

| Component | Tokens (approx) |
|-----------|-----------------|
| System prompt + instructions | ~400 |
| ASCII frame (2,800 chars) | ~1,000 |
| Joint angles + metadata | ~50 |
| FK hints (if used) | ~100 |
| **Total input** | **~1,550** |
| Output (JSON response) | ~150 |

### Full calibration run

| Scenario | Frames | Input tokens | Output tokens | Cost (Gemini Flash) |
|----------|--------|-------------|---------------|-------------------|
| 20 poses × 2 cameras (blind) | 40 | 62,000 | 6,000 | **$0.009** |
| Same + FK-hinted run | 80 | 128,000 | 12,000 | **$0.018** |
| + 5 spot-checks with Claude Sonnet | 85 | 135,750 | 12,750 | **$0.60** (Claude dominates) |
| Higher res (120×50) | 40 | 100,000 | 6,000 | **$0.012** |

**Bottom line: A full calibration run with Gemini Flash costs less than $0.02.** Even running it 50 times during development costs under $1. Cost is a non-issue.

Claude Sonnet spot-checks at ~$0.60 for 5 frames are worth it for quality comparison but shouldn't be the default.

---

## 7. Work Breakdown

### Task 1: `LLMCalibrator` module
**File:** `src/vision/llm_calibrator.py`  
**Effort:** ~2 hours  
- `AsciiConverter` integration at 80×35
- Prompt builder (blind + hinted modes)
- Gemini Flash API client (using existing API key)
- JSON response parser with validation
- ASCII→pixel coordinate scaling

### Task 2: Calibration loop integration  
**File:** Modify calibration orchestrator  
**Effort:** ~1 hour  
- Add async LLM calls alongside CV pipeline
- Store LLM results per-pose
- Don't block calibration on LLM responses

### Task 3: Comparison & reporting  
**File:** `src/vision/llm_cv_comparison.py`  
**Effort:** ~1.5 hours  
- DetectionComparison dataclass
- Aggregate metrics computation
- Summary report generation (markdown table)
- Per-pose visualization (optional: overlay both detections on ASCII frame)

### Task 4: Evaluation run  
**Effort:** ~1 hour (requires arm hardware)  
- Run full 20-pose calibration with both pipelines
- Generate comparison report
- Decide if LLM pipeline adds value (see §4.4 success criteria)

### Task 5: Resolution/prompt sweep (optional)  
**Effort:** ~2 hours  
- Test 80×35 vs 120×50 vs 160×70
- Test blind vs hinted
- Test `CHARSET_STANDARD` vs `CHARSET_DETAILED`
- 3×2×2 = 12 variants × 5 frames = 60 LLM calls (~$0.01)

### Total estimated effort: ~7.5 hours
### Total estimated API cost: < $5 for full development cycle

---

## 8. Known Limitations & Honest Assessment

### LLMs are bad at this

Let's be real: LLMs have poor spatial reasoning over text grids. Asking an LLM to find pixel coordinates in ASCII art is asking it to do something it wasn't designed for. Expected failure modes:

1. **Off-by-many errors** — LLM says col=40 when the arm is at col=55
2. **Hallucinated detections** — LLM "sees" joints in background clutter
3. **Inconsistent across runs** — same frame, different answers (mitigated by temperature=0)
4. **Matte-black arm invisible** — sparse ASCII chars for dark objects mean the arm body is mostly spaces/dots. The LLM may only find the gold accents (bright chars).

### Why do it anyway

1. **It's nearly free** — $0.02 per full run means there's no cost barrier to trying
2. **Complementary failure modes** — CV fails on low contrast; LLMs might catch structural patterns CV misses (or vice versa)
3. **Baseline for future work** — if ASCII doesn't work, we have data showing why, and can try sending actual images to multimodal models instead
4. **Confidence scoring** — even imprecise LLM agreement with CV increases trust in the CV result

### The real experiment

This isn't "can LLMs replace CV for calibration?" (answer: no). It's:
- **Can LLMs detect a robotic arm in ASCII art at all?** (detection rate metric)
- **When they detect it, are they roughly right?** (FK error metric)  
- **Does their agreement with CV correlate with CV accuracy?** (confidence metric)

If the answer to all three is "somewhat yes," the pipeline has value. If not, we archive it and move on. Total sunk cost: <$5 and a day of work.

---

## 9. Future Extensions (if experiment succeeds)

- **Multimodal mode**: Send actual JPEG frames to Gemini/Claude vision models instead of ASCII. Much better spatial reasoning, but higher cost and different architecture.
- **Temporal reasoning**: Send 3 consecutive ASCII frames showing the arm moving to a new pose. "Where did the arm move to?"
- **Anomaly detection**: LLM flags frames where the arm looks "wrong" (collision, unexpected obstacle, cable snag).
- **Auto-labeling**: Use LLM detections as training data for a lightweight ML joint detector.
