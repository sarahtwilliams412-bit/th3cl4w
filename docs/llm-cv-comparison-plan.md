# LLM vs CV Calibration Comparison Plan

**Date:** 2026-02-08
**Purpose:** Evaluate whether LLM-based joint detection on ASCII art adds value over the existing CV pipeline during calibration.

---

## 1. Comparison Architecture

### Dual Pipeline Design

```
Camera Frame (JPEG)
    ├── CV Pipeline (existing, ~5ms)
    │   ├── gpu_preprocess → ArmSegmenter → JointDetector
    │   └── → [JointDetection × 5 joints]
    │
    └── LLM Pipeline (new, ~2-5s)
        ├── AsciiConverter(80×35, color=True) → ASCII + color grid
        ├── Prompt construction (ASCII art + joint descriptions + coordinate grid)
        ├── Gemini API call
        └── Response parsing → [normalized coords × 5 joints]

Both compared against Ground Truth: FK positions projected to pixels
```

### Result Record (per pose, per camera)

```python
@dataclass
class ComparisonResult:
    pose_index: int
    camera_id: int
    timestamp: float
    joint_angles: list[float]          # actual from /api/state

    # Ground truth
    fk_pixels: list[tuple[float, float]]  # FK-projected pixel positions per joint

    # CV pipeline
    cv_detections: list[JointDetection]   # from JointDetector
    cv_latency_ms: float

    # LLM pipeline
    llm_raw_response: str                 # full API response text
    llm_normalized_coords: list[tuple[float, float] | None]  # (0-1, 0-1) per joint
    llm_pixels: list[tuple[float, float] | None]  # converted to pixel coords
    llm_latency_ms: float
    llm_input_tokens: int
    llm_output_tokens: int
    llm_cost_usd: float

    # Metrics (computed)
    cv_errors_px: list[float | None]      # per-joint pixel distance from FK
    llm_errors_px: list[float | None]     # per-joint pixel distance from FK
    cv_detected: list[bool]               # did CV find each joint?
    llm_detected: list[bool]              # did LLM find each joint?
```

### Storage

JSON files per session:
```
data/llm_comparison/
  session_YYYYMMDD_HHMMSS/
    results.json        — list of ComparisonResult dicts
    ascii_frames/       — saved ASCII art sent to LLM (for debugging)
      pose_001_cam0.txt
    summary.json        — aggregated metrics
```

No SQLite needed — 20 poses × 2 cameras = 40 records max per session.

---

## 2. Integration with Existing Calibration Flow

### Sequence

The existing 20-pose calibration (§1.2 of calibration-plan.md) runs unchanged. At each pose:

1. Command pose via `/api/command/set-all-joints`
2. Wait 2.5s for settling (existing)
3. Read actual angles from `/api/state` (existing)
4. Capture JPEG frames from both cameras via `http://localhost:8081/snap/{0,1}`
5. **CV pipeline** — run synchronously (fast, ~5ms):
   - `decode_jpeg_gpu()` → `ArmSegmenter.segment()` → `JointDetector.detect_joints()`
6. **LLM pipeline** — fire async, don't block:
   - Convert frame to ASCII (80×35 with color)
   - Queue Gemini API call
   - Continue calibration immediately
7. After all 20 poses complete, await all pending LLM responses
8. Compute comparison metrics, write results

### Key principle: LLM calls NEVER slow down calibration

The calibration needs ~50s (20 poses × 2.5s). LLM calls take ~2-5s each. Fire them all async and collect results at the end. If some timeout, mark as failed — that's data too.

---

## 3. LLM API Integration

### Client Design

```python
import google.generativeai as genai
import asyncio, time, json

class LLMJointDetector:
    """Detect arm joints from ASCII art using Gemini."""

    MODEL = "gemini-2.0-flash"  # fast + cheap, good enough for experiment
    TIMEOUT_S = 15.0
    MAX_RETRIES = 2

    # Gemini 2.0 Flash pricing (approximate)
    COST_PER_INPUT_TOKEN = 0.075 / 1_000_000   # $0.075 per 1M input tokens
    COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000    # $0.30 per 1M output tokens

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.MODEL)
        self.total_cost = 0.0
        self.total_calls = 0

    async def detect_joints(
        self, ascii_art: str, camera_description: str
    ) -> dict:
        """Send ASCII art to Gemini, get normalized joint coordinates.

        Returns dict with:
          - joints: list of {name, x, y} with x,y in 0.0-1.0 range (or null)
          - raw_response: str
          - input_tokens: int
          - output_tokens: int
          - latency_ms: float
          - cost_usd: float
        """
        prompt = self._build_prompt(ascii_art, camera_description)
        t0 = time.monotonic()

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,  # low temp for consistency
                        max_output_tokens=300,
                    ),
                )
                break
            except Exception as e:
                if attempt == self.MAX_RETRIES:
                    return self._failure_result(str(e), time.monotonic() - t0)
                await asyncio.sleep(1.0 * (attempt + 1))

        latency_ms = (time.monotonic() - t0) * 1000
        raw = response.text

        # Token counting
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        cost = (input_tokens * self.COST_PER_INPUT_TOKEN +
                output_tokens * self.COST_PER_OUTPUT_TOKEN)
        self.total_cost += cost
        self.total_calls += 1

        # Parse structured output
        joints = self._parse_response(raw)

        return {
            "joints": joints,
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }

    def _build_prompt(self, ascii_art: str, camera_desc: str) -> str:
        return f"""You are analyzing an ASCII art representation of a robotic arm (Unitree Z1/D1, 6-DOF).
The ASCII art is {80} characters wide and {35} lines tall. Each character represents a small region of the camera frame.

Camera: {camera_desc}

The arm has 5 visible joints/points:
- base: where the arm connects to its mount (bottom-center typically)
- shoulder: first major joint above the base
- elbow: middle joint where the arm bends
- wrist: joint near the end of the arm
- end_effector: the gripper tip at the very end

ASCII art of the current frame:
```
{ascii_art}
```

For each joint you can identify, report its position as normalized coordinates where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right.

Respond ONLY with this JSON (no other text):
{{"joints": [
  {{"name": "base", "x": <0.0-1.0 or null>, "y": <0.0-1.0 or null>}},
  {{"name": "shoulder", "x": <0.0-1.0 or null>, "y": <0.0-1.0 or null>}},
  {{"name": "elbow", "x": <0.0-1.0 or null>, "y": <0.0-1.0 or null>}},
  {{"name": "wrist", "x": <0.0-1.0 or null>, "y": <0.0-1.0 or null>}},
  {{"name": "end_effector", "x": <0.0-1.0 or null>, "y": <0.0-1.0 or null>}}
]}}

Use null for x and y if you cannot identify a joint. Be conservative — only report positions you're reasonably confident about."""

    def _parse_response(self, raw: str) -> list[dict]:
        """Extract joint coordinates from LLM response."""
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            return data.get("joints", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _failure_result(self, error: str, elapsed_s: float) -> dict:
        return {
            "joints": [],
            "raw_response": f"ERROR: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": elapsed_s * 1000,
            "cost_usd": 0.0,
        }
```

### Cost Estimation

Per frame at 80×35 ASCII:
- Input: ~3000 chars ASCII + ~200 chars prompt ≈ ~1000 tokens
- Output: ~200 tokens
- Cost per call: ~$0.000135 (Gemini 2.0 Flash)
- **20 poses × 2 cameras = 40 calls ≈ $0.005 per calibration session**

This is negligible. Even running 100 sessions costs under $1.

---

## 4. Scoring System

### Per-Joint Metrics

| Metric | Description | How Computed |
|--------|-------------|-------------|
| **Detection Rate** | % of poses where joint was found | `count(detected) / total_poses` |
| **Position Error (px)** | Euclidean distance from FK ground truth | `sqrt((x-fx)² + (y-fy)²)` in pixels |
| **Consistency** | Std dev of error across repeat measurements | Run same pose 5× (static), compute σ |
| **Latency** | Wall-clock time per detection | Direct measurement |
| **Cost** | USD per detection | Token count × rate |

### Aggregated Scoring

```python
def compute_score(results: list[ComparisonResult]) -> dict:
    """Compute overall LLM vs CV comparison score."""
    for pipeline in ["cv", "llm"]:
        per_joint = {}
        for joint_idx in range(5):
            errors = [r[f"{pipeline}_errors_px"][joint_idx]
                      for r in results
                      if r[f"{pipeline}_detected"][joint_idx]]
            detected = sum(1 for r in results if r[f"{pipeline}_detected"][joint_idx])
            total = len(results)

            per_joint[joint_idx] = {
                "detection_rate": detected / total,
                "mean_error_px": np.mean(errors) if errors else None,
                "median_error_px": np.median(errors) if errors else None,
                "max_error_px": np.max(errors) if errors else None,
                "std_error_px": np.std(errors) if errors else None,
            }

    return {
        "cv": {"per_joint": per_joint_cv, "mean_latency_ms": ..., "total_cost": 0.0},
        "llm": {"per_joint": per_joint_llm, "mean_latency_ms": ..., "total_cost": ...},
        "verdict": "..."  # see below
    }
```

### Decision Criteria

**LLM is worth pursuing IF any of these hold:**
1. LLM detection rate > 60% for joints where CV detection rate < 50% (LLM finds things CV misses)
2. LLM mean error < 30px when CV falls back to FK-only (LLM refines where CV can't)
3. LLM detects the matte-black arm in conditions where gold centroid matching fails

**LLM is NOT worth pursuing IF:**
1. CV detection rate > 80% across all joints (CV already works well enough)
2. LLM error > 50px on average (LLM is too imprecise even for coarse guidance)
3. LLM detection rate < 40% (can't even reliably see the arm)

**The interesting middle ground:** LLM might be useful as a *fallback* when CV confidence is low (FK_ONLY detections). In that case, LLM doesn't need to be great — just better than blind FK.

---

## 5. Build Plan — Sub-Agent Breakdown

### Agent 1: LLM Client (`src/vision/llm_joint_detector.py`)
**Scope:** API integration, prompt engineering, response parsing

- Implement `LLMJointDetector` class (see §3 above)
- Handle Gemini API key from environment (`GEMINI_API_KEY`)
- Structured JSON output parsing with fallback
- Cost tracking
- Async interface (`async def detect_joints(...)`)
- Unit tests with mocked API responses

**Files:**
- `src/vision/llm_joint_detector.py` — main implementation
- `tests/test_llm_joint_detector.py` — unit tests with mock responses

**Estimated effort:** Small. Mostly prompt engineering + JSON parsing.

### Agent 2: Comparison Engine (`src/vision/pipeline_comparator.py`)
**Scope:** Run both pipelines on same frame, compute metrics

- Accept a JPEG frame + joint angles + camera calibration
- Run CV pipeline (existing `JointDetector`)
- Run LLM pipeline (Agent 1's `LLMJointDetector`)
- Compute FK ground truth pixels
- Calculate per-joint error for both pipelines
- Return `ComparisonResult`
- Generate summary report (JSON + human-readable text)

**Files:**
- `src/vision/pipeline_comparator.py` — comparison logic
- `tests/test_pipeline_comparator.py` — tests with saved frame data

**Estimated effort:** Medium. Needs to wire together existing components.

### Agent 3: Calibration Runner (`src/calibration/dual_pipeline_runner.py`)
**Scope:** Orchestrate the 20-pose calibration with dual pipeline

- Execute the 20-pose sequence from calibration-plan.md §1.2
- At each pose: capture frames, run CV sync, fire LLM async
- Collect all results after sequence completes
- Save to `data/llm_comparison/session_*/`
- Can run standalone or integrate into existing calibration flow

**Files:**
- `src/calibration/dual_pipeline_runner.py` — orchestrator
- Reuses existing `/api/command/set-all-joints` and camera endpoints

**Estimated effort:** Medium. Async orchestration is the main complexity.

**Dependencies:** Agents 1 and 2 must complete first.

### Agent 4: Results Report (`src/calibration/comparison_report.py`)
**Scope:** Generate human-readable comparison report

- Read `results.json` from a session
- Compute all metrics from §4
- Generate markdown report with tables
- Per-joint breakdown, overall verdict
- Optionally: simple web endpoint (`/api/llm-comparison/report`) to view in browser

**Files:**
- `src/calibration/comparison_report.py` — report generator
- Template for markdown output

**Estimated effort:** Small. Mostly formatting.

**Dependencies:** Can develop in parallel with mock data; needs Agent 2's result format.

### Build Order

```
Phase 1 (parallel):
  Agent 1: LLM Client
  Agent 4: Results Report (with mock data)

Phase 2 (after Agent 1):
  Agent 2: Comparison Engine

Phase 3 (after Agent 2):
  Agent 3: Calibration Runner

Phase 4: Run experiment, review report, decide go/no-go
```

---

## 6. Success Criteria

### Minimum Bar (experiment worth continuing)

| Criteria | Threshold | Rationale |
|----------|-----------|-----------|
| LLM detects end_effector | >70% of poses | This is the easiest joint to spot |
| LLM mean error (detected joints) | <40px | Within ~5% of frame width (800px) |
| LLM finds joints CV missed | ≥3 cases across 40 frames | Demonstrates complementary value |
| Cost per session | <$0.05 | Must be negligible |

### Kill Criteria (stop the experiment)

| Criteria | Threshold |
|----------|-----------|
| LLM detection rate | <30% across all joints |
| LLM systematically wrong | Error > 100px when it claims to detect |
| API reliability | >20% timeout/error rate |
| CV already sufficient | CV detection rate >85% for all joints |

### Best Case Outcome

LLM serves as a **fallback detector** integrated into `PoseFusion`:
- When `JointDetector` returns `FK_ONLY` (no visual match), query LLM
- LLM provides coarse position estimate (±30px)
- `PoseFusion` blends LLM estimate with FK at low weight
- Net result: fewer pure FK-only joints, slightly better 3D reconstruction

### Worst Case Outcome

LLM can't reliably identify joints in ASCII art → kill the experiment, total cost <$1, move on. The existing CV pipeline continues unchanged.

---

## 7. Design Decisions

### Why 80×35 ASCII (not higher res)?
- Keeps token count ~1000 input tokens per call (~$0.0001)
- LLMs can't use pixel-level precision anyway — 80 columns gives ~1.25% X resolution
- Higher res = more tokens = more cost = same imprecise answer

### Why normalized coordinates (0-1) instead of pixel positions?
- LLMs are bad at counting characters to determine exact column/row
- Normalized coords match how humans describe positions ("about 40% from left")
- Easy to convert: `pixel_x = norm_x * frame_width`

### Why Gemini Flash instead of a larger model?
- This is a cost experiment — Flash is ~10× cheaper than Pro
- If Flash can't do it, Pro probably can't either (the task is fundamentally hard or easy)
- Can upgrade later if Flash shows promise but needs refinement

### Why async/batched LLM calls?
- Calibration takes 50s for 20 poses
- 40 LLM calls × 3s each = 120s sequential, but only ~5s if parallelized
- Never block the calibration loop

### Why JSON results instead of SQLite?
- 40 records per session — SQLite is overkill
- JSON is human-readable, easy to inspect, easy to share
- Can always load into pandas for analysis
