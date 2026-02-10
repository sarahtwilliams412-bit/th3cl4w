# ASCII Video Processing Server — Design Plan

## Overview
A dedicated FastAPI service (port 8084) that captures camera frames, converts them to ASCII art in real-time, streams live ASCII video via WebSocket, and sends ASCII snapshots to Gemini (text-only) for scene analysis — enabling LLMs to "see" without vision models.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASCII Video Server :8084                       │
│                                                                   │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐              │
│  │ Camera    │───>│ ASCII     │───>│ WebSocket    │──> Browser   │
│  │ Fetcher   │    │ Converter │    │ Streamer     │   (live feed)│
│  │ (:8081)   │    │ (reuse!)  │    │              │              │
│  └──────────┘    └─────┬─────┘    └──────────────┘              │
│                        │                                         │
│                        v                                         │
│                  ┌───────────┐    ┌──────────────┐              │
│                  │ Snapshot   │───>│ LLM Analyst  │──> Gemini   │
│                  │ Capture    │    │ (text-only)  │   Flash     │
│                  └───────────┘    └──────┬───────┘              │
│                                          │                       │
│                                   ┌──────┴───────┐              │
│                                   │ Session Mgr  │              │
│                                   │ (chat history)│              │
│                                   └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘

External:
  Camera Server :8081  ──  /snap/{id} (JPEG)  /cam/{id} (MJPEG)
  3 cameras: 0=overhead, 1=side(Brio), 2=arm-mounted
```

## Reuse Analysis

| Existing Code | What to Reuse | How |
|---|---|---|
| `src/vision/ascii_converter.py` | `AsciiConverter` class, all charsets, `decode_jpeg_to_ascii()`, `decode_jpeg_to_color_data()` | Import directly — fully production-ready |
| `src/vision/camera_pipeline.py` | `AsciiFrame` dataclass, `PipelineConfig` pattern | Reference for data structures |
| `src/vision/llm_detector.py` | Gemini SDK setup pattern, retry logic, `genai.configure()` | Adapt prompting strategy for general analysis (not just joints) |
| `tools/ascii_cameras_tui.py` | HTTP frame fetching pattern (`fetch_frame_http`) | Same approach for server-side fetching |
| `web/camera_server.py` | Camera endpoints: `/snap/{id}` for JPEG | Fetch from these endpoints |

**Key insight**: The converter and camera fetching are fully built. The new work is: streaming server, general-purpose LLM analyst (vs joint-specific), web UI, and session management.

## Module Breakdown

### `src/ascii/__init__.py`
Empty init, makes it a package.

### `src/ascii/converter.py`
**Thin wrapper** around existing `src.vision.ascii_converter`. Adds:
- `fetch_and_convert(cam_id, width, height, charset)` — fetches JPEG from :8081 and converts
- Edge-enhancement mode (optional Sobel pre-filter before ASCII mapping)
- Returns both plain text and color data

### `src/ascii/streamer.py`
Manages live ASCII feeds for all cameras:
- Background threads fetching frames from `:8081/snap/{id}` at configurable FPS
- Stores latest ASCII frame per camera
- Maintains set of WebSocket connections per camera
- Broadcasts new frames to connected clients
- Config: resolution, charset, FPS per camera or global

### `src/ascii/llm_analyst.py`
Gemini text-only analysis with expert ASCII art prompting:
- System prompt establishing ASCII art expertise
- Accepts ASCII frame text + user question
- Returns structured analysis
- Supports multiple analysis modes: describe, detect_objects, spatial_relations, arm_pose
- Uses `gemini-2.0-flash` for speed

### `src/ascii/session.py`
Conversation session management:
- Each session has an ID, chat history, associated camera
- Stores recent ASCII frames sent to LLM for context
- Supports multi-turn conversation about what the LLM "sees"
- Auto-cleanup of old sessions

## API Spec

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/ascii/stream/{cam_id}` | Current ASCII frame as JSON `{ascii, width, height, timestamp}` |
| `GET` | `/api/ascii/config` | Current settings (resolution, charset, FPS) |
| `POST` | `/api/ascii/config` | Update settings `{width, height, charset, fps}` |
| `POST` | `/api/ascii/analyze` | Send ASCII to LLM `{cam_id, question, session_id?}` → `{answer, session_id}` |
| `GET` | `/api/ascii/sessions` | List active analysis sessions |
| `GET` | `/api/ascii/sessions/{id}` | Get session history |
| `DELETE` | `/api/ascii/sessions/{id}` | Delete a session |

### WebSocket Endpoints

| Path | Description |
|---|---|
| `WS /ws/ascii/{cam_id}` | Live ASCII stream. Sends JSON frames: `{lines, width, height, timestamp, colors?}` |

Client can send config updates on the WS: `{width, height, charset, fps, color}`

## LLM Prompt Templates

### System Prompt
```
You are an expert ASCII art analyst. You specialize in interpreting ASCII art
representations of real-world camera feeds.

You are viewing a {width}x{height} character ASCII representation of a camera feed.
The character ramp used is: "{charset}" (from lightest/empty to darkest/densest).

This camera ({camera_perspective}) is observing a robotic arm workspace containing:
- A SO-ARM100/D1 6-DOF robotic arm (matte black body, gold accents at joints)
- A workspace table with various objects
- The arm has joints: base, shoulder, elbow, wrist_flex, wrist_rotate, gripper

Dense characters (like @ # % *) represent dark or detailed areas.
Sparse characters (like . : -) represent lighter or empty areas.
Space characters represent the brightest/emptiest areas.

You must analyze ONLY the raw ASCII text characters. You are NOT using any image
processing or computer vision. You read the ASCII art like a human reads ASCII art —
by recognizing patterns of characters that form shapes, edges, and objects.

Analyze carefully: look at character density patterns, edges formed by character
transitions, and spatial relationships between character clusters.
```

### Camera Perspective Context
```
Camera 0 (overhead): Looking down at the workspace from above. The arm base is
typically in the center-bottom. Objects appear as clusters of dense characters
against the lighter table surface.

Camera 1 (side/Brio): Side view showing the arm's height profile. The arm extends
upward from its base. Joint articulations are visible as bends in the character
density pattern.

Camera 2 (arm-mounted): View from the arm's end effector. Shows what the gripper
is approaching. Close objects appear as large dense character clusters.
```

### Analysis Query Template
```
Here is the current ASCII frame from {camera_name}:

```
{ascii_frame}
```

Question: {user_question}

Provide a detailed analysis based on what you can interpret from the ASCII
character patterns. Reference specific regions using approximate character
coordinates (col, row) when describing locations.
```

## UI Wireframe

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔤 ASCII Video Server                              [Config ⚙️]        │
├────────────────────────┬────────────────────────┬───────────────────────┤
│  ▶ Cam 0: Overhead     │  ▶ Cam 1: Side         │  ▶ Cam 2: Arm        │
│ ┌────────────────────┐ │ ┌────────────────────┐ │ ┌─────────────────┐  │
│ │                    │ │ │                    │ │ │                 │  │
│ │   Live ASCII       │ │ │   Live ASCII       │ │ │  Live ASCII     │  │
│ │   Feed             │ │ │   Feed             │ │ │  Feed           │  │
│ │   (monospace)      │ │ │   (monospace)      │ │ │  (monospace)    │  │
│ │                    │ │ │                    │ │ │                 │  │
│ │                    │ │ │                    │ │ │                 │  │
│ └────────────────────┘ │ └────────────────────┘ │ └─────────────────┘  │
│ [📸 Capture & Analyze] │ [📸 Capture & Analyze] │ [📸 Capture & Ask]   │
├────────────────────────┴────────────────────────┴───────────────────────┤
│  💬 ASCII Art Analysis Chat                                             │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ System: Captured frame from Cam 0 (120x60, charset: standard)      │ │
│ │ You: What objects are on the table?                                 │ │
│ │ Analyst: I can see several distinct clusters of dense characters... │ │
│ │ You: Where is the arm's gripper pointing?                           │ │
│ │ Analyst: The end effector appears to be at approximately (85, 20)..│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────┐ [Send] [New Chat]  │
│ │ Ask about what the ASCII art shows...           │                    │
│ └─────────────────────────────────────────────────┘                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Settings: Resolution [120]x[60]  Charset [Standard ▾]  FPS [5]       │
│  Charsets: Standard( .:-=+*#%@) | Detailed(68 chars) | Blocks(░▒▓█)   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Core Modules (Subagent 1)
1. `src/ascii/__init__.py` — package init
2. `src/ascii/converter.py` — wrapper around existing converter + HTTP fetch
3. `src/ascii/streamer.py` — multi-camera ASCII streaming manager
4. `src/ascii/llm_analyst.py` — Gemini text-only analyst with expert prompts
5. `src/ascii/session.py` — conversation session management
6. Tests for each module

### Phase 2: Server + UI (Subagent 2)
1. `web/ascii_server.py` — FastAPI app on :8084 with all endpoints
2. `web/static/ascii_video.html` — single-page web UI
3. Update `web/watchdog.sh` to start ascii server
4. Integration testing

### Phase 3: Verification (Planning Agent)
1. Verify imports resolve correctly
2. Check API endpoint consistency between server and UI
3. Ensure watchdog integration is correct
4. Git commit
