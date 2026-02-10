# Live Video Streaming Plan — th3cl4w D1

> Replacing JPEG snapshot polling with real-time video streaming for 3 camera feeds.

## Current State

The camera server (`web/camera_server.py`) already has two transport modes:
- **MJPEG streams** at `/cam/{0,1,2}` — `multipart/x-mixed-replace` with `--frame` boundaries
- **JPEG snapshots** at `/snap/{0,1,2}` — single frame per request

The web UI polls snapshots via `setInterval` + `<img>` src swapping. The MJPEG endpoints exist but aren't wired to the UI.

Camera threads: 3× `CameraThread` capturing at 15 fps, JPEG quality 92, V4L2 MJPG fourcc, 1920×1080. Each stores the latest encoded JPEG behind a threading lock.

---

## 1. Streaming Protocol Selection

### Comparison

| Protocol | Latency | Complexity | Browser Support | Overlay Support | CPU Cost |
|----------|---------|------------|-----------------|-----------------|----------|
| **MJPEG stream** | ~66ms (1 frame @ 15fps) | Very low | Native `<img>` | Server-side burn-in | Low |
| **WebSocket binary** | ~33–66ms | Medium | Universal | Client-side canvas | Medium |
| **WebRTC** | 30–100ms | High (STUN/ICE/SDP) | Universal | Client-side | High (VP8/H264) |
| **HLS/DASH** | 2–10 seconds | Medium | Universal | Segment-level only | Medium |

### Recommendation: MJPEG Now → WebSocket Later

**For a local-network robotics app where latency matters most:**

**Phase 1 — MJPEG (immediate win)**
- The server already serves MJPEG. Just change the UI to use `<img src="http://host:8081/cam/0">` instead of polling snapshots.
- Zero server changes. Latency drops from polling interval (200–500ms) to ~66ms.
- Works with any browser. Dead simple.
- Limitation: overlays must be burned into frames server-side (no client-side drawing on an `<img>` tag).

**Phase 2 — WebSocket binary frames (for CV overlays)**
- Switch to sending JPEG blobs over WebSocket when we need client-side canvas overlays.
- Client draws frame on `<canvas>`, then draws bounding boxes / markers on top.
- Adds: WebSocket server, client canvas renderer. Modest complexity.
- Latency comparable to MJPEG but with full overlay flexibility.

**Why not WebRTC?**
- Overkill for local network (no NAT traversal needed).
- H.264/VP8 encoding adds latency and CPU load vs raw JPEG passthrough.
- We already have JPEG frames from the cameras — re-encoding to video codec is wasteful.
- Signaling complexity (SDP offer/answer, ICE candidates) for no real benefit on LAN.

**Why not HLS/DASH?**
- Multi-second latency. Unacceptable for real-time robot control.

---

## 2. Server-Side Changes

### 2.1 Phase 1: Wire Up Existing MJPEG (No Server Changes)

The MJPEG streaming handler `_handle_mjpeg()` already works. It loops, writing `--frame` boundaries with JPEG data at `1/fps` intervals. **No server changes needed** — just update the frontend.

### 2.2 Phase 2: WebSocket Frame Server

Add a WebSocket endpoint to the FastAPI server (`web/server.py`) that taps into the same `CameraThread` frame buffers:

```python
# In server.py — add alongside existing WebSocket endpoints

from fastapi import WebSocket
import asyncio

# Import camera threads from camera_server (or share via module)
from camera_server import cameras

@app.websocket("/ws/cam/{cam_id}")
async def ws_camera_feed(websocket: WebSocket, cam_id: int):
    """Stream camera frames as binary WebSocket messages."""
    if cam_id not in cameras:
        await websocket.close(code=4004)
        return
    
    await websocket.accept()
    cam = cameras[cam_id]
    interval = 1.0 / cam.fps
    
    try:
        while True:
            frame_bytes = cam.get_frame()
            await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
```

**Architecture decision:** Rather than running two separate servers, integrate camera access into the FastAPI server. Options:

1. **Shared module** — Import `CameraThread` into `server.py`, start camera threads there, remove `camera_server.py` as a standalone process. Simplest.
2. **Keep separate** — Camera server stays on 8081 for MJPEG, FastAPI on 8080 adds WebSocket feeds by reading from the same shared `cameras` dict (requires both to run in same process or use shared memory).
3. **Proxy approach** — FastAPI proxies camera frames from 8081. Adds latency, not recommended.

**Recommendation:** Option 1 — consolidate into `server.py`. One process, one port, simpler deployment.

### 2.3 Frame Pipeline

```
V4L2 capture (hardware MJPG decode)
    │
    ▼
cv2.VideoCapture.read() → BGR numpy array
    │
    ├──► Raw frame buffer (for CV processing)
    │
    ▼
CV processing hook (optional)
    │
    ▼
cv2.imencode('.jpg', frame, quality) → JPEG bytes
    │
    ├──► MJPEG stream (multipart HTTP)
    └──► WebSocket binary message
```

### 2.4 Handling 3 Simultaneous Streams

Current architecture is already correct:
- Each camera has its own `CameraThread` with a dedicated `cv2.VideoCapture`
- Capture runs in a daemon thread, decoupled from serving
- Frame buffer is single-frame (latest only), no queue buildup
- Lock contention is minimal (lock held only during `_frame` pointer swap)

**Optimization for 3 streams at 15fps each = 45 frames/sec total:**
- JPEG encoding is the bottleneck (~5–15ms per 1080p frame on CPU)
- Current: encode happens in capture thread, blocking next capture
- Better: capture raw frame, encode on-demand or in a separate encoder thread

```python
class CameraThread:
    def __init__(self, ...):
        self._raw_frame: Optional[np.ndarray] = None  # BGR
        self._jpeg_cache: Optional[bytes] = None
        self._jpeg_dirty = True
    
    def _capture_loop(self):
        while self._running:
            ret, frame = cap.read()
            with self._lock:
                self._raw_frame = frame
                self._jpeg_dirty = True
            time.sleep(1.0 / self.fps)
    
    def get_frame(self) -> bytes:
        with self._lock:
            if self._jpeg_dirty and self._raw_frame is not None:
                _, buf = cv2.imencode('.jpg', self._raw_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                self._jpeg_cache = buf.tobytes()
                self._jpeg_dirty = False
            return self._jpeg_cache or self._no_signal_frame
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None
```

This avoids encoding frames nobody is watching and lets multiple clients share one encode.

### 2.5 GPU Acceleration (RX 580 eGPU)

The RX 580 supports OpenCL but **not** NVENC/VAAPI-style hardware JPEG encoding. Options:

- **OpenCV OpenCL (`cv2.UMat`)**: Automatic GPU offload for image processing ops (resize, color convert, blur). Already hinted at in `server.py` with `OPENCV_OPENCL_DEVICE=:GPU:0`. Use `cv2.UMat(frame)` to upload to GPU.
- **JPEG encoding**: No GPU acceleration path on AMD for JPEG. Stay with CPU `imencode`. At 1080p/quality 92, expect ~8ms per frame — fine for 3×15fps.
- **Resize on GPU**: If streaming at lower resolution (e.g., 720p for web), do `cv2.resize(cv2.UMat(frame), (1280,720))` on GPU, then encode CPU-side.
- **Future**: If switching to H.264 WebRTC, AMD VA-API can hardware-encode via `ffmpeg -vaapi_device /dev/dri/renderD128`. Not needed for MJPEG/WebSocket approach.

**Practical GPU use**: Save GPU compute for CV processing (YOLO inference, image transforms), not for streaming encode.

---

## 3. Client-Side Changes

### 3.1 Phase 1: MJPEG via `<img>` Tag

Replace snapshot polling with direct MJPEG streams:

```html
<!-- Before: polling -->
<img id="cam0" src="" />
<script>
  setInterval(() => {
    document.getElementById('cam0').src = '/snap/0?t=' + Date.now();
  }, 200);
</script>

<!-- After: MJPEG stream -->
<img id="cam0" src="http://localhost:8081/cam/0" />
<!-- That's it. Browser handles multipart/x-mixed-replace natively. -->
```

For 3 feeds:
```html
<div class="camera-grid">
  <img src="http://localhost:8081/cam/0" alt="Overhead" />
  <img src="http://localhost:8081/cam/1" alt="Arm" />
  <img src="http://localhost:8081/cam/2" alt="Side" />
</div>
```

### 3.2 Phase 2: WebSocket + Canvas (for overlays)

```javascript
class CameraFeed {
  constructor(canvasId, wsUrl) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.overlays = [];  // [{type:'rect', x, y, w, h, color, label}, ...]
    this.connect(wsUrl);
  }

  connect(url) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'blob';
    this.ws.onmessage = (e) => this.onFrame(e.data);
    this.ws.onclose = () => setTimeout(() => this.connect(url), 1000);
  }

  async onFrame(blob) {
    const bitmap = await createImageBitmap(blob);
    this.canvas.width = bitmap.width;
    this.canvas.height = bitmap.height;
    this.ctx.drawImage(bitmap, 0, 0);
    this.drawOverlays();
    bitmap.close();
  }

  drawOverlays() {
    for (const o of this.overlays) {
      this.ctx.strokeStyle = o.color || '#00ff00';
      this.ctx.lineWidth = 2;
      if (o.type === 'rect') {
        this.ctx.strokeRect(o.x, o.y, o.w, o.h);
        if (o.label) {
          this.ctx.fillStyle = o.color || '#00ff00';
          this.ctx.font = '14px monospace';
          this.ctx.fillText(o.label, o.x, o.y - 4);
        }
      } else if (o.type === 'circle') {
        this.ctx.beginPath();
        this.ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
        this.ctx.stroke();
      } else if (o.type === 'line') {
        this.ctx.beginPath();
        this.ctx.moveTo(o.x1, o.y1);
        this.ctx.lineTo(o.x2, o.y2);
        this.ctx.stroke();
      }
    }
  }

  setOverlays(overlays) {
    this.overlays = overlays;
  }
}

// Initialize 3 feeds
const overhead = new CameraFeed('canvas-overhead', 'ws://localhost:8080/ws/cam/0');
const arm = new CameraFeed('canvas-arm', 'ws://localhost:8080/ws/cam/1');
const side = new CameraFeed('canvas-side', 'ws://localhost:8080/ws/cam/2');
```

### 3.3 Handling 3 Simultaneous Video Feeds

Browser considerations:
- **HTTP/1.1 connection limit**: Browsers allow ~6 concurrent connections per hostname. 3 MJPEG streams = 3 long-lived connections. Fine.
- **Canvas rendering**: `createImageBitmap()` decodes JPEG off the main thread. Drawing 3 canvases at 15fps = 45 `drawImage` calls/sec — trivial for any modern GPU.
- **Memory**: Each 1080p JPEG blob is ~200–400KB. With single-buffering, ~1.2MB live. No concern.
- **Backpressure**: If the client can't keep up, WebSocket buffers grow. Add client-side frame dropping:

```javascript
onFrame(blob) {
  if (this._rendering) return;  // Drop frame if still drawing previous
  this._rendering = true;
  createImageBitmap(blob).then(bmp => {
    this.ctx.drawImage(bmp, 0, 0);
    this.drawOverlays();
    bmp.close();
    this._rendering = false;
  });
}
```

### 3.4 Adaptive Quality

Send quality preferences from client to server via WebSocket message or query param:

```javascript
// Client sends desired quality
ws.send(JSON.stringify({ type: 'quality', fps: 10, resolution: '720p', jpeg_quality: 70 }));
```

Server adjusts per-client:
- **Resolution tiers**: 1080p (full), 720p (default for web), 480p (low bandwidth)
- **FPS tiers**: 15 (full), 10 (normal), 5 (low)
- **JPEG quality**: 92 (full), 75 (normal), 50 (thumbnail)

Automatic downgrade: if WebSocket send buffer exceeds threshold, drop quality tier.

### 3.5 Overlay Support

Two approaches, use both:

1. **Client-side overlays** (preferred for interactivity):
   - Server sends overlay data as JSON via a separate WebSocket channel or multiplexed with frames
   - Client draws on canvas after rendering frame
   - Supports: bounding boxes, joint markers, gripper position, trajectory paths, coordinate axes
   - Pro: zero encoding overhead, interactive (hover, click)

2. **Server-side burn-in** (for MJPEG or recording):
   - CV pipeline draws directly on frame before JPEG encoding
   - Used when client can't do canvas (MJPEG `<img>` mode) or for video recording
   - Pro: what you see = what you record

Overlay data protocol (JSON over WebSocket):
```json
{
  "cam_id": 0,
  "timestamp": 1707500000.123,
  "overlays": [
    {"type": "rect", "x": 450, "y": 300, "w": 120, "h": 80, "color": "#00ff00", "label": "target_cube"},
    {"type": "circle", "x": 960, "y": 540, "r": 15, "color": "#ff0000", "label": "gripper"},
    {"type": "path", "points": [[100,100],[200,150],[300,120]], "color": "#ffff00", "label": "trajectory"}
  ]
}
```

---

## 4. Frame Processing Hook

### 4.1 Architecture

```
CameraThread._capture_loop()
    │
    ▼
raw_frame (numpy BGR) stored in CameraThread._raw_frame
    │
    ├──► FrameProcessor pipeline (async, parallel to capture)
    │       │
    │       ├── processor_1: ObjectDetector (YOLO)
    │       ├── processor_2: GripperTracker
    │       ├── processor_3: ArucoDetector
    │       └── ... (pluggable)
    │       │
    │       ▼
    │    ProcessedFrame:
    │       - annotated_frame (numpy BGR with drawings)
    │       - detections[] (structured data)
    │       - overlays[] (for client-side rendering)
    │
    ├──► JPEG encode (annotated or raw, depending on mode)
    │       │
    │       ▼
    │    WebSocket / MJPEG stream → browser
    │
    └──► Overlay JSON → WebSocket → browser canvas
```

### 4.2 FrameProcessor Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    position_3d: Optional[tuple[float, float, float]] = None  # mm, if available

@dataclass 
class Overlay:
    type: str  # 'rect', 'circle', 'line', 'path', 'text'
    data: dict  # type-specific params

@dataclass
class ProcessedFrame:
    cam_id: int
    timestamp: float
    raw_frame: np.ndarray
    annotated_frame: Optional[np.ndarray]  # frame with drawings burned in
    detections: List[Detection]
    overlays: List[Overlay]  # for client-side rendering

class FrameProcessor(ABC):
    @abstractmethod
    def process(self, cam_id: int, frame: np.ndarray, timestamp: float) -> ProcessedFrame:
        """Process a single frame. Must be fast (<target_ms)."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...
```

### 4.3 Processing Pipeline Manager

```python
import threading
import time
from collections import defaultdict

class ProcessingPipeline:
    """Manages frame processors for all cameras."""
    
    def __init__(self):
        self.processors: List[FrameProcessor] = []
        self._latest_results: dict[int, ProcessedFrame] = {}
        self._lock = threading.Lock()
        self._running = False
    
    def add_processor(self, proc: FrameProcessor):
        self.processors.append(proc)
    
    def process_frame(self, cam_id: int, frame: np.ndarray) -> ProcessedFrame:
        """Run all processors on a frame. Called from camera thread or dedicated processing thread."""
        timestamp = time.time()
        all_detections = []
        all_overlays = []
        annotated = frame.copy()
        
        for proc in self.processors:
            result = proc.process(cam_id, frame, timestamp)
            all_detections.extend(result.detections)
            all_overlays.extend(result.overlays)
            if result.annotated_frame is not None:
                annotated = result.annotated_frame
        
        combined = ProcessedFrame(
            cam_id=cam_id,
            timestamp=timestamp,
            raw_frame=frame,
            annotated_frame=annotated,
            detections=all_detections,
            overlays=all_overlays,
        )
        
        with self._lock:
            self._latest_results[cam_id] = combined
        
        return combined
```

### 4.4 Integration with CameraThread

```python
class CameraThread:
    def __init__(self, ..., pipeline: Optional[ProcessingPipeline] = None):
        self.pipeline = pipeline
        self._processed_frame: Optional[bytes] = None  # annotated JPEG
        self._overlay_data: Optional[list] = None
    
    def _capture_loop(self):
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Store raw
            with self._lock:
                self._raw_frame = frame
            
            # Process if pipeline attached
            if self.pipeline:
                result = self.pipeline.process_frame(self.device_id, frame)
                annotated_jpeg = cv2.imencode('.jpg', result.annotated_frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1].tobytes()
                with self._lock:
                    self._processed_frame = annotated_jpeg
                    self._overlay_data = [o.__dict__ for o in result.overlays]
            
            # Encode raw for non-processed stream
            raw_jpeg = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1].tobytes()
            with self._lock:
                self._frame = raw_jpeg
            
            time.sleep(1.0 / self.fps)
    
    def get_frame(self, annotated=False) -> bytes:
        with self._lock:
            if annotated and self._processed_frame:
                return self._processed_frame
            return self._frame or self._no_signal_frame
    
    def get_overlays(self) -> Optional[list]:
        with self._lock:
            return self._overlay_data
```

---

## 5. Implementation Plan

### Phase 1: Basic Streaming (Replace Polling) — 1–2 days

**Goal:** Live video in the browser with minimal changes.

1. **Update `index.html`** to use MJPEG `<img>` tags pointing at existing `/cam/{id}` endpoints
2. Remove snapshot polling JavaScript
3. Add reconnection logic (if MJPEG stream drops, retry after 2s)
4. Add camera status indicators (connected/disconnected) using `/status` endpoint
5. Test with all 3 cameras simultaneously

**Deliverable:** Live 15fps video from all 3 cameras in the web UI.

### Phase 2: Server-Side CV Overlay Pipeline — 3–5 days

**Goal:** Pluggable frame processing with overlays visible in the UI.

1. Implement `FrameProcessor` interface and `ProcessingPipeline` manager
2. Add WebSocket endpoint `/ws/cam/{id}` to `server.py` for binary frame streaming
3. Add WebSocket overlay channel `/ws/overlays` for structured detection data
4. Update UI to use `<canvas>` + WebSocket for cameras that need overlays
5. Keep MJPEG as fallback for simple viewing
6. Implement first processor: `ArucoDetector` (ArUco markers for calibration)
7. Implement second processor: `GripperTracker` (color-based gripper detection)

**Deliverable:** Canvas-based video with real-time overlay drawing.

### Phase 3: Adaptive Quality + GPU Acceleration — 2–3 days

**Goal:** Efficient streaming with quality adaptation.

1. Add client→server quality negotiation over WebSocket
2. Implement resolution tiers (1080p/720p/480p) with GPU-accelerated resize via `cv2.UMat`
3. Add server-side backpressure detection (if send buffer grows, auto-downgrade)
4. Profile CV pipeline with OpenCL acceleration on RX 580
5. Add per-camera stream toggle (pause/resume individual feeds to save bandwidth)
6. Add recording capability (save annotated MJPEG stream to file)

**Deliverable:** Adaptive quality streaming, GPU-accelerated processing pipeline.

---

## Appendix: Quick-Start Patch for Phase 1

Literally just change the camera `<img>` tags in `index.html`:

```javascript
// Replace whatever snapshot polling exists with:
function initCameraStreams() {
  const camPort = 8081;
  const host = window.location.hostname;
  
  document.getElementById('cam-overhead').src = `http://${host}:${camPort}/cam/0`;
  document.getElementById('cam-arm').src = `http://${host}:${camPort}/cam/1`;
  document.getElementById('cam-side').src = `http://${host}:${camPort}/cam/2`;
}

// Add error handling for stream disconnection
document.querySelectorAll('.camera-feed img').forEach(img => {
  img.onerror = () => {
    setTimeout(() => { img.src = img.src.split('?')[0] + '?retry=' + Date.now(); }, 2000);
  };
});
```

That's it for Phase 1. The server already supports it.
