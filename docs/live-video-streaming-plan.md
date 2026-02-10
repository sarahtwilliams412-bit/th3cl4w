# Live Video Streaming Plan — th3cl4w D1

> Replacing JPEG snapshot polling with real-time video streaming for 3 camera feeds.

## Current State

The camera server (`web/camera_server.py`) already has two transport modes:
- **MJPEG streams** at `/cam/{0,1,2}` — `multipart/x-mixed-replace` with `--frame` boundaries
- **JPEG snapshots** at `/snap/{0,1,2}` — single frame per request

The web UI (`web/static/index.html`) currently **polls snapshots** via `setInterval` + `<img>` src swapping. The MJPEG endpoints exist but aren't used by the UI.

Camera threads capture at 15 fps, encode JPEG at quality 92, serve from a shared `_frame` buffer protected by a threading lock.

---

## 1. Streaming Protocol Selection

### Comparison

| Protocol | Latency | Complexity | Browser Support | Overlay Support | CPU Cost |
|----------|---------|------------|-----------------|-----------------|----------|
| **MJPEG stream** | ~66ms (1 frame) | Very low | Native `<img>` tag | Server-side only | Low |
| **WebSocket binary** | ~33-66ms | Medium | Universal | Client or server | Medium |
| **WebRTC** | ~30-100ms | High | Universal (w/ STUN) | Client-side | High (VP8/H264 encode) |
| **HLS/DASH** | 2-10s | Medium | Universal | Segment-based | Medium |

### Recommendation: Hybrid MJPEG + WebSocket

**Phase 1: MJPEG** (already implemented server-side, just wire up the UI)
- Zero additional server work — endpoints already exist
- Set `<img src