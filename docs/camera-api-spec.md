# Camera Service API Specification

> **Service:** `camera_server.py` on `:8081`
> **Protocol:** HTTP/1.1
> **Single source of truth** for all camera hardware access.

## Physical Cameras

| ID | Device | Name | Role | Mount | Resolution | FOV | Description |
|----|--------|------|------|-------|------------|-----|-------------|
| 0 | `/dev/video0` | Overhead | `overhead` | Fixed | 1920×1080 | 78° | Logitech BRIO mounted above workspace, looking straight down. Primary camera for object detection X/Y positioning. |
| 1 | `/dev/video4` | Arm-mounted | `end_effector` | Arm | 1920×1080 | 78° | Camera attached to end-effector. Moves with arm. Used for close-up inspection and visual servo. |
| 2 | `/dev/video6` | Side | `side_profile` | Fixed | 1920×1080 | 78° | Fixed side-view camera. Used for height (Z) estimation of objects on workspace. |

## Endpoints

### `GET /cameras`

Camera registry — the canonical list of cameras and their properties.

**Response:** `application/json`

```json
{
  "0": {
    "id": 0,
    "device": "/dev/video0",
    "name": "Overhead",
    "role": "overhead",
    "mount": "fixed",
    "resolution": [1920, 1080],
    "fov_deg": 78,
    "description": "Logitech BRIO mounted above workspace, looking straight down. Primary camera for object detection X/Y positioning."
  },
  "1": { ... },
  "2": { ... }
}
```

### `GET /snap/{cam_id}`

JPEG snapshot — triggers a fresh frame read from the camera thread's buffer.

- **Response:** `image/jpeg`
- **Headers:** `Access-Control-Allow-Origin: *`, `Cache-Control: no-cache`
- **404** if `cam_id` not found.

### `GET /latest/{cam_id}`

Latest cached frame — returns the most recent buffered JPEG **without** triggering a new capture. Designed for high-frequency consumers that want to avoid hammering the camera.

- **Response:** `image/jpeg`
- **Headers:**
  - `X-Frame-Age-Ms: <milliseconds since frame was captured>`
  - `Access-Control-Allow-Origin: *`
  - `Cache-Control: no-cache`
- **404** if `cam_id` not found.
- **503** if no frame is available yet.

### `GET /cam/{cam_id}`

MJPEG stream — continuous multipart stream of JPEG frames.

- **Response:** `multipart/x-mixed-replace; boundary=frame`
- Connection stays open until client disconnects.

### `GET /status`

Camera health — per-camera connection status, resolution, FPS, and health metrics.

**Response:** `application/json`

```json
{
  "0": {
    "connected": true,
    "device_id": 0,
    "width": 1920,
    "height": 1080,
    "fps": 15,
    "health": { ... }
  },
  ...
}
```

### `GET /frame/{cam_id}?format=jpeg|raw`

Single frame with format option. `jpeg` (default) returns standard JPEG. `raw` returns raw numpy bytes (BGR, uint8) with metadata headers for CV consumers.

- **`format=jpeg`**: `image/jpeg` (same as `/snap/`)
- **`format=raw`**: `application/octet-stream`
  - `X-Frame-Width: 1920`
  - `X-Frame-Height: 1080`
  - `X-Frame-Channels: 3`
  - `X-Frame-Dtype: uint8`

### `GET /world`

World model snapshot from startup scanner (existing).

### `GET /scan`

Startup scan report (existing).

## Consumer Guidelines

### Main server (`:8080`)

Use the `cam_snap()` helper for all camera access. Proxy routes:
- `GET /snap/{cam_id}` → `:8081/snap/{cam_id}`
- `GET /cameras` → `:8081/cameras`
- `GET /latest/{cam_id}` → `:8081/latest/{cam_id}`

### Location server tracker

Use `/latest/{cam_id}` instead of `/snap/{cam_id}` for polling loops. Rate-limit frame grabs to **max 1 per camera per 2 seconds** to avoid cascading Gemini calls and 429s.

### Frontend

Access cameras through the main server proxy (`:8080/snap/{cam_id}`) or directly via `:8081` for MJPEG streams.
