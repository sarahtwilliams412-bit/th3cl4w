# Map Server — Engineering Plan

**Service:** `map_server.py` on port **8083**  
**Purpose:** Live 3D spatial map of the arm and its environment — a real-time digital twin of the entire workspace geometry.

> **Relationship to Location Server (8082):** The location server answers *"what objects are where"* (semantic labels, positions, tracking). The map server answers *"what does the 3D space look like"* (geometry, point clouds, collision volumes, arm skeleton). They complement each other: the location server feeds object positions INTO the map server, which renders them in the 3D scene alongside the environment geometry.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB BROWSER                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Three.js Map Viewer (map3d.html)                             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────────┐  │  │
│  │  │ Arm      │ │ Point    │ │ Objects │ │ Reach Envelope   │  │  │
│  │  │ Skeleton │ │ Cloud    │ │ (boxes) │ │ (wireframe)      │  │  │
│  │  └──────────┘ └──────────┘ └─────────┘ └──────────────────┘  │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             │ WebSocket (ws://host:8083/ws/map)     │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────────┐
│  MAP SERVER (FastAPI :8083) │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              WebSocket Hub (ws_hub.py)                       │   │
│  │  Broadcasts map state at 10-15 Hz to connected viewers      │   │
│  └──────────┬────────────┬────────────────┬────────────────────┘   │
│             │            │                │                         │
│  ┌──────────▼──┐  ┌──────▼───────┐  ┌────▼──────────────┐         │
│  │ Arm Model   │  │ Environment  │  │ Collision Volume  │         │
│  │ Manager     │  │ Map Manager  │  │ Manager           │         │
│  │             │  │              │  │                    │         │
│  │ FK skeleton │  │ Point cloud  │  │ AABB tree from    │         │
│  │ from DDS    │  │ + voxel grid │  │ env map + objects  │         │
│  │ joint feed  │  │ from depth   │  │ for motion planner │         │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘         │
│         │                │                    │                     │
│  ┌──────┴────────────────┴────────────────────┴─────────────┐      │
│  │                    Scene Graph (scene.py)                 │      │
│  │  Unified in-memory 3D scene:                              │      │
│  │  - Arm links + joints (updated from FK)                   │      │
│  │  - Environment point cloud / voxel grid                   │      │
│  │  - Object bounding boxes (from location server)           │      │
│  │  - Waypoints, trajectories                                │      │
│  │  - Reach envelope mesh                                    │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Data Ingest Layer                               │   │
│  │                                                              │   │
│  │  ┌────────────┐  ┌────────────────┐  ┌──────────────────┐   │   │
│  │  │ DDS Joint  │  │ Camera Depth   │  │ Location Server  │   │   │
│  │  │ Subscriber │  │ Poller         │  │ Poller           │   │   │
│  │  │ (arm state)│  │ (depth frames) │  │ (object poses)   │   │   │
│  │  │ ~50Hz      │  │ ~2-5Hz         │  │ ~5Hz             │   │   │
│  │  └─────┬──────┘  └───────┬────────┘  └────────┬─────────┘   │   │
│  │        │                 │                     │              │   │
│  │        ▼                 ▼                     ▼              │   │
│  │  main server     camera server :8081   location server :8082 │   │
│  │  :8080 /api/state   /snap/{id}         /api/objects          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### `src/map/` — Core modules

| Module | Responsibility | Reuses |
|--------|---------------|--------|
| `scene.py` | Central scene graph — holds arm model, point cloud, objects, waypoints. Thread-safe. Serializes to JSON for WebSocket broadcast. | Replaces `digital_twin.py`'s snapshot system |
| `arm_model.py` | FK-based arm skeleton. Subscribes to joint state, computes link positions via `D1Kinematics.get_joint_positions_3d()`. Generates link cylinders + joint spheres for Three.js. | Reuses `kinematics.py` directly, replaces `digital_twin.py`'s `ArmState` |
| `env_map.py` | Maintains the environment point cloud and voxel grid. Ingests depth frames, back-projects to 3D, merges incrementally. Supports both continuous camera updates and arm-sweep scans. | Wraps `pointcloud_generator.py` (backproject, merge, downsample). Wraps `depth_estimator.py`. Replaces `scan_manager.py`'s scan loop. Replaces `workspace_mapper.py`'s 2D grid with a proper 3D voxel grid. |
| `collision_map.py` | Builds collision volumes (AABBs / voxel occupancy) from env_map + object bounding boxes. Exposes `check_point(xyz)`, `check_sphere(xyz, r)`, `check_path(points, r)` for motion planner queries. | Replaces `workspace_mapper.py`'s `check_path_2d()` with full 3D |
| `ingest.py` | Async data ingest: polls main server for joint state, camera server for depth frames, location server for objects. Feeds into scene graph. | New |
| `ws_hub.py` | WebSocket connection manager. Broadcasts scene snapshots at configurable rate. Handles client subscriptions (full scene vs delta updates). | New |

### `web/map_server.py` — FastAPI application

The main entry point. Imports all modules above, wires up ingest → scene → ws_hub pipeline.

### `web/static/map3d.html` — Three.js viewer

New page (evolves from `scan3d.html`). Full 3D scene with arm + environment + objects.

---

## Data Flow

```
DDS Joint Feedback ──► Main Server :8080 /api/state
                              │
                              │ poll every 20-50ms
                              ▼
                       ┌─────────────┐
                       │  ingest.py  │
                       │  (arm poll) │
                       └──────┬──────┘
                              │ joint_angles_deg + gripper_mm
                              ▼
                       ┌─────────────┐         ┌──────────────────┐
                       │ arm_model   │────────►│                  │
                       │ .update()   │         │   scene.py       │
                       └─────────────┘         │   (Scene Graph)  │
                                               │                  │
Camera :8081 /snap/0 ──► ingest.py ──►         │  .arm_skeleton   │
  (depth frames)         env_map.py ──────────►│  .point_cloud    │
                                               │  .voxel_grid     │
Location :8082 ──────► ingest.py ──►           │  .objects[]      │
  /api/objects           scene.py ────────────►│  .collision_map  │
                                               │  .waypoints[]    │
                                               │  .reach_envelope │
                                               └────────┬─────────┘
                                                        │
                                                        │ snapshot() @ 10-15Hz
                                                        ▼
                                                ┌──────────────┐
                                                │  ws_hub.py   │
                                                │  broadcast   │──► WebSocket clients
                                                └──────────────┘

Motion Planner ◄──── collision_map.py (query API)
```

---

## API Endpoints

### REST

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/map/status` | Map server health: ingest rates, scene stats, connected clients |
| `GET` | `/api/map/scene` | Full scene snapshot as JSON (one-shot, for debugging) |
| `GET` | `/api/map/arm` | Current arm skeleton (joint positions, link geometries) |
| `GET` | `/api/map/pointcloud` | Latest merged point cloud as binary PLY or JSON summary |
| `GET` | `/api/map/pointcloud/stats` | Point count, bounds, last update time, voxel resolution |
| `GET` | `/api/map/voxels` | Occupied voxel centers as JSON array (for collision viz) |
| `GET` | `/api/map/collision/check` | Query: `?x=&y=&z=&radius=` → free/occupied/unknown |
| `POST` | `/api/map/collision/check-path` | Body: `{points: [[x,y,z],...], radius: mm}` → collision list |
| `GET` | `/api/map/objects` | Objects from location server currently in scene |
| `POST` | `/api/map/scan/start` | Begin arm-sweep 3D scan (moves arm through viewpoints) |
| `POST` | `/api/map/scan/stop` | Abort scan |
| `GET` | `/api/map/scan/status` | Scan progress |
| `GET` | `/api/map/scan/list` | Previous scans |
| `GET` | `/api/map/scan/result` | Download PLY: `?scan_id=` |
| `POST` | `/api/map/env/clear` | Clear environment point cloud |
| `POST` | `/api/map/env/config` | Set voxel size, depth range, update rate |
| `GET` | `/api/map/reach-envelope` | Pre-computed reach envelope mesh (vertices + faces) |

### WebSocket

| Path | Description |
|------|-------------|
| `ws://host:8083/ws/map` | Scene updates at 10-15Hz. Messages: `{type: "scene_update", data: {...}}` |

#### WebSocket Message Types (server → client)

```json
{
  "type": "scene_update",
  "data": {
    "arm": {
      "joints": [[x,y,z], ...],       // 8 positions (base + 7 joints)
      "links": [{                       // cylinder segments
        "start": [x,y,z],
        "end": [x,y,z],
        "radius": 0.02
      }, ...],
      "gripper_mm": 30.0,
      "ee_pose": [[4x4 matrix]]
    },
    "env": {
      "update_mode": "delta",           // "full" on first connect, "delta" after
      "new_points": [[x,y,z,r,g,b]...],// only new points since last frame
      "removed_voxels": [...],          // if any voxels were cleared
      "stats": {
        "total_points": 50000,
        "voxel_count": 12000,
        "bounds": {"min": [-0.5,-0.5,0], "max": [0.5,0.5,0.5]}
      }
    },
    "objects": [{
      "id": "cup_01",
      "label": "red cup",
      "position_mm": [200, 100, 50],
      "bbox_mm": [80, 80, 120],         // w, d, h
      "confidence": 0.92,
      "reachable": true
    }],
    "collision_voxels": [...],          // occupied voxels (sparse, only on change)
    "waypoints": [...],
    "timestamp": 1707500000.0,
    "frame": 1234
  }
}
```

#### WebSocket Message Types (client → server)

```json
{"type": "subscribe", "layers": ["arm", "env", "objects", "collision"]}
{"type": "set_update_rate", "hz": 10}
{"type": "request_full_scene"}
```

---

## Existing Code: Reuse vs Rewrite

| File | Decision | Rationale |
|------|----------|-----------|
| `kinematics.py` | **Reuse as-is** | Solid FK/IK/Jacobian. Import `D1Kinematics` directly. |
| `pointcloud_generator.py` | **Reuse as-is** | `backproject_depth()`, `merge_point_clouds()`, `voxel_downsample()`, `save_ply()` are all good. Import directly into `env_map.py`. |
| `depth_estimator.py` | **Reuse as-is** | `estimate_depth()` / `estimate_metric_depth()` work. Called by `env_map.py`. |
| `scan_manager.py` | **Wrap & extend** | Core scan loop is good. Wrap it as `MapScanManager` in `env_map.py` — add incremental cloud merging into the live scene instead of just saving to PLY. Keep the arm-command + capture + depth pipeline. |
| `digital_twin.py` | **Replace** | Its `ArmState`, `DigitalTwinSnapshot`, object tracking, and waypoint management all get absorbed into `scene.py` + `arm_model.py`. The new versions are more capable (real-time FK, proper scene graph, WebSocket streaming). Deprecate but keep importable for backward compat during transition. |
| `workspace_mapper.py` | **Replace** | 2D occupancy grid is superseded by 3D voxel grid in `env_map.py` + `collision_map.py`. The 2D grid was a stopgap. |
| `motion_planner.py` | **Integrate** | Motion planner stays in `src/planning/`. It gains a new method: `check_environment_collision(trajectory, collision_map)` that queries the map server's collision map via HTTP or direct import. |
| `scan3d.html` | **Evolve into `map3d.html`** | Keep the PLY loader, orbit controls, grid. Add arm skeleton rendering, object boxes, live WebSocket updates, layer toggles. |
| `location/reachability.py` | **Reuse** | `classify_reach()` used by map server to annotate objects and render reach envelope. |

---

## Three.js Visualization (`map3d.html`)

### Scene Graph Structure

```
Scene
├── AmbientLight
├── DirectionalLight
├── GridHelper (table surface)
├── AxesHelper (arm base)
│
├── Group: "arm"
│   ├── Mesh: base_cylinder (J0)
│   ├── Mesh: link1_cylinder (J0→J1)
│   ├── Mesh: link2_cylinder (J1→J2)
│   ├── ...
│   ├── Mesh: gripper_left
│   ├── Mesh: gripper_right
│   └── Mesh: ee_frame (small axes at end-effector)
│
├── Group: "environment"
│   ├── Points: env_pointcloud (BufferGeometry, vertex colors)
│   └── InstancedMesh: voxel_grid (optional, for collision viz)
│
├── Group: "objects"
│   ├── Mesh: object_0 (Box3 wireframe + label sprite)
│   ├── Mesh: object_1
│   └── ...
│
├── Group: "reach_envelope"
│   └── Mesh: envelope_wireframe (pre-computed hemisphere, α=0.1)
│
├── Group: "waypoints"
│   ├── Mesh: wp_0 (sphere + label)
│   └── ...
│
└── Group: "trajectory_preview"
    └── Line: trajectory_line (dotted)
```

### Update Strategy

1. **Arm** — Update joint/link positions every frame from WebSocket data. Use `CylinderGeometry` for links, `SphereGeometry` for joints. Fast: just update `position` and `quaternion` on existing meshes.

2. **Environment point cloud** — Use `BufferGeometry` with dynamic draw. On "full" update, replace entire buffer. On "delta" updates, append new points (grow buffer, or use ring buffer with max size ~200k points).

3. **Objects** — Create/update/remove box wireframes as objects appear/disappear from location server. Use `CSS2DRenderer` for floating labels.

4. **Collision voxels** — Optional overlay using `InstancedMesh` with translucent red cubes. Only shown when user enables "show collision map" toggle.

5. **Reach envelope** — Static wireframe hemisphere (radius = 550mm). Computed once, never changes.

### UI Controls

- Layer toggles: Arm | Point Cloud | Objects | Collision | Reach | Waypoints
- Update rate slider: 1-30 Hz
- Point size slider
- Camera presets: Top, Front, Side, Iso
- Scan controls: Start/Stop/Load (migrated from scan3d.html)
- Stats overlay: FPS, point count, object count, arm state

---

## Implementation Phases

### Phase 1: Skeleton + Arm Model (1-2 days)
- Create `src/map/` package with `__init__.py`
- `arm_model.py`: FK-based skeleton from joint angles. Unit test with known poses.
- `scene.py`: Minimal scene graph (arm only). `snapshot()` → JSON.
- `web/map_server.py`: FastAPI app on :8083. `/api/map/status`, `/api/map/arm`.
- `ingest.py`: Poll main server `/api/state` for joint angles.
- `/ws/map` WebSocket: broadcast arm skeleton at 10Hz.
- `map3d.html`: Render arm skeleton with orbit controls. Verify real-time update.

### Phase 2: Environment Mapping (2-3 days)
- `env_map.py`: Depth frame ingestion pipeline.
  - Poll camera server for frames.
  - Run depth estimation (reuse `depth_estimator.py`).
  - Back-project to 3D (reuse `pointcloud_generator.py`).
  - Incremental merge into voxel grid.
  - Configurable voxel size, max points, depth range.
- Wire into scene graph and WebSocket broadcast.
- `map3d.html`: Add point cloud rendering with vertex colors.
- Migrate scan endpoints from main server → map server (`/api/map/scan/*`).
- Wrap `ScanManager` for arm-sweep scans that feed directly into the live map.

### Phase 3: Collision Map (1-2 days)
- `collision_map.py`: 3D occupancy grid from voxelized point cloud.
  - `check_point(xyz)` → free/occupied/unknown
  - `check_sphere(xyz, radius)` → bool
  - `check_path(points, radius)` → collision list with distances
- REST endpoints: `/api/map/collision/check`, `/api/map/collision/check-path`.
- Integrate with `motion_planner.py`: add `check_environment_collision()` that queries collision map.
- `map3d.html`: Optional collision voxel overlay (red translucent cubes).

### Phase 4: Location Server Integration (1 day)
- `ingest.py`: Poll location server :8082 for object list.
- Merge object positions + bounding boxes into scene graph.
- Annotate objects with reachability (using `location/reachability.py`).
- `map3d.html`: Render object bounding boxes with labels.

### Phase 5: Web Visualization Polish (1-2 days)
- Reach envelope wireframe rendering.
- Waypoint display (spheres + labels + connecting lines).
- Trajectory preview (animated dotted line).
- Layer toggle UI panel.
- Camera preset buttons (Top/Front/Side/Iso).
- Stats overlay (FPS, points, objects, arm state).
- Delta updates for point cloud (avoid re-sending full cloud every frame).
- Performance tuning: cap point cloud at 200k points, LOD for distant views.

### Phase 6: Consolidation & Deprecation (1 day)
- Update main server :8080 scan endpoints to proxy to map server :8083.
- Add deprecation warnings to `digital_twin.py` and `workspace_mapper.py`.
- Update `scan3d.html` to redirect to `map3d.html`.
- Documentation: API docs, architecture diagram, startup instructions.
- Add map server to project's startup script / systemd / docker-compose.

---

## Configuration

```python
# map_server defaults (overridable via env vars or /api/map/env/config)
MAP_SERVER_PORT = 8083
MAIN_SERVER_URL = "http://localhost:8080"
CAMERA_SERVER_URL = "http://localhost:8081"
LOCATION_SERVER_URL = "http://localhost:8082"

ARM_POLL_HZ = 30          # Joint state polling rate
DEPTH_POLL_HZ = 3          # Depth frame capture rate
LOCATION_POLL_HZ = 5       # Object list polling rate
WS_BROADCAST_HZ = 15       # WebSocket broadcast rate

VOXEL_SIZE_M = 0.01         # 1cm voxels
MAX_POINTS = 200_000        # Max points in live cloud
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 2.0
CAMERA_ID = 0               # Default camera for depth
```

---

## Dependencies

Already in project:
- `fastapi`, `uvicorn`, `numpy`, `scipy`, `opencv-python`
- `transformers`, `torch` (for depth estimation)

May need:
- `httpx` (async HTTP client for polling — already used in scan_manager.py)

No new heavy dependencies required.

---

## File Tree (after implementation)

```
src/map/
├── __init__.py
├── arm_model.py          # FK-based arm skeleton
├── env_map.py            # Point cloud + voxel grid management
├── collision_map.py       # 3D collision checking
├── scene.py              # Unified scene graph
├── ingest.py             # Async data ingest from other servers
└── ws_hub.py             # WebSocket connection manager

web/
├── map_server.py          # FastAPI app on :8083
└── static/
    └── map3d.html         # Three.js full map viewer
```
