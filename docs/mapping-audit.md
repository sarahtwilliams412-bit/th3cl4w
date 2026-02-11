# Mapping Section Audit — th3cl4w

**Date:** 2026-02-10  
**Auditor:** Subagent QA  
**Scope:** Workspace Calibration (#bifocal), Object Detection (#object-detect), 3D Scan (#scan3d)

---

## Summary

The Mapping section is **surprisingly well-wired** — most buttons trace through to real endpoints with real logic. The main issues are: (1) a camera label mismatch in the HTML, (2) the "Sync to 3D" feature depends on an external map server on port 8083 that doesn't appear to be part of this codebase, (3) the 3D Inspect feature depends on CDN-loaded Three.js which will fail offline, and (4) the scan manager has a naive 1.5s sleep for arm movement completion.

**Critical issues: 0**  
**High issues: 2**  
**Medium issues: 5**  
**Low issues: 5**

---

## Panel 1: Workspace Calibration (#bifocal)

### Buttons & Controls Traced

| Control | JS Function | API Endpoint | Status |
|---------|------------|--------------|--------|
| CALIBRATE | `wsCalibScale()` | `POST /api/bifocal/check-checkerboard` → `POST /api/bifocal/calibrate-scale` (+ optional `POST /api/bifocal/diagnose-checkerboard`) | ✅ Works |
| PREVIEW | `wsCalibPreview()` | None (refreshes snap images client-side) | ✅ Works |
| CHECK ALL CAMERAS | `wsCalibCheckAll()` | `POST /api/bifocal/check-checkerboard` | ✅ Works |
| Snap buttons (per camera) | inline `window.open()` | Camera server `http://host:8081/snap/{id}` | ✅ Works |
| Square size input | Read by `wsCalibScale()` | Passed to calibrate-scale | ✅ Works |

### Issues

```
ISSUE: Workspace Calibration [LOW]
Button/Feature: Camera labels in HTML
Problem: HTML says "Camera 0 — Side", "Camera 1 — Arm", "Camera 2 — Overhead"
  which matches camera_config.py (CAM_SIDE=0, CAM_ARM=1, CAM_OVERHEAD=2). ✅ Correct.
  BUT the inline JS in wsCalibShowPreview() has a mapping: {0:'Side',1:'Arm',2:'Overhead'}
  which is consistent. No issue here actually.
Root cause: N/A
Fix: N/A
```

```
ISSUE: Workspace Calibration [LOW]
Button/Feature: Camera preview images on panel load
Problem: Camera images (wsCalibImg0/1/2) have no initial `src` — they're blank until 
  PREVIEW is clicked or MJPEG streams start. The panel shows 3 empty black boxes on first visit.
Root cause: wsCalibStartStreams() sets MJPEG URLs but is only called... let me check.
  Actually, the panel activation code (`showPanel`) doesn't auto-start streams for bifocal.
Fix: Either auto-start MJPEG streams when the bifocal panel becomes active, or load a single
  snapshot for each camera on panel show. Add to showPanel():
  `if (panelId === 'bifocal') { wsCalibStartStreams(); }`
```

```
ISSUE: Workspace Calibration [MEDIUM]
Button/Feature: wsCalibStartStreams / wsCalibStopStreams
Problem: These functions exist and set MJPEG URLs, but nothing calls wsCalibStartStreams()
  when the panel is shown, and nothing calls wsCalibStopStreams() when leaving.
  MJPEG streams consume bandwidth continuously once started.
Root cause: Missing panel lifecycle management.
Fix: In showPanel(), add bifocal panel enter/leave hooks to start/stop streams.
```

```
ISSUE: Workspace Calibration [LOW]
Button/Feature: Gemini diagnosis on checkerboard failure
Problem: Works correctly — calls /api/bifocal/diagnose-checkerboard, handles missing API key
  gracefully (server returns 400, JS catches and ignores). Good error handling.
Root cause: N/A — this is actually well implemented.
Fix: N/A
```

```
ISSUE: Workspace Calibration [MEDIUM]
Button/Feature: CALIBRATE button — checkerboard size hardcoded to 7×7
Problem: The backend uses cv2.findChessboardCorners with hardcoded (7,7) inner corners.
  If the user has a different size checkerboard, detection will fail silently.
  The UI only lets you set square_size_mm, not the grid dimensions.
Root cause: Hardcoded in bifocal_check_checkerboard() and likely in workspace_mapper.
Fix: Add grid_cols/grid_rows inputs to the UI and pass them through to the API.
  Or at minimum, document that a 7×7 inner-corner (8×8 square) checkerboard is required.
```

---

## Panel 2: Object Detection (#object-detect)

### Buttons & Controls Traced

| Control | JS Function | API Endpoint | Status |
|---------|------------|--------------|--------|
| ENABLE/DISABLE | `objPanelToggle()` | `POST /api/objects/toggle` | ✅ Works |
| DETECT NOW | `objPanelRunDetect()` | `POST /api/objects/detect/snapshot` + `GET /api/objects/list` | ✅ Works |
| CLEAR | `objPanelClear()` | `POST /api/objects/clear` | ✅ Works |
| 🔍 SCAN TABLE | `scanTable(btn)` | `POST /api/objects/scan-table` | ✅ Works |
| Sync to 3D | `objPanelSyncTo3D()` | `GET /api/objects/list` + `POST http://host:8083/api/map/objects` | ⚠️ Partial |
| Toggle 3D View | `toggleEmbeddedSim()` | Loads iframe from `http://host:8083/static/map3d.html` | ⚠️ Partial |
| Open Simulator ↗ | inline | `window.open('#simulator','_blank')` | ⚠️ Broken |
| ✓ Confirm / ✗ Reject / ⚠ False+ | `_objReview(idx, status)` | `POST /api/objects/{obj_id}/review` | ✅ Works |
| 🔍 3D Inspect | `openObjInspect3D(obj)` | None (client-side Three.js) | ✅ Works (if online) |
| 📷 Side View | `objPanelEnrich(id, label)` | `POST /api/objects/{obj_id}/enrich` | ✅ Works |
| Detection overlay canvas | `_objPanelDrawOverlay()` | N/A (client-side drawing) | ✅ Works |
| Workspace map canvas | `_objPanelDrawWorkspace()` | N/A (client-side drawing) | ✅ Works |

### Issues

```
ISSUE: Object Detection [HIGH]
Button/Feature: "Sync to 3D" button
Problem: Posts objects to http://${location.hostname}:8083/api/map/objects — a separate
  map server that is NOT part of the main th3cl4w web server (port 8080). There's no
  /api/map/objects endpoint in server.py. If port 8083 isn't running, this fails silently
  (catch block just shows error in status span).
Root cause: The map3d server is a separate service (likely served by a different process).
  No evidence it exists in this codebase — grep for 8083 returns nothing in server.py.
Fix: Either:
  1. Document that the map3d server must be running separately, or
  2. Add the /api/map/objects endpoint to server.py, or
  3. Gracefully handle the failure with a clear message like "Map 3D server not running"
```

```
ISSUE: Object Detection [HIGH]
Button/Feature: "Toggle 3D View" embedded iframe
Problem: Loads iframe from http://${location.hostname}:8083/static/map3d.html — same
  external map server dependency. If not running, shows a blank/error iframe.
Root cause: Same as above — depends on external map3d server.
Fix: Show a meaningful error when the iframe fails to load. Add an onerror/onload check.
```

```
ISSUE: Object Detection [MEDIUM]
Button/Feature: "Open Simulator ↗" button
Problem: Opens window.open('#simulator','_blank') — this opens the current page with
  a URL fragment, not the simulator panel. It should either navigate to the simulator
  panel within the app, or open a standalone simulator URL.
Root cause: '#simulator' is a fragment that doesn't correspond to any route.
  The simulator panel is shown via showPanel() with data-panel="simulator".
Fix: Change to either:
  - `showPanel(document.querySelector('[data-panel=simulator]'))` (navigate within app)
  - or remove the button (redundant with sidebar navigation)
```

```
ISSUE: Object Detection [MEDIUM]
Button/Feature: 3D Inspect (Three.js)
Problem: Three.js is loaded from CDN (cdnjs.cloudflare.com). Works online, but this
  is a robotics control app that may run on a local network without internet.
  If CDN is unreachable, openObjInspect3D() will crash with "THREE is not defined".
Root cause: External CDN dependency for a feature that should work offline.
Fix: Bundle three.min.js and OrbitControls.js locally in web/static/lib/.
```

```
ISSUE: Object Detection [LOW]
Button/Feature: Camera detection overlay annotation
Problem: The HTML comment says "Overhead camera (cam1) with bounding boxes..." but
  the overhead camera is actually cam2 (CAM_OVERHEAD=2). This is just a misleading comment
  in the HTML, not a code bug.
Root cause: Stale/incorrect comment.
Fix: Change the HTML text from "cam1" to "cam2" or just "overhead camera".
```

```
ISSUE: Object Detection [LOW]
Button/Feature: Detection state persistence
Problem: _objPanelEnabled is a JS variable that resets on page reload. The backend
  object_detector.enabled state persists in-memory but the UI won't reflect it on reload.
  However, the panel does NOT auto-fetch status on load.
Root cause: No initialization fetch on panel show.
Fix: When showing the object-detect panel, fetch /api/objects/status and sync _objPanelEnabled.
  Currently the code in showPanel has: `if (panelId === 'object-detect') _objPanelDrawWorkspace(_objPanelObjects);`
  Add: fetch /api/objects/status to sync state.
```

---

## Panel 3: 3D Scan (#scan3d)

### Buttons & Controls Traced

| Control | JS Function | API Endpoint | Status |
|---------|------------|--------------|--------|
| ▶ START SCAN | `scan3dStart(btn)` | `POST /api/scan/start` | ✅ Works (if scan_manager imports) |
| ⏹ STOP | `scan3dStop(btn)` | `POST /api/scan/stop` | ✅ Works |
| ↻ STATUS | `scan3dRefresh()` | `GET /api/scan/status` | ✅ Works |
| 📋 LIST SCANS | `scan3dList(btn)` | `GET /api/scan/list` | ✅ Works |

### Issues

```
ISSUE: 3D Scan [MEDIUM]
Button/Feature: START SCAN — arm movement
Problem: The scan manager's move_arm() function uses a hardcoded `await asyncio.sleep(1.5)`
  to wait for arm movement completion. This is a race condition — if the arm takes longer
  than 1.5s to reach the target position (likely for large moves), the scan will capture
  frames at wrong positions. If it takes less, it wastes time.
Root cause: No feedback-based motion completion detection in _get_scan_manager().
Fix: Replace the sleep with a feedback loop that checks if joint angles are within
  tolerance of the target, with a timeout. E.g.:
  ```python
  for _ in range(30):  # 3s timeout
      state = get_arm_state()
      if all(abs(state['joints'][i] - angles[i]) < 2.0 for i in range(6)):
          break
      await asyncio.sleep(0.1)
  ```
```

```
ISSUE: 3D Scan [LOW]
Button/Feature: No progress feedback during scan
Problem: The scan panel shows IDLE/running status but doesn't show progress (e.g.,
  "viewpoint 3/12" or a progress bar). The user has no idea how long the scan will take.
Root cause: The UI only calls scan3dRefresh() once after starting. No polling loop.
Fix: Add a polling interval (setInterval) that calls scan3dRefresh() every 2s while
  the scan is running. Stop polling when status shows not running.
```

---

## Architecture Review

### What Works Well
1. **Endpoint wiring is solid** — every button traces to a real API endpoint with proper error handling
2. **Object detection pipeline** is well-thought-out: overhead camera → detection → annotation → review → enrichment via side camera + Gemini
3. **Graceful degradation** — all modules use `_HAS_X` flags so missing dependencies don't crash the server
4. **Camera abstraction** — centralized camera_config.py with consistent CAM_SIDE/CAM_ARM/CAM_OVERHEAD IDs

### What's Missing or Broken

1. **Map 3D server (port 8083)** — The object detection panel has deep integration with a separate 3D map server that appears to not exist in this codebase. Either it's in a different repo, or it was planned but never built. The "Sync to 3D" and "Toggle 3D View" features are dead without it.

2. **Panel lifecycle management** — No enter/leave hooks for panels. Camera streams start but never stop. Object detection state doesn't sync on panel open. This wastes bandwidth and causes stale UI.

3. **Three.js CDN dependency** — The 3D object inspection feature loads Three.js from CDN. For a robotics app that should work on an isolated LAN, this should be bundled locally.

4. **Scan manager motion control** — The hardcoded 1.5s sleep is a hack. Should use feedback-based motion completion detection.

5. **No persistent detection state** — Objects are stored in-memory only. A page reload or server restart loses all detections. For a "mapping" feature, this should persist to disk (JSON or SQLite).

### Recommended Architecture Improvements

1. **Bundle the map3d server** into the main server, or clearly document it as a separate required service with setup instructions.

2. **Add panel lifecycle hooks** to `showPanel()`:
   - `onEnter`: start streams, fetch state, begin polling
   - `onLeave`: stop streams, cancel polling

3. **Persist detections to disk** — save to `data/detections.json` on every scan, auto-load on startup.

4. **Replace scan manager sleep** with feedback-based motion completion.

5. **Bundle all CDN dependencies** locally for offline operation.
