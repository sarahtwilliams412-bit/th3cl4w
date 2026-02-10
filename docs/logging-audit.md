# Logging Audit Report — th3cl4w

**Date:** 2026-02-09  
**Auditor:** Automated audit

---

## 1. Summary Statistics

| Metric | Count |
|--------|-------|
| Total non-trivial Python files (>20 lines, excl. tests) | ~100 |
| Files **without** logging (>20 lines) | **34** |
| Files with `print()` statements | **19** |
| Total `print()` calls that should be logger calls | **~293** |
| Servers with logging | 7/7 ✅ |
| Centralized logging config | ❌ **None** — 16+ separate `basicConfig()` calls |

---

## 2. Centralized Logging Config — MISSING

There is **no shared logging configuration module**. Instead, 16+ files each call `logging.basicConfig()` independently. Most use the format:
```
"%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```
But some use shorter formats or no format at all. This is inconsistent and fragile — only the first `basicConfig()` call takes effect in a process.

**Recommendation:** Create `src/utils/logging_config.py` with a `setup_logging()` function. All entry points (servers, scripts, tools) should call it once.

---

## 3. Per-File Assessment

### 3.1 Web Servers (Critical — need best logging)

#### `web/server.py` — Main Server (4711 lines, 112 endpoints)
- ✅ Has logging import and `getLogger(__name__)`
- ✅ Has `basicConfig` with good format
- ⚠️ **Only 28 logger calls for 112 endpoints** — most endpoints have NO entry/exit logging
- ⚠️ Many exception handlers lack `logger.exception()` or `logger.error()`
- **Recommendation:** Add INFO-level request logging to all endpoints; ensure every `except` block logs

#### `web/camera_server.py` — Camera Server (485 lines, HTTP handler)
- ✅ Has logging, 17 logger calls
- ✅ Good coverage for camera operations
- ⚠️ Uses raw `BaseHTTPRequestHandler` — no middleware logging
- **Recommendation:** Add request logging in `do_GET`

#### `web/location_server.py` — Location Server (203 lines, 6 endpoints)
- ✅ Has logging, 6 logger calls
- ✅ Decent ratio for endpoint count
- ⚠️ Could use error logging in exception handlers

#### `web/map_server.py` — Map Server (428 lines, 19 endpoints)
- ✅ Has logging
- ⚠️ **Only 4 logger calls for 19 endpoints** — very sparse
- **Recommendation:** Add entry logging and error logging to all endpoints

#### `web/ascii_server.py` — ASCII Server (308 lines, 9 endpoints)
- ✅ Has logging, 7 logger calls
- ⚠️ Needs entry logging for endpoints

#### `web/v2_server.py` — V2 Server (210 lines)
- ✅ Has logging
- ⚠️ Only 2 logger calls — very sparse

#### `web/command_smoother.py` (429 lines)
- ✅ Has logging, 19 logger calls — **best coverage in web/**

### 3.2 Source Modules (`src/`) — Files WITHOUT Logging

These files have **no logging at all** and need it added:

| File | Lines | Issue |
|------|-------|-------|
| `src/ascii/session.py` | 115 | No logging — manages ASCII sessions |
| `src/calibration/results_reporter.py` | 438 | No logging — large module! |
| `src/location/reachability.py` | 50 | No logging — small, lower priority |
| `src/map/scene.py` | 223 | No logging — scene management |
| `src/safety/limits.py` | 83 | No logging — safety-critical! |
| `src/telemetry/camera_monitor.py` | 98 | No logging — monitoring module |
| `src/telemetry/query.py` | 363 | No logging — large query module |
| `src/vision/ascii_converter.py` | 163 | No logging |
| `src/vision/fk_engine.py` | 178 | No logging |
| `src/vla/prompts.py` | 100 | No logging — data-only, acceptable |

### 3.3 Source Modules — Files WITH Logging (Good ✅)

The majority of `src/` modules have proper logging with `getLogger(__name__)`. These include all vision, planning, control, interface, introspection, and most map modules. Coverage varies but foundation is solid.

### 3.4 Files with `print()` That Should Use Logging

**In modules with existing loggers (mixed print+logger — worst pattern):**

| File | print() count | Notes |
|------|--------------|-------|
| `src/control/visual_servo.py` | 3 | Has logger, uses print too |
| `src/calibration/extrinsics_solver.py` | 7 | Has logger, uses print too |
| `calibration/capture_frames.py` | 17 | Has logger, uses print too |
| `calibration/compute_transforms.py` | 17 | Has logger, uses print too |
| `calibration/validate.py` | 11 | Has logger, uses print too |
| `scripts/run_comparison.py` | 7 | Has logger, uses print too |
| `tools/diagnose_arm.py` | 32 | Has logger, uses print too |

**In modules with NO logger (all output is print):**

| File | print() count | Notes |
|------|--------------|-------|
| `scripts/calibrate_intrinsics.py` | 21 | CLI script — print acceptable but logger preferred |
| `tools/calibrate_cameras.py` | 19 | CLI tool |
| `tools/dds_discover.py` | 41 | CLI tool — heaviest print user |
| `tools/dds_spdp_sniffer.py` | 31 | CLI tool |
| `tools/joint_calibration.py` | 23 | CLI tool |
| `tools/query_telemetry.py` | 25 | CLI tool |
| `tools/ascii_to_3d/cli.py` | 15 | CLI tool |
| `src/introspection/__init__.py` | 2 | Should be logger |

**Note:** For CLI tools (`tools/`, `scripts/`), `print()` for user-facing output is somewhat acceptable, but they should still use logging for operational messages and errors.

### 3.5 Self-Filter & Visual Hull Modules

| File | Has Logging | Notes |
|------|-------------|-------|
| `self_filter/arm_voxelizer.py` (161L) | ❌ | Needs logging |
| `self_filter/forward_kinematics.py` (125L) | ❌ | Needs logging |
| `self_filter/obstacle_extractor.py` (88L) | ❌ | Needs logging |
| `self_filter/pipeline.py` (249L) | ✅ | Good |
| `self_filter/d1_state_reader.py` (142L) | ✅ | Good |
| `visual_hull/hull_reconstructor.py` (150L) | ❌ | Needs logging |
| `visual_hull/temporal_filter.py` (90L) | ❌ | Needs logging |
| `visual_hull/pipeline.py` (208L) | ✅ | Good |

### 3.6 Frame Sync Modules
- ✅ All have proper logging (`pair_matcher.py`, `publisher.py`, `ws_server.py`)

---

## 4. Specific Recommendations

### 4.1 Create Centralized Logging Config
Create `src/utils/logging_config.py`:
```python
import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
```

### 4.2 Add Logging to All Files Without It
For each file listed in §3.2 and §3.5, add:
```python
import logging
logger = logging.getLogger(__name__)
```
Then add appropriate log calls at key points.

### 4.3 Replace print() with logger calls
- In `src/` modules: Replace ALL `print()` with `logger.info()` or `logger.debug()`
- In `tools/` and `scripts/`: Replace error/warning prints with `logger.error()`/`logger.warning()`, keep user-facing output as `print()` but add logger for operational messages
- In `calibration/`: Replace all `print()` with logger calls

### 4.4 Improve Server Endpoint Logging
- `web/server.py`: Add `logger.info("endpoint_name called")` to all 112 endpoints (or add middleware)
- `web/map_server.py`: Add logging to all 19 endpoints
- Consider adding a FastAPI middleware for automatic request/response logging

### 4.5 Ensure Exception Handlers Log
Search for bare `except` blocks and add `logger.exception()` calls.

---

## 5. Priority Order

1. **HIGH:** Create `src/utils/logging_config.py` — foundation for everything else
2. **HIGH:** Add logging to `src/safety/limits.py` — safety-critical code MUST log
3. **HIGH:** Fix mixed print+logger files in `src/` (visual_servo, extrinsics_solver, introspection/__init__)
4. **MEDIUM:** Add logging to remaining `src/` files without it (results_reporter, scene, query, etc.)
5. **MEDIUM:** Improve server endpoint logging coverage
6. **LOW:** Add logging to `self_filter/` and `visual_hull/` modules without it
7. **LOW:** Convert `tools/` and `scripts/` print statements to logging
