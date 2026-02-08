# Technical Follow-Ups — th3cl4w

*Created: 2026-02-08 | Priority: P0 (critical) → P3 (nice-to-have)*

## P0 — Must Do Next

### 1. Run 3D calibration on real hardware
- Execute `viz_calibrator.py` with live arm + cameras
- Target: <10px reprojection residual on both cameras
- Validate DH parameters match physical arm
- **Depends on:** arm powered, cameras streaming

### 2. Reduce collision detection false positives
- Current: ~70% false positive rate from stale joint feedback
- Fix: add feedback timestamp check, reject data >100ms old
- Consider: require 2+ consecutive detections before alerting
- **File:** `src/vision/collision_detector.py`

### 3. Add continuous collision monitoring
- Current: only checks post-move
- Need: background thread polling at ≥10Hz during motion
- Integrate with safety_monitor.py e-stop
- Budget: must not impact 500Hz control loop

## P1 — This Week

### 4. Add J0/J3/J5/gripper calibration support
- Current calibration only handles J1/J2/J4
- J0 locked at 0° for simplicity — unlock after main joints calibrated
- J3/J5 are wrist joints — smaller range, may need different detection approach
- Gripper: binary open/close, may not need calibration

### 5. Improve arm detection beyond frame differencing
- Frame differencing works but requires movement (can't detect static arm)
- Options: train lightweight ML model (MobileNet/YOLO), use depth camera, multi-frame temporal fusion
- Gold segment HSV detection (H:20-40) works for colored parts only

### 6. Camera auto-discovery and health monitoring
- Auto-detect camera URLs (scan ports, check /snap endpoints)
- Health heartbeat: alert if camera feed drops
- Auto-restart camera server on failure (extend watchdog)

## P2 — Soon

### 7. Performance optimization
- 500Hz control loop budget = 2ms per cycle
- Profile: FK computation, collision checking, state updates
- Consider: pre-compute collision zones, spatial indexing (octree)
- Monitor: actual loop frequency under load

### 8. Integration tests with real hardware
- Test suite currently uses mocked connections
- Need: hardware-in-the-loop test mode
- Key tests: joint limits respected, e-stop works, trajectory tracking accuracy
- Safety: start with small movements, have physical e-stop ready

### 9. Documentation
- API docs (FastAPI auto-docs may suffice for server)
- Setup guide (hardware connections, camera setup, first-run calibration)
- Architecture diagram (control flow, data flow, module dependencies)
- Calibration guide (step-by-step with photos)

## P3 — Nice to Have

### 10. CI/CD improvements
- Add hardware-in-the-loop CI stage (when arm connected)
- Performance regression tests
- Coverage tracking
- Auto-deploy web UI changes

### 11. Advanced collision avoidance
- Predictive: use trajectory to check future positions
- Dynamic: handle moving obstacles via vision
- Force feedback: detect unexpected contact via motor current

### 12. Multi-arm coordination
- Future: if second D1 added, coordinate shared workspace
- Shared collision map, turn-taking protocols
