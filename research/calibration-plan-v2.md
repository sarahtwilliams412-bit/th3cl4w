# Calibration Plan v2 — Using Overhead + Front Cameras

## Camera Setup
- **Cam 0 (front view)**: Looking at the arm from the front, shows height + left/right
- **Cam 1 (overhead)**: Looking straight down, shows forward/back + left/right — THIS IS THE KEY CAMERA for XY positioning

## Reference Markers
- **Green tape**: Reference point (on checkerboard and/or arm)
- **Blue tape**: Position marker on workspace surface
- **Orange tape**: Boundary or axis marker
- **Checkerboard**: Each square = 15/16" (23.8mm)
- **Yellow tape measure**: Linear distance reference

## Calibration Procedure

### Phase 1: Establish Coordinate Frame from Overhead
1. Arm at home [0, -90, 90, 0, -90, 0]
2. Photo both cameras — mark arm base position on overhead
3. Move J0 to -30° (should go RIGHT) — photo overhead to confirm direction
4. Move J0 to +30° (should go LEFT) — photo overhead to confirm
5. **Result**: J0 direction verified from overhead (no perspective confusion)

### Phase 2: Map J1 (shoulder) effect from both cameras
1. Home position
2. J1 = -45° — front camera shows height, overhead shows reach change
3. J1 = 0° (horizontal) — measure how far gripper extends from base
4. J1 = +30° — measure how low gripper gets (DON'T exceed +80°)
5. **Result**: J1 angle → height and reach mapping

### Phase 3: Map J2 (elbow) effect
1. With J1=0 (horizontal), sweep J2 from +90 to -45
2. Photo at each position
3. **Result**: J2 angle → reach extension mapping

### Phase 4: Locate the can from overhead
1. Use overhead camera to measure EXACT grid squares from arm base to can
2. Cross-reference with front camera for height
3. **Result**: Can position in (right, forward, height) relative to base

### Phase 5: Compute joint angles for can position
1. Using the mappings from Phase 2-3, find J0/J1/J2 that put gripper at can
2. Add safety margin (stay 5° from all limits)
3. Plan approach: position ABOVE can first, then lower

### Phase 6: Execute pickup
1. Open gripper
2. Move to approach position (above can)
3. Lower to can height
4. Close gripper
5. Lift

## Safety Rules
- Never command J1 > 80° (limit is 90°, keep 10° margin)
- Never command J2 < -85° or > 85°
- Use set_joint() only (not set_all_joints)
- Max 15° per move
- Verify feedback matches command before next move
- If any joint error > 5° from commanded, STOP and investigate
