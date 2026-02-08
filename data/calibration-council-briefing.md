# Calibration Council Briefing — 2026-02-07

## Current Calibration Results
```json
{
  "links_mm": {"base": 297.4, "shoulder": 50.0, "elbow": 50.0, "wrist1": 10.0, "wrist2": 16.1, "end": 16.1},
  "joint_viz_offsets": [0, 179.8, 180.0, 0, 177.1, 0],
  "residual": 122.83,
  "n_observations": 20,
  "camera_params": {"sx": 0.1, "sy": 0.1, "tx": 1918.4, "ty": 27.7}
}
```

## Problems
1. **122.83px average residual** — way too high, viz won't match reality
2. **Degenerate solver output** — base=297mm (real ~80mm), shoulder/elbow=50mm (real ~170mm), offsets ~180° (should be ~90°). The solver is clearly not converging to physical reality.
3. **Camera params are broken** — sx=sy=0.1, tx=1918 means the transform is mapping to a tiny corner of the image. The affine model isn't working.
4. **40 collision events** during calibration — arm hit checkerboard, clamps, tools, person standing nearby
5. **Only J1, J2, J4 (pitch joints) are calibrated** — J0 (base yaw), J3 (wrist roll), J5 (wrist roll), and gripper are completely ignored

## Why J0, J3, J5, and Gripper Are Not Calibrated

### J0 (Base Yaw)
The V1 viz is a **2D side view**. J0 rotates the arm around the vertical axis (yaw). In a side view, yaw rotation changes which "slice" of the arm you see — it foreshortens/extends the apparent link lengths. The current `drawArm()` shows J0 as a separate rotation indicator circle at the base, NOT as part of the FK chain. It's rendered as an orange arc, not affecting link positions.

**Problem:** J0 directly affects what the side-view looks like. At J0=0° you see the full arm. At J0=90° you're looking at the arm end-on and it appears as just the base. The viz ignores this entirely.

### J3 (Wrist Roll) & J5 (Wrist Roll)
These are roll joints — they rotate around the link axis. In a 2D side view, roll has zero visual effect on link positions. The viz draws them as orange rotation indicator arcs. They're cosmetic only.

**Problem:** While roll doesn't change link positions in side view, it DOES change the gripper orientation, which affects what the camera sees and where the end-effector appears.

### Gripper
Not part of the FK chain at all in the viz. Drawn as a static fork shape at the end of the last link. Opening/closing the gripper doesn't change the viz geometry.

**Problem:** Gripper aperture is visible in camera and affects detection of the end-effector.

## Workspace Observations (from camera analysis)
- Checkerboard calibration target draped across the arm's workspace, blocking movement
- Yellow bar clamps securing the base board are in the sweep path
- Metal ruler, pliers, tape measure on the work surface
- Person standing very close to the arm's range
- Unitree controller box in the reachable workspace
- Cables from the base creating snag points
- This is a cluttered garage/workshop, NOT a clean lab

## What Needs to Improve
1. The solver is fundamentally broken — degenerate output means the optimization landscape has too many local minima or the detection data is bad
2. Need to verify end-effector detection is actually finding the gripper (not random contours)
3. J0 foreshortening must be accounted for in the viz
4. Collision avoidance must work BEFORE calibration runs
5. The workspace needs to be clear for calibration to work
