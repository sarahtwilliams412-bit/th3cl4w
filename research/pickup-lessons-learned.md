# Lessons Learned — Red Bull Can Pickup Attempt (2026-02-07)

## What Went Wrong

### 1. Operated blind for 20+ minutes
I spent the entire first attempt rotating J0 the wrong direction (positive = left, not right). I should have calibrated the coordinate frame FIRST before trying anything.

### 2. Trusted a vision model for precision robotics
I used an image analysis model to estimate distances in grid squares. It was consistently wrong — off by 2-5x on distance estimates, confused by perspective, and couldn't reliably tell if the gripper was at table level. Vision models are good for "is there a can?" not "how many mm away is it?"

### 3. No systematic approach
I was guess-and-check with random joint angle tweaks instead of:
- Moving ONE joint at a time
- Measuring the effect
- Computing the correction
- Iterating

### 4. Used set_all_joints() instead of set_joint()
The calibration proved set_all_joints (funcode 2) causes freezes on large moves. Single-joint commands (funcode 1) are reliable.

### 5. Pushed past joint limits
J1 went to 99° (limit is 90°) and tripped the arm's safety. I should have built in software limits with margin — never command within 5° of a joint limit.

### 6. No safety margins in code
Every move should clamp to safe ranges BEFORE sending. I had no guard rails in my ad-hoc scripts.

### 7. Didn't use FK/IK properly
I have a kinematics module but the DH parameters are estimated (not from real hardware). Instead of trusting bad FK, I should have built a lookup table from the calibration data.

## What I Should Do Next

### Immediate
1. **Add soft joint limits** — clamp all commands to ±85° for J1/J2/J4 (5° margin from hardware limits)
2. **Build a joint-to-workspace mapping** from calibration photos — real data, not estimated DH params
3. **Use set_joint() exclusively** until set_all_joints reliability is understood
4. **Move in small increments** — max 10° per command, check feedback before next move

### For the pickup retry
1. Start from home, rotate J0 to point at can (negative = right)
2. Extend arm in small steps, checking camera between each
3. Stop at safe limits (J1 < 80°, never exceed 85°)
4. Use BOTH cameras to triangulate — don't rely on one perspective
5. Approach from above: position gripper OVER the can first, then lower

### Long-term
1. **Calibrate the DH parameters** with real measurements (not estimated)
2. **Stereo camera calibration** with the checkerboard so we have real 3D positions
3. **Closed-loop visual servoing** — compute error from camera, send correction, repeat
4. **Force/torque feedback** — gripper at 0mm means we missed, need to detect this
