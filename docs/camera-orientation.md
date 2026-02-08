# Camera Orientation Reference

## Overhead Camera (cam1, http://localhost:8081/snap/1)

### Fixed Landmarks
- **Monitor/keyboard**: LEFT edge of image
- **Plywood board with arm**: CENTER-RIGHT
- **Arm base (black circle)**: approximately pixel (820, 590)
- **Checkerboard patterns**: LEFT-CENTER, on the desk

### Arm Direction at Home (all joints 0°)
- Arm extends from base at (820, 590) toward gripper at approximately (760, 170)
- That's **toward the TOP of the image**, slightly left
- "Forward from arm" = **UP in overhead image**

### J0 Rotation Mapping (overhead pixel coords)
| J0 angle | Gripper approx (x, y) | Image direction from base |
|----------|----------------------|--------------------------|
| 0° (home) | (760, 170) | UP (slightly left) |
| +45° (CCW from above) | (760, 490) | RIGHT of home direction |
| -45° (CW from above) | (580, 290) | LEFT of home direction |

### Overhead Image → Real World
- **Image UP** = arm forward (away from operator)
- **Image DOWN** = toward operator
- **Image LEFT** = toward monitor / operator's left
- **Image RIGHT** = away from monitor / operator's right

### J0+ (CCW from above) moves gripper toward IMAGE RIGHT-DOWN
### J0- (CW from above) moves gripper toward IMAGE LEFT-UP

⚠️ The overhead camera appears to be slightly rotated/angled, so
"left in image" doesn't perfectly align with "left in real world."
Use pixel coordinates for precision, not vague directions.

---

## Front Camera (cam0, http://localhost:8081/snap/0)

### Fixed Landmarks
- **Whiteboard with quote**: LEFT, TOP
- **Checkerboard (vertical)**: CENTER background
- **Unitree box**: RIGHT, MIDDLE
- **Metal shelving**: FAR LEFT
- **Table**: spans the BOTTOM

### Arm at Home
- Arm is vertical, centered in the image
- Base at BOTTOM-CENTER, gripper at TOP-CENTER

### Direction Mapping
| Motion | Image direction |
|--------|----------------|
| J0+ (arm swings left IRL) | Gripper moves RIGHT in image |
| J0- (arm swings right IRL) | Gripper moves LEFT in image |  
| J1+ (pitch UP) | Arm tilts away from camera (less visible, foreshortens) |
| J1- (pitch DOWN/forward) | Arm tilts toward camera / extends laterally |
| J2+ (elbow extends) | Forearm extends RIGHT |
| J2- (elbow folds) | Forearm folds back |
| J4+ (wrist down) | Gripper tips RIGHT/forward |
| J4- (wrist up) | Gripper tips up/back |

⚠️ "Right in front camera" ≈ robot swinging CCW from above.
The front camera reverses left-right relative to the operator's perspective.

---

## How to Read Camera Images Reliably

1. **Always identify 2+ fixed landmarks first** before judging arm position
2. **Use pixel coordinates** for the gripper tip, not vague "left/right"
3. **Compare to known reference poses** (home, J0±45, J1±45)
4. **Cross-reference both cameras** — if overhead says one thing and front says another, check pixel coords
5. **Don't trust the vision model's "left/right" labels** — they're often wrong about the camera's relationship to real-world directions
