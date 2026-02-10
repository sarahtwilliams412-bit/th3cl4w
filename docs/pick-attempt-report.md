# Red Bull Can Pick Attempt Report — 2026-02-09

## Status: FAILED — Hardware Power Limitations

## What Happened
1. Successfully homed the arm from overcurrent-sagged position
2. Reset + Enable sequence required (enable alone doesn't work)
3. Extended arm toward Red Bull can (J0=-70°, J1=50-75°, J2=60-73°)
4. Got gripper within 2-5cm of can per camera vision analysis
5. **Arm lost power during extended reach** — overcurrent protection triggered again
6. J1 stalled at ~48° when pushing to 60+° with elbow extended

## Root Causes
1. **Overcurrent protection**: Extended arm poses (J1>50° + J2>60°) draw too much current
2. **Wrist gravity droop**: J4 can't hold target angle when arm is extended (commands +60°, actual -91°)
3. **No gravity compensation**: Software doesn't account for joint torque under load
4. **No visual servo**: Gemini API key was revoked — `/api/locate` and `/api/servo/approach` are non-functional

## Missing App Features Needed

### Critical
- [ ] **Enable sequence**: App needs "Reset then Enable" — current Enable button alone doesn't work after overcurrent
- [ ] **Gravity-aware motion planning**: Must sequence moves to minimize torque (e.g., don't extend elbow while leaning forward)
- [ ] **Power/overcurrent recovery**: Auto-detect power loss, auto-reset+re-enable, resume from safe position
- [ ] **Gemini API key rotation**: Need new key for vision-guided tasks

### Important  
- [ ] **Visual servo without Gemini**: Use OpenCV HSV detection for the Red Bull can (red color) instead of LLM
- [ ] **Safe reach planner**: Pre-compute reachable workspace considering torque limits
- [ ] **Incremental approach**: Move-snap-verify loop using cameras at each step
- [ ] **Gripper contact detection**: Current adaptive close doesn't reliably detect contact
- [ ] **Joint torque monitoring**: Track when joints can't reach targets (stall detection)

### Nice to Have
- [ ] **Gravity compensation model**: Compute expected torque per pose, reject dangerous configs
- [ ] **Smooth trajectory**: S-curve profiles that keep acceleration within motor limits
- [ ] **Pick-and-place macro**: One-button "pick object at X,Y" with full approach/grasp/lift sequence

## Key Operational Learnings
- `reset` THEN `enable` is required after power loss
- J1 stalls around 48° when J2 > 60° (combined torque too high)
- J4 (wrist) completely loses position when arm is extended — gravity wins
- Maximum safe reach config: J1≈50°, J2≈50°, J4 unreliable
- Can was ~15-25cm from gripper at max safe extension — may need to reposition can closer

## Recommended Next Steps
1. Get new Gemini API key for vision
2. Add reset+enable combo to the UI
3. Add power-loss auto-recovery
4. Move Red Bull can closer to arm base (within safe reach envelope)
5. Implement OpenCV-based can detection as Gemini backup
