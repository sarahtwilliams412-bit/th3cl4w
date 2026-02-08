# Full Comparison Run Notes — 2026-02-08

## Run Metadata
- **Session:** cal_1770588119
- **Start:** ~14:07 PST (inferred from file timestamps at 14:11)
- **Duration:** 231.6s (~3m 52s)
- **Poses:** 20 unique poses × 2 cameras = 40 observations
- **Recommendation from system:** `archive` (not usable)

## Timeline
- ~14:07 PST — Run started (session cal_1770588119)
- Poses 0–19 executed sequentially across 2 cameras each
- ~14:11 PST — Run completed, all files written
- 14:21 PST — Notes agent began observation (run already finished)

## Pose Sequence (commanded → actual angles)
| Pose | Actual Joint Angles (deg) |
|------|--------------------------|
| 0 | [0.4, 0.5, 0.5, -0.3, 0.6, 0.0] — home |
| 1 | [29.5, 0.6, 0.5, -0.1, 0.5, 0.1] — base +30 |
| 2 | [-29.4, 0.6, 0.4, 0.0, 0.4, 0.1] — base -30 |
| 3 | [-0.1, -29.7, 0.4, 0.0, 0.4, 0.1] — shoulder -30 |
| 4 | [-0.1, -60.3, 0.3, 0.2, 0.3, 0.1] — shoulder -60 |
| 5 | [-0.2, 0.2, 45.3, -0.1, 0.4, 0.0] — elbow +45 |
| 6 | [-0.1, 0.1, -44.5, 0.0, 0.3, 0.0] — elbow -45 |
| 7 | [-0.2, -29.9, 30.3, 0.1, 0.4, 0.1] — shoulder-30 elbow+30 |
| 8 | [0.0, -44.9, 45.3, 0.0, -29.1, 0.1] — multi-joint |
| 9 | [29.5, -30.4, 30.7, -0.1, 0.1, 0.1] — multi-joint |
| 10 | [-29.7, -30.3, 30.7, 0.0, 0.1, 0.1] — multi-joint |
| 11 | [-0.2, 30.7, 0.8, 0.0, 0.2, 0.0] — shoulder +30 |
| 12 | [-0.1, -29.5, 45.1, 0.0, 44.9, 0.0] — multi-joint |
| 13 | [44.7, -44.9, 30.7, 0.1, 0.8, 0.1] — multi-joint |
| 14 | [-44.5, -44.9, 30.7, 0.1, 0.7, 0.1] — multi-joint |
| 15 | [-0.2, 0.2, 0.6, 0.0, -44.1, 0.1] — wrist -44 |
| 16 | [-0.1, -29.6, 0.5, 0.1, 45.1, 0.1] — shoulder-30 wrist+45 |
| 17 | [59.8, 0.4, 30.2, 0.0, 0.8, 0.1] — base+60 elbow+30 |
| 18 | [-59.5, 0.4, 30.2, 0.0, 0.4, 0.1] — base-60 elbow+30 |
| 19 | [-0.1, -60.2, 60.1, 0.0, -44.1, 0.1] — multi-joint extreme |

## CV Detection Results
- **Overall detection rate: 1.5%** (3 detections out of 200 joint observations)
- Only camera 1 produced any detections (camera 0 = zero)
- Detections occurred only on poses 9, 12, 13:
  - Pose 9, cam 1: 1/5 joints detected (end_effector, error 4.6px)
  - Pose 12, cam 1: 1/5 joints detected (end_effector, error 19.6px)
  - Pose 13, cam 1: 1/5 joints detected (elbow, error 27.6px)
- **CV latency:** avg 672ms, min 609ms, max 795ms (consistent)

## LLM Detection Results
- **LLM was never invoked** — 0 tokens, 0 latency, 0 detections
- This suggests the LLM detection path may not be wired up, or is disabled/skipped
- Cost: $0.00

## Observations
1. **LLM path is non-functional** — Zero LLM calls were made across all 40 observations. This is the most critical finding. The comparison is one-sided.
2. **CV detection is near-zero** — 1.5% detection rate means the CV pipeline is not reliably finding joints. Only end_effector and elbow were ever detected, and only on camera 1.
3. **Camera 0 produced zero detections** — Possible occlusion, bad angle, or calibration issue with that camera.
4. **Joint position accuracy is reasonable** — Commanded vs actual angles are within ~0.5° for most joints, showing good motor control.
5. **CV latency is stable** — 609-795ms range with 672ms average suggests consistent processing time even when nothing is detected.
6. **All 20 poses completed without errors** — The arm moved through the full sequence successfully.
7. **Arm ended at approximately [-0.2, 0.3, 0.4, 0.0, -44.4, 0.1]** — close to pose 15/19 wrist position, suggesting it didn't return to home.

## Issues
1. **CRITICAL: LLM detection never ran** — 0 tokens consumed, 0 latency recorded. The comparison is invalid without LLM data.
2. **CV detection rate too low to calibrate** — 1.5% gives almost no data points to compute transforms.
3. **System recommended "archive"** — correctly identifying this run as not useful.
4. **Notes agent spawned after run completed** — run took ~4 minutes but agent started at 14:21, ~10 min after completion. Real-time observation was not possible.

## Summary
- **Total duration:** 231.6s (~3m 52s)
- **Poses completed:** 20/20 ✅
- **CV detection rate:** 1.5% (3/200 joints)
- **LLM detection rate:** 0.0% (never called)
- **LLM success rate:** N/A (0 calls made)
- **LLM cost:** $0.00
- **Verdict:** Run completed mechanically but produced no useful calibration data. Both detection methods effectively failed — CV found almost nothing, LLM was never invoked. The run should be investigated for why the LLM path didn't execute before re-running.
