# Calibration Comparison Report
**Session:** cal_1770595726
**Date:** 2026-02-08T16:09:48.393785
**Duration:** 51.9s

## Summary

- **Total poses:** 5
- **CV mean detection rate:** 0.0%
- **LLM mean detection rate:** 100.0%
- **CV mean error:** 0.0 px
- **LLM mean error:** 1001.2 px
- **CV mean latency:** 91.7 ms
- **LLM mean latency:** 1807.7 ms
- **Recommendation:** `archive`

## Per-Joint Analysis

| Joint | CV Det% | LLM Det% | CV Err(px) | LLM Err(px) | Agreement |
|-------|---------|----------|------------|-------------|-----------|
| base           |   0.0% |  100.0% |          — |      1405.1 |   0.0% |
| shoulder       |   0.0% |  100.0% |          — |      1256.1 |   0.0% |
| elbow          |   0.0% |  100.0% |          — |       995.5 |   0.0% |
| wrist          |   0.0% |  100.0% |          — |       745.3 |   0.0% |
| end_effector   |   0.0% |  100.0% |          — |       603.8 |   0.0% |

## Per-Pose Detail

| Pose | Cam | CV Det | LLM Det | CV Err(mean) | LLM Err(mean) |
|------|-----|--------|---------|--------------|---------------|
|    0 | 0 | 0/5 | 5/5 |            — |        1033.9 |
|    1 | 0 | 0/5 | 5/5 |            — |         986.2 |
|    2 | 0 | 0/5 | 5/5 |            — |         978.2 |
|    3 | 0 | 0/5 | 5/5 |            — |        1033.9 |
|    4 | 0 | 0/5 | 5/5 |            — |         973.6 |

## Cost Analysis

- **Total LLM tokens:** 11,863
- **Estimated cost:** $0.0016
- **Cost per pose:** $0.000311
- **Cost per detection:** $0.000062

## Recommendation

**Verdict: archive**

### Where LLM Helped (CV missed, LLM found)

- Pose 0, joint `base`
- Pose 0, joint `shoulder`
- Pose 0, joint `elbow`
- Pose 0, joint `wrist`
- Pose 0, joint `end_effector`
- Pose 1, joint `base`
- Pose 1, joint `shoulder`
- Pose 1, joint `elbow`
- Pose 1, joint `wrist`
- Pose 1, joint `end_effector`
- Pose 2, joint `base`
- Pose 2, joint `shoulder`
- Pose 2, joint `elbow`
- Pose 2, joint `wrist`
- Pose 2, joint `end_effector`
- Pose 3, joint `base`
- Pose 3, joint `shoulder`
- Pose 3, joint `elbow`
- Pose 3, joint `wrist`
- Pose 3, joint `end_effector`
- Pose 4, joint `base`
- Pose 4, joint `shoulder`
- Pose 4, joint `elbow`
- Pose 4, joint `wrist`
- Pose 4, joint `end_effector`

### Where LLM Failed

- Pose 0, joint `base`: error 1405.1px
- Pose 0, joint `shoulder`: error 1256.1px
- Pose 0, joint `elbow`: error 1048.5px
- Pose 0, joint `wrist`: error 798.9px
- Pose 0, joint `end_effector`: error 660.9px
- Pose 1, joint `base`: error 1405.1px
- Pose 1, joint `shoulder`: error 1256.1px
- Pose 1, joint `elbow`: error 970.9px
- Pose 1, joint `wrist`: error 720.0px
- Pose 1, joint `end_effector`: error 579.1px
- Pose 2, joint `base`: error 1405.1px
- Pose 2, joint `shoulder`: error 1256.1px
- Pose 2, joint `elbow`: error 970.9px
- Pose 2, joint `wrist`: error 720.0px
- Pose 2, joint `end_effector`: error 538.8px
- Pose 3, joint `base`: error 1405.1px
- Pose 3, joint `shoulder`: error 1256.1px
- Pose 3, joint `elbow`: error 1048.5px
- Pose 3, joint `wrist`: error 798.9px
- Pose 3, joint `end_effector`: error 660.9px
- Pose 4, joint `base`: error 1405.1px
- Pose 4, joint `shoulder`: error 1256.1px
- Pose 4, joint `elbow`: error 938.8px
- Pose 4, joint `wrist`: error 689.0px
- Pose 4, joint `end_effector`: error 579.1px

## Visualizations

### Detection Rate by Joint
```
base         CV  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0%
             LLM |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| 100%

shoulder     CV  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0%
             LLM |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| 100%

elbow        CV  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0%
             LLM |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| 100%

wrist        CV  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0%
             LLM |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| 100%

end_effector CV  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0%
             LLM |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| 100%

```

### CV vs LLM Error (px)
```
  (no paired error data)
```