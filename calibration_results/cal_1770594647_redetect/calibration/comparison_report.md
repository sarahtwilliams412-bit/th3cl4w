# Calibration Comparison Report
**Session:** calibration
**Date:** 2026-02-08T16:10:40.144998
**Duration:** 0.0s

## Summary

- **Total poses:** 20
- **CV mean detection rate:** 0.0%
- **LLM mean detection rate:** 100.0%
- **CV mean error:** 0.0 px
- **LLM mean error:** 1004.9 px
- **CV mean latency:** 97.4 ms
- **LLM mean latency:** 1778.2 ms
- **Recommendation:** `archive`

## Per-Joint Analysis

| Joint | CV Det% | LLM Det% | CV Err(px) | LLM Err(px) | Agreement |
|-------|---------|----------|------------|-------------|-----------|
| base           |   0.0% |  100.0% |          — |      1395.1 |   0.0% |
| shoulder       |   0.0% |  100.0% |          — |      1262.7 |   0.0% |
| elbow          |   0.0% |  100.0% |          — |      1010.4 |   0.0% |
| wrist          |   0.0% |  100.0% |          — |       753.3 |   0.0% |
| end_effector   |   0.0% |  100.0% |          — |       603.2 |   0.0% |

## Per-Pose Detail

| Pose | Cam | CV Det | LLM Det | CV Err(mean) | LLM Err(mean) |
|------|-----|--------|---------|--------------|---------------|
|    0 | 0 | 0/5 | 5/5 |            — |        1017.5 |
|    1 | 0 | 0/5 | 5/5 |            — |         992.6 |
|    2 | 0 | 0/5 | 5/5 |            — |         986.2 |
|    3 | 0 | 0/5 | 5/5 |            — |        1008.2 |
|    4 | 0 | 0/5 | 5/5 |            — |        1089.7 |
|    5 | 0 | 0/5 | 5/5 |            — |         992.6 |
|    6 | 0 | 0/5 | 5/5 |            — |         986.2 |
|    7 | 0 | 0/5 | 5/5 |            — |         981.7 |
|    8 | 0 | 0/5 | 5/5 |            — |        1006.6 |
|    9 | 0 | 0/5 | 5/5 |            — |         986.2 |
|   10 | 0 | 0/5 | 5/5 |            — |         992.6 |
|   11 | 0 | 0/5 | 5/5 |            — |         992.6 |
|   12 | 0 | 0/5 | 5/5 |            — |        1099.6 |
|   13 | 0 | 0/5 | 5/5 |            — |         986.2 |
|   14 | 0 | 0/5 | 5/5 |            — |         992.6 |
|   15 | 0 | 0/5 | 5/5 |            — |         986.9 |
|   16 | 0 | 0/5 | 5/5 |            — |         992.6 |
|   17 | 0 | 0/5 | 5/5 |            — |         978.2 |
|   18 | 0 | 0/5 | 5/5 |            — |         978.2 |
|   19 | 0 | 0/5 | 5/5 |            — |        1051.7 |

## Cost Analysis

- **Total LLM tokens:** 46,454
- **Estimated cost:** $0.0061
- **Cost per pose:** $0.000305
- **Cost per detection:** $0.000061

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
- Pose 5, joint `base`
- Pose 5, joint `shoulder`
- Pose 5, joint `elbow`
- Pose 5, joint `wrist`
- Pose 5, joint `end_effector`
- Pose 6, joint `base`
- Pose 6, joint `shoulder`
- Pose 6, joint `elbow`
- Pose 6, joint `wrist`
- Pose 6, joint `end_effector`
- Pose 7, joint `base`
- Pose 7, joint `shoulder`
- Pose 7, joint `elbow`
- Pose 7, joint `wrist`
- Pose 7, joint `end_effector`
- Pose 8, joint `base`
- Pose 8, joint `shoulder`
- Pose 8, joint `elbow`
- Pose 8, joint `wrist`
- Pose 8, joint `end_effector`
- Pose 9, joint `base`
- Pose 9, joint `shoulder`
- Pose 9, joint `elbow`
- Pose 9, joint `wrist`
- Pose 9, joint `end_effector`
- Pose 10, joint `base`
- Pose 10, joint `shoulder`
- Pose 10, joint `elbow`
- Pose 10, joint `wrist`
- Pose 10, joint `end_effector`
- Pose 11, joint `base`
- Pose 11, joint `shoulder`
- Pose 11, joint `elbow`
- Pose 11, joint `wrist`
- Pose 11, joint `end_effector`
- Pose 12, joint `base`
- Pose 12, joint `shoulder`
- Pose 12, joint `elbow`
- Pose 12, joint `wrist`
- Pose 12, joint `end_effector`
- Pose 13, joint `base`
- Pose 13, joint `shoulder`
- Pose 13, joint `elbow`
- Pose 13, joint `wrist`
- Pose 13, joint `end_effector`
- Pose 14, joint `base`
- Pose 14, joint `shoulder`
- Pose 14, joint `elbow`
- Pose 14, joint `wrist`
- Pose 14, joint `end_effector`
- Pose 15, joint `base`
- Pose 15, joint `shoulder`
- Pose 15, joint `elbow`
- Pose 15, joint `wrist`
- Pose 15, joint `end_effector`
- Pose 16, joint `base`
- Pose 16, joint `shoulder`
- Pose 16, joint `elbow`
- Pose 16, joint `wrist`
- Pose 16, joint `end_effector`
- Pose 17, joint `base`
- Pose 17, joint `shoulder`
- Pose 17, joint `elbow`
- Pose 17, joint `wrist`
- Pose 17, joint `end_effector`
- Pose 18, joint `base`
- Pose 18, joint `shoulder`
- Pose 18, joint `elbow`
- Pose 18, joint `wrist`
- Pose 18, joint `end_effector`
- Pose 19, joint `base`
- Pose 19, joint `shoulder`
- Pose 19, joint `elbow`
- Pose 19, joint `wrist`
- Pose 19, joint `end_effector`

### Where LLM Failed

- Pose 0, joint `base`: error 1405.1px
- Pose 0, joint `shoulder`: error 1256.1px
- Pose 0, joint `elbow`: error 1048.5px
- Pose 0, joint `wrist`: error 798.9px
- Pose 0, joint `end_effector`: error 579.1px
- Pose 1, joint `base`: error 1405.1px
- Pose 1, joint `shoulder`: error 1256.1px
- Pose 1, joint `elbow`: error 970.9px
- Pose 1, joint `wrist`: error 720.0px
- Pose 1, joint `end_effector`: error 611.0px
- Pose 2, joint `base`: error 1405.1px
- Pose 2, joint `shoulder`: error 1256.1px
- Pose 2, joint `elbow`: error 970.9px
- Pose 2, joint `wrist`: error 720.0px
- Pose 2, joint `end_effector`: error 579.1px
- Pose 3, joint `base`: error 1405.1px
- Pose 3, joint `shoulder`: error 1285.7px
- Pose 3, joint `elbow`: error 1001.6px
- Pose 3, joint `wrist`: error 751.1px
- Pose 3, joint `end_effector`: error 597.8px
- Pose 4, joint `base`: error 1371.1px
- Pose 4, joint `shoulder`: error 1285.7px
- Pose 4, joint `elbow`: error 1211.6px
- Pose 4, joint `wrist`: error 856.0px
- Pose 4, joint `end_effector`: error 724.2px
- Pose 5, joint `base`: error 1405.1px
- Pose 5, joint `shoulder`: error 1256.1px
- Pose 5, joint `elbow`: error 970.9px
- Pose 5, joint `wrist`: error 720.0px
- Pose 5, joint `end_effector`: error 611.0px
- Pose 6, joint `base`: error 1405.1px
- Pose 6, joint `shoulder`: error 1256.1px
- Pose 6, joint `elbow`: error 970.9px
- Pose 6, joint `wrist`: error 720.0px
- Pose 6, joint `end_effector`: error 579.1px
- Pose 7, joint `base`: error 1397.2px
- Pose 7, joint `shoulder`: error 1263.9px
- Pose 7, joint `elbow`: error 960.8px
- Pose 7, joint `wrist`: error 716.9px
- Pose 7, joint `end_effector`: error 569.6px
- Pose 8, joint `base`: error 1405.1px
- Pose 8, joint `shoulder`: error 1277.2px
- Pose 8, joint `elbow`: error 1001.6px
- Pose 8, joint `wrist`: error 751.1px
- Pose 8, joint `end_effector`: error 597.8px
- Pose 9, joint `base`: error 1405.1px
- Pose 9, joint `shoulder`: error 1256.1px
- Pose 9, joint `elbow`: error 970.9px
- Pose 9, joint `wrist`: error 720.0px
- Pose 9, joint `end_effector`: error 579.1px
- Pose 10, joint `base`: error 1405.1px
- Pose 10, joint `shoulder`: error 1256.1px
- Pose 10, joint `elbow`: error 970.9px
- Pose 10, joint `wrist`: error 720.0px
- Pose 10, joint `end_effector`: error 611.0px
- Pose 11, joint `base`: error 1371.1px
- Pose 11, joint `shoulder`: error 1241.5px
- Pose 11, joint `elbow`: error 1001.6px
- Pose 11, joint `wrist`: error 751.1px
- Pose 11, joint `end_effector`: error 597.8px
- Pose 12, joint `base`: error 1341.3px
- Pose 12, joint `shoulder`: error 1256.1px
- Pose 12, joint `elbow`: error 1211.6px
- Pose 12, joint `wrist`: error 943.1px
- Pose 12, joint `end_effector`: error 746.0px
- Pose 13, joint `base`: error 1405.1px
- Pose 13, joint `shoulder`: error 1256.1px
- Pose 13, joint `elbow`: error 970.9px
- Pose 13, joint `wrist`: error 720.0px
- Pose 13, joint `end_effector`: error 579.1px
- Pose 14, joint `base`: error 1405.1px
- Pose 14, joint `shoulder`: error 1256.1px
- Pose 14, joint `elbow`: error 970.9px
- Pose 14, joint `wrist`: error 720.0px
- Pose 14, joint `end_effector`: error 611.0px
- Pose 15, joint `base`: error 1371.1px
- Pose 15, joint `shoulder`: error 1285.7px
- Pose 15, joint `elbow`: error 988.9px
- Pose 15, joint `wrist`: error 721.4px
- Pose 15, joint `end_effector`: error 567.3px
- Pose 16, joint `base`: error 1405.1px
- Pose 16, joint `shoulder`: error 1256.1px
- Pose 16, joint `elbow`: error 970.9px
- Pose 16, joint `wrist`: error 720.0px
- Pose 16, joint `end_effector`: error 611.0px
- Pose 17, joint `base`: error 1405.1px
- Pose 17, joint `shoulder`: error 1256.1px
- Pose 17, joint `elbow`: error 970.9px
- Pose 17, joint `wrist`: error 720.0px
- Pose 17, joint `end_effector`: error 538.8px
- Pose 18, joint `base`: error 1405.1px
- Pose 18, joint `shoulder`: error 1256.1px
- Pose 18, joint `elbow`: error 970.9px
- Pose 18, joint `wrist`: error 720.0px
- Pose 18, joint `end_effector`: error 538.8px
- Pose 19, joint `base`: error 1379.1px
- Pose 19, joint `shoulder`: error 1285.7px
- Pose 19, joint `elbow`: error 1101.5px
- Pose 19, joint `wrist`: error 856.0px
- Pose 19, joint `end_effector`: error 636.1px

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