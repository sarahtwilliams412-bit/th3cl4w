# LLM vs CV Joint Detection — Full Comparison Report

**Date:** 2026-02-08T14:23:35-0800
**Poses tested:** 20
**Cameras:** 2 (front + overhead)

## Summary

| Metric | LLM (Gemini) | CV Pipeline |
|--------|-------------|-------------|
| Detection rate | 100.0% | 10.0% |
| Successful detections | 40/40 | 4/40 |
| Total API calls | 40 | N/A |
| Total tokens | 102999 | N/A |
| Avg tokens/call | 2575 | N/A |

## Per-Pose Results

### Pose 0: commanded=[0, 0, 0, 0, 0, 0], actual=[0.4, 0.7, 0.7, 0.3, 0.1, -2.4]

**cam0:** LLM=✅ (5 joints, 2500 tok, 1749ms) | CV=❌ (5 joints, 719ms)

**cam1:** LLM=✅ (5 joints, 2610 tok, 1574ms) | CV=❌ (5 joints, 670ms)

### Pose 1: commanded=[30, 0, 0, 0, 0, 0], actual=[29.5, 0.7, 0.7, -0.1, 0.3, 0.0]

**cam0:** LLM=✅ (5 joints, 2496 tok, 1790ms) | CV=❌ (5 joints, 853ms)

**cam1:** LLM=✅ (5 joints, 2593 tok, 1544ms) | CV=❌ (5 joints, 668ms)

### Pose 2: commanded=[-30, 0, 0, 0, 0, 0], actual=[-29.4, 0.7, 0.7, 0.0, 0.3, 0.1]

**cam0:** LLM=✅ (5 joints, 2539 tok, 1532ms) | CV=❌ (5 joints, 667ms)

**cam1:** LLM=✅ (5 joints, 2602 tok, 1613ms) | CV=❌ (5 joints, 682ms)

### Pose 3: commanded=[0, -30, 0, 0, 0, 0], actual=[0.0, -29.7, 0.6, 0.0, 0.3, 0.1]

**cam0:** LLM=✅ (5 joints, 2541 tok, 1663ms) | CV=❌ (5 joints, 681ms)

**cam1:** LLM=✅ (5 joints, 2604 tok, 1717ms) | CV=❌ (5 joints, 676ms)

### Pose 4: commanded=[0, -60, 0, 0, 0, 0], actual=[-0.1, -60.3, 0.6, 0.1, 0.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2548 tok, 1571ms) | CV=✅ (5 joints, 666ms)

**cam1:** LLM=✅ (5 joints, 2622 tok, 1590ms) | CV=✅ (5 joints, 775ms)

### Pose 5: commanded=[0, 0, 45, 0, 0, 0], actual=[-0.1, 0.2, 45.3, 0.0, 0.3, 0.1]

**cam0:** LLM=✅ (5 joints, 2552 tok, 1765ms) | CV=❌ (5 joints, 702ms)

**cam1:** LLM=✅ (5 joints, 2609 tok, 1726ms) | CV=❌ (5 joints, 688ms)

### Pose 6: commanded=[0, 0, -45, 0, 0, 0], actual=[-0.2, 0.1, -44.5, 0.0, 0.2, 0.1]

**cam0:** LLM=✅ (5 joints, 2551 tok, 1802ms) | CV=❌ (5 joints, 661ms)

**cam1:** LLM=✅ (5 joints, 2604 tok, 1443ms) | CV=❌ (5 joints, 710ms)

### Pose 7: commanded=[0, -30, 30, 0, 0, 0], actual=[-0.1, -29.8, 30.2, 0.1, 0.3, 0.1]

**cam0:** LLM=✅ (5 joints, 2513 tok, 1634ms) | CV=❌ (5 joints, 730ms)

**cam1:** LLM=✅ (5 joints, 2600 tok, 1538ms) | CV=❌ (5 joints, 693ms)

### Pose 8: commanded=[0, -45, 45, 0, -30, 0], actual=[0.0, -45.0, 45.0, 0.0, -29.3, 0.1]

**cam0:** LLM=✅ (5 joints, 2536 tok, 1987ms) | CV=❌ (5 joints, 700ms)

**cam1:** LLM=✅ (5 joints, 2603 tok, 1476ms) | CV=❌ (5 joints, 775ms)

### Pose 9: commanded=[30, -30, 30, 0, 0, 0], actual=[29.5, -29.9, 30.7, 0.0, 0.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2540 tok, 1714ms) | CV=❌ (5 joints, 687ms)

**cam1:** LLM=✅ (5 joints, 2601 tok, 1501ms) | CV=✅ (5 joints, 804ms)

### Pose 10: commanded=[-30, -30, 30, 0, 0, 0], actual=[-29.5, -29.9, 30.7, 0.0, 0.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2528 tok, 1536ms) | CV=❌ (5 joints, 848ms)

**cam1:** LLM=✅ (5 joints, 2593 tok, 1722ms) | CV=❌ (5 joints, 691ms)

### Pose 11: commanded=[0, 30, 0, 0, 0, 0], actual=[-0.2, 30.5, 0.7, 0.0, 0.2, 0.1]

**cam0:** LLM=✅ (5 joints, 2622 tok, 1749ms) | CV=❌ (5 joints, 693ms)

**cam1:** LLM=✅ (5 joints, 2620 tok, 1583ms) | CV=❌ (5 joints, 688ms)

### Pose 12: commanded=[0, -30, 45, 0, 45, 0], actual=[-0.2, -30.0, 45.3, -0.1, 44.9, 0.1]

**cam0:** LLM=✅ (5 joints, 2521 tok, 1506ms) | CV=❌ (5 joints, 808ms)

**cam1:** LLM=✅ (5 joints, 2605 tok, 1641ms) | CV=❌ (5 joints, 682ms)

### Pose 13: commanded=[45, -45, 30, 0, 0, 0], actual=[44.7, -45.2, 30.7, 0.0, 0.9, 0.1]

**cam0:** LLM=✅ (5 joints, 2559 tok, 1776ms) | CV=✅ (5 joints, 701ms)

**cam1:** LLM=✅ (5 joints, 2601 tok, 1658ms) | CV=❌ (5 joints, 820ms)

### Pose 14: commanded=[-45, -45, 30, 0, 0, 0], actual=[-44.5, -45.2, 30.7, 0.0, 0.8, 0.1]

**cam0:** LLM=✅ (5 joints, 2499 tok, 1753ms) | CV=❌ (5 joints, 701ms)

**cam1:** LLM=✅ (5 joints, 2620 tok, 1534ms) | CV=❌ (5 joints, 867ms)

### Pose 15: commanded=[0, 0, 0, 0, -45, 0], actual=[-0.2, 0.4, 0.6, -0.1, -44.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2578 tok, 1576ms) | CV=❌ (5 joints, 688ms)

**cam1:** LLM=✅ (5 joints, 2607 tok, 1552ms) | CV=❌ (5 joints, 713ms)

### Pose 16: commanded=[0, -30, 0, 0, 45, 0], actual=[-0.2, -29.9, 0.4, 0.0, 45.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2559 tok, 1557ms) | CV=❌ (5 joints, 806ms)

**cam1:** LLM=✅ (5 joints, 2606 tok, 1650ms) | CV=❌ (5 joints, 710ms)

### Pose 17: commanded=[60, 0, 30, 0, 0, 0], actual=[59.8, 0.1, 30.2, 0.0, 0.9, 0.1]

**cam0:** LLM=✅ (5 joints, 2578 tok, 1584ms) | CV=❌ (5 joints, 731ms)

**cam1:** LLM=✅ (5 joints, 2613 tok, 1667ms) | CV=❌ (5 joints, 720ms)

### Pose 18: commanded=[-60, 0, 30, 0, 0, 0], actual=[-59.5, 0.1, 30.2, 0.0, 0.4, 0.1]

**cam0:** LLM=✅ (5 joints, 2594 tok, 1548ms) | CV=❌ (5 joints, 713ms)

**cam1:** LLM=✅ (5 joints, 2604 tok, 1599ms) | CV=❌ (5 joints, 716ms)

### Pose 19: commanded=[0, -60, 60, 0, -45, 0], actual=[-0.1, -60.2, 60.0, 0.0, -44.1, 0.1]

**cam0:** LLM=✅ (5 joints, 2542 tok, 1580ms) | CV=❌ (5 joints, 709ms)

**cam1:** LLM=✅ (5 joints, 2586 tok, 1632ms) | CV=❌ (5 joints, 809ms)

## Latency

| | LLM | CV |
|---|---|---|
| Avg | 1633ms | 726ms |
| Min | 1443ms | 661ms |
| Max | 1987ms | 867ms |
