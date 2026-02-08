# LLM vs CV Joint Detection — Offline Comparison Report

**Session:** cal_1770590048
**Date:** 2026-02-08T14:41:35-0800
**Poses:** 20 (overhead camera only)
**Note:** Checkerboard calibration pattern present in scene

## Summary

| Metric | LLM (Gemini) | CV Pipeline |
|--------|-------------|-------------|
| Detection rate | 95.0% | 25.0% |
| Successful detections | 19/20 | 5/20 |
| Total API calls | 20 | N/A |
| Total tokens | 49556 | N/A |
| Avg tokens/call | 2478 | N/A |

## Per-Pose Results

### Pose 0: commanded=[0, 0, 0, 0, 0, 0], actual=[-0.2, 0.1, 0.7, 0.0, 0.1, 0.1]

LLM=✅ (5 joints, 2609 tok, 2008ms) | CV=❌ (5 joints, 823ms)

### Pose 1: commanded=[30, 0, 0, 0, 0, 0], actual=[29.7, 0.1, 0.8, 0.0, 0.2, 0.1]

LLM=✅ (5 joints, 2608 tok, 2137ms) | CV=❌ (5 joints, 703ms)

### Pose 2: commanded=[-30, 0, 0, 0, 0, 0], actual=[-29.7, 0.1, 0.7, 0.0, 0.3, 0.1]

LLM=✅ (5 joints, 2612 tok, 1865ms) | CV=❌ (5 joints, 693ms)

### Pose 3: commanded=[0, -30, 0, 0, 0, 0], actual=[-0.1, -29.7, 0.7, 0.0, 0.3, 0.1]

LLM=✅ (5 joints, 2611 tok, 1876ms) | CV=✅ (5 joints, 753ms)

### Pose 4: commanded=[0, -60, 0, 0, 0, 0], actual=[-0.1, -60.4, 0.6, 0.2, 0.1, 0.1]

LLM=✅ (5 joints, 2605 tok, 1642ms) | CV=❌ (5 joints, 754ms)

### Pose 5: commanded=[0, 0, 45, 0, 0, 0], actual=[-0.2, 0.3, 45.3, -0.1, 0.2, 0.1]

LLM=✅ (5 joints, 2605 tok, 3525ms) | CV=✅ (5 joints, 692ms)

### Pose 6: commanded=[0, 0, -45, 0, 0, 0], actual=[-0.1, 0.2, -44.5, 0.0, 0.2, 0.1]

LLM=✅ (5 joints, 2618 tok, 1877ms) | CV=✅ (5 joints, 775ms)

### Pose 7: commanded=[0, -30, 30, 0, 0, 0], actual=[-0.1, -29.9, 30.1, 0.0, 0.3, 0.1]

LLM=✅ (5 joints, 2623 tok, 1987ms) | CV=❌ (5 joints, 698ms)

### Pose 8: commanded=[0, -45, 45, 0, -30, 0], actual=[0.0, -45.1, 45.1, 0.0, -29.1, 0.1]

LLM=✅ (5 joints, 2611 tok, 1696ms) | CV=✅ (5 joints, 697ms)

### Pose 9: commanded=[30, -30, 30, 0, 0, 0], actual=[29.8, -30.3, 30.5, -0.1, 0.1, 0.1]

LLM=✅ (5 joints, 2593 tok, 1864ms) | CV=❌ (5 joints, 692ms)

### Pose 10: commanded=[-30, -30, 30, 0, 0, 0], actual=[-29.5, -30.3, 30.5, 0.0, 0.1, 0.1]

LLM=❌ (0 joints, 0 tok, 5425ms) | CV=❌ (5 joints, 687ms)

### Pose 11: commanded=[0, 30, 0, 0, 0, 0], actual=[-0.2, 30.5, 0.6, 0.0, 0.3, 0.1]

LLM=✅ (5 joints, 2621 tok, 6307ms) | CV=❌ (5 joints, 686ms)

### Pose 12: commanded=[0, -30, 45, 0, 45, 0], actual=[-0.1, -30.0, 45.3, -0.1, 44.9, 0.1]

LLM=✅ (5 joints, 2625 tok, 1646ms) | CV=❌ (5 joints, 698ms)

### Pose 13: commanded=[45, -45, 30, 0, 0, 0], actual=[44.7, -45.0, 30.7, 0.0, 0.8, 0.1]

LLM=✅ (5 joints, 2608 tok, 1832ms) | CV=❌ (5 joints, 769ms)

### Pose 14: commanded=[-45, -45, 30, 0, 0, 0], actual=[-44.5, -45.1, 30.7, 0.1, 0.7, 0.1]

LLM=✅ (5 joints, 2608 tok, 2006ms) | CV=❌ (5 joints, 694ms)

### Pose 15: commanded=[0, 0, 0, 0, -45, 0], actual=[-0.2, 0.2, 0.4, -0.1, -44.1, 0.1]

LLM=✅ (5 joints, 2604 tok, 1647ms) | CV=✅ (5 joints, 692ms)

### Pose 16: commanded=[0, -30, 0, 0, 45, 0], actual=[-0.2, -29.7, 0.4, 0.0, 45.1, 0.1]

LLM=✅ (5 joints, 2593 tok, 1640ms) | CV=❌ (5 joints, 721ms)

### Pose 17: commanded=[60, 0, 30, 0, 0, 0], actual=[59.6, 0.3, 30.1, 0.0, 0.8, 0.1]

LLM=✅ (5 joints, 2610 tok, 1721ms) | CV=❌ (5 joints, 824ms)

### Pose 18: commanded=[-60, 0, 30, 0, 0, 0], actual=[-59.5, 0.4, 30.1, 0.0, 0.4, 0.1]

LLM=✅ (5 joints, 2591 tok, 1846ms) | CV=❌ (5 joints, 689ms)

### Pose 19: commanded=[0, -60, 60, 0, -45, 0], actual=[-0.1, -60.3, 60.1, 0.0, -44.1, 0.1]

LLM=✅ (5 joints, 2601 tok, 1777ms) | CV=❌ (5 joints, 697ms)

## Latency

| | LLM | CV |
|---|---|---|
| Avg | 2316ms | 722ms |
| Min | 1640ms | 686ms |
| Max | 6307ms | 824ms |

## vs Previous Run (cal_1770588119)

The previous run had NO checkerboard pattern:
- LLM: 100% detection rate
- CV: 10% detection rate

This run includes checkerboard calibration patterns which should improve CV detection
through better visual contrast and reference features.
