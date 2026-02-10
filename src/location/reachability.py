"""Reachability checking for the D1 arm's workspace envelope."""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np


class ReachStatus(Enum):
    REACHABLE = "reachable"
    OUT_OF_RANGE = "out_of_range"
    TOO_CLOSE = "too_close"
    MARGINAL = "marginal"


# D1 workspace constants
ARM_MAX_REACH_MM = 550.0
ARM_SAFE_REACH_MM = 500.0
ARM_MIN_REACH_MM = 80.0
ARM_BASE_POSITION = np.array([0.0, 0.0, 0.0])


def classify_reach(position_mm: np.ndarray) -> tuple[ReachStatus, float]:
    """Classify whether a 3D position is within the arm's reach envelope.

    Args:
        position_mm: (3,) XYZ position in mm relative to arm base.

    Returns:
        (ReachStatus, distance_mm) tuple.
    """
    xy_dist = float(np.linalg.norm(position_mm[:2]))

    if xy_dist < ARM_MIN_REACH_MM:
        return ReachStatus.TOO_CLOSE, xy_dist
    elif xy_dist > ARM_MAX_REACH_MM:
        return ReachStatus.OUT_OF_RANGE, xy_dist
    elif xy_dist > ARM_SAFE_REACH_MM:
        return ReachStatus.MARGINAL, xy_dist
    else:
        return ReachStatus.REACHABLE, xy_dist


def is_reachable(position_mm: np.ndarray) -> bool:
    """Quick check: is this position within the arm's reach?"""
    status, _ = classify_reach(position_mm)
    return status in (ReachStatus.REACHABLE, ReachStatus.MARGINAL)
