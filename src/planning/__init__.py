"""
Planning module for the D1 arm.

Provides trajectory generation and interpolation.
"""

from src.planning.trajectory import JointTrajectory, TrajectoryPoint

__all__ = [
    "JointTrajectory",
    "TrajectoryPoint",
]
