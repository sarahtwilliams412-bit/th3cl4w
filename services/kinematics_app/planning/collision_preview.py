"""
Collision Preview for Bifocal Workspace Mapping.

Takes a proposed trajectory and checks it against the workspace occupancy grid
to predict collisions BEFORE the arm moves. Planning only — does not affect
arm movement.

Uses FK from kinematics to get arm link positions along the trajectory,
then queries the workspace mapper for obstacles.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.kinematics.kinematics import D1Kinematics
from .motion_planner import Trajectory, NUM_ARM_JOINTS

logger = logging.getLogger("th3cl4w.planning.collision_preview")


@dataclass
class CollisionHit:
    """A predicted collision along a trajectory."""

    trajectory_index: int
    time_s: float
    joint_angles_deg: list[float]
    link_index: int  # which link segment hit
    link_point_mm: list[float]  # 3D position of the link point
    obstacle_point_mm: list[float]  # 3D position of the nearest obstacle
    distance_mm: float  # distance to obstacle
    severity: str  # "collision", "warning", "clear"


@dataclass
class PreviewResult:
    """Result of a collision preview check."""

    clear: bool  # True if no collisions
    hits: list[CollisionHit]
    checked_points: int
    elapsed_ms: float
    trajectory_points: int
    summary: str


class CollisionPreview:
    """Preview arm trajectories against the workspace occupancy grid."""

    def __init__(
        self,
        kinematics: Optional[D1Kinematics] = None,
        warning_distance_mm: float = 50.0,
        collision_distance_mm: float = 20.0,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.warning_dist = warning_distance_mm
        self.collision_dist = collision_distance_mm

    def preview_trajectory(
        self,
        trajectory: Trajectory,
        workspace_mapper,  # WorkspaceMapper — avoid circular import
        step: int = 5,  # check every Nth trajectory point for speed
    ) -> PreviewResult:
        """Check a full trajectory for collisions against the workspace map.

        Args:
            trajectory: The planned trajectory to check.
            workspace_mapper: WorkspaceMapper with current occupancy grid.
            step: Check every Nth point for speed (1 = check all).

        Returns:
            PreviewResult with collision details.
        """
        t0 = time.monotonic()

        hits: list[CollisionHit] = []
        checked = 0

        for i in range(0, len(trajectory.points), step):
            pt = trajectory.points[i]
            angles_deg = pt.positions[:NUM_ARM_JOINTS]
            angles_rad = np.deg2rad(angles_deg)

            # Get 3D positions of all joint frames via FK
            q7 = np.zeros(7)
            q7[:6] = angles_rad
            joint_positions = self.kinematics.get_joint_positions_3d(q7)

            # Check each link segment midpoint and endpoint
            for link_idx in range(len(joint_positions)):
                link_pt = joint_positions[link_idx]
                link_mm = link_pt * 1000  # meters to mm

                status = workspace_mapper.check_point(link_mm)
                checked += 1

                if status == "occupied":
                    hits.append(
                        CollisionHit(
                            trajectory_index=i,
                            time_s=pt.time,
                            joint_angles_deg=[round(float(a), 1) for a in angles_deg],
                            link_index=link_idx,
                            link_point_mm=[round(float(x), 1) for x in link_mm],
                            obstacle_point_mm=[round(float(x), 1) for x in link_mm],
                            distance_mm=0.0,
                            severity="collision",
                        )
                    )

            # Also check midpoints between links for better coverage
            for link_idx in range(len(joint_positions) - 1):
                mid_pt = (joint_positions[link_idx] + joint_positions[link_idx + 1]) / 2
                mid_mm = mid_pt * 1000

                status = workspace_mapper.check_point(mid_mm)
                checked += 1

                if status == "occupied":
                    hits.append(
                        CollisionHit(
                            trajectory_index=i,
                            time_s=pt.time,
                            joint_angles_deg=[round(float(a), 1) for a in angles_deg],
                            link_index=link_idx,
                            link_point_mm=[round(float(x), 1) for x in mid_mm],
                            obstacle_point_mm=[round(float(x), 1) for x in mid_mm],
                            distance_mm=0.0,
                            severity="collision",
                        )
                    )

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Deduplicate hits by trajectory index (keep worst severity per point)
        seen_indices = set()
        unique_hits = []
        for h in hits:
            if h.trajectory_index not in seen_indices:
                seen_indices.add(h.trajectory_index)
                unique_hits.append(h)

        n_collisions = sum(1 for h in unique_hits if h.severity == "collision")
        n_warnings = sum(1 for h in unique_hits if h.severity == "warning")

        if n_collisions > 0:
            summary = f"{n_collisions} collision(s) detected"
        elif n_warnings > 0:
            summary = f"{n_warnings} close approach(es)"
        else:
            summary = "Path clear"

        return PreviewResult(
            clear=len(unique_hits) == 0,
            hits=unique_hits,
            checked_points=checked,
            elapsed_ms=round(elapsed_ms, 1),
            trajectory_points=len(trajectory.points),
            summary=summary,
        )

    def preview_single_pose(
        self,
        joint_angles_deg: np.ndarray,
        workspace_mapper,
    ) -> PreviewResult:
        """Quick check of a single arm pose against the workspace map."""
        t0 = time.monotonic()

        angles_rad = np.deg2rad(np.asarray(joint_angles_deg[:NUM_ARM_JOINTS]))
        q7 = np.zeros(7)
        q7[:6] = angles_rad

        joint_positions = self.kinematics.get_joint_positions_3d(q7)

        hits: list[CollisionHit] = []
        checked = 0

        for link_idx, pos in enumerate(joint_positions):
            pos_mm = pos * 1000
            status = workspace_mapper.check_point(pos_mm)
            checked += 1

            if status == "occupied":
                hits.append(
                    CollisionHit(
                        trajectory_index=0,
                        time_s=0.0,
                        joint_angles_deg=[round(float(a), 1) for a in joint_angles_deg[:6]],
                        link_index=link_idx,
                        link_point_mm=[round(float(x), 1) for x in pos_mm],
                        obstacle_point_mm=[round(float(x), 1) for x in pos_mm],
                        distance_mm=0.0,
                        severity="collision",
                    )
                )

        elapsed_ms = (time.monotonic() - t0) * 1000

        return PreviewResult(
            clear=len(hits) == 0,
            hits=hits,
            checked_points=checked,
            elapsed_ms=round(elapsed_ms, 1),
            trajectory_points=1,
            summary=f"{'Clear' if not hits else f'{len(hits)} collision(s)'}",
        )

    def get_arm_envelope(
        self,
        joint_angles_deg: np.ndarray,
    ) -> list[list[float]]:
        """Get the 3D positions of all arm links for a given pose.

        Returns list of [x, y, z] in mm for visualization.
        """
        angles_rad = np.deg2rad(np.asarray(joint_angles_deg[:NUM_ARM_JOINTS]))
        q7 = np.zeros(7)
        q7[:6] = angles_rad

        positions = self.kinematics.get_joint_positions_3d(q7)
        return [[round(float(x) * 1000, 1) for x in p] for p in positions]
