"""
World Model — kinematic replay of what the arm did.

Takes an Episode from the ReplayBuffer and reconstructs:
- End-effector position and orientation at each timestep
- Link positions for each joint frame
- Commanded vs actual trajectory comparison
- Workspace boundary proximity
- Near-collision detection between links

This is the arm's internal simulation of itself — its ability to
"re-watch" its own movements through the lens of its kinematic model
and understand spatially what happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from shared.kinematics.kinematics import D1Kinematics

logger = logging.getLogger("th3cl4w.introspection.world_model")

# D1 workspace radius in meters.
# The DH parameters give a fully-extended reach of ~0.6515m, so the true
# spherical workspace bound from the FK is larger than the rated 550mm.
WORKSPACE_RADIUS_M = 0.66


@dataclass
class FrameState:
    """Reconstructed spatial state at one instant."""

    ts: float
    joint_positions_rad: np.ndarray  # (7,) what the joints actually were
    ee_position: np.ndarray  # (3,) end-effector xyz in meters
    ee_orientation: np.ndarray  # (3, 3) rotation matrix
    link_positions: list[np.ndarray]  # list of (3,) for each joint frame origin
    ee_distance_from_base: float  # meters
    workspace_margin: float  # how far from workspace boundary (positive = inside)
    min_link_clearance: float  # min distance between non-adjacent links


@dataclass
class TrajectoryReconstruction:
    """Full kinematic reconstruction of an episode."""

    frames: list[FrameState] = field(default_factory=list)
    total_ee_path_length: float = 0.0  # meters, total distance traveled
    max_ee_speed: float = 0.0  # m/s peak speed
    mean_ee_speed: float = 0.0  # m/s average speed
    workspace_violations: int = 0  # times EE went outside workspace
    near_collision_count: int = 0  # times links got dangerously close
    min_workspace_margin: float = float("inf")
    min_link_clearance: float = float("inf")

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def ee_positions_array(self) -> np.ndarray:
        """(N, 3) array of end-effector positions."""
        if not self.frames:
            return np.empty((0, 3))
        return np.array([f.ee_position for f in self.frames])

    def ee_distances_array(self) -> np.ndarray:
        """(N,) array of EE distances from base."""
        if not self.frames:
            return np.empty(0)
        return np.array([f.ee_distance_from_base for f in self.frames])

    def times_array(self) -> np.ndarray:
        if not self.frames:
            return np.empty(0)
        return np.array([f.ts for f in self.frames])


class WorldModel:
    """Replays an episode through the kinematic model.

    Uses D1Kinematics to reconstruct spatial information from recorded
    joint angles, giving the arm a "mind's eye" view of what it did.
    """

    def __init__(
        self,
        kinematics: D1Kinematics | None = None,
        collision_clearance_m: float = 0.03,
    ) -> None:
        self.kin = kinematics or D1Kinematics()
        self.collision_clearance = collision_clearance_m

    def reconstruct(self, episode: "Episode") -> TrajectoryReconstruction:
        """Reconstruct the full spatial trajectory from an episode.

        Parameters
        ----------
        episode : Episode from the replay buffer containing joint snapshots.

        Returns
        -------
        TrajectoryReconstruction with per-frame spatial data and aggregate metrics.
        """
        from .replay_buffer import Episode

        recon = TrajectoryReconstruction()

        if not episode.joint_snapshots:
            logger.warning("Episode %s has no joint snapshots", episode.episode_id)
            return recon

        prev_ee_pos = None
        ee_speeds: list[float] = []
        prev_ts = None

        for snap in episode.joint_snapshots:
            positions_rad = snap.positions  # already in radians from DDS feedback

            # Forward kinematics
            T_ee = self.kin.forward_kinematics(positions_rad)
            ee_pos = T_ee[:3, 3].copy()
            ee_rot = T_ee[:3, :3].copy()

            # All link positions
            link_positions = self.kin.get_joint_positions_3d(positions_rad)

            # Distance from base
            ee_dist = float(np.linalg.norm(ee_pos))

            # Workspace margin
            workspace_margin = WORKSPACE_RADIUS_M - ee_dist

            # Link clearance (non-adjacent pairs)
            min_clearance = float("inf")
            n_links = len(link_positions)
            for i in range(n_links):
                for j in range(i + 2, n_links):
                    dist = float(np.linalg.norm(link_positions[i] - link_positions[j]))
                    min_clearance = min(min_clearance, dist)

            frame = FrameState(
                ts=snap.ts,
                joint_positions_rad=positions_rad,
                ee_position=ee_pos,
                ee_orientation=ee_rot,
                link_positions=link_positions,
                ee_distance_from_base=ee_dist,
                workspace_margin=workspace_margin,
                min_link_clearance=min_clearance,
            )
            recon.frames.append(frame)

            # Aggregate metrics
            if workspace_margin < 0:
                recon.workspace_violations += 1
            if min_clearance < self.collision_clearance:
                recon.near_collision_count += 1
            recon.min_workspace_margin = min(recon.min_workspace_margin, workspace_margin)
            recon.min_link_clearance = min(recon.min_link_clearance, min_clearance)

            # Path length and speed
            if prev_ee_pos is not None and prev_ts is not None:
                step_dist = float(np.linalg.norm(ee_pos - prev_ee_pos))
                recon.total_ee_path_length += step_dist
                dt = snap.ts - prev_ts
                if dt > 0:
                    speed = step_dist / dt
                    ee_speeds.append(speed)

            prev_ee_pos = ee_pos
            prev_ts = snap.ts

        if ee_speeds:
            recon.max_ee_speed = max(ee_speeds)
            recon.mean_ee_speed = sum(ee_speeds) / len(ee_speeds)

        logger.info(
            "Reconstructed %d frames: path=%.3fm, max_speed=%.3fm/s, "
            "workspace_violations=%d, near_collisions=%d",
            recon.n_frames,
            recon.total_ee_path_length,
            recon.max_ee_speed,
            recon.workspace_violations,
            recon.near_collision_count,
        )

        return recon

    def compute_tracking_error(
        self,
        episode: "Episode",
        reconstruction: TrajectoryReconstruction,
    ) -> dict:
        """Compare commanded targets against actual end-effector positions.

        Returns per-joint tracking error statistics.
        """
        if not episode.commands or not reconstruction.frames:
            return {"mean_error_rad": 0.0, "max_error_rad": 0.0, "per_joint": {}}

        # Build a timeline of actual positions keyed by timestamp
        actual_by_ts = {f.ts: f.joint_positions_rad for f in reconstruction.frames}
        sorted_ts = sorted(actual_by_ts.keys())

        errors_per_joint: dict[int, list[float]] = {}

        for cmd in episode.commands:
            if cmd.joint_id is None or cmd.target_value is None:
                continue

            # Find the closest actual state after this command (with some delay)
            target_ts = cmd.ts + 0.1  # 100ms expected settling
            closest_ts = min(sorted_ts, key=lambda t: abs(t - target_ts), default=None)
            if closest_ts is None:
                continue

            actual = actual_by_ts[closest_ts]
            jid = cmd.joint_id
            if jid < len(actual):
                err = abs(cmd.target_value - actual[jid])
                errors_per_joint.setdefault(jid, []).append(err)

        per_joint_stats = {}
        all_errors = []
        for jid, errs in errors_per_joint.items():
            per_joint_stats[jid] = {
                "mean": float(np.mean(errs)),
                "max": float(np.max(errs)),
                "std": float(np.std(errs)),
                "n_samples": len(errs),
            }
            all_errors.extend(errs)

        return {
            "mean_error_rad": float(np.mean(all_errors)) if all_errors else 0.0,
            "max_error_rad": float(np.max(all_errors)) if all_errors else 0.0,
            "per_joint": per_joint_stats,
        }

    def summarize_motion(self, reconstruction: TrajectoryReconstruction) -> dict:
        """High-level motion summary for the feedback generator."""
        if not reconstruction.frames:
            return {"motion": "none", "details": "No frames recorded"}

        first = reconstruction.frames[0]
        last = reconstruction.frames[-1]
        displacement = float(np.linalg.norm(last.ee_position - first.ee_position))

        # Compute joint range of motion
        positions = np.array([f.joint_positions_rad for f in reconstruction.frames])
        joint_rom = np.ptp(positions, axis=0)  # range per joint

        return {
            "duration_s": last.ts - first.ts,
            "ee_displacement_m": displacement,
            "ee_path_length_m": reconstruction.total_ee_path_length,
            "path_efficiency": displacement / max(reconstruction.total_ee_path_length, 1e-9),
            "max_speed_ms": reconstruction.max_ee_speed,
            "mean_speed_ms": reconstruction.mean_ee_speed,
            "joint_range_of_motion_rad": joint_rom.tolist(),
            "workspace_violations": reconstruction.workspace_violations,
            "near_collisions": reconstruction.near_collision_count,
            "min_workspace_margin_m": reconstruction.min_workspace_margin,
            "start_ee_pos": first.ee_position.tolist(),
            "end_ee_pos": last.ee_position.tolist(),
        }
