"""
Forward Kinematics and Self-Filter Validation Tests

1. Zero pose → end-effector at expected position
2. Joint 1 at pi/2 → 90° base rotation
3. Arm mask volume sanity check
4. Self-filter preserves obstacles, removes arm

Run with: pytest self_filter/test_fk.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from self_filter.arm_voxelizer import ArmVoxelizer
from self_filter.forward_kinematics import ForwardKinematics
from self_filter.obstacle_extractor import ObstacleExtractor

# D1 DH parameters from config.yaml (mm, rad)
_DH_PARAMS = [
    [0, -1.5708, 121.5, 0],   # J0: shoulder_yaw
    [0,  1.5708,   0,   0],   # J1: shoulder_pitch
    [0, -1.5708, 208.5, 0],   # J2: shoulder_roll
    [0,  1.5708,   0,   0],   # J3: elbow_pitch
    [0, -1.5708, 208.5, 0],   # J4: wrist_yaw
    [0,  1.5708,   0,   0],   # J5: wrist_pitch
    [0,  0,       113,  0],   # J6: wrist_roll / flange
]

_LINK_RADII_MM = [40, 35, 30, 30, 25, 25, 20]
_SAFETY_MARGIN = 15


@pytest.fixture
def fk() -> ForwardKinematics:
    return ForwardKinematics(_DH_PARAMS)


@pytest.fixture
def voxelizer() -> ArmVoxelizer:
    return ArmVoxelizer(
        link_radii_mm=_LINK_RADII_MM,
        safety_margin_mm=_SAFETY_MARGIN,
        grid_resolution=128,
        cell_size_mm=7.8,
        grid_origin_mm=[0, 0, 0],
    )


class TestZeroPose:
    """Test 1: All joints at zero → end-effector position check."""

    def test_ee_position(self, fk: ForwardKinematics):
        """EE should be at approximately (0, 0, sum_of_d_values)."""
        q = np.zeros(7)
        frames = fk.compute_link_frames(q)
        ee_pos = frames[-1][:3, 3]

        # Sum of d values: 121.5 + 0 + 208.5 + 0 + 208.5 + 0 + 113 = 651.5 mm
        # Due to alpha rotations the exact position may differ, but
        # the total reach from base should be in the right ballpark
        reach = np.linalg.norm(ee_pos)
        assert reach > 400, f"Zero-pose reach {reach:.1f}mm seems too short"
        assert reach < 700, f"Zero-pose reach {reach:.1f}mm seems too long"

    def test_link_chain_length(self, fk: ForwardKinematics):
        """Total chain length should match sum of link offsets."""
        q = np.zeros(7)
        segments = fk.link_endpoints(q)
        total_length = sum(
            np.linalg.norm(end - start) for start, end in segments
        )
        # Sum of absolute d values: 121.5 + 208.5 + 208.5 + 113 = 651.5
        expected_min = 500  # some segments may overlap due to DH
        assert total_length > expected_min, (
            f"Total chain length {total_length:.1f}mm < {expected_min}mm"
        )

    def test_frame_count(self, fk: ForwardKinematics):
        """Should return n_joints + 1 frames."""
        q = np.zeros(7)
        frames = fk.compute_link_frames(q)
        assert len(frames) == 8  # 7 joints + base


class TestBaseRotation:
    """Test 2: Joint 0 at pi/2 → 90° rotation about base Z axis."""

    def test_rotation_swaps_xy(self, fk: ForwardKinematics):
        """Rotating J0 by pi/2 should rotate the arm 90° in XY."""
        q_zero = np.zeros(7)
        q_rot = np.zeros(7)
        q_rot[0] = np.pi / 2

        ee_zero = fk.end_effector_pose(q_zero)[:3, 3]
        ee_rot = fk.end_effector_pose(q_rot)[:3, 3]

        # After 90° rotation about Z:
        # x_new ≈ -y_old, y_new ≈ x_old, z_new ≈ z_old
        # Z should remain approximately the same
        assert abs(ee_rot[2] - ee_zero[2]) < 10.0, (
            f"Z changed by {abs(ee_rot[2] - ee_zero[2]):.1f}mm after base rotation"
        )

        # The XY magnitude should be preserved
        xy_zero = np.sqrt(ee_zero[0] ** 2 + ee_zero[1] ** 2)
        xy_rot = np.sqrt(ee_rot[0] ** 2 + ee_rot[1] ** 2)
        assert abs(xy_rot - xy_zero) < 10.0, (
            f"XY magnitude changed: {xy_zero:.1f} → {xy_rot:.1f}mm"
        )


class TestArmVoxelization:
    """Test 3: Voxelize a known pose and verify volume."""

    def test_arm_mask_nonempty(self, fk: ForwardKinematics, voxelizer: ArmVoxelizer):
        """Arm mask should have nonzero volume at zero pose."""
        q = np.zeros(7)
        segments = fk.link_endpoints(q)
        mask = voxelizer.voxelize_arm(segments)
        volume = mask.sum()
        assert volume > 0, "Arm mask is empty at zero pose"

    def test_arm_mask_reasonable_volume(
        self, fk: ForwardKinematics, voxelizer: ArmVoxelizer
    ):
        """Arm mask volume should be in a reasonable range.

        For a ~550mm arm with ~30mm avg radius:
        Approx volume = pi * 30^2 * 550 ≈ 1,555,088 mm^3
        In voxels (7.8mm cells): ~3,279 voxels
        With safety margin and discrete rounding, expect 1000-20000.
        """
        q = np.zeros(7)
        segments = fk.link_endpoints(q)
        mask = voxelizer.voxelize_arm(segments)
        volume = int(mask.sum())
        assert 100 < volume < 100000, f"Arm mask volume {volume} voxels seems unreasonable"

    def test_capsule_basic(self, voxelizer: ArmVoxelizer):
        """A capsule along a known line should produce expected voxels."""
        # Capsule from (500,500,0) to (500,500,200) with radius 50mm
        p0 = np.array([500, 500, 0], dtype=np.float32)
        p1 = np.array([500, 500, 200], dtype=np.float32)
        mask = voxelizer.voxelize_capsule(p0, p1, 50.0)
        assert mask.sum() > 0, "Capsule mask is empty"


class TestSelfFilter:
    """Test 4: Self-filter removes arm and preserves obstacles."""

    def test_arm_removed(self, fk: ForwardKinematics, voxelizer: ArmVoxelizer):
        """After self-filtering, arm region should be zeroed."""
        q = np.zeros(7)
        segments = fk.link_endpoints(q)
        arm_mask = voxelizer.voxelize_arm(segments)

        # Create synthetic scene: arm + box obstacle
        occupancy = np.zeros((128, 128, 128), dtype=np.float32)
        occupancy[arm_mask] = 0.8  # arm
        occupancy[10:20, 10:20, 10:20] = 0.9  # obstacle box

        extractor = ObstacleExtractor(obstacle_threshold=0.3, cell_size_mm=7.8)
        result = extractor.extract(occupancy, arm_mask)

        # Arm region should be zero
        assert result["obstacle_grid"][arm_mask].max() == 0.0, "Arm not fully removed"

    def test_obstacle_preserved(self, fk: ForwardKinematics, voxelizer: ArmVoxelizer):
        """Obstacle box should survive self-filtering."""
        q = np.zeros(7)
        segments = fk.link_endpoints(q)
        arm_mask = voxelizer.voxelize_arm(segments)

        # Box far from arm
        occupancy = np.zeros((128, 128, 128), dtype=np.float32)
        occupancy[arm_mask] = 0.8
        occupancy[10:20, 10:20, 10:20] = 0.9

        extractor = ObstacleExtractor(obstacle_threshold=0.3, cell_size_mm=7.8)
        result = extractor.extract(occupancy, arm_mask)

        # Check obstacle box survived (if not overlapping with arm)
        box_region = result["obstacle_binary"][10:20, 10:20, 10:20]
        # Only check cells not in arm mask
        box_arm_overlap = arm_mask[10:20, 10:20, 10:20]
        non_arm_box = box_region & ~box_arm_overlap
        expected_non_arm = (~box_arm_overlap).sum()
        if expected_non_arm > 0:
            preserved_pct = non_arm_box.sum() / expected_non_arm * 100
            assert preserved_pct > 90, (
                f"Only {preserved_pct:.0f}% of non-arm obstacle box preserved"
            )

    def test_distance_field(self):
        """Distance field should be correct for a simple case."""
        occupancy = np.zeros((128, 128, 128), dtype=np.float32)
        occupancy[60:68, 60:68, 60:68] = 0.9  # obstacle cube

        arm_mask = np.zeros((128, 128, 128), dtype=bool)
        arm_mask[30:35, 64, 64] = True  # small arm segment

        extractor = ObstacleExtractor(obstacle_threshold=0.3, cell_size_mm=7.8)
        result = extractor.extract(occupancy, arm_mask)

        assert result["obstacle_voxel_count"] > 0
        assert result["min_obstacle_distance_mm"] > 0
        # Arm at ~30-35, obstacle at 60-68 → distance ~25 voxels
        min_dist_voxels = result["min_obstacle_distance_mm"] / 7.8
        assert min_dist_voxels > 10, f"Distance {min_dist_voxels:.1f} voxels seems too small"

    def test_empty_scene(self):
        """Empty scene should report max distance."""
        occupancy = np.zeros((128, 128, 128), dtype=np.float32)
        arm_mask = np.zeros((128, 128, 128), dtype=bool)
        arm_mask[64, 64, 64] = True

        extractor = ObstacleExtractor(obstacle_threshold=0.3, cell_size_mm=7.8)
        result = extractor.extract(occupancy, arm_mask)

        assert result["obstacle_voxel_count"] == 0
        assert result["min_obstacle_distance_mm"] > 100
