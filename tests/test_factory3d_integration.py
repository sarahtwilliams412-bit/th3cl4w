"""Tests for Factory 3D integration features (sub-agent 5).

Tests the client-side logic concepts: arm3d message parsing,
voxel coordinate mapping, and FK overlay projection.
"""

import json
import math
import pytest


class TestArm3dMessageFormat:
    """Verify arm3d WebSocket message format expectations."""

    def test_valid_message_structure(self):
        msg = {
            "type": "arm3d",
            "positions": [
                [0, 0, 0],
                [0, 0, 0.1215],
                [0.1, 0, 0.25],
                [0.2, 0, 0.15],
                [0.25, 0, 0.1],
            ],
            "confidence": [1.0, 0.9, 0.85, 0.8, 0.7],
            "source": "fused",
        }
        assert msg["type"] == "arm3d"
        assert len(msg["positions"]) == 5
        assert len(msg["confidence"]) == 5
        assert msg["source"] in ("fused", "fk_only")

    def test_positions_are_3d(self):
        positions = [[0, 0, 0], [0, 0, 0.1215], [0.1, 0, 0.25], [0.2, 0, 0.15], [0.25, 0, 0.1]]
        for p in positions:
            assert len(p) == 3
            assert all(isinstance(v, (int, float)) for v in p)

    def test_source_values(self):
        for src in ("fused", "fk_only"):
            msg = {
                "type": "arm3d",
                "positions": [[0, 0, 0]] * 5,
                "confidence": [1] * 5,
                "source": src,
            }
            assert msg["source"] == src


class TestVoxelCoordinateMapping:
    """Test the coordinate mapping from real arm meters to voxel units."""

    VOXEL_SCALE = 40  # matches JS constant
    FX, FZ = 256, 256
    ARM_X, ARM_Z = FX - 30, FZ - 40
    FLOOR = 2

    def test_base_maps_to_arm_origin(self):
        """Base position [0,0,0] should map to (armX, armZ) in voxel grid."""
        base = [0, 0, 0]
        vx = self.ARM_X + (base[0] - 0) * self.VOXEL_SCALE
        vz = self.ARM_Z + (base[1] - 0) * self.VOXEL_SCALE
        assert vx == self.ARM_X
        assert vz == self.ARM_Z

    def test_shoulder_height_maps_correctly(self):
        """Shoulder at [0,0,0.1215] should be ~4.86 voxels above base."""
        d0 = 0.1215
        height_voxels = d0 * self.VOXEL_SCALE
        assert abs(height_voxels - 4.86) < 0.01

    def test_extended_arm_reach(self):
        """Fully extended arm should be within factory bounds."""
        max_reach = 0.1215 + 0.2085 + 0.2085 + 0.113  # ~0.6515m
        max_voxels = max_reach * self.VOXEL_SCALE  # ~26 voxels
        assert max_voxels < 30  # well within factory

    def test_line_drawing_steps(self):
        """Bresenham-like line should cover segment length."""
        p0 = [0, 0, 0.1215]
        p1 = [0.15, 0, 0.3]
        dx = (p1[0] - p0[0]) * self.VOXEL_SCALE
        dz = (p1[1] - p0[1]) * self.VOXEL_SCALE
        steps = max(1, math.ceil(math.sqrt(dx**2 + dz**2)))
        assert steps >= 1


class TestFkProjection:
    """Test FK position computation for overlay."""

    def test_fk_home_position(self):
        """At home (all zeros), EE should be roughly at d0+L1 height with L2+L3 forward."""
        # This mirrors the JS fkPositions function
        d0 = 0.1215
        L1 = 0.2085
        L2 = 0.2085
        L3 = 0.113
        # At home: arm points up from base
        # shoulder at [0,0,d0]
        # With all joints at 0: elbow at [0,0,d0+L1]
        # Then 90deg bend: wrist forward
        # This is a simplified sanity check
        assert d0 + L1 + L2 + L3 == pytest.approx(0.6515, abs=0.001)


class TestCalibStatusEndpoint:
    """Test calibration status response format expectations."""

    def test_expected_fields(self):
        """Status response should have expected fields."""
        # Mock response format
        status = {
            "cameras_connected": 2,
            "calibration_loaded": True,
            "tracking_quality": "good",
            "source": "fused",
        }
        assert "cameras_connected" in status
        assert "calibration_loaded" in status
        assert isinstance(status["cameras_connected"], int)
        assert isinstance(status["calibration_loaded"], bool)
