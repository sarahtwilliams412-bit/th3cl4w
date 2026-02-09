"""Tests for fk_engine — verifying Python FK matches JS fkPositions() exactly."""

import math
import pytest
import numpy as np
from src.vision.fk_engine import fk_positions, project_to_camera_pinhole, D0, L1, L2, L3


def approx(a, b, tol=1e-6):
    """Assert two 3D points are approximately equal."""
    for i in range(3):
        assert abs(a[i] - b[i]) < tol, f"axis {i}: {a[i]} != {b[i]} (diff={abs(a[i]-b[i])})"


class TestZeroPose:
    """Home position (all joints 0°) = vertical arm pointing straight up."""

    def test_positions(self):
        pts = fk_positions([0, 0, 0, 0, 0, 0])
        base, shoulder, elbow, wrist, ee = pts

        approx(base, [0, 0, 0])
        approx(shoulder, [0, 0, D0])
        # At home, Ry(0) means Z stays Z, so elbow = shoulder + [0,0,L1]
        approx(elbow, [0, 0, D0 + L1])
        # Then Ry(pi/2) rotates Z->X, so wrist = elbow + [L2, 0, 0]
        # Wait — let's trace: R = Rz(0)@Ry(0) = I, then R = I @ Ry(pi/2)
        # Ry(pi/2) @ [0,0,L2] = [L2*sin(pi/2), 0, L2*cos(pi/2)] = [L2, 0, 0]
        # Hmm but the JS says 90° home bend. Let me verify:
        # Ry(a) = [[cos,-,sin],[0,1,0],[-sin,0,cos]]
        # Ry(pi/2)@[0,0,L2] = [sin(pi/2)*L2, 0, cos(pi/2)*L2] = [L2, 0, ~0]
        # So wrist is at elbow + [L2, 0, 0] = [L2, 0, D0+L1]
        approx(wrist, [L2, 0, D0 + L1])
        # Then Ry(0)@[0,0,L3] with current R which includes Ry(pi/2):
        # same rotation applied: ee = wrist + R@[0,0,L3] = wrist + [L3, 0, 0]
        approx(ee, [L2 + L3, 0, D0 + L1])

    def test_total_height_at_home(self):
        """At home, the highest point should be elbow (shoulder + L1)."""
        pts = fk_positions([0, 0, 0, 0, 0, 0])
        assert pts[2][2] == pytest.approx(D0 + L1, abs=1e-6)


class TestSingleJointMotion:
    """Test each joint individually at ±45°."""

    def test_j0_plus45_yaw(self):
        """J0 +45° = base yaw, arm should swing in XY plane."""
        pts = fk_positions([45, 0, 0, 0, 0, 0])
        # Elbow should still be at height D0+L1, but rotated in XY
        assert pts[2][2] == pytest.approx(D0 + L1, abs=1e-6)
        # After Rz(45°), the X component of wrist offset rotates
        # Wrist offset was [L2, 0, 0] at home; after Rz(45°): [L2*cos45, L2*sin45, 0]
        c45 = math.cos(math.radians(45))
        s45 = math.sin(math.radians(45))
        approx(pts[3], [L2 * c45, L2 * s45, D0 + L1])

    def test_j0_minus45_yaw(self):
        pts = fk_positions([-45, 0, 0, 0, 0, 0])
        c45 = math.cos(math.radians(45))
        s45 = math.sin(math.radians(45))
        approx(pts[3], [L2 * c45, -L2 * s45, D0 + L1])

    def test_j1_plus45_pitch(self):
        """J1 +45° = shoulder pitch."""
        pts = fk_positions([0, 45, 0, 0, 0, 0])
        # R = Ry(45°), so elbow = shoulder + Ry(45°)@[0,0,L1]
        # = [L1*sin(45°), 0, D0 + L1*cos(45°)]
        s45 = math.sin(math.radians(45))
        c45 = math.cos(math.radians(45))
        approx(pts[2], [L1 * s45, 0, D0 + L1 * c45])

    def test_j1_minus45_pitch(self):
        pts = fk_positions([0, -45, 0, 0, 0, 0])
        s45 = math.sin(math.radians(45))
        c45 = math.cos(math.radians(45))
        approx(pts[2], [-L1 * s45, 0, D0 + L1 * c45])

    def test_j2_plus45_elbow(self):
        """J2 +45° = elbow extends forward more."""
        pts = fk_positions([0, 0, 45, 0, 0, 0])
        # R after shoulder = I, then R = Ry(90° + 45°) = Ry(135°)
        # wrist = elbow + Ry(135°)@[0,0,L2]
        a = math.radians(135)
        dx = L2 * math.sin(a)
        dz = L2 * math.cos(a)
        approx(pts[3], [dx, 0, D0 + L1 + dz])

    def test_j2_minus45_elbow(self):
        pts = fk_positions([0, 0, -45, 0, 0, 0])
        a = math.radians(45)  # 90 - 45
        dx = L2 * math.sin(a)
        dz = L2 * math.cos(a)
        approx(pts[3], [dx, 0, D0 + L1 + dz])

    def test_j4_plus45_wrist(self):
        """J4 +45° = wrist pitch."""
        pts = fk_positions([0, 0, 0, 0, 45, 0])
        # At home (j2=0), R includes Ry(pi/2), then Ry(45°) extra
        # Total rotation for EE offset: Ry(90°+45°) = Ry(135°)
        a = math.radians(135)
        wrist = pts[3]
        ee_expected = [wrist[0] + L3 * math.sin(a), wrist[1], wrist[2] + L3 * math.cos(a)]
        approx(pts[4], ee_expected)

    def test_j4_minus45_wrist(self):
        pts = fk_positions([0, 0, 0, 0, -45, 0])
        a = math.radians(45)  # 90 - 45
        wrist = pts[3]
        ee_expected = [wrist[0] + L3 * math.sin(a), wrist[1], wrist[2] + L3 * math.cos(a)]
        approx(pts[4], ee_expected)


class TestReadyPose:
    """A typical 'ready' pose with multiple joints active."""

    def test_combined_pose(self):
        """Test J0=30, J1=-45, J2=45 combined."""
        pts = fk_positions([30, -45, 45, 0, 0, 0])

        # Verify it's a valid kinematic chain (distances between joints match link lengths)
        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        assert dist(pts[0], pts[1]) == pytest.approx(D0, abs=1e-6)
        assert dist(pts[1], pts[2]) == pytest.approx(L1, abs=1e-6)
        assert dist(pts[2], pts[3]) == pytest.approx(L2, abs=1e-6)
        assert dist(pts[3], pts[4]) == pytest.approx(L3, abs=1e-6)


class TestLinkLengths:
    """Link lengths should be preserved for any joint configuration."""

    @pytest.mark.parametrize(
        "joints",
        [
            [0, 0, 0, 0, 0, 0],
            [45, -30, 60, 0, 20, 0],
            [-90, 45, -45, 30, -30, 15],
            [135, 85, 135, 135, 85, 135],
        ],
    )
    def test_link_distances(self, joints):
        pts = fk_positions(joints)

        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        assert dist(pts[0], pts[1]) == pytest.approx(D0, abs=1e-6)
        assert dist(pts[1], pts[2]) == pytest.approx(L1, abs=1e-6)
        assert dist(pts[2], pts[3]) == pytest.approx(L2, abs=1e-6)
        assert dist(pts[3], pts[4]) == pytest.approx(L3, abs=1e-6)


class TestJ3Roll:
    """J3 (forearm roll) should not change wrist position, only EE direction."""

    def test_j3_doesnt_move_wrist(self):
        pts_0 = fk_positions([0, 0, 0, 0, 0, 0])
        pts_90 = fk_positions([0, 0, 0, 90, 0, 0])
        approx(pts_0[3], pts_90[3])  # wrist unchanged

    def test_j3_changes_ee_with_j4(self):
        """J3 only matters when combined with J4 pitch."""
        pts_a = fk_positions([0, 0, 0, 0, 45, 0])
        pts_b = fk_positions([0, 0, 0, 90, 45, 0])
        # EE should differ
        diff = sum((pts_a[4][i] - pts_b[4][i]) ** 2 for i in range(3))
        assert diff > 0.001  # meaningfully different


class TestProjection:
    """Test pinhole projection."""

    def test_simple_projection(self):
        """Point at [0, 0, 1] with identity camera should project to (cx, cy)."""
        pts = project_to_camera_pinhole(
            [[0, 0, 1]],
            fx=500,
            fy=500,
            cx=320,
            cy=240,
            rvec=[0, 0, 0],
            tvec=[0, 0, 0],
        )
        assert pts[0][0] == pytest.approx(320, abs=1e-3)
        assert pts[0][1] == pytest.approx(240, abs=1e-3)

    def test_behind_camera(self):
        """Point behind camera returns None."""
        pts = project_to_camera_pinhole(
            [[0, 0, -1]],
            fx=500,
            fy=500,
            cx=320,
            cy=240,
            rvec=[0, 0, 0],
            tvec=[0, 0, 0],
        )
        assert pts[0] is None


class TestShortInput:
    """FK should handle fewer than 6 joints gracefully."""

    def test_empty(self):
        pts = fk_positions([])
        # Same as all zeros
        pts_zero = fk_positions([0, 0, 0, 0, 0, 0])
        for i in range(5):
            approx(pts[i], pts_zero[i])

    def test_three_joints(self):
        pts = fk_positions([30, -20, 10])
        pts_full = fk_positions([30, -20, 10, 0, 0, 0])
        for i in range(5):
            approx(pts[i], pts_full[i])
