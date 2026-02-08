"""Tests for the calibration runner."""

import asyncio
import json
import tempfile
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.calibration.calibration_runner import (
    CALIBRATION_POSES,
    CalibrationRunner,
    CalibrationSession,
    CalibrationError,
    PoseCapture,
    _compute_increments,
    MAX_INCREMENT_DEG,
    JOINT_LIMITS_SAFE,
)


def run(coro):
    """Helper to run async tests without pytest-asyncio."""
    return asyncio.run(coro)


# ---- Pose list tests ----

def test_calibration_poses_has_20_entries():
    assert len(CALIBRATION_POSES) == 20


def test_all_poses_have_6_joints():
    for i, pose in enumerate(CALIBRATION_POSES):
        assert len(pose) == 6, f"Pose {i} has {len(pose)} joints"


def test_all_poses_within_safe_limits():
    for i, pose in enumerate(CALIBRATION_POSES):
        for j, angle in enumerate(pose):
            lo, hi = JOINT_LIMITS_SAFE[j]
            assert lo <= angle <= hi, (
                f"Pose {i}, J{j}: {angle}° outside safe [{lo}, {hi}]"
            )


# ---- Increment computation tests ----

def test_compute_increments_small_move():
    """Moves ≤10° should be a single step."""
    steps = _compute_increments(0, 5)
    assert steps == [5]


def test_compute_increments_exact_10():
    steps = _compute_increments(0, 10)
    assert steps == [10]


def test_compute_increments_large_move():
    """60° move should break into ≤10° steps."""
    steps = _compute_increments(0, 60)
    assert len(steps) >= 6
    # Each step should be ≤10° from previous
    prev = 0
    for s in steps:
        assert abs(s - prev) <= MAX_INCREMENT_DEG + 0.01
        prev = s
    assert steps[-1] == 60  # Final step is exact target


def test_compute_increments_negative():
    steps = _compute_increments(30, -30)
    assert len(steps) >= 6
    prev = 30
    for s in steps:
        assert abs(s - prev) <= MAX_INCREMENT_DEG + 0.01
        prev = s
    assert steps[-1] == -30


def test_compute_increments_no_move():
    """If already at target, should still return target (handled by caller)."""
    steps = _compute_increments(45, 45)
    assert steps == [45]


# ---- Session save/load roundtrip ----

def test_session_save_load_roundtrip():
    session = CalibrationSession(
        start_time=1000.0,
        end_time=1050.0,
        total_poses=2,
    )
    session.captures = [
        PoseCapture(
            pose_index=0,
            commanded_angles=(0, 0, 0, 0, 0, 0),
            actual_angles=[0.1, -0.2, 0.0, 0.0, 0.0, 0.0],
            cam0_jpeg=b"\xff\xd8test0",
            cam1_jpeg=b"\xff\xd8test1",
            timestamp=1001.0,
        ),
        PoseCapture(
            pose_index=1,
            commanded_angles=(30, 0, 0, 0, 0, 0),
            actual_angles=[29.8, 0.1, 0.0, 0.0, 0.0, 0.0],
            cam0_jpeg=b"img0",
            cam1_jpeg=b"img1",
            timestamp=1005.0,
        ),
    ]

    runner = CalibrationRunner()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    runner.save_session(session, path)
    loaded = runner.load_session(path)

    assert loaded.start_time == session.start_time
    assert loaded.end_time == session.end_time
    assert loaded.total_poses == session.total_poses
    assert len(loaded.captures) == 2
    assert loaded.captures[0].commanded_angles == (0, 0, 0, 0, 0, 0)
    assert loaded.captures[0].actual_angles == [0.1, -0.2, 0.0, 0.0, 0.0, 0.0]
    assert loaded.captures[0].cam0_jpeg == b"\xff\xd8test0"
    assert loaded.captures[1].cam1_jpeg == b"img1"


# ---- command_pose tests with mocked HTTP ----

def test_command_pose_incremental():
    """Verify command_pose sends ≤10° increments."""
    async def _test():
        runner = CalibrationRunner()
        runner.settle_time = 0  # Speed up test
        commands_sent = []

        async def mock_get_joints():
            # Simulate arm tracking: return the last commanded angle for each joint
            angles = [0.0] * 6
            for c in reversed(commands_sent):
                if angles[c["id"]] == 0.0:
                    angles[c["id"]] = c["angle"]
            return angles

        async def mock_set_joint(joint_id, angle):
            commands_sent.append({"id": joint_id, "angle": angle})
            return True

        runner.get_joint_angles = mock_get_joints
        runner.set_single_joint = mock_set_joint

        await runner.command_pose((60, 0, 0, 0, 0, 0))

        # J0 moved from 0 to 60: should have multiple commands
        j0_commands = [c for c in commands_sent if c["id"] == 0]
        assert len(j0_commands) >= 6  # 60/10 = 6 steps minimum

        # Each step ≤ 10°
        prev = 0
        for c in j0_commands:
            assert abs(c["angle"] - prev) <= MAX_INCREMENT_DEG + 0.01
            prev = c["angle"]
        assert j0_commands[-1]["angle"] == 60

    run(_test())


def test_command_pose_safety_limit_violation():
    """Pose outside safe limits should raise CalibrationError."""
    async def _test():
        runner = CalibrationRunner()
        runner.get_joint_angles = AsyncMock(return_value=[0.0] * 6)

        with pytest.raises(CalibrationError, match="outside safe limits"):
            await runner.command_pose((0, 90, 0, 0, 0, 0))  # J1 limit is ±85

    run(_test())


def test_command_pose_tracking_error_stops():
    """If feedback differs >15° from commanded, should raise."""
    async def _test():
        runner = CalibrationRunner()
        runner.settle_time = 0

        runner.get_joint_angles = AsyncMock(return_value=[0.0] * 6)
        runner.set_single_joint = AsyncMock(return_value=True)

        # Target 20° on J0, but arm reports 0° → 20° tracking error > 15°
        with pytest.raises(CalibrationError, match="tracking error"):
            await runner.command_pose((20, 0, 0, 0, 0, 0))

    run(_test())


# ---- Arm offline handling ----

def test_handles_arm_offline():
    """When arm is unreachable, should raise a clear error."""
    async def _test():
        runner = CalibrationRunner(arm_host="192.0.2.1", arm_port=1)

        with pytest.raises(Exception):
            await runner.get_joint_angles()

    run(_test())


# ---- Progress tracking ----

def test_progress_defaults():
    runner = CalibrationRunner()
    p = runner.progress
    assert p["running"] is False
    assert p["current_pose"] == -1
    assert p["total_poses"] == 20


# ---- Abort ----

def test_abort_sets_flag():
    runner = CalibrationRunner()
    assert runner._abort is False
    runner.abort()
    assert runner._abort is True
