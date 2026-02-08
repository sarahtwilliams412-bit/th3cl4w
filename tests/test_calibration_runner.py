"""Tests for the calibration runner."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.calibration.calibration_runner import (
    CalibrationRunner,
    CalibrationSession,
    CalibrationError,
    PoseCapture,
    CALIBRATION_POSES,
    JOINT_LIMITS_SAFE,
    MAX_INCREMENT_DEG,
    _compute_increments,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_runner(**kwargs):
    return CalibrationRunner(arm_host='localhost', arm_port=8080, cam_host='localhost', cam_port=8081, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────

class TestComputeIncrements:
    def test_small_move(self):
        steps = _compute_increments(0.0, 5.0)
        assert steps == [5.0]

    def test_exact_10(self):
        steps = _compute_increments(0.0, 10.0)
        assert steps == [10.0]

    def test_large_move(self):
        steps = _compute_increments(0.0, 30.0)
        assert all(abs(steps[i] - steps[i-1] if i > 0 else steps[0]) <= MAX_INCREMENT_DEG + 0.1 for i in range(len(steps)))
        assert steps[-1] == 30.0

    def test_negative_move(self):
        steps = _compute_increments(10.0, -20.0)
        assert steps[-1] == -20.0
        assert len(steps) >= 3

    def test_no_move(self):
        steps = _compute_increments(5.0, 5.0)
        assert steps == [5.0]


class TestCalibrationPoses:
    def test_20_poses(self):
        assert len(CALIBRATION_POSES) == 20

    def test_all_within_limits(self):
        for pose in CALIBRATION_POSES:
            for j_id in range(6):
                lo, hi = JOINT_LIMITS_SAFE[j_id]
                assert lo <= pose[j_id] <= hi, f"Pose {pose} J{j_id}={pose[j_id]} outside [{lo},{hi}]"

    def test_home_is_first(self):
        assert CALIBRATION_POSES[0] == (0, 0, 0, 0, 0, 0)


class TestRunnerInit:
    def test_defaults(self):
        runner = _make_runner()
        assert runner.arm_base == "http://localhost:8080"
        assert runner.cam_base == "http://localhost:8081"
        assert runner.settle_time == 2.5

    def test_progress(self):
        runner = _make_runner()
        p = runner.progress
        assert p["running"] is False
        assert p["total_poses"] == 20


class TestCommandPose:
    def test_rejects_out_of_limits(self):
        runner = _make_runner()
        runner.get_joint_angles = AsyncMock(return_value=[0.0]*6)

        with pytest.raises(CalibrationError, match="outside safe limits"):
            asyncio.get_event_loop().run_until_complete(
                runner.command_pose((0, 0, 0, 0, 0, 200))  # J5=200 exceeds ±130
            )

    def test_abort_during_move(self):
        runner = _make_runner()
        runner.get_joint_angles = AsyncMock(return_value=[0.0]*6)
        runner.set_single_joint = AsyncMock(return_value=True)
        runner._abort = True

        with pytest.raises(CalibrationError, match="aborted"):
            asyncio.get_event_loop().run_until_complete(
                runner.command_pose((60, 0, 0, 0, 0, 0))
            )


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        runner = _make_runner()
        session = CalibrationSession(
            start_time=1000.0,
            end_time=1100.0,
            total_poses=2,
        )
        session.captures.append(PoseCapture(
            pose_index=0,
            commanded_angles=(0, 0, 0, 0, 0, 0),
            actual_angles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            cam0_jpeg=b"fake_cam0",
            cam1_jpeg=b"fake_cam1",
            timestamp=1050.0,
        ))

        path = str(tmp_path / "session.json")
        runner.save_session(session, path)

        loaded = runner.load_session(path)
        assert loaded.total_poses == 2
        assert len(loaded.captures) == 1
        assert loaded.captures[0].cam0_jpeg == b"fake_cam0"
        assert loaded.captures[0].actual_angles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
