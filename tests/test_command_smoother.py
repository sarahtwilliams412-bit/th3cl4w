"""Tests for the CommandSmoother."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web.command_smoother import CommandSmoother


@pytest.fixture
def mock_arm():
    arm = MagicMock()
    arm.get_joint_angles.return_value = [0.0] * 6
    arm.get_gripper_position.return_value = 0.0
    arm.set_joint.return_value = True
    arm.set_all_joints.return_value = True
    arm.set_gripper.return_value = True
    return arm


@pytest.fixture
def smoother(mock_arm):
    return CommandSmoother(mock_arm, rate_hz=10.0, smoothing_factor=0.5, max_step_deg=15.0)


class TestCommandSmootherInit:
    def test_initial_state(self, smoother):
        assert not smoother.running
        assert smoother._current == [0.0] * 6
        assert smoother._ticks == 0

    def test_sync_positions(self, smoother, mock_arm):
        mock_arm.get_joint_angles.return_value = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        mock_arm.get_gripper_position.return_value = 25.0
        smoother.sync_current_positions()
        assert smoother._current == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        assert smoother._current_gripper == 25.0


class TestSmootherTick:
    def test_no_target_no_command(self, smoother, mock_arm):
        """No targets set = no commands sent."""
        smoother._tick()
        mock_arm.set_joint.assert_not_called()
        mock_arm.set_all_joints.assert_not_called()

    def test_single_joint_interpolates(self, smoother, mock_arm):
        """Setting a target should NOT jump immediately to it."""
        smoother.set_joint_target(0, 100.0)
        smoother._tick()
        # With alpha=0.5, step=50.0 but max_step=15.0 caps it
        assert smoother._current[0] == pytest.approx(15.0, abs=0.1)
        mock_arm.set_joint.assert_called_once_with(0, pytest.approx(15.0, abs=0.1))

    def test_converges_to_target(self, smoother, mock_arm):
        """After enough ticks, should reach the target."""
        smoother.set_joint_target(2, 60.0)
        for _ in range(50):
            smoother._tick()
        assert smoother._current[2] == pytest.approx(60.0, abs=0.1)

    def test_max_step_limits_velocity(self, mock_arm):
        """Max step should cap movement per tick."""
        s = CommandSmoother(mock_arm, smoothing_factor=1.0, max_step_deg=5.0)
        s.set_joint_target(0, 100.0)
        s._tick()
        # alpha=1.0 wants to jump 100°, but max_step=5° caps it
        assert s._current[0] == pytest.approx(5.0, abs=0.1)

    def test_multiple_joints_batch(self, smoother, mock_arm):
        """3+ joints should send set_all_joints instead of individual."""
        smoother.set_joint_target(0, 10.0)
        smoother.set_joint_target(1, 20.0)
        smoother.set_joint_target(2, 30.0)
        smoother._tick()
        mock_arm.set_all_joints.assert_called_once()
        mock_arm.set_joint.assert_not_called()

    def test_gripper_smoothing(self, smoother, mock_arm):
        smoother.set_gripper_target(40.0)
        smoother._tick()
        # alpha=0.5 wants 20.0, but max_gripper_step=5.0 caps it
        assert smoother._current_gripper == pytest.approx(5.0, abs=0.5)
        mock_arm.set_gripper.assert_called_once()

    def test_dirty_cleared_on_arrival(self, smoother, mock_arm):
        """Once target is reached, joint should stop being dirty."""
        smoother.set_joint_target(0, 0.01)  # very close to current (0.0)
        smoother._tick()
        assert 0 not in smoother._dirty_joints

    def test_set_all_joints_target(self, smoother, mock_arm):
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        smoother.set_all_joints_target(angles)
        assert smoother._dirty_joints == {0, 1, 2, 3, 4, 5}

    def test_target_update_mid_motion(self, smoother, mock_arm):
        """Changing target mid-motion should redirect smoothly."""
        smoother.set_joint_target(0, 100.0)
        smoother._tick()  # moves toward 100
        pos_after_first = smoother._current[0]
        smoother.set_joint_target(0, -50.0)  # reverse direction
        smoother._tick()
        # Should now be moving toward -50 from wherever we were
        assert smoother._current[0] < pos_after_first

    def test_stats(self, smoother, mock_arm):
        smoother.set_joint_target(0, 10.0)
        smoother._tick()
        s = smoother.stats
        assert s["ticks"] == 0  # ticks incremented by _loop, not _tick
        assert s["commands_sent"] == 1


class TestSmootherAsync:
    def test_start_stop(self, smoother):
        async def _run():
            await smoother.start()
            assert smoother.running
            await asyncio.sleep(0.15)
            await smoother.stop()
            assert not smoother.running
        asyncio.run(_run())

    def test_smoothing_over_time(self, mock_arm):
        async def _run():
            s = CommandSmoother(mock_arm, rate_hz=100, smoothing_factor=0.3, max_step_deg=50.0)
            await s.start()
            s.set_joint_target(0, 90.0)
            await asyncio.sleep(0.3)
            await s.stop()
            assert s._current[0] == pytest.approx(90.0, abs=1.0)
        asyncio.run(_run())


class TestPassthrough:
    def test_alpha_one_is_passthrough(self, mock_arm):
        """With alpha=1.0 and high max_step, acts as passthrough."""
        s = CommandSmoother(mock_arm, smoothing_factor=1.0, max_step_deg=999.0)
        s.set_joint_target(0, 45.0)
        s._tick()
        assert s._current[0] == pytest.approx(45.0, abs=0.1)
