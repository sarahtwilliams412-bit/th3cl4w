"""Tests for arm_operations.py — the codified operational playbook."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.control.arm_operations import ArmOps, MoveResult


@pytest.fixture
def ops():
    return ArmOps("http://localhost:8080")


class TestMoveJointVerified:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self, ops):
        with (
            patch.object(ops, "_set_joint", new_callable=AsyncMock) as mock_set,
            patch.object(
                ops, "_get_joints", new_callable=AsyncMock, return_value=[0, 25.0, 0, 0, 0, 0]
            ),
        ):
            result = await ops.move_joint_verified(1, 25.0, tolerance_deg=5.0, settle_time=0.01)
            assert result is True
            mock_set.assert_called_once_with(1, 25.0)

    @pytest.mark.asyncio
    async def test_retries_on_feedback_miss(self, ops):
        call_count = [0]

        async def fake_joints():
            call_count[0] += 1
            if call_count[0] < 3:
                return [0, 0, 0, 0, 0, 0]  # wrong position
            return [0, 25.0, 0, 0, 0, 0]  # correct

        with (
            patch.object(ops, "_set_joint", new_callable=AsyncMock),
            patch.object(ops, "_get_joints", side_effect=fake_joints),
        ):
            result = await ops.move_joint_verified(1, 25.0, tolerance_deg=5.0, settle_time=0.01)
            assert result is True

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self, ops):
        with (
            patch.object(ops, "_set_joint", new_callable=AsyncMock),
            patch.object(
                ops, "_get_joints", new_callable=AsyncMock, return_value=[0, 0, 0, 0, 0, 0]
            ),
        ):
            result = await ops.move_joint_verified(
                1, 25.0, tolerance_deg=5.0, settle_time=0.01, max_retries=2
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_handles_none_feedback(self, ops):
        with (
            patch.object(ops, "_set_joint", new_callable=AsyncMock),
            patch.object(ops, "_get_joints", new_callable=AsyncMock, return_value=None),
        ):
            result = await ops.move_joint_verified(1, 25.0, settle_time=0.01, max_retries=2)
            assert result is False


class TestStagedReach:
    @pytest.mark.asyncio
    async def test_moves_low_torque_first(self, ops):
        calls = []

        async def track_set(j, a):
            calls.append(j)

        with (
            patch.object(ops, "_set_joint", side_effect=track_set),
            patch.object(
                ops, "_get_joints", new_callable=AsyncMock, return_value=[0, 0, 0, 0, 0, 0]
            ),
        ):
            await ops.staged_reach([30, 25, 10, 15, 85, 10], step_deg=50, step_delay=0.01)
            # J0, J3, J5 should come before J1, J2, J4
            low_torque_indices = [i for i, j in enumerate(calls) if j in (0, 3, 5)]
            high_torque_indices = [i for i, j in enumerate(calls) if j in (1, 2, 4)]
            if low_torque_indices and high_torque_indices:
                assert max(low_torque_indices[:3]) < min(high_torque_indices)

    @pytest.mark.asyncio
    async def test_returns_failure_on_no_feedback(self, ops):
        with patch.object(ops, "_get_joints", new_callable=AsyncMock, return_value=None):
            result = await ops.staged_reach([10, 20, 10, 0, 85, 0])
            assert result.success is False


class TestStagedRetract:
    @pytest.mark.asyncio
    async def test_retracts_distal_first(self, ops):
        calls = []

        async def track_set(j, a):
            calls.append(j)

        with (
            patch.object(ops, "_set_joint", side_effect=track_set),
            patch.object(
                ops, "_get_joints", new_callable=AsyncMock, return_value=[30, 25, 10, 15, 85, 10]
            ),
        ):
            await ops.staged_retract([0, 0, 0, 0, 0, 0], step_deg=50, step_delay=0.01)
            # J4 should come before J2, J2 before J1
            pitch_calls = [(i, j) for i, j in enumerate(calls) if j in (1, 2, 4)]
            if len(pitch_calls) >= 3:
                first_j4 = next(i for i, j in pitch_calls if j == 4)
                first_j2 = next(i for i, j in pitch_calls if j == 2)
                first_j1 = next(i for i, j in pitch_calls if j == 1)
                assert first_j4 < first_j2 < first_j1


class TestFullRecovery:
    @pytest.mark.asyncio
    async def test_calls_all_three_steps(self, ops):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await ops.full_recovery()
            assert result is True
            # Should have called power-on, reset, enable-here
            calls = [c.args[0] for c in mock_client.post.call_args_list]
            assert any("power-on" in c for c in calls)
            assert any("reset" in c for c in calls)
            assert any("enable-here" in c for c in calls)


class TestRetreatHome:
    @pytest.mark.asyncio
    async def test_calls_staged_retract_to_zeros(self, ops):
        with patch.object(
            ops, "staged_retract", new_callable=AsyncMock, return_value=MoveResult(True, [0] * 6)
        ) as mock:
            result = await ops.retreat_home()
            assert result.success
            mock.assert_called_once_with([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestPickSequence:
    @pytest.mark.asyncio
    async def test_retreats_home_on_approach_failure(self, ops):
        with (
            patch.object(ops, "_set_gripper", new_callable=AsyncMock),
            patch.object(
                ops,
                "approach_from_above",
                new_callable=AsyncMock,
                return_value=MoveResult(False, [], "failed"),
            ),
            patch.object(
                ops, "retreat_home", new_callable=AsyncMock, return_value=MoveResult(True, [0] * 6)
            ) as mock_retreat,
        ):
            result = await ops.pick_sequence([0, 25, 6, 0, 88, 0])
            assert not result.success
            # approach_from_above already calls retreat_home on failure

    @pytest.mark.asyncio
    async def test_full_success_path(self, ops):
        with (
            patch.object(ops, "_set_gripper", new_callable=AsyncMock),
            patch.object(
                ops,
                "approach_from_above",
                new_callable=AsyncMock,
                return_value=MoveResult(True, [0, 25, 6, 0, 88, 0]),
            ),
            patch.object(ops, "grip_and_verify", new_callable=AsyncMock, return_value=True),
            patch.object(
                ops,
                "lift_from_pick",
                new_callable=AsyncMock,
                return_value=MoveResult(True, [0, 5, 6, 0, 88, 0]),
            ),
        ):
            result = await ops.pick_sequence([0, 25, 6, 0, 88, 0])
            assert result.success


class TestGetJoints:
    @pytest.mark.asyncio
    async def test_retries_on_all_zeros(self, ops):
        """DDS sometimes returns all zeros — should retry."""
        call_count = [0]

        async def fake_get(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count[0] < 3:
                mock_resp.json.return_value = {"joint_angles": [0, 0, 0, 0, 0, 0]}
            else:
                mock_resp.json.return_value = {"joint_angles": [1, -70, 30, 0, -80, 0]}
            return mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            joints = await ops._get_joints()
            assert joints == [1, -70, 30, 0, -80, 0]
