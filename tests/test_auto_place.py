"""Tests for the autonomous place pipeline."""

import math
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.planning.auto_place import AutoPlace, AutoPlacePhase, AutoPlaceState, PlaceResult


class TestPlanPlaceJoints:
    """Test the geometric joint planner for place targets."""

    def test_reuses_autopick_planner(self):
        """plan_place_joints should produce same result as AutoPick.plan_joints."""
        from src.planning.auto_pick import AutoPick
        for x, y in [(100, 0), (50, 50), (200, -100), (0, 150)]:
            assert AutoPlace.plan_place_joints(x, y) == AutoPick.plan_joints(x, y)

    def test_j0_base_yaw(self):
        """J0 should be atan2(y, x) in degrees."""
        joints = AutoPlace.plan_place_joints(100, 100)
        assert abs(joints[0] - 45.0) < 0.1

    def test_j4_always_90(self):
        """J4 should always be 90° for top-down place."""
        for x, y in [(50, 0), (100, 50), (200, -100)]:
            joints = AutoPlace.plan_place_joints(x, y)
            assert joints[4] == 90.0

    def test_j1_scales_with_distance(self):
        """J1 should increase with horizontal distance."""
        j_near = AutoPlace.plan_place_joints(50, 0)
        j_far = AutoPlace.plan_place_joints(200, 0)
        assert j_near[1] < j_far[1]

    def test_origin_gives_zero_reach(self):
        """Target at origin should give J1=J2≈0."""
        joints = AutoPlace.plan_place_joints(0, 0)
        assert joints[1] == 0.0
        assert joints[2] == 0.0


class TestAutoPlaceState:
    """Test AutoPlace state management."""

    def test_initial_state_idle(self):
        """AutoPlace should start in IDLE phase."""
        ap = AutoPlace()
        assert ap.state.phase == AutoPlacePhase.IDLE
        assert not ap.running

    def test_get_status_structure(self):
        """get_status should return expected keys."""
        ap = AutoPlace()
        status = ap.get_status()
        assert "phase" in status
        assert "target_xy_mm" in status
        assert "planned_joints" in status
        assert "running" in status
        assert "log" in status
        assert status["phase"] == "idle"
        assert status["running"] is False

    def test_stop_sets_flag(self):
        """stop() should set the stop flag."""
        ap = AutoPlace()
        ap.stop()
        assert ap._stop_requested is True
        assert len(ap.state.log) == 1
        assert "Stop" in ap.state.log[0]


class TestAutoPlaceExecution:
    """Test the place execution pipeline with mocked arm ops."""

    @pytest.fixture
    def mock_ap(self):
        ap = AutoPlace(server_url="http://fake:8080")
        ap.ops = MagicMock()
        ap.ops.staged_reach = AsyncMock(return_value=MagicMock(success=True, final_joints=[0]*6, error=""))
        ap.ops._set_gripper = AsyncMock()
        ap.ops.lift_from_pick = AsyncMock(return_value=MagicMock(success=True, final_joints=[0]*6, error=""))
        ap.ops.retreat_home = AsyncMock(return_value=MagicMock(success=True, final_joints=[0]*6, error=""))
        return ap

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_ap):
        """Full place execution should succeed with mocked ops."""
        result = await mock_ap.execute(100.0, 50.0)
        assert result.success is True
        assert result.phase == AutoPlacePhase.DONE
        assert result.target_xy_mm == (100.0, 50.0)
        assert len(result.joints) == 6

    @pytest.mark.asyncio
    async def test_execute_calls_staged_reach_twice(self, mock_ap):
        """Should call staged_reach for transport (hover) and lower."""
        await mock_ap.execute(100.0, 0.0)
        assert mock_ap.ops.staged_reach.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_releases_gripper(self, mock_ap):
        """Should open gripper to release object."""
        await mock_ap.execute(100.0, 0.0)
        mock_ap.ops._set_gripper.assert_called_once_with(60.0)

    @pytest.mark.asyncio
    async def test_execute_retracts(self, mock_ap):
        """Should retract after releasing."""
        await mock_ap.execute(100.0, 0.0)
        mock_ap.ops.lift_from_pick.assert_called_once()
        mock_ap.ops.retreat_home.assert_called()

    @pytest.mark.asyncio
    async def test_transport_failure_raises(self, mock_ap):
        """If transport fails, should raise and record failure."""
        mock_ap.ops.staged_reach = AsyncMock(return_value=MagicMock(success=False, error="overcurrent"))
        with pytest.raises(RuntimeError, match="Transport failed"):
            await mock_ap.execute(100.0, 0.0)

    @pytest.mark.asyncio
    async def test_stop_during_execution(self, mock_ap):
        """Requesting stop should abort the pipeline."""
        original_reach = mock_ap.ops.staged_reach

        async def slow_reach(*a, **kw):
            mock_ap.stop()
            return MagicMock(success=True, final_joints=[0]*6, error="")

        mock_ap.ops.staged_reach = slow_reach
        result = await mock_ap._run(100.0, 0.0)
        assert result.phase == AutoPlacePhase.STOPPED

    @pytest.mark.asyncio
    async def test_start_while_running_raises(self, mock_ap):
        """Starting a second place while one is running should raise."""
        mock_ap._running = True
        with pytest.raises(RuntimeError, match="Place already in progress"):
            await mock_ap.start(100.0, 0.0)
