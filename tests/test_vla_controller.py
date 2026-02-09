"""Tests for VLA controller."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.vla.vla_controller import VLAController, TaskState, TaskResult
from src.vla.vla_model import VLABackend, Observation, ActionPlan


class MockVLABackend(VLABackend):
    """Mock backend for testing the controller logic."""

    def __init__(self, plans=None):
        self._plans = plans or []
        self._call_count = 0

    @property
    def name(self):
        return "mock"

    async def plan(self, observation, task, history=None):
        return self._next_plan()

    async def verify(self, observation, task, actions_taken):
        return self._next_plan()

    def _next_plan(self):
        if self._call_count < len(self._plans):
            plan = self._plans[self._call_count]
            self._call_count += 1
            return plan
        # Default: task done
        return ActionPlan(
            phase="done",
            actions=[{"type": "done", "reason": "mock complete"}],
            confidence=1.0,
        )


class TestVLAControllerState:
    def test_initial_state(self):
        ctrl = VLAController()
        assert ctrl.state == TaskState.IDLE
        assert not ctrl.is_busy

    def test_abort(self):
        ctrl = VLAController()
        ctrl.abort()
        assert ctrl._abort is True

    def test_status(self):
        ctrl = VLAController(backend=MockVLABackend())
        status = ctrl.get_status()
        assert status["state"] == "idle"
        assert status["backend"] == "mock"


class TestVLAControllerExecution:
    @pytest.mark.asyncio
    async def test_immediate_done(self):
        """Model says done on first call → task succeeds immediately."""
        backend = MockVLABackend(plans=[
            ActionPlan(
                phase="done",
                actions=[{"type": "done", "reason": "already at target"}],
                confidence=1.0,
            ),
        ])

        ctrl = VLAController(backend=backend)

        # Mock the observation and arm API calls
        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=True,
        )

        with patch.object(ctrl, '_observe', return_value=mock_obs):
            result = await ctrl.execute("test task")

        assert result.success
        assert result.task == "test task"

    @pytest.mark.asyncio
    async def test_plan_then_done(self):
        """Model plans actions, then says done after verify."""
        backend = MockVLABackend(plans=[
            ActionPlan(
                phase="approach",
                actions=[
                    {"type": "joint", "id": 0, "delta": 5.0, "reason": "rotate"},
                    {"type": "verify", "reason": "check"},
                ],
                confidence=0.7,
            ),
            ActionPlan(
                phase="done",
                actions=[{"type": "done", "reason": "aligned"}],
                confidence=0.95,
            ),
        ])

        ctrl = VLAController(backend=backend)

        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=True,
        )

        with patch.object(ctrl, '_observe', return_value=mock_obs), \
             patch.object(ctrl, '_execute_joint', return_value=True):
            result = await ctrl.execute("pick up can")

        assert result.success
        assert result.actions_executed == 1
        assert result.observations_made == 2

    @pytest.mark.asyncio
    async def test_arm_not_enabled(self):
        """Should fail immediately if arm is not enabled."""
        backend = MockVLABackend()
        ctrl = VLAController(backend=backend)

        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=False,  # NOT enabled
        )

        with patch.object(ctrl, '_observe', return_value=mock_obs):
            result = await ctrl.execute("test")

        assert not result.success
        assert "not enabled" in result.error

    @pytest.mark.asyncio
    async def test_max_steps_reached(self):
        """Should fail after hitting max steps."""
        # Backend always returns more actions, never done
        plans = [
            ActionPlan(
                phase="approach",
                actions=[
                    {"type": "joint", "id": 0, "delta": 5.0, "reason": "keep going"},
                ],
                confidence=0.5,
            )
        ] * 50

        backend = MockVLABackend(plans=plans)
        ctrl = VLAController(backend=backend, max_steps=3)

        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=True,
        )

        with patch.object(ctrl, '_observe', return_value=mock_obs), \
             patch.object(ctrl, '_execute_joint', return_value=True):
            result = await ctrl.execute("impossible task")

        assert not result.success
        assert result.actions_executed <= 3

    @pytest.mark.asyncio
    async def test_abort_during_execution(self):
        """Should abort cleanly when abort() is called."""
        backend = MockVLABackend(plans=[
            ActionPlan(
                phase="approach",
                actions=[
                    {"type": "joint", "id": 0, "delta": 5.0},
                    {"type": "joint", "id": 1, "delta": -5.0},
                    {"type": "joint", "id": 2, "delta": 5.0},
                ],
                confidence=0.5,
            ),
        ])

        ctrl = VLAController(backend=backend)

        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=True,
        )

        async def slow_execute(jid, angle):
            ctrl.abort()  # Abort after first action
            return True

        with patch.object(ctrl, '_observe', return_value=mock_obs), \
             patch.object(ctrl, '_execute_joint', side_effect=slow_execute):
            result = await ctrl.execute("test")

        assert not result.success
        assert result.final_phase == "aborted" or "aborted" in result.message.lower()

    @pytest.mark.asyncio
    async def test_observation_failure(self):
        """Should handle observation errors gracefully."""
        backend = MockVLABackend()
        ctrl = VLAController(backend=backend)

        with patch.object(ctrl, '_observe', side_effect=Exception("Camera offline")):
            result = await ctrl.execute("test")

        assert not result.success
        assert "Observation failed" in result.error

    @pytest.mark.asyncio
    async def test_gripper_action(self):
        """Should execute gripper actions."""
        backend = MockVLABackend(plans=[
            ActionPlan(
                phase="grasp",
                actions=[
                    {"type": "gripper", "position_mm": 55.0, "reason": "open"},
                    {"type": "verify", "reason": "check"},
                ],
                confidence=0.9,
            ),
            ActionPlan(
                phase="done",
                actions=[{"type": "done", "reason": "grasped"}],
                confidence=1.0,
            ),
        ])

        ctrl = VLAController(backend=backend)

        mock_obs = Observation(
            cam0_jpeg=b"fake", cam1_jpeg=b"fake",
            joints=[0.0] * 6, gripper_mm=0.0, enabled=True,
        )

        with patch.object(ctrl, '_observe', return_value=mock_obs), \
             patch.object(ctrl, '_execute_gripper', return_value=True):
            result = await ctrl.execute("grasp object")

        assert result.success
        assert result.actions_executed == 1
