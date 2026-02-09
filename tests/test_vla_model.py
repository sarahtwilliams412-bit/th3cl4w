"""Tests for VLA model backends."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.vla.vla_model import (
    Observation,
    ActionPlan,
    GeminiVLABackend,
    OctoVLABackend,
)


@pytest.fixture
def sample_observation():
    return Observation(
        cam0_jpeg=b"\xff\xd8\xff\xe0" + b"\x00" * 100,  # minimal JPEG header
        cam1_jpeg=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
        joints=[0.0, -20.0, 30.0, 0.0, 45.0, 0.0],
        gripper_mm=10.0,
        enabled=True,
    )


class TestObservation:
    def test_observation_defaults(self):
        obs = Observation(
            cam0_jpeg=b"test",
            cam1_jpeg=b"test",
            joints=[0.0] * 6,
            gripper_mm=0.0,
        )
        assert obs.enabled is True
        assert obs.timestamp > 0

    def test_observation_preserves_joints(self, sample_observation):
        assert sample_observation.joints[1] == -20.0
        assert sample_observation.gripper_mm == 10.0


class TestActionPlan:
    def test_empty_plan(self):
        plan = ActionPlan()
        assert not plan.is_done
        assert not plan.needs_verify
        assert plan.joint_actions == []
        assert plan.gripper_actions == []

    def test_done_detection(self):
        plan = ActionPlan(
            actions=[{"type": "done", "reason": "complete"}],
            phase="done",
        )
        assert plan.is_done

    def test_verify_detection(self):
        plan = ActionPlan(
            actions=[
                {"type": "joint", "id": 0, "delta": 5.0},
                {"type": "verify", "reason": "check"},
            ],
        )
        assert plan.needs_verify

    def test_action_filtering(self):
        plan = ActionPlan(
            actions=[
                {"type": "joint", "id": 0, "delta": 5.0},
                {"type": "gripper", "position_mm": 50.0},
                {"type": "joint", "id": 1, "delta": -3.0},
                {"type": "verify"},
            ],
        )
        assert len(plan.joint_actions) == 2
        assert len(plan.gripper_actions) == 1

    def test_error_plan(self):
        plan = ActionPlan(error="API timeout")
        assert plan.error == "API timeout"
        assert not plan.is_done


class TestGeminiVLABackend:
    def test_parse_valid_response(self):
        """Test parsing a well-formed JSON response."""
        # We need to test the _parse_response method directly
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            backend = GeminiVLABackend.__new__(GeminiVLABackend)
            backend.api_key = "test"

        response_json = json.dumps(
            {
                "reasoning": "Can is to the right, need to rotate base",
                "scene_description": "Red bull can visible on table",
                "gripper_position": {"cam1": {"u": 760, "v": 135}},
                "target_position": {"cam1": {"u": 900, "v": 600}},
                "actions": [
                    {"type": "joint", "id": 0, "delta": 8.0, "reason": "rotate right"},
                    {"type": "joint", "id": 1, "delta": -5.0, "reason": "lean forward"},
                    {"type": "verify", "reason": "check position"},
                ],
                "phase": "approach",
                "confidence": 0.75,
                "estimated_remaining_steps": 8,
            }
        )

        plan = backend._parse_response(response_json)
        assert plan.phase == "approach"
        assert plan.confidence == 0.75
        assert len(plan.actions) == 3
        assert plan.error is None

    def test_parse_with_markdown_fences(self):
        """Gemini sometimes wraps JSON in ```json ... ```."""
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            backend = GeminiVLABackend.__new__(GeminiVLABackend)

        response = (
            '```json\n{"reasoning": "test", "actions": [], "phase": "done", "confidence": 1.0}\n```'
        )
        plan = backend._parse_response(response)
        assert plan.phase == "done"
        assert plan.error is None

    def test_parse_invalid_json(self):
        """Invalid JSON should return error plan, not crash."""
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            backend = GeminiVLABackend.__new__(GeminiVLABackend)

        plan = backend._parse_response("this is not json at all")
        assert plan.error is not None
        assert "JSON parse error" in plan.error

    def test_parse_empty_response(self):
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            backend = GeminiVLABackend.__new__(GeminiVLABackend)

        plan = backend._parse_response("")
        assert plan.error is not None


class TestOctoVLABackend:
    def test_raises_not_implemented(self, sample_observation):
        backend = OctoVLABackend()
        assert backend.name == "octo-small"

    @pytest.mark.asyncio
    async def test_plan_raises(self, sample_observation):
        backend = OctoVLABackend()
        with pytest.raises(NotImplementedError):
            await backend.plan(sample_observation, "test task")

    @pytest.mark.asyncio
    async def test_verify_raises(self, sample_observation):
        backend = OctoVLABackend()
        with pytest.raises(NotImplementedError):
            await backend.verify(sample_observation, "test task", [])
