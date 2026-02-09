"""VLA model backends — abstract interface + implementations.

Supports multiple backends:
- GeminiVLABackend: Uses Gemini Flash multimodal API (works NOW)
- OctoVLABackend: Placeholder for Octo-Small fine-tuned model (future)
"""

import abc
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Sensor observation from the arm."""

    cam0_jpeg: bytes  # Front camera JPEG
    cam1_jpeg: bytes  # Overhead camera JPEG
    joints: List[float]  # 6 joint angles in degrees
    gripper_mm: float  # Gripper opening in mm
    enabled: bool = True
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ActionPlan:
    """Planned actions from the model."""

    reasoning: str = ""
    scene_description: str = ""
    gripper_position: Optional[Dict] = None  # pixel coords per camera
    target_position: Optional[Dict] = None  # pixel coords per camera
    actions: List[Dict[str, Any]] = field(default_factory=list)
    phase: str = "unknown"
    confidence: float = 0.0
    estimated_remaining_steps: int = -1
    raw_response: str = ""
    inference_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_done(self) -> bool:
        return self.phase == "done" or any(a.get("type") == "done" for a in self.actions)

    @property
    def needs_verify(self) -> bool:
        return any(a.get("type") == "verify" for a in self.actions)

    @property
    def joint_actions(self) -> List[Dict]:
        return [a for a in self.actions if a.get("type") == "joint"]

    @property
    def gripper_actions(self) -> List[Dict]:
        return [a for a in self.actions if a.get("type") == "gripper"]


class VLABackend(abc.ABC):
    """Abstract base class for VLA model backends."""

    @abc.abstractmethod
    async def plan(
        self,
        observation: Observation,
        task: str,
        history: Optional[List[str]] = None,
    ) -> ActionPlan:
        """Given observation + task, produce an action plan."""
        ...

    @abc.abstractmethod
    async def verify(
        self,
        observation: Observation,
        task: str,
        actions_taken: List[str],
    ) -> ActionPlan:
        """After executing actions, verify and plan next steps."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class GeminiVLABackend(VLABackend):
    """Gemini Flash multimodal as VLA backbone.

    Sends camera images + joint state + task description to Gemini,
    gets back structured JSON action plans.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for GeminiVLABackend")

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(
            model_name,
            system_instruction=self._get_system_prompt(),
        )
        self._model_name = model_name
        logger.info("GeminiVLABackend initialized with model=%s", model_name)

    def _get_system_prompt(self) -> str:
        from src.vla.prompts import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    def _build_observe_prompt(
        self,
        obs: Observation,
        task: str,
        history: Optional[List[str]] = None,
    ) -> str:
        from src.vla.prompts import OBSERVE_TEMPLATE

        hist_str = "\n".join(history[-10:]) if history else "None (first step)"
        return OBSERVE_TEMPLATE.format(
            j0=obs.joints[0],
            j1=obs.joints[1],
            j2=obs.joints[2],
            j3=obs.joints[3],
            j4=obs.joints[4],
            j5=obs.joints[5],
            gripper=obs.gripper_mm,
            enabled=obs.enabled,
            task=task,
            history=hist_str,
        )

    def _build_verify_prompt(
        self,
        obs: Observation,
        task: str,
        actions_taken: List[str],
    ) -> str:
        from src.vla.prompts import VERIFY_TEMPLATE

        return VERIFY_TEMPLATE.format(
            actions_taken="\n".join(actions_taken[-5:]),
            j0=obs.joints[0],
            j1=obs.joints[1],
            j2=obs.joints[2],
            j3=obs.joints[3],
            j4=obs.joints[4],
            j5=obs.joints[5],
            gripper=obs.gripper_mm,
            task=task,
        )

    def _parse_response(self, text: str) -> ActionPlan:
        """Parse Gemini JSON response into ActionPlan."""
        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Gemini response as JSON: %s\nRaw: %s", e, text[:500])
            return ActionPlan(
                error=f"JSON parse error: {e}",
                raw_response=text,
                reasoning="Failed to parse model response",
            )

        return ActionPlan(
            reasoning=data.get("reasoning", ""),
            scene_description=data.get("scene_description", ""),
            gripper_position=data.get("gripper_position"),
            target_position=data.get("target_position"),
            actions=data.get("actions", []),
            phase=data.get("phase", "unknown"),
            confidence=data.get("confidence", 0.0),
            estimated_remaining_steps=data.get("estimated_remaining_steps", -1),
            raw_response=text,
        )

    async def _call_model(self, obs: Observation, prompt: str) -> ActionPlan:
        """Send images + prompt to Gemini and parse response."""
        t0 = time.monotonic()

        cam0_b64 = base64.b64encode(obs.cam0_jpeg).decode()
        cam1_b64 = base64.b64encode(obs.cam1_jpeg).decode()

        contents = [
            {"mime_type": "image/jpeg", "data": cam0_b64},
            "Camera 0 (front/side view) above.",
            {"mime_type": "image/jpeg", "data": cam1_b64},
            "Camera 1 (overhead view) above.",
            prompt,
        ]

        try:
            response = self._model.generate_content(contents)
            text = response.text.strip()
            plan = self._parse_response(text)
            plan.inference_time_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Gemini VLA inference: %.0fms, phase=%s, %d actions, confidence=%.2f",
                plan.inference_time_ms,
                plan.phase,
                len(plan.actions),
                plan.confidence,
            )
            return plan
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.error("Gemini VLA call failed after %.0fms: %s", elapsed, e)
            return ActionPlan(
                error=str(e),
                inference_time_ms=elapsed,
                reasoning=f"Model call failed: {e}",
            )

    async def plan(
        self,
        observation: Observation,
        task: str,
        history: Optional[List[str]] = None,
    ) -> ActionPlan:
        prompt = self._build_observe_prompt(observation, task, history)
        return await self._call_model(observation, prompt)

    async def verify(
        self,
        observation: Observation,
        task: str,
        actions_taken: List[str],
    ) -> ActionPlan:
        prompt = self._build_verify_prompt(observation, task, actions_taken)
        return await self._call_model(observation, prompt)


class OctoVLABackend(VLABackend):
    """Placeholder for Octo-Small model backend.

    When fine-tuned and deployed, this will provide real-time (~200ms) inference
    for action chunking. Currently raises NotImplementedError.

    To use:
    1. Collect demonstrations with DataCollector
    2. Fine-tune Octo-Small on our D1 data
    3. Save model to data/models/octo-small-d1/
    4. Initialize this backend
    """

    def __init__(self, model_path: str = "data/models/octo-small-d1"):
        self._model_path = model_path
        self._model = None
        logger.info("OctoVLABackend initialized (model_path=%s)", model_path)

    @property
    def name(self) -> str:
        return "octo-small"

    async def plan(
        self, observation: Observation, task: str, history: Optional[List[str]] = None
    ) -> ActionPlan:
        raise NotImplementedError(
            "Octo backend not yet available. Collect demonstrations first, "
            "then fine-tune. See docs/vla-architecture.md"
        )

    async def verify(
        self, observation: Observation, task: str, actions_taken: List[str]
    ) -> ActionPlan:
        raise NotImplementedError("Octo backend not yet available")
