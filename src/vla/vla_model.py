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

        from google import genai as _genai
        from google.genai import types as _gtypes

        self._client = _genai.Client(api_key=self.api_key)
        self._model_name = model_name
        self._config = _gtypes.GenerateContentConfig(
            system_instruction=self._get_system_prompt(),
        )
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

        from google.genai import types as _gtypes

        contents = [
            _gtypes.Part.from_bytes(data=obs.cam0_jpeg, mime_type="image/jpeg"),
            "Camera 0 (front/side view) above.",
            _gtypes.Part.from_bytes(data=obs.cam1_jpeg, mime_type="image/jpeg"),
            "Camera 1 (overhead view) above.",
            prompt,
        ]

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._config,
            )
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
    """Local VLA model backend using OpenVLA / Octo architecture.

    Provides real-time (~200ms) inference for closed-loop manipulation
    without requiring external API calls. Supports pre-training on NVIDIA
    Kitchen-Sim-Demos and fine-tuning on D1 demonstrations.

    Training pipeline:
    1. Pre-train on NVIDIA Kitchen-Sim-Demos via LeRobot format
    2. Fine-tune on D1 teleoperation demonstrations
    3. Save model to data/models/octo-d1/
    4. Initialize this backend with the model path

    Action output format:
    The model predicts action chunks — sequences of (state_dim,) vectors
    representing joint deltas + gripper, which are decoded into the standard
    ActionPlan format for the ActionDecoder pipeline.
    """

    # Default model configurations
    STATE_DIM = 7     # 6 joints + gripper
    ACTION_DIM = 7    # 6 joint deltas + gripper delta
    ACTION_HORIZON = 4  # Predict 4 steps ahead (action chunking)

    def __init__(
        self,
        model_path: str = "data/models/octo-d1",
        device: str = "auto",
        action_horizon: int = 4,
    ):
        self._model_path = model_path
        self._device = device
        self._action_horizon = action_horizon
        self._model = None
        self._processor = None
        self._loaded = False
        logger.info("OctoVLABackend initialized (model_path=%s)", model_path)

    @property
    def name(self) -> str:
        return "octo-d1"

    def _load_model(self):
        """Lazy-load the model on first inference call."""
        if self._loaded:
            return

        import os

        model_dir = os.path.join(self._model_path)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model not found at {model_dir}. To create it:\n"
                f"1. Collect demonstrations: python -m scripts.download_nvidia_kitchen\n"
                f"2. Train model: python -m scripts.train_vla\n"
                f"3. Or use Gemini backend as fallback: VLAController(backend=GeminiVLABackend())"
            )

        try:
            import torch

            # Determine device
            if self._device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try loading as a PyTorch model checkpoint
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                self._action_horizon = config.get("action_horizon", self._action_horizon)

            weights_path = os.path.join(model_dir, "model.pt")
            if os.path.exists(weights_path):
                self._model = torch.load(weights_path, map_location=self._device)
                if hasattr(self._model, "eval"):
                    self._model.eval()
                self._loaded = True
                logger.info(
                    "Octo model loaded from %s (device=%s, action_horizon=%d)",
                    model_dir, self._device, self._action_horizon,
                )
                return

            # Try loading as HuggingFace model
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor
                self._processor = AutoProcessor.from_pretrained(model_dir)
                self._model = AutoModelForVision2Seq.from_pretrained(model_dir)
                self._model.to(self._device)
                self._model.eval()
                self._loaded = True
                logger.info("Loaded VLA model from HuggingFace format: %s", model_dir)
                return
            except Exception:
                pass

            raise FileNotFoundError(
                f"No loadable model found at {model_dir}. "
                f"Expected model.pt or HuggingFace model files."
            )

        except ImportError as e:
            raise ImportError(
                f"PyTorch required for Octo backend: {e}. "
                f"Install with: pip install torch torchvision"
            ) from e

    def _preprocess_observation(self, obs: Observation) -> dict:
        """Convert Observation to model input tensors."""
        import torch
        import io
        from PIL import Image

        # Decode JPEG images
        img0 = Image.open(io.BytesIO(obs.cam0_jpeg)).convert("RGB")
        img1 = Image.open(io.BytesIO(obs.cam1_jpeg)).convert("RGB")

        # Resize to model input size (224x224 default for most VLAs)
        img0 = img0.resize((224, 224))
        img1 = img1.resize((224, 224))

        import numpy as np

        img0_tensor = torch.from_numpy(np.array(img0)).permute(2, 0, 1).float() / 255.0
        img1_tensor = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0

        # Stack images as (2, 3, 224, 224)
        images = torch.stack([img0_tensor, img1_tensor]).unsqueeze(0)  # (1, 2, 3, H, W)

        # Build state vector
        state = torch.tensor(
            obs.joints + [obs.gripper_mm], dtype=torch.float32,
        ).unsqueeze(0)  # (1, 7)

        return {
            "images": images.to(self._device),
            "state": state.to(self._device),
        }

    def _decode_action_chunk(
        self,
        action_chunk: "Any",
        observation: Observation,
        task: str,
    ) -> ActionPlan:
        """Convert model output (action chunk) to ActionPlan format."""
        import torch
        import numpy as np

        if isinstance(action_chunk, torch.Tensor):
            actions_np = action_chunk.detach().cpu().numpy()
        else:
            actions_np = np.asarray(action_chunk)

        # actions_np shape: (horizon, action_dim) or (1, horizon, action_dim)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]  # Remove batch dim
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)  # Single step

        actions_list = []
        for step_idx in range(min(len(actions_np), self._action_horizon)):
            action_vec = actions_np[step_idx]

            # First 6 values: joint deltas in degrees
            for j in range(min(6, len(action_vec))):
                delta = float(action_vec[j])
                if abs(delta) >= 0.5:
                    actions_list.append({
                        "type": "joint",
                        "id": j,
                        "delta": round(delta, 1),
                        "reason": f"octo step {step_idx}: j{j} delta",
                    })

            # 7th value: gripper
            if len(action_vec) > 6:
                gripper_val = float(action_vec[6])
                # Interpret as absolute gripper position in mm
                gripper_mm = max(0.0, min(65.0, gripper_val))
                actions_list.append({
                    "type": "gripper",
                    "position_mm": round(gripper_mm, 1),
                    "reason": f"octo step {step_idx}: gripper",
                })

            # Add verify checkpoint between action chunk steps
            if step_idx < len(actions_np) - 1:
                actions_list.append({
                    "type": "verify",
                    "reason": f"checkpoint after action chunk step {step_idx}",
                })

        # Determine phase from action patterns
        phase = self._infer_phase(actions_np, observation)

        return ActionPlan(
            reasoning=f"Octo model predicted {len(actions_np)}-step action chunk for: {task}",
            scene_description="Processed by local VLA model",
            actions=actions_list,
            phase=phase,
            confidence=0.7,  # Default confidence for local model
            estimated_remaining_steps=max(1, self._action_horizon),
        )

    def _infer_phase(self, actions_np: "Any", obs: Observation) -> str:
        """Infer the current manipulation phase from action patterns."""
        import numpy as np

        if actions_np.size == 0:
            return "unknown"

        # Check gripper actions
        if actions_np.shape[1] > 6:
            gripper_values = actions_np[:, 6]
            gripper_closing = np.any(gripper_values < 20.0)
            gripper_opening = np.any(gripper_values > 50.0)
        else:
            gripper_closing = False
            gripper_opening = False

        # Check joint movements
        joint_deltas = np.abs(actions_np[:, :6])
        total_movement = np.sum(joint_deltas)

        if gripper_closing and obs.gripper_mm > 30:
            return "grasp"
        elif gripper_opening and obs.gripper_mm < 30:
            return "place"
        elif total_movement > 30:
            return "approach"
        elif total_movement < 5:
            return "align"
        else:
            return "approach"

    async def plan(
        self,
        observation: Observation,
        task: str,
        history: Optional[List[str]] = None,
    ) -> ActionPlan:
        """Generate action plan using the local VLA model."""
        t0 = time.monotonic()

        try:
            self._load_model()
        except (FileNotFoundError, ImportError) as e:
            return ActionPlan(
                error=str(e),
                reasoning=f"Model loading failed: {e}",
                inference_time_ms=(time.monotonic() - t0) * 1000,
            )

        try:
            import torch

            inputs = self._preprocess_observation(observation)

            with torch.no_grad():
                if hasattr(self._model, "predict_action"):
                    # Standard Octo/OpenVLA interface
                    action_chunk = self._model.predict_action(
                        images=inputs["images"],
                        state=inputs["state"],
                        task=task,
                    )
                elif callable(self._model):
                    # Generic callable model
                    action_chunk = self._model(inputs["images"], inputs["state"])
                else:
                    return ActionPlan(
                        error="Model does not have a callable interface",
                        reasoning="Model loaded but no predict_action or __call__ method found",
                        inference_time_ms=(time.monotonic() - t0) * 1000,
                    )

            plan = self._decode_action_chunk(action_chunk, observation, task)
            plan.inference_time_ms = (time.monotonic() - t0) * 1000

            logger.info(
                "Octo VLA inference: %.0fms, phase=%s, %d actions",
                plan.inference_time_ms, plan.phase, len(plan.actions),
            )
            return plan

        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.error("Octo VLA inference failed after %.0fms: %s", elapsed, e)
            return ActionPlan(
                error=str(e),
                inference_time_ms=elapsed,
                reasoning=f"Local model inference failed: {e}",
            )

    async def verify(
        self,
        observation: Observation,
        task: str,
        actions_taken: List[str],
    ) -> ActionPlan:
        """Verify and re-plan using the local model.

        For the local model, verify is the same as plan — the model
        processes the current observation without explicit history context
        (the observation itself encodes the result of previous actions).
        """
        return await self.plan(observation, task, history=actions_taken)
