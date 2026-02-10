"""
ASCII Art LLM Analyst — Gemini text-only analysis of ASCII camera feeds.

The LLM is prompted as an expert ASCII art reader. It receives raw ASCII text
and analyzes it without any vision/image models — pure text pattern recognition.
"""

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("th3cl4w.ascii.llm_analyst")

try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_GENAI = True
except ImportError:
    genai = None
    genai_types = None
    _HAS_GENAI = False

CAMERA_PERSPECTIVES = {
    0: "overhead — looking straight down at the workspace table from above",
    1: "side view (MX Brio) — viewing the arm from the side, showing height profile",
    2: "arm-mounted — view from the robot arm's end effector / gripper",
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert ASCII art analyst. You specialize in interpreting ASCII art representations of real-world camera feeds from a robotic arm workspace.

You are viewing a {width}x{height} character ASCII representation captured by a camera.
Character ramp used: "{charset}" (from lightest/empty space to darkest/densest characters).

Camera perspective: {perspective}

This camera observes a robotic arm workspace containing:
- A SO-ARM100/D1 6-DOF robotic arm (matte black body, gold accents at joints)
- A workspace table with various objects the arm can manipulate
- The arm joints: base (rotation), shoulder, elbow, wrist_flex, wrist_rotate, gripper

How to read the ASCII art:
- Dense characters (@, #, %, *, &) = dark/solid areas (objects, arm body)
- Medium characters (=, +, :, -) = edges, transitions, medium-toned areas
- Light characters (., ') = subtle details, light surfaces
- Spaces = brightest/emptiest areas (background, white surfaces)

You must analyze ONLY the raw ASCII text characters — you are NOT using any image processing or computer vision. You read ASCII art like a skilled human: recognizing patterns of characters that form shapes, edges, and objects.

When describing locations, use approximate (col, row) coordinates where (0,0) is top-left.
Be specific and analytical. Reference character patterns you observe."""


class AsciiAnalyst:
    """Gemini-powered ASCII art analyst for robotic arm workspace."""

    MODEL = "gemini-2.0-flash"
    MAX_HISTORY = 20  # max messages in a chat session

    def __init__(self, api_key: Optional[str] = None):
        if not _HAS_GENAI:
            raise RuntimeError("google-genai required: pip install google-genai")

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("No Gemini API key: set GEMINI_API_KEY or pass api_key")

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = self.MODEL
        self.total_tokens = 0
        self.total_calls = 0

    def _build_system_prompt(self, width: int, height: int, charset: str, cam_id: int) -> str:
        perspective = CAMERA_PERSPECTIVES.get(cam_id, f"camera {cam_id}")
        return SYSTEM_PROMPT_TEMPLATE.format(
            width=width, height=height, charset=charset, perspective=perspective
        )

    def _create_config(self, system_prompt: str):
        return genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=1500,
        )

    async def analyze(
        self,
        ascii_text: str,
        question: str,
        cam_id: int = 0,
        width: int = 120,
        height: int = 60,
        charset: str = " .:-=+*#%@",
        history: Optional[list[dict]] = None,
    ) -> dict:
        """Analyze an ASCII frame with a question.

        Args:
            ascii_text: Multi-line ASCII art string
            question: User's question about the scene
            cam_id: Camera ID for perspective context
            width/height: ASCII dimensions
            charset: Character ramp used
            history: Optional chat history [{role, text}, ...]

        Returns:
            {answer, tokens_used, latency_ms, model}
        """
        t0 = time.monotonic()
        system_prompt = self._build_system_prompt(width, height, charset, cam_id)
        config = self._create_config(system_prompt)

        # Build the user message with the ASCII frame
        user_msg = f"Here is the current ASCII frame from camera {cam_id}:\n\n```\n{ascii_text}\n```\n\nQuestion: {question}"

        # Build chat history if provided
        contents = []
        if history:
            for msg in history[-self.MAX_HISTORY:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=msg["text"])]))
        contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_msg)]))

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model_name,
                contents=contents,
                config=config,
            )

            answer = response.text or ""
            tokens = 0
            try:
                um = response.usage_metadata
                tokens = (um.prompt_token_count or 0) + (um.candidates_token_count or 0)
            except Exception:
                pass

            self.total_tokens += tokens
            self.total_calls += 1
            latency_ms = (time.monotonic() - t0) * 1000

            logger.info("Analysis cam=%d: %d tokens, %.0fms", cam_id, tokens, latency_ms)

            return {
                "answer": answer,
                "tokens_used": tokens,
                "latency_ms": round(latency_ms, 1),
                "model": self._model_name,
            }

        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.error("Gemini analysis failed: %s", e)
            return {
                "answer": f"Analysis failed: {e}",
                "tokens_used": 0,
                "latency_ms": round(latency_ms, 1),
                "model": self._model_name,
                "error": str(e),
            }

    def get_stats(self) -> dict:
        return {
            "model": self._model_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }
