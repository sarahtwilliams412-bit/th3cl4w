"""
th3cl4w Vision — LLM-based Joint Detection via ASCII Art

Sends ASCII art renderings of camera frames to Gemini and parses
joint position responses. Designed to run alongside the CV pipeline
as an experimental second opinion.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None

from .ascii_converter import AsciiConverter, CHARSET_DETAILED

logger = logging.getLogger(__name__)

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "end_effector"]

CAMERA_DESCRIPTIONS = {
    0: "front view — camera facing the arm from the front",
    1: "overhead view — camera looking down at the arm from above",
}


@dataclass
class LLMJointPosition:
    name: str  # base, shoulder, elbow, wrist, end_effector
    norm_x: float  # 0-1 normalized
    norm_y: float  # 0-1 normalized
    pixel_x: int  # scaled to camera resolution
    pixel_y: int  # scaled to camera resolution
    confidence: str  # high/medium/low from LLM


@dataclass
class LLMDetectionResult:
    joints: list[LLMJointPosition]
    camera_id: int
    model: str
    tokens_used: int
    latency_ms: float
    raw_response: str
    success: bool
    error: str | None = None


class LLMJointDetector:
    """Detect arm joints from ASCII art using Gemini."""

    DEFAULT_MODEL = "gemini-2.0-flash-001"
    TIMEOUT_S = 15.0
    MAX_RETRIES = 2

    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        ascii_width: int = 80,
        ascii_height: int = 35,
        api_key: str | None = None,
        camera_width: int = 1920,
        camera_height: int = 1080,
    ):
        if genai is None:
            raise RuntimeError(
                "google-genai package required: pip install google-genai"
            )

        self.model_name = model
        self.ascii_width = ascii_width
        self.ascii_height = ascii_height
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.converter = AsciiConverter(
            width=ascii_width,
            height=ascii_height,
            charset=CHARSET_DETAILED,
            invert=True,
        )

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("No Gemini API key: set GEMINI_API_KEY env var or pass api_key")

        self._client = genai.Client(api_key=resolved_key)
        self._config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=400,
            response_mime_type="application/json",
        )

        self.total_tokens = 0
        self.total_calls = 0

    def _build_prompt(
        self,
        ascii_text: str,
        camera_id: int,
        joint_angles: list[float] | None = None,
        fk_hints: dict | None = None,
    ) -> str:
        camera_desc = CAMERA_DESCRIPTIONS.get(camera_id, f"camera {camera_id}")

        angle_section = ""
        if joint_angles:
            labels = [
                "J0(base)",
                "J1(shoulder)",
                "J2(elbow)",
                "J3(wrist_flex)",
                "J4(wrist_rot)",
                "J5(gripper)",
            ]
            parts = [
                f"{labels[i]}={joint_angles[i]:.1f}°" for i in range(min(len(joint_angles), 6))
            ]
            angle_section = f"\nCurrent joint angles: {' '.join(parts)}\n"

        hint_section = ""
        if fk_hints:
            hint_section = "\nExpected approximate positions (from forward kinematics):\n"
            for name in JOINT_NAMES:
                if name in fk_hints:
                    h = fk_hints[name]
                    hint_section += f"- {name}: x≈{h['x']:.2f}, y≈{h['y']:.2f}\n"
            hint_section += "These are predictions — report what you actually see.\n"

        return f"""You are analyzing an ASCII art rendering of a robotic arm captured by a camera.

Camera: {camera_desc}
Frame resolution: {self.ascii_width} columns × {self.ascii_height} rows
Original image: {self.camera_width}×{self.camera_height} pixels
Each ASCII character represents approximately {self.camera_width / self.ascii_width:.0f}×{self.camera_height / self.ascii_height:.0f} pixels.

The arm is a 6-DOF robotic arm (SO-ARM100 / D1). It has 5 key points:
- base: where the arm connects to its mount (typically bottom-center)
- shoulder: first major joint above the base
- elbow: middle joint where the arm bends
- wrist: joint near the end of the arm
- end_effector: the gripper tip at the very end

The arm is MATTE BLACK (appears as dim/sparse characters like . : - ).
Gold accents at joints may appear as brighter characters (* # % @).
{angle_section}{hint_section}
ASCII art of the current frame:
```
{ascii_text}
```

Identify each joint's position as normalized coordinates where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right of the frame.

Respond with JSON:
{{"joints": [
  {{"name": "base", "x": 0.5, "y": 0.9, "confidence": "high"}},
  {{"name": "shoulder", "x": 0.5, "y": 0.7, "confidence": "medium"}},
  {{"name": "elbow", "x": 0.4, "y": 0.5, "confidence": "medium"}},
  {{"name": "wrist", "x": 0.3, "y": 0.4, "confidence": "low"}},
  {{"name": "end_effector", "x": 0.25, "y": 0.35, "confidence": "low"}}
]}}

Use confidence "high", "medium", or "low". Report all 5 joints — use your best estimate even if uncertain."""

    def _parse_response(self, raw: str, camera_id: int) -> list[LLMJointPosition]:
        """Parse Gemini JSON response into LLMJointPosition list."""
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        joints_data = data.get("joints", [])

        positions = []
        for j in joints_data:
            name = j.get("name", "")
            x = j.get("x")
            y = j.get("y")
            confidence = j.get("confidence", "low")

            if x is None or y is None:
                continue
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue

            norm_x = max(0.0, min(1.0, float(x)))
            norm_y = max(0.0, min(1.0, float(y)))

            positions.append(
                LLMJointPosition(
                    name=name,
                    norm_x=norm_x,
                    norm_y=norm_y,
                    pixel_x=int(norm_x * self.camera_width),
                    pixel_y=int(norm_y * self.camera_height),
                    confidence=confidence,
                )
            )

        return positions

    async def detect_joints(
        self,
        jpeg_bytes: bytes,
        camera_id: int,
        joint_angles: list[float] | None = None,
        fk_hints: dict | None = None,
    ) -> LLMDetectionResult:
        """Convert JPEG to ASCII, send to Gemini, parse joint positions."""
        t0 = time.monotonic()

        try:
            ascii_text = self.converter.decode_jpeg_to_ascii(jpeg_bytes)
        except Exception as e:
            return LLMDetectionResult(
                joints=[],
                camera_id=camera_id,
                model=self.model_name,
                tokens_used=0,
                latency_ms=(time.monotonic() - t0) * 1000,
                raw_response="",
                success=False,
                error=f"ASCII conversion failed: {e}",
            )

        prompt = self._build_prompt(ascii_text, camera_id, joint_angles, fk_hints)

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=self._config,
                )
                break
            except Exception as e:
                if attempt == self.MAX_RETRIES:
                    latency_ms = (time.monotonic() - t0) * 1000
                    logger.error("Gemini API failed after %d retries: %s", self.MAX_RETRIES + 1, e)
                    return LLMDetectionResult(
                        joints=[],
                        camera_id=camera_id,
                        model=self.model_name,
                        tokens_used=0,
                        latency_ms=latency_ms,
                        raw_response=str(e),
                        success=False,
                        error=f"API error: {e}",
                    )
                await asyncio.sleep(1.0 * (attempt + 1))

        latency_ms = (time.monotonic() - t0) * 1000
        raw = response.text or ""

        # Token counting
        tokens_used = 0
        try:
            um = response.usage_metadata
            tokens_used = (um.prompt_token_count or 0) + (um.candidates_token_count or 0)
        except Exception:
            pass

        self.total_tokens += tokens_used
        self.total_calls += 1

        # Parse
        try:
            joints = self._parse_response(raw, camera_id)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM response: %s", e)
            return LLMDetectionResult(
                joints=[],
                camera_id=camera_id,
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                raw_response=raw,
                success=False,
                error=f"Parse error: {e}",
            )

        logger.info(
            "LLM detection cam=%d: %d joints, %d tokens, %.0fms",
            camera_id,
            len(joints),
            tokens_used,
            latency_ms,
        )

        return LLMDetectionResult(
            joints=joints,
            camera_id=camera_id,
            model=self.model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            raw_response=raw,
            success=True,
        )

    async def detect_joints_batch(self, frames: list[dict]) -> list[LLMDetectionResult]:
        """Run multiple detections concurrently.

        Each frame dict should have:
          - jpeg_bytes: bytes
          - camera_id: int
          - joint_angles: list[float] (optional)
          - fk_hints: dict (optional)
        """
        tasks = [
            self.detect_joints(
                jpeg_bytes=f["jpeg_bytes"],
                camera_id=f["camera_id"],
                joint_angles=f.get("joint_angles"),
                fk_hints=f.get("fk_hints"),
            )
            for f in frames
        ]
        return list(await asyncio.gather(*tasks))
