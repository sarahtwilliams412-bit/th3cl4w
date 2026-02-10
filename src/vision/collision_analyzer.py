"""
Collision Analyzer — Camera + Vision Model analysis for arm collisions.

Captures snapshots from both cameras, sends them to Google Gemini for
analysis, and saves collision data for review.
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("th3cl4w.collision_analyzer")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "collisions"
CAMERA_BASE = "http://localhost:8081"


@dataclass
class CollisionAnalysis:
    """Result of a collision analysis."""

    analysis_text: str
    cam0_path: Optional[str] = None
    cam1_path: Optional[str] = None
    timestamp: str = ""
    vision_available: bool = False


class CollisionAnalyzer:
    """
    Captures camera snapshots and analyzes collisions using Google Gemini vision.

    Falls back to saving images only if no API key is available.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, camera_base: str = CAMERA_BASE):
        self._camera_base = camera_base
        self._api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None
        self._model = None

        if self._api_key:
            try:
                from google import genai as _genai

                self._client = _genai.Client(api_key=self._api_key)
                self._model = self._client.models
                logger.info("Gemini vision model initialized")
            except Exception as e:
                logger.warning("Failed to initialize Gemini: %s", e)
                self._client = None
                self._model = None

    @property
    def vision_available(self) -> bool:
        return self._model is not None

    def analyze(
        self,
        joint_id: int,
        commanded_deg: float,
        actual_deg: float,
    ) -> CollisionAnalysis:
        """
        Capture snapshots and analyze the collision.

        This is synchronous — designed to be called from an async context
        via asyncio.to_thread() or similar.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DATA_DIR / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        # Capture snapshots
        cam0_bytes = self._capture_camera(0)
        cam1_bytes = self._capture_camera(1)

        cam0_path = None
        cam1_path = None

        if cam0_bytes:
            cam0_path = str(out_dir / "cam0.jpg")
            Path(cam0_path).write_bytes(cam0_bytes)

        if cam1_bytes:
            cam1_path = str(out_dir / "cam1.jpg")
            Path(cam1_path).write_bytes(cam1_bytes)

        # Analyze with vision model
        analysis_text = "Vision analysis unavailable — images saved for manual review"
        vision_used = False

        if self._model and (cam0_bytes or cam1_bytes):
            try:
                analysis_text = self._analyze_with_gemini(
                    joint_id,
                    commanded_deg,
                    actual_deg,
                    cam0_bytes,
                    cam1_bytes,
                )
                vision_used = True
            except Exception as e:
                logger.error("Gemini analysis failed: %s", e)
                analysis_text = f"Vision analysis failed: {e} — images saved for manual review"

        # Save analysis
        analysis_data = {
            "timestamp": ts,
            "joint_id": joint_id,
            "commanded_deg": commanded_deg,
            "actual_deg": actual_deg,
            "error_deg": abs(commanded_deg - actual_deg),
            "analysis": analysis_text,
            "vision_used": vision_used,
            "cam0": cam0_path,
            "cam1": cam1_path,
        }
        (out_dir / "analysis.json").write_text(json.dumps(analysis_data, indent=2))

        return CollisionAnalysis(
            analysis_text=analysis_text,
            cam0_path=cam0_path,
            cam1_path=cam1_path,
            timestamp=ts,
            vision_available=vision_used,
        )

    def _capture_camera(self, cam_id: int) -> Optional[bytes]:
        """Capture a snapshot from the camera server."""
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self._camera_base}/snap/{cam_id}")
                if resp.status_code == 200:
                    return resp.content
                logger.warning("Camera %d snapshot failed: %d", cam_id, resp.status_code)
        except Exception as e:
            logger.warning("Camera %d capture error: %s", cam_id, e)
        return None

    def _analyze_with_gemini(
        self,
        joint_id: int,
        commanded_deg: float,
        actual_deg: float,
        cam0_bytes: Optional[bytes],
        cam1_bytes: Optional[bytes],
    ) -> str:
        """Send images to Gemini for collision analysis."""
        prompt = (
            f"This robotic arm (Unitree D1) is stuck. "
            f"Joint {joint_id} was commanded to {commanded_deg:.1f}° but is at {actual_deg:.1f}°. "
            f"Look at the camera views and describe:\n"
            f"1) What is the arm hitting or blocked by?\n"
            f"2) Is there a risk of damage?\n"
            f"3) Suggested corrective action.\n"
            f"Be concise (2-3 sentences max)."
        )

        parts = [prompt]
        if self._client is not None:
            # Real genai client — wrap image bytes in Part objects
            from google.generativeai import types as _gtypes

            for img_bytes in [cam0_bytes, cam1_bytes]:
                if img_bytes:
                    parts.append(_gtypes.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        response = self._model.generate_content(
            model="gemini-2.0-flash",
            contents=parts,
        )
        return response.text
