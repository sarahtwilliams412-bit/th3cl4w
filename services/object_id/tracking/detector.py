"""Unified detector — wraps CV (HSV) and LLM (Gemini) detection.

Primary: fast HSV color detection via realtime_detector
Secondary: ObjectDetector from object_detection (multi-color)
Deep scan: Gemini LLM vision for unknown objects (periodic)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.location.detector")

# Use centralized rate limiter
from shared.utils.gemini_limiter import gemini_limiter


def gemini_rate_limited() -> bool:
    """Return True if we should skip Gemini calls due to rate limiting."""
    return gemini_limiter.is_limited


@dataclass
class DetectionResult:
    """Unified detection result from any detector."""

    label: str
    centroid_px: tuple[int, int]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    source: str  # "hsv", "cv", "llm"
    camera_id: int
    mask_area: int = 0


class UnifiedDetector:
    """Wraps multiple detection backends behind a single interface.

    - `detect_fast(frame, camera_id)` — HSV color detection, <30ms
    - `detect_cv(frame, camera_id)` — OpenCV multi-color detector
    - `detect_llm(jpeg_bytes, camera_id, target)` — Gemini vision
    """

    def __init__(self):
        # Lazy imports to avoid circular deps
        self._hsv_detector = None
        self._cv_detector = None

    def _get_hsv_detector(self):
        if self._hsv_detector is None:
            from ..detection.realtime_detector import detect_object

            self._hsv_detector = detect_object
        return self._hsv_detector

    def _get_cv_detector(self):
        if self._cv_detector is None:
            from ..detection.object_detection import ObjectDetector

            self._cv_detector = ObjectDetector(min_area=300)
        return self._cv_detector

    def detect_fast(
        self,
        frame: np.ndarray,
        camera_id: int,
        targets: list[str] = None,
    ) -> list[DetectionResult]:
        """Fast HSV detection. Returns results for each target color."""
        if frame is None or frame.size == 0:
            return []

        if targets is None:
            targets = ["redbull", "red", "blue", "green", "yellow"]

        detect_fn = self._get_hsv_detector()
        results = []

        for target in targets:
            det = detect_fn(frame, target=target)
            if det.found:
                results.append(
                    DetectionResult(
                        label=det.label,
                        centroid_px=det.centroid_px,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        source="hsv",
                        camera_id=camera_id,
                        mask_area=det.mask_area,
                    )
                )

        return results

    def detect_cv(self, frame: np.ndarray, camera_id: int) -> list[DetectionResult]:
        """Multi-color OpenCV detection."""
        if frame is None or frame.size == 0:
            return []

        detector = self._get_cv_detector()
        detections = detector.detect(frame)
        results = []

        for det in detections:
            results.append(
                DetectionResult(
                    label=det.label,
                    centroid_px=det.centroid_2d,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    source="cv",
                    camera_id=camera_id,
                    mask_area=int(det.area),
                )
            )

        return results

    async def detect_llm(
        self,
        jpeg_bytes: bytes,
        camera_id: int,
        target: str = "all objects",
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> list[DetectionResult]:
        """LLM-based detection using Gemini. Async, slower but more capable."""
        if not gemini_limiter.acquire():
            logger.debug("Skipping LLM detection — rate-limited")
            return []

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, skipping LLM detection")
            return []

        try:
            from google import genai as _genai
            from google.genai import types as _gtypes

            _client = _genai.Client(api_key=api_key)

            prompt = (
                f"Find all visible objects in this {image_width}x{image_height} camera image. "
                f"For each object, return a JSON array of objects with: "
                f'{{"label": "<name>", "u": <center_x_pixel>, "v": <center_y_pixel>, '
                f'"w": <width_pixels>, "h": <height_pixels>, "confidence": <0-1>}}. '
                f"Return ONLY the JSON array, no other text."
            )

            response = _client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    _gtypes.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )

            # Parse response
            text = response.text.strip()
            # Extract JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                logger.warning("LLM response not parseable: %s", text[:200])
                return []

            items = json.loads(match.group())
            results = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                label = item.get("label", "unknown")
                u = int(item.get("u", 0))
                v = int(item.get("v", 0))
                w = int(item.get("w", 50))
                h = int(item.get("h", 50))
                conf = float(item.get("confidence", 0.5))

                results.append(
                    DetectionResult(
                        label=label,
                        centroid_px=(u, v),
                        bbox=(u - w // 2, v - h // 2, w, h),
                        confidence=conf,
                        source="llm",
                        camera_id=camera_id,
                    )
                )

            logger.info("LLM detected %d objects on cam %d", len(results), camera_id)
            gemini_limiter.record_success()
            return results

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or ("Resource" in err_str and "exhausted" in err_str):
                gemini_limiter.record_429()
            else:
                logger.error("LLM detection failed: %s", e)
            return []
