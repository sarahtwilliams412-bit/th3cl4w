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

# --- Rate-limit backoff state for Gemini API ---
_rate_limit_backoff_s: float = 0.0       # current backoff (0 = no backoff)
_rate_limit_until: float = 0.0           # timestamp until which we should not call Gemini
_consecutive_429s: int = 0               # consecutive 429 errors
_BACKOFF_INITIAL_S = 60.0
_BACKOFF_MAX_S = 600.0                   # 10 minutes
_BACKOFF_MULTIPLIER = 2.0
_PAUSE_AFTER_CONSECUTIVE = 5             # pause scanning after this many consecutive 429s
_PAUSE_DURATION_S = 600.0                # 10 minute pause


def gemini_rate_limited() -> bool:
    """Return True if we should skip Gemini calls due to rate limiting."""
    return time.time() < _rate_limit_until


def _record_429():
    """Record a 429 error and update backoff state."""
    global _rate_limit_backoff_s, _rate_limit_until, _consecutive_429s
    _consecutive_429s += 1
    if _consecutive_429s >= _PAUSE_AFTER_CONSECUTIVE:
        _rate_limit_backoff_s = _PAUSE_DURATION_S
        logger.warning("Hit %d consecutive 429s — pausing Gemini for %ds",
                        _consecutive_429s, int(_PAUSE_DURATION_S))
    elif _rate_limit_backoff_s == 0:
        _rate_limit_backoff_s = _BACKOFF_INITIAL_S
    else:
        _rate_limit_backoff_s = min(_rate_limit_backoff_s * _BACKOFF_MULTIPLIER, _BACKOFF_MAX_S)
    _rate_limit_until = time.time() + _rate_limit_backoff_s
    logger.info("Gemini 429 backoff: %.0fs (consecutive: %d)", _rate_limit_backoff_s, _consecutive_429s)


def _record_success():
    """Reset backoff state on successful Gemini call."""
    global _rate_limit_backoff_s, _rate_limit_until, _consecutive_429s
    _rate_limit_backoff_s = 0.0
    _rate_limit_until = 0.0
    _consecutive_429s = 0


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
            from src.vision.realtime_detector import detect_object
            self._hsv_detector = detect_object
        return self._hsv_detector

    def _get_cv_detector(self):
        if self._cv_detector is None:
            from src.vision.object_detection import ObjectDetector
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
                results.append(DetectionResult(
                    label=det.label,
                    centroid_px=det.centroid_px,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    source="hsv",
                    camera_id=camera_id,
                    mask_area=det.mask_area,
                ))

        return results

    def detect_cv(
        self, frame: np.ndarray, camera_id: int
    ) -> list[DetectionResult]:
        """Multi-color OpenCV detection."""
        if frame is None or frame.size == 0:
            return []

        detector = self._get_cv_detector()
        detections = detector.detect(frame)
        results = []

        for det in detections:
            results.append(DetectionResult(
                label=det.label,
                centroid_px=det.centroid_2d,
                bbox=det.bbox,
                confidence=det.confidence,
                source="cv",
                camera_id=camera_id,
                mask_area=int(det.area),
            ))

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
        if gemini_rate_limited():
            logger.debug("Skipping LLM detection — rate-limited for %.0fs more",
                         _rate_limit_until - time.time())
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
            match = re.search(r'\[.*\]', text, re.DOTALL)
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

                results.append(DetectionResult(
                    label=label,
                    centroid_px=(u, v),
                    bbox=(u - w // 2, v - h // 2, w, h),
                    confidence=conf,
                    source="llm",
                    camera_id=camera_id,
                ))

            logger.info("LLM detected %d objects on cam %d", len(results), camera_id)
            _record_success()
            return results

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Resource" in err_str and "exhausted" in err_str:
                _record_429()
            else:
                logger.error("LLM detection failed: %s", e)
            return []
