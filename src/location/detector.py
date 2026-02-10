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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, skipping LLM detection")
            return []

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            b64 = base64.b64encode(jpeg_bytes).decode()
            prompt = (
                f"Find all visible objects in this {image_width}x{image_height} camera image. "
                f"For each object, return a JSON array of objects with: "
                f'{{"label": "<name>", "u": <center_x_pixel>, "v": <center_y_pixel>, '
                f'"w": <width_pixels>, "h": <height_pixels>, "confidence": <0-1>}}. '
                f"Return ONLY the JSON array, no other text."
            )

            response = model.generate_content([
                {"mime_type": "image/jpeg", "data": b64},
                prompt,
            ])

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
            return results

        except Exception as e:
            logger.error("LLM detection failed: %s", e)
            return []
