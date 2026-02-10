"""
Vision-based Object Labeler — Uses Gemini Flash to identify objects from side camera.

Takes detected objects from the overhead camera's color/contour pipeline and
enriches them with semantic labels by sending the side camera frame to a
vision LLM (Gemini 2.0 Flash).

Results are cached and rate-limited to avoid excessive API calls.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.object_labeler")

# Rate limit: max 1 API call per this many seconds
_MIN_CALL_INTERVAL_S = 5.0

# Cache: don't re-label if objects moved less than this
_CACHE_MOVE_THRESHOLD_MM = 20.0

# Side camera horizontal FOV mapped to workspace X range
_SIDE_CAM_WS_X_MIN = -400.0
_SIDE_CAM_WS_X_MAX = 400.0

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "object_ontology.json"


@dataclass
class _CachedLabel:
    """Cached label result for a detected object position."""

    x_mm: float
    y_mm: float
    label: str
    category: str
    llm_confidence: float
    timestamp: float


def _load_ontology() -> dict:
    """Load the object ontology from disk."""
    try:
        with open(_ONTOLOGY_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load ontology: %s", e)
        return {"objects": {}, "categories": {}}


def _get_ontology_labels(ontology: dict) -> list[str]:
    """Get list of standard labels from ontology for prompting."""
    return list(ontology.get("objects", {}).keys())


def _build_prompt(ontology_labels: list[str]) -> str:
    """Build the vision prompt for Gemini."""
    labels_str = ", ".join(ontology_labels[:30])  # cap to avoid huge prompts
    return (
        "You are a robotic arm vision system. Identify ALL objects visible on the "
        "table/workspace in this image. For each object, provide:\n"
        "- label: a snake_case identifier (use these standard labels when applicable: "
        f"{labels_str})\n"
        "- category: one of beverage_can, bottle, cup_mug, hand_tool, office_supply, "
        "small_box, ball, electronic, or 'unknown'\n"
        "- estimated_width_mm: estimated width in millimeters\n"
        "- estimated_height_mm: estimated height in millimeters\n"
        "- color: dominant color name\n"
        "- x_position: horizontal position as fraction 0.0 (left) to 1.0 (right)\n"
        "- confidence: your confidence 0.0 to 1.0\n\n"
        "Respond ONLY with a JSON array of objects. No markdown, no explanation.\n"
        'Example: [{"label":"coca_cola_can","category":"beverage_can",'
        '"estimated_width_mm":66,"estimated_height_mm":122,"color":"red",'
        '"x_position":0.3,"confidence":0.9}]'
    )


class ObjectLabeler:
    """Labels detected objects using Gemini Flash vision API.

    Thread-safe. Caches results and rate-limits API calls.
    """

    def __init__(self):
        self._ontology = _load_ontology()
        self._ontology_labels = _get_ontology_labels(self._ontology)
        self._prompt = _build_prompt(self._ontology_labels)
        self._cache: list[_CachedLabel] = []
        self._last_call_time: float = 0.0
        self._api_key: Optional[str] = None
        self._model = None
        self._init_attempted = False

    def _ensure_api(self):
        """Lazy-init the Gemini client."""
        if self._init_attempted:
            return
        self._init_attempted = True

        # Try .env first, then environment
        try:
            from dotenv import load_dotenv

            env_path = Path(__file__).resolve().parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass

        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            logger.warning("No GEMINI_API_KEY found — labeler will use color-based fallback")
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini Flash vision labeler initialized")
        except Exception as e:
            logger.warning("Failed to init Gemini client: %s — using fallback", e)
            self._model = None

    def reload_ontology(self):
        """Reload the ontology from disk (after edits)."""
        self._ontology = _load_ontology()
        self._ontology_labels = _get_ontology_labels(self._ontology)
        self._prompt = _build_prompt(self._ontology_labels)

    @property
    def ontology(self) -> dict:
        return self._ontology

    def label_objects(
        self,
        side_frame: Optional[np.ndarray],
        detected_objects: list,
    ) -> list:
        """Label detected objects using the side camera frame + Gemini vision.

        Modifies detected_objects in-place, adding/updating:
          - label (semantic name from ontology or color fallback)
          - category
          - llm_confidence

        Returns the same list for convenience.
        """
        if not detected_objects:
            return detected_objects

        # Check cache — skip if objects haven't moved
        if self._cache_valid(detected_objects):
            self._apply_cache(detected_objects)
            return detected_objects

        # Try LLM labeling
        if side_frame is not None:
            self._ensure_api()
            if self._model is not None and self._can_call():
                llm_labels = self._call_gemini(side_frame)
                if llm_labels:
                    self._match_and_apply(detected_objects, llm_labels)
                    self._update_cache(detected_objects)
                    return detected_objects

        # Fallback: color-based labels (keep existing label which is color+shape)
        for obj in detected_objects:
            if not hasattr(obj, "category") or not getattr(obj, "category", None):
                # Add fallback attributes
                if not hasattr(obj, "llm_confidence"):
                    object.__setattr__(obj, "llm_confidence", 0.0)
                if not hasattr(obj, "category"):
                    object.__setattr__(obj, "category", "unknown")
        return detected_objects

    def _can_call(self) -> bool:
        """Check rate limit."""
        now = time.monotonic()
        return (now - self._last_call_time) >= _MIN_CALL_INTERVAL_S

    def _call_gemini(self, frame: np.ndarray) -> list[dict]:
        """Send frame to Gemini and parse response."""
        try:
            import PIL.Image

            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb)

            # Resize if too large (save tokens/bandwidth)
            max_dim = 1024
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                pil_img = pil_img.resize(new_size, PIL.Image.LANCZOS)

            self._last_call_time = time.monotonic()
            response = self._model.generate_content(
                [self._prompt, pil_img],
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )

            text = response.text.strip()
            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            if isinstance(result, list):
                logger.info("Gemini identified %d objects", len(result))
                return result
            return []

        except Exception as e:
            logger.warning("Gemini vision call failed: %s", e)
            return []

    def _match_and_apply(self, detected_objects: list, llm_labels: list[dict]):
        """Match LLM-identified objects to DetectedObjects by horizontal position and color."""
        ontology_objects = self._ontology.get("objects", {})

        for obj in detected_objects:
            # Normalize object's x position to 0-1 range
            norm_x = (obj.x_mm - _SIDE_CAM_WS_X_MIN) / (_SIDE_CAM_WS_X_MAX - _SIDE_CAM_WS_X_MIN)
            norm_x = max(0.0, min(1.0, norm_x))

            # Find best matching LLM label by position
            best_match = None
            best_score = 0.0

            # Get object's color name for matching
            obj_color = (
                self._color_name_from_bgr(obj.color_bgr) if hasattr(obj, "color_bgr") else ""
            )

            for llm_obj in llm_labels:
                llm_x = llm_obj.get("x_position", 0.5)
                pos_dist = abs(norm_x - llm_x)
                if pos_dist > 0.25:  # too far horizontally
                    continue

                # Score: position proximity + color match bonus
                pos_score = 1.0 - (pos_dist / 0.25)
                color_score = (
                    0.3
                    if llm_obj.get("color", "").lower() in obj_color.lower()
                    or obj_color.lower() in llm_obj.get("color", "").lower()
                    else 0.0
                )
                score = pos_score * 0.7 + color_score

                if score > best_score:
                    best_score = score
                    best_match = llm_obj

            if best_match and best_score > 0.3:
                label = best_match.get("label", obj.label)
                category = best_match.get("category", "unknown")
                confidence = best_match.get("confidence", 0.5)

                # Validate against ontology
                if label in ontology_objects:
                    ont_entry = ontology_objects[label]
                    category = ont_entry.get("category", category)

                obj.label = label
                # Set extra fields — use object.__setattr__ since these may not be in dataclass
                try:
                    obj.category = category
                except AttributeError:
                    object.__setattr__(obj, "category", category)
                try:
                    obj.llm_confidence = float(confidence)
                except AttributeError:
                    object.__setattr__(obj, "llm_confidence", float(confidence))

                # Remove matched LLM label to avoid double-matching
                llm_labels.remove(best_match)

    def _cache_valid(self, detected_objects: list) -> bool:
        """Check if cached labels are still valid (objects haven't moved much)."""
        if not self._cache or len(self._cache) != len(detected_objects):
            return False

        # Check if all objects are within threshold of cached positions
        for obj in detected_objects:
            matched = False
            for cached in self._cache:
                dx = obj.x_mm - cached.x_mm
                dy = obj.y_mm - cached.y_mm
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < _CACHE_MOVE_THRESHOLD_MM:
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _apply_cache(self, detected_objects: list):
        """Apply cached labels to detected objects."""
        for obj in detected_objects:
            for cached in self._cache:
                dx = obj.x_mm - cached.x_mm
                dy = obj.y_mm - cached.y_mm
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < _CACHE_MOVE_THRESHOLD_MM:
                    obj.label = cached.label
                    try:
                        obj.category = cached.category
                    except AttributeError:
                        object.__setattr__(obj, "category", cached.category)
                    try:
                        obj.llm_confidence = cached.llm_confidence
                    except AttributeError:
                        object.__setattr__(obj, "llm_confidence", cached.llm_confidence)
                    break

    def _update_cache(self, detected_objects: list):
        """Update cache with current labels."""
        self._cache = []
        for obj in detected_objects:
            self._cache.append(
                _CachedLabel(
                    x_mm=obj.x_mm,
                    y_mm=obj.y_mm,
                    label=obj.label,
                    category=getattr(obj, "category", "unknown"),
                    llm_confidence=getattr(obj, "llm_confidence", 0.0),
                    timestamp=time.monotonic(),
                )
            )

    @staticmethod
    def _color_name_from_bgr(bgr: tuple) -> str:
        """Simple BGR to color name."""
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v < 40:
            return "dark"
        if s < 40:
            return "white" if v > 180 else "gray"
        if h < 8 or h >= 165:
            return "red"
        if h < 22:
            return "orange"
        if h < 35:
            return "yellow"
        if h < 80:
            return "green"
        if h < 130:
            return "blue"
        if h < 150:
            return "purple"
        return "pink"
