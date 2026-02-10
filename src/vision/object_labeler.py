"""
Vision-based Object Labeler — Uses Gemini Flash to identify each detected object.

For every detected object, crops the region from BOTH overhead and side camera
frames, draws a bright indicator circle on the target object so Gemini knows
which one to look at, and sends both views to Gemini 2.0 Flash for identification.

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

# Rate limit: minimum seconds between API call batches
_MIN_CALL_INTERVAL_S = 2.0

# Per-object call spacing to avoid hitting Gemini rate limits
_PER_OBJECT_DELAY_S = 0.5

# Cache: don't re-label if objects moved less than this
_CACHE_MOVE_THRESHOLD_MM = 20.0

# Side camera horizontal FOV mapped to workspace X range
_SIDE_CAM_WS_X_MIN = -400.0
_SIDE_CAM_WS_X_MAX = 400.0

# Crop padding around detected objects (fraction of bounding box size)
_CROP_PAD_FACTOR = 0.6

# Max crop dimension before resizing for Gemini
_MAX_CROP_DIM = 512

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


def _build_per_object_prompt(ontology_labels: list[str], color_hint: str, shape_hint: str) -> str:
    """Build a focused prompt for identifying a single object from two camera views."""
    labels_str = ", ".join(ontology_labels[:30])
    return (
        "You are a robotic arm vision system. I am showing you TWO camera views "
        "(overhead and side) of the SAME workspace. In each image, the specific "
        "object I want you to identify is marked with a bright GREEN CIRCLE.\n\n"
        "Focus ONLY on the object inside the green circle. Ignore all other objects.\n\n"
        f"Detection hints: the object appears {color_hint} in color and has a "
        f"{shape_hint} shape from overhead.\n\n"
        "Identify what this object is. Use these standard labels when applicable: "
        f"{labels_str}\n\n"
        "Respond ONLY with a single JSON object (not an array):\n"
        '{"label": "snake_case_name", "category": "one of: beverage_can, bottle, '
        'cup_mug, hand_tool, office_supply, small_box, ball, electronic, food_item, '
        'toy, container, or unknown", "confidence": 0.0-1.0}\n\n'
        "Examples:\n"
        '{"label": "coca_cola_can", "category": "beverage_can", "confidence": 0.92}\n'
        '{"label": "tennis_ball", "category": "ball", "confidence": 0.85}\n'
        '{"label": "blue_marker", "category": "office_supply", "confidence": 0.78}\n\n'
        "Respond ONLY with JSON. No markdown, no explanation."
    )


def _crop_and_mark(
    frame: np.ndarray,
    cx: int,
    cy: int,
    box_w: int,
    box_h: int,
) -> Optional[np.ndarray]:
    """Crop a region around (cx, cy) from the frame and draw a green circle indicator.

    The crop is padded around the bounding box, and a bright green circle is drawn
    around the target object center so Gemini knows exactly which object to look at.
    """
    if frame is None:
        return None

    fh, fw = frame.shape[:2]
    if fw == 0 or fh == 0:
        return None

    # Calculate padded crop region
    pad_w = int(box_w * _CROP_PAD_FACTOR)
    pad_h = int(box_h * _CROP_PAD_FACTOR)

    # Minimum crop size so we get enough context
    min_crop = 120
    crop_w = max(box_w + 2 * pad_w, min_crop)
    crop_h = max(box_h + 2 * pad_h, min_crop)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(fw, x1 + crop_w)
    y2 = min(fh, y1 + crop_h)

    # Adjust if we hit an edge
    if x2 - x1 < min_crop and fw >= min_crop:
        if x1 == 0:
            x2 = min(fw, min_crop)
        else:
            x1 = max(0, x2 - min_crop)
    if y2 - y1 < min_crop and fh >= min_crop:
        if y1 == 0:
            y2 = min(fh, min_crop)
        else:
            y1 = max(0, y2 - min_crop)

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    # Draw a bright green circle around the target object center within the crop
    local_cx = cx - x1
    local_cy = cy - y1
    radius = max(box_w, box_h) // 2 + 8
    cv2.circle(crop, (local_cx, local_cy), radius, (0, 255, 0), 3, cv2.LINE_AA)
    # Also draw a small crosshair
    cross = 6
    cv2.line(crop, (local_cx - cross, local_cy), (local_cx + cross, local_cy),
             (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(crop, (local_cx, local_cy - cross), (local_cx, local_cy + cross),
             (0, 255, 0), 2, cv2.LINE_AA)

    # Resize if too large
    ch, cw = crop.shape[:2]
    if max(cw, ch) > _MAX_CROP_DIM:
        ratio = _MAX_CROP_DIM / max(cw, ch)
        crop = cv2.resize(crop, (int(cw * ratio), int(ch * ratio)), interpolation=cv2.INTER_AREA)

    return crop


def _estimate_side_position(
    obj_x_mm: float,
    obj_height_mm: float,
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int]:
    """Estimate where an object appears in the side camera view.

    Returns (cx, cy, est_width, est_height) in pixel coordinates.
    """
    # Horizontal: map workspace X to side camera horizontal position
    norm_x = (obj_x_mm - _SIDE_CAM_WS_X_MIN) / (_SIDE_CAM_WS_X_MAX - _SIDE_CAM_WS_X_MIN)
    norm_x = max(0.05, min(0.95, norm_x))
    cx = int(norm_x * frame_w)

    # Vertical: table surface is ~85% down; object center is above that
    table_y_frac = 0.85
    table_y = int(frame_h * table_y_frac)
    # Map object height to pixel extent above table
    max_height_mm = 300.0
    height_frac = min(obj_height_mm / max_height_mm, 1.0)
    pixel_height = int(height_frac * table_y)
    est_h = max(pixel_height, 30)
    cy = table_y - est_h // 2

    # Estimated width in side view (rough)
    est_w = max(int(est_h * 0.7), 30)

    return cx, cy, est_w, est_h


class ObjectLabeler:
    """Labels detected objects using per-object Gemini Flash vision API calls.

    For each object, crops both overhead and side camera views around that
    specific object, draws a green indicator circle, and sends both images
    to Gemini to identify what it is.

    Thread-safe. Caches results and rate-limits API calls.
    """

    def __init__(self):
        self._ontology = _load_ontology()
        self._ontology_labels = _get_ontology_labels(self._ontology)
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

    @property
    def ontology(self) -> dict:
        return self._ontology

    def label_objects(
        self,
        overhead_frame: Optional[np.ndarray],
        side_frame: Optional[np.ndarray],
        detected_objects: list,
    ) -> list:
        """Label each detected object using cropped dual-camera views + Gemini.

        For every object in detected_objects:
          1. Crops the overhead frame around the object's bounding box
          2. Estimates the object's position in the side frame and crops there
          3. Draws a green circle on both crops to highlight the target object
          4. Sends both crops to Gemini asking "what is this specific object?"
          5. Updates the object's label, category, and llm_confidence

        Args:
            overhead_frame: BGR frame from overhead camera (cam0).
            side_frame: BGR frame from side camera (cam2). Optional.
            detected_objects: List of DetectedObject instances from the detector.

        Returns the same list for convenience.
        """
        if not detected_objects:
            return detected_objects

        # Check cache — skip if objects haven't moved
        if self._cache_valid(detected_objects):
            self._apply_cache(detected_objects)
            return detected_objects

        # Need at least one frame
        if overhead_frame is None and side_frame is None:
            self._apply_fallback(detected_objects)
            return detected_objects

        self._ensure_api()
        if self._model is None or not self._can_call():
            self._apply_fallback(detected_objects)
            return detected_objects

        # Label each object individually with dedicated Gemini calls
        any_labeled = False
        ontology_objects = self._ontology.get("objects", {})

        for obj in detected_objects:
            images = []
            color_hint = self._color_name_from_bgr(obj.color_bgr) if hasattr(obj, "color_bgr") else "unknown"
            shape_hint = getattr(obj, "shape", "irregular")

            # Crop from overhead camera
            if overhead_frame is not None and hasattr(obj, "bbox_overhead"):
                bx, by, bw, bh = obj.bbox_overhead
                if bw > 0 and bh > 0:
                    oh_cx = bx + bw // 2
                    oh_cy = by + bh // 2
                    oh_crop = _crop_and_mark(overhead_frame, oh_cx, oh_cy, bw, bh)
                    if oh_crop is not None:
                        images.append(("overhead", oh_crop))

            # Crop from side camera
            if side_frame is not None:
                sh, sw = side_frame.shape[:2]
                s_cx, s_cy, s_w, s_h = _estimate_side_position(
                    obj.x_mm, getattr(obj, "height_mm", 50.0), sh, sw
                )
                side_crop = _crop_and_mark(side_frame, s_cx, s_cy, s_w, s_h)
                if side_crop is not None:
                    images.append(("side", side_crop))

            if not images:
                continue

            # Call Gemini for this specific object
            result = self._call_gemini_per_object(images, color_hint, shape_hint)
            if result:
                label = result.get("label", obj.label)
                category = result.get("category", "unknown")
                confidence = result.get("confidence", 0.5)

                # Validate against ontology
                if label in ontology_objects:
                    ont_entry = ontology_objects[label]
                    category = ont_entry.get("category", category)

                # Clean up label for display: snake_case -> readable
                obj.label = label
                obj.category = category
                obj.llm_confidence = float(confidence)
                any_labeled = True

                logger.info("Object %d labeled as '%s' (category=%s, conf=%.2f)",
                            obj.obj_id, label, category, confidence)
            else:
                # Keep color-based fallback label
                if not obj.category:
                    obj.category = "unknown"
                if not obj.llm_confidence:
                    obj.llm_confidence = 0.0

            # Small delay between per-object calls to avoid rate limits
            if len(detected_objects) > 1:
                time.sleep(_PER_OBJECT_DELAY_S)

        if any_labeled:
            self._update_cache(detected_objects)

        return detected_objects

    def _can_call(self) -> bool:
        """Check rate limit."""
        now = time.monotonic()
        return (now - self._last_call_time) >= _MIN_CALL_INTERVAL_S

    def _call_gemini_per_object(
        self,
        images: list[tuple[str, np.ndarray]],
        color_hint: str,
        shape_hint: str,
    ) -> Optional[dict]:
        """Send cropped images of a single object to Gemini and parse response."""
        try:
            import PIL.Image

            prompt = _build_per_object_prompt(self._ontology_labels, color_hint, shape_hint)
            content_parts = [prompt]

            for view_name, crop_bgr in images:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(rgb)
                content_parts.append(pil_img)

            self._last_call_time = time.monotonic()
            response = self._model.generate_content(
                content_parts,
                generation_config={"temperature": 0.1, "max_output_tokens": 256},
            )

            text = response.text.strip()
            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            if isinstance(result, dict) and "label" in result:
                return result
            # If Gemini returned an array, take the first item
            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], dict) else None
            return None

        except Exception as e:
            logger.warning("Gemini per-object call failed: %s", e)
            return None

    def _apply_fallback(self, detected_objects: list):
        """Apply fallback attributes when Gemini isn't available."""
        for obj in detected_objects:
            if not getattr(obj, "category", None):
                obj.category = "unknown"
            if not getattr(obj, "llm_confidence", None):
                obj.llm_confidence = 0.0

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
                    obj.category = cached.category
                    obj.llm_confidence = cached.llm_confidence
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
