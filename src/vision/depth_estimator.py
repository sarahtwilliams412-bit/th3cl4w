"""Monocular depth estimation using Depth Anything v2 (or MiDaS fallback).

Provides `estimate_depth(frame_bgr) -> depth_map` (HxW float32, relative depth).
Gracefully degrades if torch/transformers aren't installed.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_processor = None
_backend: Optional[str] = None  # "depth_anything" or "midas"


def _load_model():
    """Lazy-load the depth model on first call. Caches globally."""
    global _model, _processor, _backend

    if _model is not None:
        return True

    # Try Depth Anything v2 first
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        logger.info("Loading Depth Anything v2 (%s)...", model_name)
        _processor = AutoImageProcessor.from_pretrained(model_name)
        _model = AutoModelForDepthEstimation.from_pretrained(model_name)
        _model.eval()
        _backend = "depth_anything"
        logger.info("Depth Anything v2 loaded successfully")
        return True
    except Exception as e:
        logger.warning("Depth Anything v2 unavailable: %s", e)

    # Fallback to MiDaS
    try:
        import torch

        logger.info("Loading MiDaS (small)...")
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _processor = midas_transforms.small_transform
        _backend = "midas"
        logger.info("MiDaS loaded successfully")
        return True
    except Exception as e:
        logger.warning("MiDaS unavailable: %s", e)

    logger.error("No depth estimation model available. Install: pip install transformers torch torchvision")
    return False


def is_available() -> bool:
    """Check if depth estimation is available (loads model if needed)."""
    return _load_model()


def get_backend() -> Optional[str]:
    """Return the active backend name, or None if unavailable."""
    _load_model()
    return _backend


def estimate_depth(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Estimate relative depth from a BGR image.

    Parameters
    ----------
    frame_bgr : HxWx3 uint8 BGR image (OpenCV format)

    Returns
    -------
    depth_map : HxW float32 array with relative depth values (larger = farther).
                None if model is unavailable.
    """
    if not _load_model():
        return None

    import torch
    import cv2

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if _backend == "depth_anything":
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        inputs = _processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    elif _backend == "midas":
        input_batch = _processor(frame_rgb)
        if isinstance(input_batch, dict):
            input_batch = input_batch["pixel_values"]
        if not isinstance(input_batch, torch.Tensor):
            input_batch = torch.from_numpy(input_batch)
        if input_batch.dim() == 3:
            input_batch = input_batch.unsqueeze(0)
        with torch.no_grad():
            prediction = _model(input_batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    else:
        return None

    # Normalize to 0-1 range (relative depth)
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


def estimate_metric_depth(
    frame_bgr: np.ndarray,
    known_distance_m: float = 0.5,
    reference_depth_ratio: float = 0.5,
) -> Optional[np.ndarray]:
    """Estimate metric depth by scaling relative depth using a known reference distance.

    Parameters
    ----------
    frame_bgr : BGR image
    known_distance_m : known distance from camera to workspace (meters)
    reference_depth_ratio : the relative depth value that should map to known_distance_m

    Returns
    -------
    depth_map : HxW float32 in meters
    """
    rel_depth = estimate_depth(frame_bgr)
    if rel_depth is None:
        return None

    # Scale: reference_depth_ratio maps to known_distance_m
    # Avoid division by zero
    scale = known_distance_m / max(reference_depth_ratio, 1e-6)
    metric = rel_depth * scale

    return metric
