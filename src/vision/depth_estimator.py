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

    logger.error(
        "No depth estimation model available. Install: pip install transformers torch torchvision"
    )
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
        depth = (
            torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

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
        depth = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
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
    # Check if a calibrated model is available first
    calibrated = _load_calibrated_model()
    if calibrated is not None:
        return calibrated(frame_bgr)

    rel_depth = estimate_depth(frame_bgr)
    if rel_depth is None:
        return None

    # Scale: reference_depth_ratio maps to known_distance_m
    # Avoid division by zero
    scale = known_distance_m / max(reference_depth_ratio, 1e-6)
    metric = rel_depth * scale

    return metric


# ---------------------------------------------------------------------------
# Calibrated metric depth from sim ground truth
# ---------------------------------------------------------------------------

_calibrated_model = None
_calibrated_loaded = False


def _load_calibrated_model():
    """Load a depth model fine-tuned on MuJoCo ground-truth depth maps.

    The NVIDIA Kitchen-Sim-Demos provide MuJoCo scenes that can render
    perfect ground-truth depth. This calibrated model was trained on
    paired (RGB, metric_depth) data from those scenes, producing
    metric depth directly without the single-reference-point scaling.

    Returns a callable (frame_bgr -> depth_meters) or None.
    """
    global _calibrated_model, _calibrated_loaded

    if _calibrated_loaded:
        return _calibrated_model

    _calibrated_loaded = True

    import os
    model_path = os.path.join("data", "models", "depth_calibrated", "model.pt")
    if not os.path.exists(model_path):
        return None

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")

        # Expected: a fine-tuned Depth Anything v2 Small with metric head
        if "scale" in checkpoint and "shift" in checkpoint:
            # Simple affine calibration: metric = scale * relative + shift
            scale = float(checkpoint["scale"])
            shift = float(checkpoint["shift"])

            def _calibrated_metric(frame_bgr):
                rel = estimate_depth(frame_bgr)
                if rel is None:
                    return None
                return rel * scale + shift

            _calibrated_model = _calibrated_metric
            logger.info(
                "Loaded calibrated depth model (affine: scale=%.4f, shift=%.4f)",
                scale, shift,
            )
            return _calibrated_model

        # Full model checkpoint
        if hasattr(checkpoint, "eval"):
            checkpoint.eval()
            _calibrated_model = lambda frame: _run_calibrated_model(checkpoint, frame)
            logger.info("Loaded calibrated depth model (full checkpoint)")
            return _calibrated_model

    except Exception as e:
        logger.debug("Calibrated depth model not available: %s", e)

    return None


def _run_calibrated_model(model, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Run the calibrated depth model on a frame."""
    try:
        import torch
        import cv2
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if _processor is not None:
            inputs = _processor(images=pil_img, return_tensors="pt")
        else:
            return None

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "predicted_depth"):
                depth = outputs.predicted_depth
            else:
                depth = outputs

        depth = (
            torch.nn.functional.interpolate(
                depth.unsqueeze(1) if depth.dim() == 2 else depth.unsqueeze(0),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return depth.astype(np.float32)
    except Exception as e:
        logger.warning("Calibrated model inference failed: %s", e)
        return None


def render_sim_depth_pair(
    mujoco_model_path: str,
    joint_angles_rad: np.ndarray,
    camera_name: str = "agentview",
    width: int = 640,
    height: int = 480,
) -> Optional[tuple]:
    """Render an RGB + ground-truth depth pair from a MuJoCo scene.

    This is used to generate training data for the calibrated depth model.
    Requires MuJoCo to be installed.

    Parameters
    ----------
    mujoco_model_path : Path to MuJoCo XML scene model
    joint_angles_rad : Robot joint configuration for the scene
    camera_name : MuJoCo camera to render from
    width, height : Render resolution

    Returns
    -------
    (rgb, depth) tuple of HxWx3 uint8 and HxW float32 (meters), or None
    """
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        data = mujoco.MjData(model)

        # Set joint angles if the model has matching joints
        for i, angle in enumerate(joint_angles_rad):
            if i < model.nq:
                data.qpos[i] = angle

        mujoco.mj_forward(model, data)

        # Render RGB
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()

        # Render depth
        renderer.enable_depth_rendering(True)
        renderer.update_scene(data, camera=camera_name)
        depth = renderer.render()

        renderer.close()
        return rgb, depth.astype(np.float32)

    except ImportError:
        logger.warning("MuJoCo not available for depth ground truth rendering")
        return None
    except Exception as e:
        logger.warning("Sim depth rendering failed: %s", e)
        return None
