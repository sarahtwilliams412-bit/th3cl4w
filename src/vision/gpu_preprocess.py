"""
th3cl4w Vision — GPU-Accelerated Frame Preprocessing

Shared utility for GPU-accelerated (OpenCL) frame decoding, resizing,
and color conversion. Falls back to CPU transparently when no GPU is available.
"""

import cv2
import numpy as np

# Initialize OpenCL
_has_opencl = cv2.ocl.haveOpenCL()
if _has_opencl:
    cv2.ocl.setUseOpenCL(True)


def decode_jpeg_gpu(jpeg_bytes: bytes) -> cv2.UMat:
    """Decode JPEG bytes and upload to GPU memory."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG")
    return cv2.UMat(frame) if _has_opencl else frame


def resize_gpu(frame, width, height):
    """GPU-accelerated resize."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale_gpu(frame):
    """GPU-accelerated BGR to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_numpy(frame):
    """Convert UMat back to numpy if needed."""
    return frame.get() if isinstance(frame, cv2.UMat) else frame


def gpu_status() -> dict:
    """Return GPU compute status."""
    return {
        "opencl_available": _has_opencl,
        "opencl_enabled": cv2.ocl.useOpenCL() if _has_opencl else False,
        "device": cv2.ocl.Device.getDefault().name() if _has_opencl else None,
    }
