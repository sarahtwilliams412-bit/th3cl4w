"""
th3cl4w Vision — GPU-Accelerated Frame Preprocessing

Shared utility for GPU-accelerated (OpenCL) frame decoding, resizing,
and color conversion. Falls back to CPU transparently when no GPU is available.

On AMD GPUs with Mesa rusticl, the HSV color-conversion kernel may fail to
build (missing ``convert_uchar_sat_rte``).  We probe once at import time and
route HSV through CPU when the kernel is broken, avoiding repeated stderr spam.
"""

import logging
import os
import select

import cv2
import numpy as np

logger = logging.getLogger("th3cl4w.vision.gpu_preprocess")

# ── OpenCL initialisation ────────────────────────────────────────────

_has_opencl = cv2.ocl.haveOpenCL()
if _has_opencl:
    cv2.ocl.setUseOpenCL(True)


def _probe_hsv_kernel() -> bool:
    """Test whether the OpenCL HSV kernel compiles on this GPU.

    On AMD GPUs with Mesa rusticl, the HSV kernel fails to build.  OpenCV
    falls back to CPU silently but the driver dumps build errors to
    stdout/stderr.  We capture output via a pipe during the first (cached)
    build attempt so subsequent calls are clean.

    Returns True if the GPU kernel compiled successfully.
    """
    try:
        probe = cv2.UMat(np.zeros((2, 2, 3), dtype=np.uint8))
        r_fd, w_fd = os.pipe()
        saved_fds = (os.dup(1), os.dup(2))
        os.dup2(w_fd, 1)
        os.dup2(w_fd, 2)
        os.close(w_fd)
        try:
            cv2.cvtColor(probe, cv2.COLOR_BGR2HSV)
        finally:
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            os.close(saved_fds[0])
            os.close(saved_fds[1])

        # Read captured output (non-blocking)
        ready, _, _ = select.select([r_fd], [], [], 0.1)
        captured = os.read(r_fd, 16384).decode(errors="replace") if ready else ""
        os.close(r_fd)

        if "BUILD_PROGRAM_FAILURE" in captured:
            logger.warning(
                "OpenCL HSV kernel build failed (rusticl bug) — HSV will use CPU. "
                "Other GPU operations remain accelerated."
            )
            return False
        return True
    except Exception:
        logger.warning("OpenCL HSV probe raised an exception — HSV will use CPU")
        return False


# Probe once at import; result cached for the process lifetime.
_hsv_gpu_ok = _probe_hsv_kernel() if _has_opencl else False


def decode_jpeg_gpu(jpeg_bytes: bytes) -> cv2.UMat:
    """Decode JPEG bytes and upload to GPU memory."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG")
    return cv2.UMat(frame) if _has_opencl else frame


def resize_gpu(frame, width: int, height: int):
    """GPU-accelerated resize."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale_gpu(frame):
    """GPU-accelerated BGR to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_hsv(frame):
    """BGR → HSV, using GPU when the kernel is available.

    Falls back to CPU on devices where the OpenCL HSV kernel fails to
    build (e.g. AMD RX 580 with Mesa rusticl).
    """
    if _hsv_gpu_ok:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # CPU path — avoids triggering the broken kernel again
    np_frame = frame.get() if isinstance(frame, cv2.UMat) else frame
    hsv_np = cv2.cvtColor(np_frame, cv2.COLOR_BGR2HSV)
    # Re-upload if caller expects UMat
    return cv2.UMat(hsv_np) if isinstance(frame, cv2.UMat) else hsv_np


def to_numpy(frame):
    """Convert UMat back to numpy if needed."""
    return frame.get() if isinstance(frame, cv2.UMat) else frame


def gpu_status() -> dict:
    """Return GPU compute status."""
    return {
        "opencl_available": _has_opencl,
        "opencl_enabled": cv2.ocl.useOpenCL() if _has_opencl else False,
        "hsv_gpu": _hsv_gpu_ok,
        "device": cv2.ocl.Device.getDefault().name() if _has_opencl else None,
    }
