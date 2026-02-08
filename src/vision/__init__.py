"""
th3cl4w Vision Module — Phase 1: Stereo Vision

Provides stereo camera calibration, depth estimation, and basic object detection.
Requires opencv-python (cv2) which may not be installed in all environments.
"""

try:
    from .calibration import StereoCalibrator
    from .stereo_depth import StereoDepthEstimator
    from .object_detection import ObjectDetector
except ImportError:
    StereoCalibrator = None  # type: ignore[assignment,misc]
    StereoDepthEstimator = None  # type: ignore[assignment,misc]
    ObjectDetector = None  # type: ignore[assignment,misc]

__all__ = ["StereoCalibrator", "StereoDepthEstimator", "ObjectDetector"]
