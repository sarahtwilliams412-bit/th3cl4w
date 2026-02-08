"""
th3cl4w Vision Module — Phase 1: Stereo Vision

Provides stereo camera calibration, depth estimation, and basic object detection.
"""

from .calibration import StereoCalibrator
from .stereo_depth import StereoDepthEstimator
from .object_detection import ObjectDetector

__all__ = ["StereoCalibrator", "StereoDepthEstimator", "ObjectDetector"]
