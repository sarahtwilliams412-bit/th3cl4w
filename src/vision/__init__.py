"""
th3cl4w Vision Module — Stereo Vision + Workspace Mapping

Provides stereo camera calibration, depth estimation, object detection,
and bifocal workspace mapping for arm planning.
Requires opencv-python (cv2) which may not be installed in all environments.
"""

try:
    from .calibration import StereoCalibrator
    from .stereo_depth import StereoDepthEstimator
    from .object_detection import ObjectDetector
    from .workspace_mapper import WorkspaceMapper
except ImportError:
    StereoCalibrator = None  # type: ignore[assignment,misc]
    StereoDepthEstimator = None  # type: ignore[assignment,misc]
    ObjectDetector = None  # type: ignore[assignment,misc]
    WorkspaceMapper = None  # type: ignore[assignment,misc]

__all__ = ["StereoCalibrator", "StereoDepthEstimator", "ObjectDetector", "WorkspaceMapper"]
