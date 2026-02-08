"""
th3cl4w Vision Module — Stereo Vision + Workspace Mapping + Arm Tracking

Provides stereo camera calibration, depth estimation, object detection,
bifocal workspace mapping, dual-camera arm tracking, and visual grasp planning.
Requires opencv-python (cv2) which may not be installed in all environments.
"""

try:
    from .calibration import StereoCalibrator
    from .stereo_depth import StereoDepthEstimator
    from .object_detection import ObjectDetector
    from .workspace_mapper import WorkspaceMapper
    from .arm_tracker import DualCameraArmTracker, TrackedObject
    from .grasp_planner import VisualGraspPlanner, GraspPlan
except ImportError:
    StereoCalibrator = None  # type: ignore[assignment,misc]
    StereoDepthEstimator = None  # type: ignore[assignment,misc]
    ObjectDetector = None  # type: ignore[assignment,misc]
    WorkspaceMapper = None  # type: ignore[assignment,misc]
    DualCameraArmTracker = None  # type: ignore[assignment,misc]
    TrackedObject = None  # type: ignore[assignment,misc]
    VisualGraspPlanner = None  # type: ignore[assignment,misc]
    GraspPlan = None  # type: ignore[assignment,misc]

__all__ = [
    "StereoCalibrator",
    "StereoDepthEstimator",
    "ObjectDetector",
    "WorkspaceMapper",
    "DualCameraArmTracker",
    "TrackedObject",
    "VisualGraspPlanner",
    "GraspPlan",
]
