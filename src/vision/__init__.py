"""
th3cl4w Vision Module — Independent Camera Views + Object Detection

Provides object detection, workspace mapping, dual-camera arm tracking,
visual grasp planning, and scene analysis.
Each camera operates independently (no stereo pair).
Requires opencv-python (cv2) which may not be installed in all environments.
"""

try:
    from .object_detection import ObjectDetector
    from .workspace_mapper import WorkspaceMapper
    from .arm_tracker import DualCameraArmTracker, TrackedObject
    from .grasp_planner import VisualGraspPlanner, GraspPlan
    from .scene_analyzer import SceneAnalyzer, SceneDescription, SceneObject
except ImportError:
    ObjectDetector = None  # type: ignore[assignment,misc]
    WorkspaceMapper = None  # type: ignore[assignment,misc]
    DualCameraArmTracker = None  # type: ignore[assignment,misc]
    TrackedObject = None  # type: ignore[assignment,misc]
    VisualGraspPlanner = None  # type: ignore[assignment,misc]
    GraspPlan = None  # type: ignore[assignment,misc]
    SceneAnalyzer = None  # type: ignore[assignment,misc]
    SceneDescription = None  # type: ignore[assignment,misc]
    SceneObject = None  # type: ignore[assignment,misc]

__all__ = [
    "ObjectDetector",
    "WorkspaceMapper",
    "DualCameraArmTracker",
    "TrackedObject",
    "VisualGraspPlanner",
    "GraspPlan",
    "SceneAnalyzer",
    "SceneDescription",
    "SceneObject",
]
