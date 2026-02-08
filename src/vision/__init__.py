"""
th3cl4w Vision Module — Independent Camera Views + Object Detection

Provides object detection, workspace mapping, dual-camera arm tracking,
visual grasp planning, scene analysis, claw position prediction,
object dimension estimation, world model, and startup scanning.
Each camera operates independently (no stereo pair).
Requires opencv-python (cv2) which may not be installed in all environments.
"""

try:
    from .object_detection import ObjectDetector
    from .workspace_mapper import WorkspaceMapper
    from .arm_tracker import DualCameraArmTracker, TrackedObject
    from .grasp_planner import VisualGraspPlanner, GraspPlan
    from .scene_analyzer import SceneAnalyzer, SceneDescription, SceneObject
    from .claw_position import ClawPositionPredictor
    from .dimension_estimator import ObjectDimensionEstimator, DimensionEstimate
    from .world_model import WorldModel, WorldModelSnapshot, WorldObject
    from .startup_scanner import StartupScanner, StartupScanReport
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
    ClawPositionPredictor = None  # type: ignore[assignment,misc]
    ObjectDimensionEstimator = None  # type: ignore[assignment,misc]
    DimensionEstimate = None  # type: ignore[assignment,misc]
    WorldModel = None  # type: ignore[assignment,misc]
    WorldModelSnapshot = None  # type: ignore[assignment,misc]
    WorldObject = None  # type: ignore[assignment,misc]
    StartupScanner = None  # type: ignore[assignment,misc]
    StartupScanReport = None  # type: ignore[assignment,misc]

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
    "ClawPositionPredictor",
    "ObjectDimensionEstimator",
    "DimensionEstimate",
    "WorldModel",
    "WorldModelSnapshot",
    "WorldObject",
    "StartupScanner",
    "StartupScanReport",
]
