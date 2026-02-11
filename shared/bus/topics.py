"""Topic name constants for the message bus.

All services should use these constants rather than hardcoded strings
to ensure consistency across the system.
"""


class Topics:
    """Message bus topic names."""

    # Arm state (published by Control Plane at ~10Hz)
    ARM_STATE = "arm.state"
    ARM_COMMAND = "arm.command"
    ARM_SAFETY = "arm.safety"
    ARM_ESTOP = "arm.estop"

    # Camera (published by Camera service)
    CAMERA_FRAME = "camera.frame"          # camera.frame.{cam_id}
    CAMERA_STATUS = "camera.status"

    # Object detection (published by Object ID)
    OBJECTS_DETECTED = "objects.detected"
    OBJECTS_UPDATED = "objects.updated"
    OBJECTS_REMOVED = "objects.removed"

    # World model (published by World Model service)
    WORLD_UPDATED = "world.updated"

    # Local model (published by Local Model service)
    LOCAL_UPDATED = "local.updated"

    # Mapping (published by Mapping service)
    MAP_SCENE = "map.scene"
    MAP_COLLISION = "map.collision"

    # Planning (published by Kinematics App)
    PLAN_REQUESTED = "plan.requested"
    PLAN_COMPUTED = "plan.computed"
    PLAN_EXECUTING = "plan.executing"

    # Tasks (published by Tasker)
    TASK_QUEUED = "task.queued"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    # Simulation (published by Simulation service)
    SIM_STATE = "sim.state"
    SIM_RESULT = "sim.result"

    # Telemetry (published by Telemetry service)
    TELEMETRY_EVENT = "telemetry.event"

    @classmethod
    def camera_frame(cls, cam_id: int) -> str:
        """Topic for a specific camera's frame notifications."""
        return f"{cls.CAMERA_FRAME}.{cam_id}"
