"""Event type definitions for the message bus."""

from enum import Enum


class EventType(str, Enum):
    """All event types published on the message bus."""

    # Arm state events
    ARM_STATE = "arm.state"
    ARM_COMMAND = "arm.command"
    ARM_SAFETY = "arm.safety"
    ARM_ESTOP = "arm.estop"

    # Camera events
    CAMERA_FRAME = "camera.frame"
    CAMERA_STATUS = "camera.status"

    # Object detection events
    OBJECTS_DETECTED = "objects.detected"
    OBJECTS_UPDATED = "objects.updated"
    OBJECTS_REMOVED = "objects.removed"

    # World model events
    WORLD_UPDATED = "world.updated"
    LOCAL_UPDATED = "local.updated"

    # Mapping events
    MAP_SCENE = "map.scene"
    MAP_COLLISION = "map.collision"

    # Planning events
    PLAN_REQUESTED = "plan.requested"
    PLAN_COMPUTED = "plan.computed"
    PLAN_EXECUTING = "plan.executing"

    # Task events
    TASK_QUEUED = "task.queued"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    # Simulation events
    SIM_STATE = "sim.state"
    SIM_RESULT = "sim.result"

    # Telemetry events
    TELEMETRY_EVENT = "telemetry.event"
