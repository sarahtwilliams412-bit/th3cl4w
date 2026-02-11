"""Pydantic models for task commands and plan requests."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of tasks the system can execute."""
    PICK = "pick"
    PLACE = "place"
    SCAN = "scan"
    MOVE_TO = "move_to"
    HOME = "home"
    CUSTOM = "custom"
    TEXT_COMMAND = "text_command"


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    QUEUED = "queued"
    PLANNING = "planning"
    REHEARSING = "rehearsing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskCommand(BaseModel):
    """A task to be submitted to the Tasker service."""

    task_type: TaskType = Field(description="Type of task")
    target: str = Field(default="", description="Target object label or position description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
    priority: int = Field(default=0, ge=0, le=10, description="Priority (0=normal, 10=urgent)")
    require_sim_rehearsal: bool = Field(
        default=True,
        description="Whether to simulate before real execution"
    )
    text_command: str = Field(default="", description="Natural language command (for text_command type)")

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "pick",
                "target": "red ball",
                "parameters": {"approach_height_mm": 120.0},
                "priority": 0,
                "require_sim_rehearsal": True,
            }
        }


class PlanRequest(BaseModel):
    """Request to the Kinematics App to compute a motion plan."""

    task_id: str = Field(description="ID of the parent task")
    task_type: TaskType
    target_object_id: Optional[str] = None
    target_position_mm: Optional[list[float]] = None
    target_orientation: Optional[list[float]] = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    """Result of motion planning from the Kinematics App."""

    task_id: str
    success: bool
    trajectory: list[list[float]] = Field(
        default_factory=list,
        description="List of joint angle waypoints (degrees)"
    )
    estimated_duration_s: float = 0.0
    collision_free: bool = True
    error_message: str = ""
