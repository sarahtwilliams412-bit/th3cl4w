"""Pydantic models for gripper command and state messages."""

from pydantic import BaseModel, Field


class GripperCommandMessage(BaseModel):
    """Gripper command broadcast on the message bus."""

    position_mm: float = Field(description="Target gripper opening in mm (0-65)")
    speed: float = Field(default=1.0, ge=0.0, le=1.0, description="Gripper speed (0-1)")
    force_limit: float = Field(default=0.5, ge=0.0, le=1.0, description="Force limit (0-1)")
    timestamp: float = Field(description="Unix timestamp")


class GripperStateMessage(BaseModel):
    """Gripper state broadcast on the message bus."""

    position_mm: float = Field(description="Current gripper opening in mm (0-65)")
    force: float = Field(default=0.0, description="Current force estimate")
    is_gripping: bool = Field(default=False, description="Whether the gripper is holding an object")
    stalled: bool = Field(default=False, description="Whether the gripper is stalled")
    timestamp: float = Field(description="Unix timestamp")
