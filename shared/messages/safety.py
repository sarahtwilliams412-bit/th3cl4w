"""Pydantic models for safety alert messages."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SafetyViolationType(str, Enum):
    """Types of safety violations."""
    POSITION_LIMIT = "position_limit"
    VELOCITY_LIMIT = "velocity_limit"
    TORQUE_LIMIT = "torque_limit"
    COLLISION = "collision"
    STALL = "stall"
    ESTOP = "estop"
    WORKSPACE_BOUNDARY = "workspace_boundary"


class SafetyAlertMessage(BaseModel):
    """Safety alert broadcast on the message bus."""

    violation_type: SafetyViolationType = Field(description="Type of safety violation")
    severity: str = Field(default="warning", description="Severity: info, warning, critical")
    joint_id: Optional[int] = Field(default=None, description="Affected joint (if applicable)")
    value: Optional[float] = Field(default=None, description="Measured value that triggered alert")
    limit: Optional[float] = Field(default=None, description="Limit that was exceeded")
    message: str = Field(default="", description="Human-readable description")
    auto_action: Optional[str] = Field(default=None, description="Action taken: estop, clamp, none")
    timestamp: float = Field(description="Unix timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "violation_type": "torque_limit",
                "severity": "warning",
                "joint_id": 3,
                "value": 11.5,
                "limit": 10.0,
                "message": "Joint 3 torque exceeds limit (11.5 > 10.0 Nm)",
                "auto_action": "clamp",
                "timestamp": 1700000000.0,
            }
        }
