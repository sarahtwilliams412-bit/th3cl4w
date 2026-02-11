"""Pydantic models for detected objects."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ObjectCategory(str, Enum):
    """Categories for detected objects."""
    TARGET = "target"
    OBSTACLE = "obstacle"
    BOUNDARY = "boundary"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


class ReachStatus(str, Enum):
    """Reachability classification for objects."""
    REACHABLE = "reachable"
    OUT_OF_RANGE = "out_of_range"
    TOO_CLOSE = "too_close"
    MARGINAL = "marginal"


class DetectedObjectMessage(BaseModel):
    """Object detected by the Object ID service, published to message bus."""

    id: str = Field(description="Unique object identifier")
    label: str = Field(description="Object label/name (e.g., 'red ball', 'redbull can')")
    position_mm: list[float] = Field(description="[x, y, z] position in mm relative to arm base")
    dimensions_mm: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="[width, height, depth] in mm"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")
    category: ObjectCategory = Field(default=ObjectCategory.UNKNOWN)
    reach_status: ReachStatus = Field(default=ReachStatus.REACHABLE)
    distance_mm: float = Field(default=0.0, description="Distance from arm base in mm")
    source_camera_id: int = Field(default=-1, description="Camera that detected this object")
    observation_count: int = Field(default=1, description="Number of times observed")
    characteristics: dict = Field(
        default_factory=dict,
        description="Object characteristics: color, shape, material, etc."
    )
    first_seen: float = Field(default=0.0, description="Unix timestamp of first detection")
    last_seen: float = Field(default=0.0, description="Unix timestamp of latest detection")
    stable: bool = Field(default=False, description="Whether position has stabilized")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "obj_001",
                "label": "red ball",
                "position_mm": [200.0, 100.0, 50.0],
                "dimensions_mm": [40.0, 40.0, 40.0],
                "confidence": 0.92,
                "category": "target",
                "reach_status": "reachable",
                "distance_mm": 245.0,
                "source_camera_id": 2,
                "observation_count": 15,
                "characteristics": {"color": "red", "shape": "spherical"},
                "first_seen": 1700000000.0,
                "last_seen": 1700000030.0,
                "stable": True,
            }
        }
