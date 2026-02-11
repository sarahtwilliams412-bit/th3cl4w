"""Pydantic models for scene/map data."""

from typing import Optional

from pydantic import BaseModel, Field


class SceneObject(BaseModel):
    """An object in the 3D scene."""
    id: str
    label: str
    position: list[float] = Field(description="[x, y, z] in meters")
    dimensions: list[float] = Field(default=[0.0, 0.0, 0.0], description="[w, h, d] in meters")
    color: list[float] = Field(default=[0.5, 0.5, 0.5], description="[r, g, b] 0-1")
    category: str = "unknown"


class ArmSkeleton(BaseModel):
    """Arm skeleton for 3D visualization."""
    joint_positions: list[list[float]] = Field(description="List of [x, y, z] for each joint")
    link_radii: list[float] = Field(default_factory=list, description="Visual radius per link")


class SceneUpdateMessage(BaseModel):
    """Scene state broadcast by the Mapping service."""

    arm: Optional[ArmSkeleton] = None
    objects: list[SceneObject] = Field(default_factory=list)
    point_cloud_size: int = Field(default=0, description="Number of points in environment cloud")
    bounds_min: list[float] = Field(default=[-1.0, -1.0, 0.0])
    bounds_max: list[float] = Field(default=[1.0, 1.0, 1.0])
    timestamp: float = 0.0
