"""Pydantic models for service health event messages."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ServiceHealthMessage(BaseModel):
    """Service health heartbeat broadcast on the message bus."""

    service_name: str = Field(description="Name of the reporting service")
    status: str = Field(description="Status: healthy, degraded, unhealthy")
    port: int = Field(description="Service port number")
    uptime_s: float = Field(default=0.0, description="Uptime in seconds")
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="Dependency statuses: {name: 'healthy'|'unavailable'}"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Service-specific metrics (e.g., queue_depth, detection_count)"
    )
    timestamp: float = Field(description="Unix timestamp")
