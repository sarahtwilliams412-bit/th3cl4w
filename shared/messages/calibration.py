"""Pydantic models for calibration event messages."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class CalibrationResultMessage(BaseModel):
    """Calibration result broadcast on the message bus."""

    run_id: str = Field(description="Unique calibration run ID")
    calibration_type: str = Field(description="Type: intrinsic, extrinsic, joint_mapping, hand_eye")
    camera_id: Optional[int] = Field(default=None, description="Camera ID (if applicable)")
    success: bool = Field(description="Whether calibration succeeded")
    reprojection_error: Optional[float] = Field(
        default=None, description="RMS reprojection error in pixels"
    )
    num_images_used: int = Field(default=0, description="Number of images used")
    result_path: Optional[str] = Field(default=None, description="Path to saved result file")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_s: float = Field(default=0.0, description="Calibration duration in seconds")
    timestamp: float = Field(description="Unix timestamp")
