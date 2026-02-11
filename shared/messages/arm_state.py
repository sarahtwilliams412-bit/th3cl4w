"""Pydantic models for arm state messages."""

from pydantic import BaseModel, Field


class ArmStateMessage(BaseModel):
    """Arm state broadcast on the message bus at 10Hz."""

    joint_angles_deg: list[float] = Field(description="6 arm joint angles in degrees")
    joint_velocities: list[float] = Field(description="7 joint velocities in rad/s")
    joint_torques: list[float] = Field(description="7 joint torques in Nm")
    gripper_mm: float = Field(description="Gripper opening in mm (0-65)")
    gripper_force: float = Field(default=0.0, description="Gripper force estimate")
    powered: bool = Field(default=False, description="Whether arm is powered on")
    enabled: bool = Field(default=False, description="Whether arm is enabled for motion")
    estop: bool = Field(default=False, description="Whether e-stop is active")
    sim_mode: bool = Field(default=False, description="Whether running in simulation mode")
    connected: bool = Field(default=False, description="Whether DDS connection is live")
    timestamp: float = Field(description="Unix timestamp of this state reading")

    class Config:
        json_schema_extra = {
            "example": {
                "joint_angles_deg": [0.0, -45.0, 0.0, 90.0, 0.0, -45.0],
                "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "joint_torques": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "gripper_mm": 30.0,
                "gripper_force": 0.0,
                "powered": True,
                "enabled": True,
                "estop": False,
                "sim_mode": False,
                "connected": True,
                "timestamp": 1700000000.0,
            }
        }
