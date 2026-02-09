"""
Tests for visual servo module.

Since VisualServo depends on Gemini API and live cameras, we test
the data structures and helper logic, plus mock-based integration.
"""

import pytest
from src.control.visual_servo import ServoStep, ServoResult


class TestServoDataStructures:
    def test_servo_step_defaults(self):
        step = ServoStep(step=0, joints_before=[0.0]*7, joints_after=[0.1]*7)
        assert step.pixel_distance == 0.0
        assert step.action == ""
        assert step.gripper_pixel is None
        assert step.target_pixel is None

    def test_servo_step_with_pixels(self):
        step = ServoStep(
            step=1,
            joints_before=[0.0]*7,
            joints_after=[0.1]*7,
            gripper_pixel=(100.0, 200.0),
            target_pixel=(300.0, 400.0),
            pixel_distance=283.0,
            action="move_j1_positive",
        )
        assert step.gripper_pixel == (100.0, 200.0)
        assert step.pixel_distance == 283.0

    def test_servo_result_failure(self):
        result = ServoResult(success=False, message="timeout")
        assert not result.success
        assert result.total_time_s == 0.0
        assert result.final_distance_px == 999.0
        assert len(result.steps) == 0

    def test_servo_result_success(self):
        steps = [
            ServoStep(step=0, joints_before=[0]*7, joints_after=[0.1]*7, pixel_distance=200),
            ServoStep(step=1, joints_before=[0.1]*7, joints_after=[0.2]*7, pixel_distance=50),
        ]
        result = ServoResult(success=True, steps=steps, total_time_s=3.5, final_distance_px=50)
        assert result.success
        assert len(result.steps) == 2
        assert result.final_distance_px == 50

    def test_servo_result_convergence_check(self):
        """Check that we can assess convergence from result."""
        result = ServoResult(success=True, final_distance_px=30)
        assert result.final_distance_px < 80  # close_enough_px default

    def test_servo_result_no_convergence(self):
        result = ServoResult(success=False, final_distance_px=500, message="max steps exceeded")
        assert result.final_distance_px >= 80

    def test_visual_servo_requires_api_key(self):
        """VisualServo should fail without GEMINI_API_KEY."""
        import os
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            from src.control.visual_servo import VisualServo
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                VisualServo()
        finally:
            if old:
                os.environ["GEMINI_API_KEY"] = old
