"""Tests for multi-view fusion: side height estimator, arm camera aligner, controller."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.side_height_estimator import SideHeightEstimator
from src.vision.arm_camera_aligner import ArmCameraAligner
from src.control.multiview_controller import MultiviewController, PickPhase


# ── Helpers ──

def make_side_frame(obj_color_bgr=(0, 0, 200), obj_y=400, obj_h=80):
    """Create a synthetic side-view frame with a colored object at given Y."""
    frame = np.full((1080, 1920, 3), (200, 200, 200), dtype=np.uint8)  # grey bg
    # Draw object (red rectangle)
    x, w = 800, 120
    cv2.rectangle(frame, (x, obj_y), (x + w, obj_y + obj_h), obj_color_bgr, -1)
    return frame


def make_arm_frame(obj_cx=960, obj_cy=540, obj_r=60, color_bgr=(0, 0, 200)):
    """Create a synthetic arm-cam frame with a colored circle at given position."""
    frame = np.full((1080, 1920, 3), (180, 180, 180), dtype=np.uint8)
    cv2.circle(frame, (obj_cx, obj_cy), obj_r, color_bgr, -1)
    return frame


# ── Side Height Estimator ──

class TestSideHeightEstimator:
    def test_calibrate_and_convert(self, tmp_path):
        cal_file = tmp_path / "cal.json"
        est = SideHeightEstimator(calibration_path=str(cal_file))
        assert not est.calibrated

        # Calibrate with known points: pixel_y=100 → z=200mm, pixel_y=900 → z=0mm
        points = [(100, 200.0), (900, 0.0)]
        est.calibrate(points)
        assert est.calibrated
        assert cal_file.exists()

        # Check conversion
        z_top = est.pixel_y_to_z_mm(100)
        z_bot = est.pixel_y_to_z_mm(900)
        assert abs(z_top - 200.0) < 0.1
        assert abs(z_bot - 0.0) < 0.1

        # Mid-point
        z_mid = est.pixel_y_to_z_mm(500)
        assert abs(z_mid - 100.0) < 0.1

    def test_load_calibration(self, tmp_path):
        cal_file = tmp_path / "cal.json"
        cal_file.write_text(json.dumps({
            "image_height": 1080,
            "slope": -0.25,
            "intercept": 225.0,
            "calibrated": True,
            "points": [{"pixel_y": 100, "z_mm": 200}, {"pixel_y": 900, "z_mm": 0}],
        }))
        est = SideHeightEstimator(calibration_path=str(cal_file))
        assert est.calibrated
        assert len(est.reference_points) == 2

    def test_detect_redbull(self, tmp_path):
        est = SideHeightEstimator(calibration_path=str(tmp_path / "cal.json"))
        # Red object in side view
        frame = make_side_frame(obj_color_bgr=(0, 0, 200), obj_y=400, obj_h=80)
        det = est.detect_target(frame, target="redbull")
        assert det is not None
        assert abs(det["bottom_y"] - 480) <= 5  # obj_y + obj_h (±morphology)
        assert det["area"] > 0

    def test_estimate_height(self, tmp_path):
        cal_file = tmp_path / "cal.json"
        est = SideHeightEstimator(calibration_path=str(cal_file))
        est.calibrate([(100, 200.0), (900, 0.0)])

        frame = make_side_frame(obj_color_bgr=(0, 0, 200), obj_y=400, obj_h=80)
        z = est.estimate_height(frame, target="redbull")
        assert z is not None
        # bottom_y = 480, z = slope*480 + intercept
        # slope = -0.25, intercept = 225 => z = -0.25*480 + 225 = 105
        assert 80 < z < 130  # reasonable range

    def test_no_detection_returns_none(self, tmp_path):
        est = SideHeightEstimator(calibration_path=str(tmp_path / "cal.json"))
        est.calibrate([(100, 200.0), (900, 0.0)])
        # Plain grey frame, no red objects
        frame = np.full((1080, 1920, 3), (180, 180, 180), dtype=np.uint8)
        z = est.estimate_height(frame, target="redbull")
        assert z is None


# ── Arm Camera Aligner ──

class TestArmCameraAligner:
    def test_centered_object(self):
        aligner = ArmCameraAligner()
        # Object at center
        frame = make_arm_frame(obj_cx=960, obj_cy=540)
        result = aligner.compute_alignment(frame)
        assert result["detected"]
        assert result["centered"]
        assert result["distance_px"] < aligner.tolerance_px

    def test_off_center_object(self):
        aligner = ArmCameraAligner()
        # Object far from center
        frame = make_arm_frame(obj_cx=200, obj_cy=200)
        result = aligner.compute_alignment(frame)
        assert result["detected"]
        assert not result["centered"]
        assert result["distance_px"] > 100
        # Correction should point toward the object
        dx, dy = result["correction_mm"]
        assert dx < 0  # object is left of center → move left
        assert dy < 0  # object is above center → move up

    def test_no_object(self):
        aligner = ArmCameraAligner()
        frame = np.full((1080, 1920, 3), (180, 180, 180), dtype=np.uint8)
        result = aligner.compute_alignment(frame)
        assert not result["detected"]
        assert not result["centered"]

    def test_correction_magnitude(self):
        aligner = ArmCameraAligner(px_to_mm=0.2)
        frame = make_arm_frame(obj_cx=1060, obj_cy=640)
        result = aligner.compute_alignment(frame)
        assert result["detected"]
        dx, dy = result["correction_mm"]
        # 100px offset * 0.2 mm/px = 20mm
        assert abs(dx - 20.0) < 5
        assert abs(dy - 20.0) < 5


# ── MultiviewController State Machine ──

class TestMultiviewController:
    def test_initial_state(self):
        ctrl = MultiviewController()
        status = ctrl.get_status()
        assert status["phase"] == "idle"
        assert not status["running"]

    def test_abort(self):
        ctrl = MultiviewController()
        ctrl.abort()
        assert ctrl._abort_event.is_set()

    def test_phase_transitions_success(self):
        """Test that a successful pick goes through all phases."""
        async def _run():
            ctrl = MultiviewController()
            phases_visited = []

            original_transition = ctrl._transition

            def tracking_transition(phase):
                phases_visited.append(phase)
                original_transition(phase)

            ctrl._transition = tracking_transition

            ctrl._phase_a_overhead_xy = AsyncMock(return_value=True)
            ctrl._phase_b_side_z_approach = AsyncMock(return_value=True)
            ctrl._phase_c_arm_fine_align = AsyncMock(return_value=True)
            ctrl._phase_d_side_descend = AsyncMock(return_value=True)
            ctrl._phase_e_verify_grasp = AsyncMock(return_value=True)
            ctrl._phase_f_lift_verify = AsyncMock(return_value=True)

            await ctrl.execute_pick("redbull", target_xy_mm=(100, 200))

            assert PickPhase.A_OVERHEAD_XY in phases_visited
            assert PickPhase.B_SIDE_Z_APPROACH in phases_visited
            assert PickPhase.C_ARM_FINE_ALIGN in phases_visited
            assert PickPhase.D_SIDE_DESCEND in phases_visited
            assert PickPhase.E_VERIFY_GRASP in phases_visited
            assert PickPhase.F_LIFT_VERIFY in phases_visited
            assert PickPhase.COMPLETE in phases_visited

        asyncio.run(_run())

    def test_phase_failure_stops_sequence(self):
        async def _run():
            ctrl = MultiviewController()
            ctrl._phase_a_overhead_xy = AsyncMock(return_value=True)
            ctrl._phase_b_side_z_approach = AsyncMock(return_value=False)
            await ctrl.execute_pick("redbull", target_xy_mm=(100, 200))
            assert ctrl.state.phase == PickPhase.FAILED

        asyncio.run(_run())

    def test_abort_during_execution(self):
        async def _run():
            ctrl = MultiviewController()

            async def slow_phase():
                await asyncio.sleep(0.1)
                return True

            ctrl._phase_a_overhead_xy = slow_phase

            task = asyncio.create_task(ctrl.execute_pick("redbull", target_xy_mm=(100, 200)))
            await asyncio.sleep(0.01)
            ctrl.abort()
            await task

            assert ctrl.state.phase in (PickPhase.ABORTED, PickPhase.FAILED, PickPhase.COMPLETE)

        asyncio.run(_run())

    def test_get_status_format(self):
        ctrl = MultiviewController()
        status = ctrl.get_status()
        assert "phase" in status
        assert "target" in status
        assert "running" in status
        assert "elapsed_s" in status
        assert "side_calibrated" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
