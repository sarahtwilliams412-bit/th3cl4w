"""Tests for the joint mapping calibration module."""

import json
import tempfile
from pathlib import Path

import pytest


def test_joint_mapper_identity():
    """Default mapper should be identity (no remapping)."""
    from src.calibration.joint_mapper import JointMapper

    with tempfile.TemporaryDirectory() as td:
        mapper = JointMapper(mapping_path=Path(td) / "mapping.json")
        for i in range(6):
            assert mapper.ui_to_dds(i) == i
            assert mapper.dds_to_ui(i) == i
        assert not mapper.calibrated


def test_joint_mapper_remap():
    """Test non-identity mapping."""
    from src.calibration.joint_mapper import JointMapper

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mapping.json"
        mapper = JointMapper(mapping_path=path)
        # Save a swapped mapping: UI J0 -> DDS 2, UI J2 -> DDS 0
        mapper.save(
            ui_to_dds={0: 2, 1: 1, 2: 0, 3: 3, 4: 4, 5: 5},
            labels={0: "J0", 1: "J1", 2: "J2", 3: "J3", 4: "J4", 5: "J5"},
        )
        assert mapper.calibrated
        assert mapper.ui_to_dds(0) == 2
        assert mapper.ui_to_dds(2) == 0
        assert mapper.dds_to_ui(2) == 0
        assert mapper.dds_to_ui(0) == 2


def test_joint_mapper_remap_angles():
    """Test angle list remapping."""
    from src.calibration.joint_mapper import JointMapper

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mapping.json"
        mapper = JointMapper(mapping_path=path)
        mapper.save(
            ui_to_dds={0: 2, 1: 1, 2: 0, 3: 3, 4: 4, 5: 5},
            labels={},
        )
        ui_angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        dds_angles = mapper.remap_angles_ui_to_dds(ui_angles)
        # UI J0=10 -> DDS[2]=10, UI J2=30 -> DDS[0]=30
        assert dds_angles[2] == 10.0
        assert dds_angles[0] == 30.0
        assert dds_angles[1] == 20.0

        # Reverse
        back = mapper.remap_angles_dds_to_ui(dds_angles)
        assert back == pytest.approx(ui_angles)


def test_joint_mapper_save_load():
    """Test persistence: save then load from file."""
    from src.calibration.joint_mapper import JointMapper

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mapping.json"
        m1 = JointMapper(mapping_path=path)
        m1.save(
            ui_to_dds={0: 3, 1: 1, 2: 2, 3: 0, 4: 5, 5: 4},
            labels={0: "base", 3: "roll"},
        )

        # Load fresh
        m2 = JointMapper(mapping_path=path)
        assert m2.calibrated
        assert m2.ui_to_dds(0) == 3
        assert m2.ui_to_dds(3) == 0
        assert m2.dds_to_ui(3) == 0


def test_joint_mapper_get_info():
    """Test get_mapping_info returns expected structure."""
    from src.calibration.joint_mapper import JointMapper

    with tempfile.TemporaryDirectory() as td:
        mapper = JointMapper(mapping_path=Path(td) / "mapping.json")
        info = mapper.get_mapping_info()
        assert "ui_to_dds" in info
        assert "dds_to_ui" in info
        assert "calibrated" in info
        assert len(info["ui_to_dds"]) == 6


def test_calibrator_analyze_diff():
    """Test frame differencing analysis."""
    import numpy as np
    from src.calibration.joint_mapping_calibrator import JointMappingCalibrator

    cal = JointMappingCalibrator()

    # Create synthetic before/after frames with movement in upper region
    before = np.zeros((480, 640, 3), dtype=np.uint8)
    after = before.copy()
    # Add a white rectangle in the upper portion (simulating movement)
    after[50:150, 200:400] = 255

    result = cal._analyze_diff(0, before, after, None, None)
    assert result.moved
    assert result.diff_score > 0
    assert result.dds_index == 0


def test_calibrator_no_movement():
    """Test detection of no movement."""
    import numpy as np
    from src.calibration.joint_mapping_calibrator import JointMappingCalibrator

    cal = JointMappingCalibrator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = cal._analyze_diff(3, frame, frame, frame, frame)
    assert not result.moved
    assert result.diff_score == 0.0
