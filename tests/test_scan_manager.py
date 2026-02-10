"""Tests for scan manager."""

import pytest
import numpy as np


class TestScanManager:
    def test_import(self):
        from src.vision.scan_manager import ScanManager, ScanStatus, SCAN_POSES

        assert len(SCAN_POSES) == 8

    def test_status_dict(self):
        from src.vision.scan_manager import ScanStatus

        s = ScanStatus()
        d = s.to_dict()
        assert d["running"] is False
        assert d["phase"] == "idle"

    def test_validate_poses(self):
        from src.vision.scan_manager import _validate_pose

        assert _validate_pose({"J0": 0, "J1": 30, "J2": 40, "J4": 50})
        assert not _validate_pose({"J0": 0, "J1": 100})  # J1 limit is ±80

    def test_list_scans_empty(self):
        from src.vision.scan_manager import ScanManager

        scans = ScanManager.list_scans()
        assert isinstance(scans, list)

    def test_get_scan_ply_none(self):
        from src.vision.scan_manager import ScanManager

        assert ScanManager.get_scan_ply("nonexistent") is None
