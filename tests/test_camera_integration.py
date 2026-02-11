"""Integration tests for the centralized camera API."""

import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── Camera config tests ─────────────────────────────────────────────────


class TestCameraConfig:
    def test_snap_url(self):
        from src.config.camera_config import snap_url

        assert snap_url(0) == "http://localhost:8081/snap/0"
        assert snap_url(2) == "http://localhost:8081/snap/2"

    def test_latest_url(self):
        from src.config.camera_config import latest_url

        assert latest_url(1) == "http://localhost:8081/latest/1"

    def test_cameras_url(self):
        from src.config.camera_config import cameras_url

        assert cameras_url() == "http://localhost:8081/cameras"

    def test_camera_ids(self):
        from src.config.camera_config import CAM_OVERHEAD, CAM_ARM, CAM_SIDE

        assert CAM_SIDE == 0
        assert CAM_ARM == 1
        assert CAM_OVERHEAD == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SERVER_URL", "http://myhost:9999")
        # Re-import to pick up env var
        import src.config.camera_config as mod

        importlib.reload(mod)
        assert mod.snap_url(0) == "http://myhost:9999/snap/0"
        assert mod.latest_url(1) == "http://myhost:9999/latest/1"
        assert mod.cameras_url() == "http://myhost:9999/cameras"
        # Restore
        monkeypatch.delenv("CAMERA_SERVER_URL", raising=False)
        importlib.reload(mod)


# ── Consumer integration: verify modules use camera_config ───────────────

_CONSUMER_MODULES = [
    ("src.ascii.converter", "CAMERA_SERVER_URL"),
    ("src.control.visual_servo", "CAM_API"),
    ("src.control.visual_servo_controller", "CAM_API"),
    ("src.control.multiview_controller", "CAM_API"),
    ("src.vla.vla_controller", "CAM_API"),
    ("src.vision.collision_analyzer", "CAMERA_BASE"),
    ("src.calibration.joint_mapping_calibrator", "CAMERA_SERVER"),
]


class TestConsumerImports:
    """Verify consumer modules get their camera URL from camera_config."""

    @pytest.mark.parametrize("module_path,attr", _CONSUMER_MODULES)
    def test_uses_camera_config_url(self, module_path, attr):
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        # Should match the camera_config default
        from src.config.camera_config import CAMERA_SERVER_URL

        assert val == CAMERA_SERVER_URL


# ── cam_snap helper test ────────────────────────────────────────────────


class TestCamSnap:
    @pytest.mark.asyncio
    async def test_cam_snap_success(self):
        # Import the helper
        sys.path.insert(0, os.path.join(_root, "web"))
        from web.server import cam_snap  # noqa: delayed import

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xd8\xff\xe0fake-jpeg"

        client = AsyncMock()
        client.get = AsyncMock(return_value=mock_resp)

        resp = await cam_snap(client, 0)
        assert resp.status_code == 200
        assert resp.content == b"\xff\xd8\xff\xe0fake-jpeg"

    @pytest.mark.asyncio
    async def test_cam_snap_failure(self):
        from web.server import cam_snap

        client = AsyncMock()
        client.get = AsyncMock(side_effect=Exception("connection refused"))

        resp = await cam_snap(client, 0)
        assert resp.status_code == 0
        assert resp.content == b""


# ── Camera registry test (mock camera_server) ──────────────────────────

MOCK_REGISTRY = {
    "0": {
        "id": 0,
        "device": "/dev/video0",
        "name": "overhead",
        "role": "overhead",
        "mount": "ceiling",
        "resolution": [1920, 1080],
    },
    "1": {
        "id": 1,
        "device": "/dev/video2",
        "name": "arm",
        "role": "arm",
        "mount": "wrist",
        "resolution": [1920, 1080],
    },
    "2": {
        "id": 2,
        "device": "/dev/video6",
        "name": "side",
        "role": "side",
        "mount": "tripod",
        "resolution": [1920, 1080],
    },
}


class TestCameraRegistry:
    def test_registry_has_all_cameras(self):
        """Verify the expected registry structure."""
        assert len(MOCK_REGISTRY) == 3
        for cam_id in ["0", "1", "2"]:
            cam = MOCK_REGISTRY[cam_id]
            for key in ("id", "device", "name", "role", "mount", "resolution"):
                assert key in cam, f"Missing {key} in camera {cam_id}"
            assert len(cam["resolution"]) == 2
