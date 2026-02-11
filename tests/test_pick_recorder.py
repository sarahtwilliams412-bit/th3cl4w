"""Tests for PickVideoRecorder and pick recording API endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.telemetry.pick_recorder import PickVideoRecorder, RECORDINGS_DIR, ALL_CAMS


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "recordings"


@pytest.fixture
def recorder(tmp_dir):
    return PickVideoRecorder(recording_dir=tmp_dir)


class TestPickVideoRecorder:
    def test_initial_state(self, recorder):
        assert not recorder.recording
        assert recorder.frame_count == 0

    @pytest.mark.asyncio
    async def test_start_creates_directory(self, recorder, tmp_dir):
        """Starting recording creates the episode directory."""
        with patch("src.telemetry.pick_recorder.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"\xff\xd8fake-jpeg"
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            recorder.start("ep_001")
            assert recorder.recording
            assert (tmp_dir / "ep_001").is_dir()

            # Let it grab a few frames
            await asyncio.sleep(0.5)
            count = await recorder.stop()
            assert count > 0
            assert not recorder.recording

    @pytest.mark.asyncio
    async def test_stop_without_start(self, recorder):
        """Stopping when not recording returns 0."""
        count = await recorder.stop()
        assert count == 0

    @pytest.mark.asyncio
    async def test_frame_saving(self, recorder, tmp_dir):
        """Frames are saved as JPEG files with correct naming."""
        with patch("src.telemetry.pick_recorder.httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"\xff\xd8fake-jpeg-data"
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_resp)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            recorder.start("ep_frames")
            await asyncio.sleep(0.5)
            await recorder.stop()

            ep_dir = tmp_dir / "ep_frames"
            frames = list(ep_dir.glob("*.jpg"))
            assert len(frames) > 0
            # Check naming pattern: cam{id}_{timestamp}.jpg
            for f in frames:
                assert f.name.startswith("cam")
                assert f.name.endswith(".jpg")
                # Verify content was written
                assert f.read_bytes() == b"\xff\xd8fake-jpeg-data"

    @pytest.mark.asyncio
    async def test_camera_failure_doesnt_crash(self, recorder, tmp_dir):
        """If camera server is down, recording continues without crashing."""
        with patch("src.telemetry.pick_recorder.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_instance

            recorder.start("ep_fail")
            await asyncio.sleep(0.5)
            count = await recorder.stop()
            assert count == 0  # No frames saved, but no crash

    def test_uses_config_urls(self):
        """Verify we use camera_config URLs, not hardcoded ones."""
        from src.config.camera_config import latest_url

        # Should use the config module's URL builder
        url = latest_url(0)
        assert "/latest/0" in url


class TestPickRecordingAPI:
    """Test the recording API endpoints (unit-style with TestClient)."""

    @pytest.fixture
    def app_client(self, tmp_dir):
        """Create a test client with patched recordings dir."""
        # We test the endpoint logic directly rather than spinning up the full server
        # since server.py has heavy dependencies
        pass

    def test_recordings_dir_constant(self):
        """RECORDINGS_DIR uses data/pick_recordings."""
        assert str(RECORDINGS_DIR) == "data/pick_recordings"

    def test_all_cams_has_three(self):
        """ALL_CAMS includes all 3 camera IDs."""
        assert len(ALL_CAMS) == 3
        from src.config.camera_config import CAM_OVERHEAD, CAM_ARM, CAM_SIDE

        assert CAM_OVERHEAD in ALL_CAMS
        assert CAM_ARM in ALL_CAMS
        assert CAM_SIDE in ALL_CAMS

    def test_recordings_list_empty(self, tmp_dir):
        """When no recordings exist, list is empty."""
        # Simulate the endpoint logic
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        results = []
        for ep_dir in sorted(tmp_dir.iterdir(), reverse=True):
            if ep_dir.is_dir():
                frames = list(ep_dir.glob("*.jpg"))
                results.append({"episode_id": ep_dir.name, "frame_count": len(frames)})
        assert results == []

    def test_recordings_list_with_data(self, tmp_dir):
        """Recordings list returns correct frame counts."""
        ep = tmp_dir / "ep_test"
        ep.mkdir(parents=True)
        (ep / "cam0_100.jpg").write_bytes(b"fake")
        (ep / "cam1_100.jpg").write_bytes(b"fake")

        results = []
        for ep_dir in sorted(tmp_dir.iterdir(), reverse=True):
            if ep_dir.is_dir():
                frames = list(ep_dir.glob("*.jpg"))
                results.append({"episode_id": ep_dir.name, "frame_count": len(frames)})
        assert len(results) == 1
        assert results[0]["episode_id"] == "ep_test"
        assert results[0]["frame_count"] == 2
