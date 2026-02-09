"""Tests for GripperContactDetector."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.control.contact_detector import (
    GripperContactDetector,
    ContactResult,
    ContactStatus,
    GripResult,
    OBJECT_PROFILES,
    GRIPPER_MAX_MM,
)


@pytest.fixture
def detector():
    return GripperContactDetector(
        api_base="http://test:8080",
        poll_hz=100,  # fast polling for tests
        stable_duration_s=0.05,  # short stability window
    )


def _mock_state_response(gripper_mm: float):
    """Create a mock httpx response with gripper state."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"gripper": gripper_mm}
    return resp


def _mock_ok_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    return resp


class TestCloseAndDetect:
    """Test the close_and_detect method."""

    @pytest.mark.asyncio
    async def test_contact_detected(self, detector):
        """Gripper stabilizes above threshold → contact."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate gripper closing then stopping at 50mm (object)
            if call_count <= 2:
                return _mock_state_response(55.0)
            return _mock_state_response(50.0)  # stabilizes here

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect(target_mm=10, object_min_mm=25)

        assert result.contacted is True
        assert result.status == ContactStatus.CONTACT
        assert result.stable_mm == pytest.approx(50.0, abs=3.0)

    @pytest.mark.asyncio
    async def test_no_contact(self, detector):
        """Gripper reaches target → no contact."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Gripper closes all the way
            mm = max(10.0, 60.0 - call_count * 10)
            return _mock_state_response(mm)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect(target_mm=10, object_min_mm=25)

        assert result.contacted is False
        assert result.status == ContactStatus.NO_CONTACT

    @pytest.mark.asyncio
    async def test_zero_readings_filtered(self, detector):
        """Zero readings from DDS should be ignored."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return _mock_state_response(0.0)  # DDS glitch
            return _mock_state_response(50.0)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect(
                target_mm=10, object_min_mm=25, timeout_s=0.5
            )

        # Should still detect contact (zeros filtered out)
        assert result.contacted is True
        # No zero readings in the result
        for _, mm in result.readings:
            assert mm != 0.0

    @pytest.mark.asyncio
    async def test_timeout(self, detector):
        """Detection times out if gripper never stabilizes or reaches target."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Oscillates — never stable
            return _mock_state_response(40.0 + (call_count % 5) * 3)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect(
                target_mm=10, object_min_mm=25, timeout_s=0.2
            )

        assert result.time_s == pytest.approx(0.2, abs=0.05)

    @pytest.mark.asyncio
    async def test_with_profile(self, detector):
        """Object profile adjusts detection threshold."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_state_response(50.0)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect(
                target_mm=10, profile="redbull", timeout_s=0.5
            )

        assert result.contacted is True

    @pytest.mark.asyncio
    async def test_command_failure(self, detector):
        """If gripper command fails, return timeout result."""
        async def mock_post(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            return resp

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.close_and_detect()

        assert result.contacted is False
        assert result.status == ContactStatus.TIMEOUT


class TestAdaptiveGrip:
    """Test the adaptive_grip method."""

    @pytest.mark.asyncio
    async def test_contact_at_step(self, detector):
        """Object resistance detected at a step."""
        post_calls = []

        async def mock_get(*args, **kwargs):
            # Always return 50mm (object blocks gripper)
            return _mock_state_response(50.0)

        async def mock_post(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            post_calls.append(url)
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.adaptive_grip(initial_mm=15, object_min_mm=25)

        assert result.contacted is True
        assert result.final_mm == pytest.approx(50.0, abs=1.0)
        # Should have stopped early (not gone through all steps)
        assert result.steps_taken <= 3

    @pytest.mark.asyncio
    async def test_no_contact_all_steps(self, detector):
        """Gripper closes fully — no object."""
        step_positions = iter([60.0, 50.0, 40.0, 30.0, 20.0, 15.0])

        async def mock_get(*args, **kwargs):
            try:
                return _mock_state_response(next(step_positions))
            except StopIteration:
                return _mock_state_response(15.0)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            result = await detector.adaptive_grip(initial_mm=15, object_min_mm=25)

        assert result.contacted is False


class TestObjectProfiles:
    """Test object profile definitions."""

    def test_redbull_profile(self):
        p = OBJECT_PROFILES["redbull"]
        assert p.expected_at(50.0)
        assert not p.expected_at(30.0)

    def test_generic_profile(self):
        p = OBJECT_PROFILES["generic"]
        assert p.expected_at(40.0)
        assert not p.expected_at(10.0)

    def test_all_profiles_valid(self):
        for name, p in OBJECT_PROFILES.items():
            assert p.min_contact_mm < p.max_contact_mm
            assert p.min_contact_mm >= 0
            assert p.max_contact_mm <= GRIPPER_MAX_MM


class TestLastResult:
    """Test that last_result is tracked."""

    @pytest.mark.asyncio
    async def test_last_result_updated(self, detector):
        assert detector.last_result is None

        async def mock_get(*args, **kwargs):
            return _mock_state_response(5.0)

        async def mock_post(*args, **kwargs):
            return _mock_ok_response()

        with patch("httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = mock_get
            client.post = mock_post
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client

            await detector.close_and_detect(target_mm=10)

        assert detector.last_result is not None
