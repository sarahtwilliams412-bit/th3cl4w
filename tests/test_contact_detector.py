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


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@pytest.fixture
def detector():
    return GripperContactDetector(
        api_base="http://test:8080",
        poll_hz=100,
        stable_duration_s=0.05,
    )


def _mock_state_response(gripper_mm: float):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"gripper": gripper_mm}
    resp.headers = {"content-type": "application/json"}
    return resp


def _mock_ok_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    return resp


def _patch_client(get_fn, post_fn):
    """Context manager that patches httpx.AsyncClient with mock get/post."""
    client = AsyncMock()
    client.get = get_fn
    client.post = post_fn
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=client)


class TestCloseAndDetect:

    def test_contact_detected(self, detector):
        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_state_response(55.0)
            return _mock_state_response(50.0)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect(target_mm=10, object_min_mm=25))

        assert result.contacted is True
        assert result.status == ContactStatus.CONTACT
        assert result.stable_mm == pytest.approx(50.0, abs=3.0)

    def test_no_contact(self, detector):
        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            mm = max(10.0, 60.0 - call_count * 10)
            return _mock_state_response(mm)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect(target_mm=10, object_min_mm=25))

        assert result.contacted is False
        assert result.status == ContactStatus.NO_CONTACT

    def test_zero_readings_filtered(self, detector):
        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return _mock_state_response(0.0)
            return _mock_state_response(50.0)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect(target_mm=10, object_min_mm=25, timeout_s=0.5))

        assert result.contacted is True
        for _, mm in result.readings:
            assert mm != 0.0

    def test_timeout(self, detector):
        call_count = 0

        async def mock_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            return _mock_state_response(40.0 + (call_count % 5) * 3)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect(target_mm=10, object_min_mm=25, timeout_s=0.2))

        assert result.time_s == pytest.approx(0.2, abs=0.1)

    def test_with_profile(self, detector):
        async def mock_get(*a, **kw):
            return _mock_state_response(50.0)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect(target_mm=10, profile="redbull", timeout_s=0.5))

        assert result.contacted is True

    def test_command_failure(self, detector):
        async def mock_post(*a, **kw):
            resp = MagicMock()
            resp.status_code = 500
            return resp

        async def mock_get(*a, **kw):
            return _mock_state_response(50.0)

        with _patch_client(mock_get, mock_post):
            result = _run(detector.close_and_detect())

        assert result.contacted is False
        assert result.status == ContactStatus.TIMEOUT


class TestAdaptiveGrip:

    def test_contact_at_step(self, detector):
        async def mock_get(*a, **kw):
            return _mock_state_response(50.0)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.adaptive_grip(initial_mm=15, object_min_mm=25))

        assert result.contacted is True
        assert result.final_mm == pytest.approx(50.0, abs=1.0)

    def test_no_contact_all_steps(self, detector):
        last_target = [65.0]

        async def mock_get(*a, **kw):
            # Gripper always reaches commanded position (no object)
            return _mock_state_response(last_target[0])

        async def mock_post(*a, **kw):
            # Extract position from json body if present
            json_data = kw.get("json", {})
            if "position" in json_data:
                last_target[0] = json_data["position"]
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            result = _run(detector.adaptive_grip(initial_mm=15, object_min_mm=25))

        assert result.contacted is False


class TestObjectProfiles:

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

    def test_last_result_updated(self, detector):
        assert detector.last_result is None

        async def mock_get(*a, **kw):
            return _mock_state_response(5.0)

        async def mock_post(*a, **kw):
            return _mock_ok_response()

        with _patch_client(mock_get, mock_post):
            _run(detector.close_and_detect(target_mm=10))

        assert detector.last_result is not None
