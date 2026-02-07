"""Tests for D1DDSConnection — mocks the CycloneDDS layer."""

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stub so tests run without cyclonedds installed
# ---------------------------------------------------------------------------

@dataclass
class _FakeArmString:
    data_: str = ""


class FakeDataReader:
    """Simulates a CycloneDDS DataReader."""

    def __init__(self):
        self._samples: List[_FakeArmString] = []
        self._lock = threading.Lock()

    def inject(self, payload: str) -> None:
        with self._lock:
            self._samples.append(_FakeArmString(data_=payload))

    def take(self, N: int = 32) -> List[_FakeArmString]:
        with self._lock:
            out = self._samples[:N]
            self._samples = self._samples[N:]
            return out


class FakeDataWriter:
    def __init__(self):
        self.written: List[Any] = []

    def write(self, sample: Any) -> None:
        self.written.append(sample)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_reader():
    return FakeDataReader()


@pytest.fixture()
def fake_writer():
    return FakeDataWriter()


@pytest.fixture()
def connection(fake_reader, fake_writer):
    """Return a connected D1DDSConnection with mocked DDS internals."""
    with (
        patch("src.interface.d1_dds_connection.DomainParticipant"),
        patch("src.interface.d1_dds_connection.Topic"),
        patch("src.interface.d1_dds_connection.DataReader", return_value=fake_reader),
        patch("src.interface.d1_dds_connection.DataWriter", return_value=fake_writer),
    ):
        from src.interface.d1_dds_connection import D1DDSConnection

        conn = D1DDSConnection()
        assert conn.connect(interface_name="test0")
        yield conn
        conn.disconnect()


# ---------------------------------------------------------------------------
# JSON encoding / command tests
# ---------------------------------------------------------------------------

class TestCommands:
    def test_enable_motors(self, connection, fake_writer):
        assert connection.enable_motors()
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 5
        assert payload["data"]["mode"] == 1

    def test_disable_motors(self, connection, fake_writer):
        assert connection.disable_motors()
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 5
        assert payload["data"]["mode"] == 0

    def test_power_on(self, connection, fake_writer):
        assert connection.power_on()
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 6
        assert payload["data"]["power"] == 1

    def test_power_off(self, connection, fake_writer):
        assert connection.power_off()
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 6
        assert payload["data"]["power"] == 0

    def test_reset_to_zero(self, connection, fake_writer):
        assert connection.reset_to_zero()
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 7

    def test_set_joint(self, connection, fake_writer):
        assert connection.set_joint(3, 45.0, delay_ms=100)
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 1
        assert payload["data"]["id"] == 3
        assert payload["data"]["angle"] == 45.0
        assert payload["data"]["delay_ms"] == 100

    def test_set_joint_invalid_id(self, connection):
        with pytest.raises(ValueError):
            connection.set_joint(7, 0.0)
        with pytest.raises(ValueError):
            connection.set_joint(-1, 0.0)

    def test_set_all_joints(self, connection, fake_writer):
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        assert connection.set_all_joints(angles)
        payload = json.loads(fake_writer.written[-1].data_)
        assert payload["funcode"] == 2
        for i, a in enumerate(angles):
            assert payload["data"][f"angle{i}"] == a

    def test_set_all_joints_wrong_count(self, connection):
        with pytest.raises(ValueError):
            connection.set_all_joints([1.0, 2.0])

    def test_seq_increments(self, connection, fake_writer):
        connection.enable_motors()
        connection.disable_motors()
        s1 = json.loads(fake_writer.written[-2].data_)["seq"]
        s2 = json.loads(fake_writer.written[-1].data_)["seq"]
        assert s2 == s1 + 1


# ---------------------------------------------------------------------------
# Feedback parsing tests
# ---------------------------------------------------------------------------

class TestFeedback:
    def _wait_for_cache(self, connection, attr="joint_angles", timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with connection._cache.lock:
                if getattr(connection._cache, attr) is not None:
                    return
            time.sleep(0.01)
        raise TimeoutError(f"Cache attribute {attr!r} not populated in time")

    def test_joint_angle_feedback(self, connection, fake_reader):
        fb = {
            "seq": 10,
            "address": 2,
            "funcode": 1,
            "data": {
                "angle0": 0.7, "angle1": -90.4, "angle2": 92.1,
                "angle3": -4.6, "angle4": -98.4, "angle5": 7.5, "angle6": -24.9,
            },
        }
        fake_reader.inject(json.dumps(fb))
        self._wait_for_cache(connection, "joint_angles")

        angles = connection.get_joint_angles()
        assert angles is not None
        np.testing.assert_allclose(
            angles,
            [0.7, -90.4, 92.1, -4.6, -98.4, 7.5, -24.9],
        )

    def test_status_feedback(self, connection, fake_reader):
        fb = {
            "seq": 11,
            "address": 2,
            "funcode": 3,
            "data": {"enable_status": 1, "power_status": 1, "error_status": 0},
        }
        fake_reader.inject(json.dumps(fb))
        self._wait_for_cache(connection, "status")

        status = connection.get_status()
        assert status == {"enable_status": 1, "power_status": 1, "error_status": 0}

    def test_get_state(self, connection, fake_reader):
        fb = {
            "seq": 1,
            "address": 2,
            "funcode": 1,
            "data": {f"angle{i}": float(i * 10) for i in range(7)},
        }
        fake_reader.inject(json.dumps(fb))
        self._wait_for_cache(connection, "joint_angles")

        state = connection.get_state()
        assert state is not None
        assert state.joint_positions.shape == (7,)
        np.testing.assert_allclose(state.joint_positions, [0, 10, 20, 30, 40, 50, 60])
        assert state.gripper_position == 60.0
        assert state.joint_velocities.sum() == 0.0

    def test_get_state_none_before_feedback(self, connection):
        assert connection.get_state() is None
        assert connection.get_joint_angles() is None
        assert connection.get_status() is None

    def test_malformed_feedback_ignored(self, connection, fake_reader):
        fake_reader.inject("not json at all")
        time.sleep(0.05)
        assert connection.get_joint_angles() is None


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_disconnect(self, connection):
        assert connection.is_connected
        connection.disconnect()
        assert not connection.is_connected

    def test_send_when_disconnected(self):
        from src.interface.d1_dds_connection import D1DDSConnection
        conn = D1DDSConnection()
        assert not conn.send_command({"funcode": 1})

    def test_connect_twice_returns_true(self, connection):
        assert connection.connect(interface_name="test0")
