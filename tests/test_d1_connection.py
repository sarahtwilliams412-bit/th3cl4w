"""Tests for D1 arm connection interface."""

import struct
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.interface.d1_connection import (
    D1Command,
    D1Connection,
    D1State,
    NUM_JOINTS,
    STATE_PACKET_FORMAT,
    STATE_PACKET_SIZE,
    COMMAND_PACKET_FORMAT,
    COMMAND_PACKET_SIZE,
)

# --- D1State dataclass ---


class TestD1State:
    def test_creation(self):
        state = D1State(
            joint_positions=np.zeros(7),
            joint_velocities=np.ones(7),
            joint_torques=np.full(7, 0.5),
            gripper_position=0.75,
            timestamp=1000.0,
        )
        assert state.joint_positions.shape == (7,)
        assert state.gripper_position == 0.75
        assert state.timestamp == 1000.0


# --- D1Command dataclass ---


class TestD1Command:
    def test_defaults(self):
        cmd = D1Command(mode=0)
        assert cmd.mode == 0
        assert cmd.joint_positions is None
        assert cmd.joint_velocities is None
        assert cmd.joint_torques is None
        assert cmd.gripper_position is None

    def test_with_values(self):
        pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        cmd = D1Command(mode=1, joint_positions=pos, gripper_position=0.5)
        assert cmd.mode == 1
        np.testing.assert_array_equal(cmd.joint_positions, pos)
        assert cmd.gripper_position == 0.5


# --- _parse_state ---


class TestParseState:
    def _make_state_packet(self, positions, velocities, torques, gripper, timestamp):
        """Build a binary state packet from arrays."""
        values = (*positions, *velocities, *torques, gripper, timestamp)
        return struct.pack(STATE_PACKET_FORMAT, *values)

    def test_parse_valid_packet(self):
        conn = D1Connection()
        positions = [float(i) for i in range(7)]
        velocities = [float(i + 10) for i in range(7)]
        torques = [float(i + 20) for i in range(7)]
        gripper = 0.8
        timestamp = 12345.6789

        data = self._make_state_packet(positions, velocities, torques, gripper, timestamp)
        state = conn._parse_state(data)

        np.testing.assert_array_almost_equal(state.joint_positions, positions, decimal=5)
        np.testing.assert_array_almost_equal(state.joint_velocities, velocities, decimal=5)
        np.testing.assert_array_almost_equal(state.joint_torques, torques, decimal=5)
        assert abs(state.gripper_position - gripper) < 1e-5
        assert abs(state.timestamp - timestamp) < 1e-4

    def test_parse_packet_too_short(self):
        conn = D1Connection()
        with pytest.raises(ValueError, match="too short"):
            conn._parse_state(b"\x00" * 10)

    def test_parse_packet_with_extra_bytes(self):
        """Extra trailing bytes should be ignored."""
        conn = D1Connection()
        positions = [1.0] * 7
        velocities = [2.0] * 7
        torques = [3.0] * 7
        data = self._make_state_packet(positions, velocities, torques, 0.5, 100.0)
        data += b"\xff" * 32  # extra trailing bytes

        state = conn._parse_state(data)
        np.testing.assert_array_almost_equal(state.joint_positions, positions, decimal=5)


# --- _encode_command ---


class TestEncodeCommand:
    def test_encode_idle_command(self):
        conn = D1Connection()
        cmd = D1Command(mode=0)
        data = conn._encode_command(cmd)

        assert len(data) == COMMAND_PACKET_SIZE
        values = struct.unpack(COMMAND_PACKET_FORMAT, data)
        assert values[0] == 0  # mode
        # All positions/velocities/torques/gripper should be zero
        for v in values[1:]:
            assert v == 0.0

    def test_encode_position_command(self):
        conn = D1Connection()
        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        cmd = D1Command(mode=1, joint_positions=positions, gripper_position=0.9)
        data = conn._encode_command(cmd)

        assert len(data) == COMMAND_PACKET_SIZE
        values = struct.unpack(COMMAND_PACKET_FORMAT, data)
        assert values[0] == 1  # mode
        for i in range(7):
            assert abs(values[1 + i] - positions[i]) < 1e-5
        # gripper is last
        assert abs(values[-1] - 0.9) < 1e-5

    def test_roundtrip_encode_parse_state(self):
        """Encode a command and verify the struct layout is consistent."""
        conn = D1Connection()
        cmd = D1Command(
            mode=2,
            joint_positions=np.ones(7) * 1.0,
            joint_velocities=np.ones(7) * 2.0,
            joint_torques=np.ones(7) * 3.0,
            gripper_position=0.5,
        )
        data = conn._encode_command(cmd)
        values = struct.unpack(COMMAND_PACKET_FORMAT, data)

        assert values[0] == 2
        # positions: indices 1-7
        for v in values[1:8]:
            assert abs(v - 1.0) < 1e-5
        # velocities: indices 8-14
        for v in values[8:15]:
            assert abs(v - 2.0) < 1e-5
        # torques: indices 15-21
        for v in values[15:22]:
            assert abs(v - 3.0) < 1e-5
        # gripper: index 22
        assert abs(values[22] - 0.5) < 1e-5


# --- D1Connection connect/disconnect ---


class TestD1Connection:
    def test_initial_state(self):
        conn = D1Connection()
        assert not conn.is_connected
        assert conn._cmd_socket is None
        assert conn._state_socket is None

    def test_custom_params(self):
        conn = D1Connection(ip="10.0.0.1", command_port=9000, state_port=9001)
        assert conn.ip == "10.0.0.1"
        assert conn.command_port == 9000
        assert conn.state_port == 9001

    @patch("src.interface.d1_connection.socket.socket")
    def test_connect_success(self, mock_socket_cls):
        """Test that connect() succeeds when ping gets a response."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        # recvfrom returns some data (simulating arm response)
        mock_sock.recvfrom.return_value = (b"\x01", ("192.168.123.18", 8082))

        conn = D1Connection()
        result = conn.connect()

        assert result is True
        assert conn.is_connected

    @patch("src.interface.d1_connection.socket.socket")
    def test_connect_ping_timeout(self, mock_socket_cls):
        """Test that connect() fails when ping times out."""
        import socket as real_socket

        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.side_effect = real_socket.timeout("timed out")

        conn = D1Connection()
        result = conn.connect()

        assert result is False
        assert not conn.is_connected
        # Sockets should be cleaned up
        assert conn._cmd_socket is None
        assert conn._state_socket is None

    @patch("src.interface.d1_connection.socket.socket")
    def test_disconnect_cleans_up(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recvfrom.return_value = (b"\x01", ("192.168.123.18", 8082))

        conn = D1Connection()
        conn.connect()
        conn.disconnect()

        assert not conn.is_connected
        assert conn._cmd_socket is None
        assert conn._state_socket is None

    @patch("src.interface.d1_connection.socket.socket")
    def test_connect_bind_failure_cleans_up(self, mock_socket_cls):
        """If bind fails, both sockets should be cleaned up (no resource leak)."""
        call_count = 0

        def make_socket(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if call_count == 2:
                # Second socket (state) fails on bind
                mock.bind.side_effect = OSError("Address already in use")
            return mock

        mock_socket_cls.side_effect = make_socket

        conn = D1Connection()
        result = conn.connect()

        assert result is False
        assert not conn.is_connected
        assert conn._cmd_socket is None
        assert conn._state_socket is None

    def test_get_state_when_not_connected(self):
        conn = D1Connection()
        assert conn.get_state() is None

    def test_send_command_when_not_connected(self):
        conn = D1Connection()
        cmd = D1Command(mode=0)
        assert conn.send_command(cmd) is False


# --- Packet format constants ---


class TestPacketFormats:
    def test_state_packet_size(self):
        # 7+7+7+1 floats (4 bytes each) + 1 double (8 bytes) = 22*4 + 8 = 96
        expected = 22 * 4 + 8
        assert STATE_PACKET_SIZE == expected

    def test_command_packet_size(self):
        # 1 int (4 bytes) + (7+7+7+1) floats (4 bytes each) = 4 + 22*4 = 92
        expected = 4 + 22 * 4
        assert COMMAND_PACKET_SIZE == expected
