"""
D1 Arm Connection Interface

Handles low-level communication with the Unitree D1 arm over Ethernet.
"""

import logging
import socket
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NUM_JOINTS = 7  # 6 arm + 1 gripper

# State packet layout: 7 positions + 7 velocities + 7 torques + 1 gripper + 1 timestamp = 23 floats
STATE_PACKET_FORMAT = f"<{NUM_JOINTS}f{NUM_JOINTS}f{NUM_JOINTS}f1f1d"
STATE_PACKET_SIZE = struct.calcsize(STATE_PACKET_FORMAT)

# Command packet layout: 1 mode byte + 7 positions + 7 velocities + 7 torques + 1 gripper = 22 floats + 1 int
COMMAND_PACKET_FORMAT = f"<i{NUM_JOINTS}f{NUM_JOINTS}f{NUM_JOINTS}f1f"
COMMAND_PACKET_SIZE = struct.calcsize(COMMAND_PACKET_FORMAT)

# Pre-allocated zero array for default command fields (avoids per-call allocation)
_ZEROS_7 = np.zeros(NUM_JOINTS, dtype=np.float64)


@dataclass
class D1State:
    """Current state of the D1 arm."""

    joint_positions: np.ndarray  # 7 joints (6 arm + 1 gripper)
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_position: float  # 0.0 (closed) to 1.0 (open)
    timestamp: float


@dataclass
class D1Command:
    """Command to send to the D1 arm."""

    mode: int  # 0=idle, 1=position, 2=velocity, 3=torque
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    gripper_position: Optional[float] = None


class D1Connection:
    """
    Low-level connection to Unitree D1 arm.

    Uses UDP communication on the standard Unitree port.
    """

    DEFAULT_IP = "192.168.123.18"  # D1 default IP
    DEFAULT_PORT = 8082  # Command port
    STATE_PORT = 8083  # State feedback port
    PING_TIMEOUT = 1.0  # Seconds to wait for ping response

    def __init__(
        self,
        ip: str = DEFAULT_IP,
        command_port: int = DEFAULT_PORT,
        state_port: int = STATE_PORT,
    ):
        self.ip = ip
        self.command_port = command_port
        self.state_port = state_port

        self._cmd_socket: Optional[socket.socket] = None
        self._state_socket: Optional[socket.socket] = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to D1 arm."""
        try:
            # Command socket (UDP)
            self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._cmd_socket.settimeout(1.0)

            # State socket (UDP, bind to receive)
            self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._state_socket.bind(("0.0.0.0", self.state_port))
            self._state_socket.settimeout(0.1)

            # Test connection with ping
            self._connected = self._ping()
            if not self._connected:
                self.disconnect()
            return self._connected

        except Exception as e:
            logger.error("Connection failed: %s", e)
            self.disconnect()
            return False

    def disconnect(self):
        """Close connection to D1 arm."""
        if self._cmd_socket:
            self._cmd_socket.close()
            self._cmd_socket = None
        if self._state_socket:
            self._state_socket.close()
            self._state_socket = None
        self._connected = False

    def _ping(self) -> bool:
        """Test if arm is reachable by sending a zero-length probe packet."""
        if not self._cmd_socket:
            return False
        try:
            # Send a zero-byte probe to the command port and wait for any
            # UDP response (or ICMP port-unreachable) within the timeout.
            self._cmd_socket.sendto(b"\x00", (self.ip, self.command_port))
            self._cmd_socket.settimeout(self.PING_TIMEOUT)
            self._cmd_socket.recvfrom(1024)
            return True
        except socket.timeout:
            logger.warning("Ping timeout — no response from %s:%d", self.ip, self.command_port)
            return False
        except OSError as e:
            logger.warning("Ping failed: %s", e)
            return False

    def get_state(self) -> Optional[D1State]:
        """Read current arm state."""
        if not self._connected or not self._state_socket:
            return None

        try:
            data, addr = self._state_socket.recvfrom(STATE_PACKET_SIZE + 64)
            return self._parse_state(data)
        except socket.timeout:
            return None

    def send_command(self, cmd: D1Command) -> bool:
        """Send command to arm."""
        if not self._connected or not self._cmd_socket:
            return False

        try:
            data = self._encode_command(cmd)
            self._cmd_socket.sendto(data, (self.ip, self.command_port))
            return True
        except Exception as e:
            logger.error("Send failed: %s", e)
            return False

    def _parse_state(self, data: bytes) -> D1State:
        """Parse state packet from arm.

        Expected binary layout (little-endian):
          7 floats  — joint positions
          7 floats  — joint velocities
          7 floats  — joint torques
          1 float   — gripper position
          1 double  — timestamp
        """
        if len(data) < STATE_PACKET_SIZE:
            logger.warning(
                "State packet too short: got %d bytes, expected %d",
                len(data),
                STATE_PACKET_SIZE,
            )
            raise ValueError(
                f"State packet too short: got {len(data)} bytes, expected {STATE_PACKET_SIZE}"
            )

        values = struct.unpack(STATE_PACKET_FORMAT, data[:STATE_PACKET_SIZE])
        idx = 0
        joint_positions = np.array(values[idx : idx + NUM_JOINTS], dtype=np.float64)
        idx += NUM_JOINTS
        joint_velocities = np.array(values[idx : idx + NUM_JOINTS], dtype=np.float64)
        idx += NUM_JOINTS
        joint_torques = np.array(values[idx : idx + NUM_JOINTS], dtype=np.float64)
        idx += NUM_JOINTS
        gripper_position = float(values[idx])
        idx += 1
        timestamp = float(values[idx])

        return D1State(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            gripper_position=gripper_position,
            timestamp=timestamp,
        )

    def _encode_command(self, cmd: D1Command) -> bytes:
        """Encode command to wire format.

        Binary layout (little-endian):
          1 int32   — mode
          7 floats  — joint positions
          7 floats  — joint velocities
          7 floats  — joint torques
          1 float   — gripper position
        """
        positions = cmd.joint_positions if cmd.joint_positions is not None else _ZEROS_7
        velocities = cmd.joint_velocities if cmd.joint_velocities is not None else _ZEROS_7
        torques = cmd.joint_torques if cmd.joint_torques is not None else _ZEROS_7
        gripper = cmd.gripper_position if cmd.gripper_position is not None else 0.0

        # Pack mode int, then 3×7 floats + 1 gripper float using pre-built buffer
        return struct.pack(
            COMMAND_PACKET_FORMAT,
            cmd.mode,
            positions[0], positions[1], positions[2], positions[3],
            positions[4], positions[5], positions[6],
            velocities[0], velocities[1], velocities[2], velocities[3],
            velocities[4], velocities[5], velocities[6],
            torques[0], torques[1], torques[2], torques[3],
            torques[4], torques[5], torques[6],
            gripper,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


# Convenience functions
def connect_d1(ip: str = D1Connection.DEFAULT_IP) -> D1Connection:
    """Connect to D1 arm with default settings."""
    conn = D1Connection(ip=ip)
    if conn.connect():
        logger.info("Connected to D1 at %s", ip)
        return conn
    else:
        raise ConnectionError(f"Failed to connect to D1 at {ip}")
