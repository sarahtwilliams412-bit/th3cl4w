"""
D1 Arm State Reader

Read-only interface to the Unitree D1 arm's joint angles.
Polls at a configurable rate via UDP. Never sends motion commands.

The D1 has known firmware issues with high-frequency commands;
10Hz polling for state reading is safe.
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# D1 state packet format (matches src/interface/d1_connection.py)
NUM_JOINTS = 7  # 6 arm + 1 gripper
STATE_PACKET_FORMAT = f"<{NUM_JOINTS}f{NUM_JOINTS}f{NUM_JOINTS}f1f1d"
STATE_PACKET_SIZE = struct.calcsize(STATE_PACKET_FORMAT)


class D1StateReader:
    """Read-only interface to D1 arm joint state.

    This is a stripped-down reader that only reads joint angles.
    It NEVER sends motion commands — only state request packets.

    Parameters
    ----------
    ip : str
        D1 arm IP address.
    port : int
        UDP port for state communication.
    timeout_ms : int
        Socket timeout in milliseconds.
    """

    def __init__(
        self,
        ip: str = "192.168.123.100",
        port: int = 8082,
        timeout_ms: int = 100,
    ):
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_ms / 1000.0
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._last_angles: Optional[np.ndarray] = None
        self._last_read_time: float = 0.0
        self._consecutive_failures: int = 0

    def connect(self) -> bool:
        """Establish UDP socket for state reading."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(self.timeout_s)
            # Bind to any available port for receiving responses
            self._socket.bind(("0.0.0.0", 0))
            self._connected = True
            logger.info("D1 state reader connected to %s:%d", self.ip, self.port)
            return True
        except OSError as e:
            logger.error("Failed to create state reader socket: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the UDP socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._connected = False

    def read_joint_angles(self) -> Optional[np.ndarray]:
        """Read current joint angles from the D1 arm.

        Returns
        -------
        np.ndarray or None
            float64[7] joint angles in radians (6 arm + 1 gripper),
            or None if the read fails.
        """
        if not self._connected or not self._socket:
            return self._last_angles

        try:
            # Send a state request packet (zero-byte probe)
            self._socket.sendto(b"\x00", (self.ip, self.port))

            # Wait for response
            data, _ = self._socket.recvfrom(STATE_PACKET_SIZE + 64)

            if len(data) < STATE_PACKET_SIZE:
                logger.debug("State packet too short: %d bytes", len(data))
                self._consecutive_failures += 1
                return self._last_angles

            # Parse state packet
            values = struct.unpack(STATE_PACKET_FORMAT, data[:STATE_PACKET_SIZE])
            joint_positions = np.array(values[:NUM_JOINTS], dtype=np.float64)

            self._last_angles = joint_positions
            self._last_read_time = time.monotonic()
            self._consecutive_failures = 0

            return joint_positions

        except socket.timeout:
            self._consecutive_failures += 1
            if self._consecutive_failures % 50 == 1:
                logger.warning(
                    "D1 state read timeout (%d consecutive failures)",
                    self._consecutive_failures,
                )
            return self._last_angles

        except OSError as e:
            self._consecutive_failures += 1
            logger.debug("D1 state read error: %s", e)
            return self._last_angles

    def is_connected(self) -> bool:
        """Check if the reader is connected and receiving data."""
        if not self._connected:
            return False
        # Consider disconnected if no successful read in 2 seconds
        if self._last_read_time > 0:
            return (time.monotonic() - self._last_read_time) < 2.0
        return self._consecutive_failures < 20

    @property
    def last_angles(self) -> Optional[np.ndarray]:
        """Last successfully read joint angles."""
        return self._last_angles
