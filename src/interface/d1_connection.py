"""
D1 Arm Connection Interface

Handles low-level communication with the Unitree D1 arm over Ethernet.
"""

import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


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
            self._state_socket.bind(("0.0.0.0", self.state_port))
            self._state_socket.settimeout(0.1)
            
            # Test connection with ping
            self._connected = self._ping()
            return self._connected
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close connection to D1 arm."""
        if self._cmd_socket:
            self._cmd_socket.close()
        if self._state_socket:
            self._state_socket.close()
        self._connected = False
    
    def _ping(self) -> bool:
        """Test if arm is reachable."""
        # TODO: Implement actual ping protocol
        return True
    
    def get_state(self) -> Optional[D1State]:
        """Read current arm state."""
        if not self._connected or not self._state_socket:
            return None
            
        try:
            data, addr = self._state_socket.recvfrom(1024)
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
            print(f"Send failed: {e}")
            return False
    
    def _parse_state(self, data: bytes) -> D1State:
        """Parse state packet from arm."""
        # TODO: Implement actual protocol parsing
        # Placeholder implementation
        return D1State(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_torques=np.zeros(7),
            gripper_position=0.0,
            timestamp=time.time(),
        )
    
    def _encode_command(self, cmd: D1Command) -> bytes:
        """Encode command to wire format."""
        # TODO: Implement actual protocol encoding
        # Placeholder implementation
        return b""
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# Convenience functions
def connect_d1(ip: str = D1Connection.DEFAULT_IP) -> D1Connection:
    """Connect to D1 arm with default settings."""
    conn = D1Connection(ip=ip)
    if conn.connect():
        print(f"Connected to D1 at {ip}")
        return conn
    else:
        raise ConnectionError(f"Failed to connect to D1 at {ip}")
