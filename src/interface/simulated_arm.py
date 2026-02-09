"""
Simulated D1 Arm — drop-in replacement for D1DDSConnection.

Keeps joint state in memory and interpolates toward commanded positions
at ~10Hz (driven by callers polling get_joint_angles). No DDS, no hardware.

Used for:
  - Offline development and UI testing
  - 3D visualization without a physical arm
  - Integration testing of the command pipeline
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .d1_connection import D1State, NUM_JOINTS

logger = logging.getLogger(__name__)

# Default joint limits (degrees) — 6 arm joints
_DEFAULT_LIMITS = {
    0: (-135.0, 135.0),
    1: (-20.0, 90.0),
    2: (-135.0, 45.0),
    3: (-135.0, 135.0),
    4: (-80.0, 80.0),
    5: (-135.0, 135.0),
}

_GRIPPER_MIN = 0.0
_GRIPPER_MAX = 65.0

# How fast the simulated arm moves toward targets (fraction per call)
_DEFAULT_INTERP_FACTOR = 0.15


class SimulatedArm:
    """In-memory simulated arm implementing the same interface as D1DDSConnection.

    Joint angles interpolate smoothly toward commanded targets each time
    ``get_joint_angles()`` is called, simulating realistic motion lag.

    Thread-safe: all state access is guarded by a lock.
    """

    def __init__(
        self,
        joint_limits: Optional[Dict[int, tuple]] = None,
        interp_factor: float = _DEFAULT_INTERP_FACTOR,
        num_arm_joints: int = 6,
    ) -> None:
        self._lock = threading.Lock()
        self._limits = joint_limits or dict(_DEFAULT_LIMITS)
        self._interp = interp_factor
        self._num_arm = num_arm_joints

        # Current and target state
        self._angles = [0.0] * self._num_arm
        self._target_angles = [0.0] * self._num_arm
        self._gripper = 0.0
        self._target_gripper = 0.0

        # Arm state flags
        self._powered = False
        self._enabled = False
        self._error = 0
        self._connected = True

        # Sequence counter (mirrors DDS connection)
        self._seq = 0

        # Timestamp of last state update
        self._last_update = time.monotonic()

        # Background feedback thread (optional, for consumers that expect periodic updates)
        self._feedback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, interface_name: str = "sim", domain_id: int = 0) -> bool:
        """No-op connect — always succeeds."""
        self._connected = True
        logger.info("SimulatedArm connected (interface=%s)", interface_name)
        return True

    def disconnect(self) -> None:
        """Disconnect the simulated arm."""
        self._stop_event.set()
        if self._feedback_thread is not None:
            self._feedback_thread.join(timeout=2.0)
            self._feedback_thread = None
        self._connected = False
        logger.info("SimulatedArm disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Power / enable
    # ------------------------------------------------------------------

    def power_on(self, _correlation_id: Optional[str] = None, **kwargs) -> bool:
        with self._lock:
            self._powered = True
        logger.info("SimulatedArm: power ON")
        return True

    def power_off(self, _correlation_id: Optional[str] = None, **kwargs) -> bool:
        with self._lock:
            self._enabled = False
            self._powered = False
        logger.info("SimulatedArm: power OFF")
        return True

    def enable_motors(self, _correlation_id: Optional[str] = None, **kwargs) -> bool:
        with self._lock:
            if not self._powered:
                logger.warning("SimulatedArm: cannot enable — not powered")
                return False
            self._enabled = True
        logger.info("SimulatedArm: motors ENABLED")
        return True

    def disable_motors(self, _correlation_id: Optional[str] = None, **kwargs) -> bool:
        with self._lock:
            self._enabled = False
        logger.info("SimulatedArm: motors DISABLED")
        return True

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def reset_to_zero(self, _correlation_id: Optional[str] = None, **kwargs) -> bool:
        with self._lock:
            self._target_angles = [0.0] * self._num_arm
            self._target_gripper = 0.0
        return True

    def set_joint(
        self,
        joint_id: int,
        angle_deg: float,
        delay_ms: int = 0,
        _correlation_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        if not (0 <= joint_id < self._num_arm):
            return False
        lo, hi = self._limits.get(joint_id, (-135.0, 135.0))
        with self._lock:
            self._target_angles[joint_id] = max(lo, min(hi, angle_deg))
        return True

    def set_all_joints(
        self,
        angles_deg: List[float],
        mode: int = 0,
        _correlation_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        n = min(len(angles_deg), self._num_arm)
        with self._lock:
            for i in range(n):
                lo, hi = self._limits.get(i, (-135.0, 135.0))
                self._target_angles[i] = max(lo, min(hi, angles_deg[i]))
        return True

    def set_gripper(
        self, position_mm: float, _correlation_id: Optional[str] = None, **kwargs
    ) -> bool:
        with self._lock:
            self._target_gripper = max(_GRIPPER_MIN, min(_GRIPPER_MAX, position_mm))
        return True

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Accept raw command dicts (same as DDS connection)."""
        funcode = cmd.get("funcode")
        data = cmd.get("data", {}) or {}
        self._seq += 1

        if funcode == 1:  # single joint
            return self.set_joint(data.get("id", 0), data.get("angle", 0.0))
        elif funcode == 2:  # all joints
            angles = [data.get(f"angle{i}", 0.0) for i in range(self._num_arm)]
            return self.set_all_joints(angles, mode=data.get("mode", 0))
        elif funcode == 5:  # enable/disable
            if data.get("mode") == 1:
                return self.enable_motors()
            else:
                return self.disable_motors()
        elif funcode == 6:  # power
            if data.get("power") == 1:
                return self.power_on()
            else:
                return self.power_off()
        elif funcode == 7:  # reset
            return self.reset_to_zero()
        return True

    # ------------------------------------------------------------------
    # State reading — interpolates toward targets each call
    # ------------------------------------------------------------------

    def _interpolate(self) -> None:
        """Move current angles toward targets (must hold lock)."""
        for i in range(self._num_arm):
            diff = self._target_angles[i] - self._angles[i]
            self._angles[i] += diff * self._interp
        self._gripper += (self._target_gripper - self._gripper) * self._interp
        self._last_update = time.monotonic()

    def get_joint_angles(self) -> Optional[np.ndarray]:
        """Return current joint angles as (7,) array (6 arm + gripper)."""
        with self._lock:
            self._interpolate()
            # Return 7 joints to match DDS: 6 arm + gripper as joint 6
            return np.array(self._angles + [self._gripper], dtype=np.float64)

    def get_reliable_joint_angles(self) -> Optional[np.ndarray]:
        """Same as get_joint_angles — sim has no zero-read issues."""
        return self.get_joint_angles()

    def get_gripper_position(self) -> float:
        with self._lock:
            return round(self._gripper, 2)

    def get_status(self) -> Optional[Dict[str, int]]:
        with self._lock:
            return {
                "power_status": 1 if self._powered else 0,
                "enable_status": 1 if self._enabled else 0,
                "error_status": self._error,
            }

    def get_state(self) -> Optional[D1State]:
        """Return a D1State matching DDS connection interface."""
        angles = self.get_joint_angles()
        if angles is None:
            return None
        return D1State(
            joint_positions=angles,
            joint_velocities=np.zeros(NUM_JOINTS, dtype=np.float64),
            joint_torques=np.zeros(NUM_JOINTS, dtype=np.float64),
            gripper_position=float(angles[6]) if len(angles) > 6 else 0.0,
            timestamp=self._last_update,
        )

    def get_reliable_state(self) -> Optional[D1State]:
        """Same as get_state — sim always reliable."""
        return self.get_state()

    def is_feedback_fresh(self, max_age_s: float = 2.0) -> bool:
        """Sim feedback is always fresh."""
        return True

    def get_joint_freshness(self) -> Dict[int, float]:
        """All joints always fresh in sim."""
        return {i: 0.0 for i in range(NUM_JOINTS)}

    def get_feedback_health(self) -> dict:
        """Return healthy feedback metrics."""
        return {
            "total_samples": 0,
            "zero_reads": 0,
            "zero_rate": 0.0,
            "last_good_age_s": 0.0,
            "stale": False,
        }

    @property
    def feedback_monitor(self):
        """Return a dummy monitor that reports healthy status."""
        return _DummyFeedbackMonitor()

    # ------------------------------------------------------------------
    # Background feedback (optional — starts a thread that periodically
    # interpolates state, useful if something polls get_state at low rate)
    # ------------------------------------------------------------------

    def start_feedback_loop(self, rate_hz: float = 10.0) -> None:
        """Start background interpolation at the given rate."""
        if self._feedback_thread is not None:
            return
        self._stop_event.clear()
        interval = 1.0 / rate_hz

        def _loop():
            while not self._stop_event.is_set():
                with self._lock:
                    self._interpolate()
                self._stop_event.wait(interval)

        self._feedback_thread = threading.Thread(target=_loop, daemon=True, name="sim-feedback")
        self._feedback_thread.start()
        logger.info("SimulatedArm feedback loop started at %.0fHz", rate_hz)


class _DummyFeedbackMonitor:
    """Stub feedback monitor for simulated arm."""

    def is_feedback_fresh(self, max_age_s: float = 2.0) -> bool:
        return True

    def get_health(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            total_samples=0,
            zero_reads=0,
            zero_rate=0.0,
            last_good_age_s=0.0,
            stale=False,
        )

    def get_reliable_angles(self) -> Optional[np.ndarray]:
        return None

    def get_reliable_gripper(self) -> Optional[float]:
        return None

    def get_recent_samples(self, n: int = 10) -> list:
        return []

    def record_sample(self, data: dict, seq: int = 0) -> None:
        pass
