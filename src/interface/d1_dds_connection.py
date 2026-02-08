"""
D1 Arm DDS Connection Interface

Communicates with the Unitree D1 arm via CycloneDDS, using the ArmString_
IDL type on topics rt/arm_Command and rt/arm_Feedback.

Requires: cyclonedds, python 3.12 (cyclonedds is incompatible with 3.14).
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

from .d1_connection import D1State, NUM_JOINTS

try:
    from src.telemetry import get_collector, EventType

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IDL type — must match the Unitree DDS schema exactly
# ---------------------------------------------------------------------------


@dataclass
class ArmString_(IdlStruct, typename="unitree_arm.msg.dds_.ArmString_"):
    data_: str = ""


# ---------------------------------------------------------------------------
# Feedback cache
# ---------------------------------------------------------------------------


@dataclass
class _FeedbackCache:
    """Thread-safe cache for the latest feedback from the arm."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    joint_angles: Optional[Dict[str, float]] = None  # angle0..angle6
    status: Optional[Dict[str, int]] = None  # enable_status, power_status, error_status
    last_seq: int = 0
    last_update: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class D1DDSConnection:
    """
    DDS-based connection to the Unitree D1 arm.

    Usage::

        conn = D1DDSConnection()
        conn.connect(interface_name="eno1")
        conn.power_on()
        conn.enable_motors()
        angles = conn.get_joint_angles()
        conn.set_joint(0, 45.0)
        conn.disconnect()
    """

    FEEDBACK_TOPIC = "rt/arm_Feedback"
    COMMAND_TOPIC = "rt/arm_Command"
    DOMAIN_ID = 0
    # How often the reader thread polls for new samples (seconds)
    _POLL_INTERVAL = 0.005  # 5 ms — fast enough for real-time feedback

    def __init__(self, collector=None) -> None:
        self._dp: Optional[DomainParticipant] = None
        self._reader: Optional[DataReader] = None
        self._writer: Optional[DataWriter] = None
        self._connected = False
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._cache = _FeedbackCache()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._prev_angles: Optional[Dict[str, float]] = None
        self._last_stale_warn: float = 0.0
        # Use explicitly passed collector to avoid singleton import-path issues
        self._collector = collector

    def _get_collector(self):
        """Return the telemetry collector — prefer explicit instance, fall back to singleton."""
        if self._collector is not None:
            return self._collector
        if _HAS_TELEMETRY:
            return get_collector()
        return None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, interface_name: str = "eno1", domain_id: int = DOMAIN_ID) -> bool:
        """Initialise the DDS participant, topics, reader, writer and start
        the background feedback reader thread.

        Args:
            interface_name: Network interface connected to the arm.
            domain_id: DDS domain id (default 0).

        Returns:
            True on success.
        """
        if self._connected:
            logger.warning("Already connected — call disconnect() first")
            return True

        try:
            os.environ["CYCLONEDDS_URI"] = (
                "<CycloneDDS>"
                "  <Domain>"
                "    <General>"
                "      <Interfaces>"
                f'        <NetworkInterface name="{interface_name}" />'
                "      </Interfaces>"
                "    </General>"
                "  </Domain>"
                "</CycloneDDS>"
            )

            self._dp = DomainParticipant(domain_id=domain_id)

            topic_fb = Topic(self._dp, self.FEEDBACK_TOPIC, ArmString_)
            self._reader = DataReader(self._dp, topic_fb)

            topic_cmd = Topic(self._dp, self.COMMAND_TOPIC, ArmString_)
            self._writer = DataWriter(self._dp, topic_cmd)

            self._stop_event.clear()
            self._reader_thread = threading.Thread(
                target=self._feedback_loop, daemon=True, name="d1-dds-feedback"
            )
            self._reader_thread.start()

            self._connected = True
            logger.info("DDS connection established on interface %s", interface_name)
            return True

        except Exception:
            logger.exception("Failed to establish DDS connection")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """Tear down DDS entities and stop the background thread."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._reader = None
        self._writer = None
        self._dp = None
        self._connected = False
        logger.info("DDS connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Sequence counter
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    # ------------------------------------------------------------------
    # Low-level publish
    # ------------------------------------------------------------------

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Publish a raw JSON command dict to rt/arm_Command.

        The ``seq`` field is auto-populated if not present.
        """
        correlation_id = cmd.pop("_correlation_id", None)

        if not self._connected or self._writer is None:
            logger.error("Cannot send command — not connected")
            tc = self._get_collector()
            if tc is not None:
                tc.emit(
                    "dds",
                    EventType.ERROR,
                    {"error": "not_connected", "cmd": str(cmd)},
                    correlation_id,
                )
            return False

        if "seq" not in cmd:
            cmd["seq"] = self._next_seq()

        try:
            payload = json.dumps(cmd, separators=(",", ":"))
            self._writer.write(ArmString_(data_=payload))
            logger.debug("Published command: %s", payload)
            tc = self._get_collector()
            if tc is not None:
                data = cmd.get("data", {}) or {}
                tc.log_dds_command(
                    seq=cmd.get("seq", 0),
                    funcode=cmd.get("funcode", 0),
                    joint_id=data.get("id"),
                    target_value=data.get("angle"),
                    data=data,
                    correlation_id=correlation_id,
                    raw_len=len(payload),
                )
            return True
        except Exception:
            logger.exception("Failed to publish command")
            tc = self._get_collector()
            if tc is not None:
                tc.emit(
                    "dds",
                    EventType.ERROR,
                    {"error": "publish_failed", "seq": cmd.get("seq")},
                    correlation_id,
                )
            return False

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def enable_motors(self, _correlation_id: Optional[str] = None) -> bool:
        """Enable all motors."""
        cmd: Dict[str, Any] = {"address": 1, "funcode": 5, "data": {"mode": 1}}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def disable_motors(self, _correlation_id: Optional[str] = None) -> bool:
        """Disable all motors."""
        cmd: Dict[str, Any] = {"address": 1, "funcode": 5, "data": {"mode": 0}}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def power_on(self, _correlation_id: Optional[str] = None) -> bool:
        """Power on the arm."""
        cmd: Dict[str, Any] = {"address": 1, "funcode": 6, "data": {"power": 1}}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def power_off(self, _correlation_id: Optional[str] = None) -> bool:
        """Power off the arm."""
        cmd: Dict[str, Any] = {"address": 1, "funcode": 6, "data": {"power": 0}}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def reset_to_zero(self, _correlation_id: Optional[str] = None) -> bool:
        """Reset all joints to the zero position."""
        cmd: Dict[str, Any] = {"address": 1, "funcode": 7}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def set_joint(
        self,
        joint_id: int,
        angle_deg: float,
        delay_ms: int = 0,
        _correlation_id: Optional[str] = None,
    ) -> bool:
        """Move a single joint to the given angle (degrees).

        Args:
            joint_id: Joint index 0–6.
            angle_deg: Target angle in degrees.
            delay_ms: Optional motion delay in milliseconds.
        """
        if not 0 <= joint_id <= 6:
            raise ValueError(f"joint_id must be 0–6, got {joint_id}")
        cmd: Dict[str, Any] = {
            "address": 1,
            "funcode": 1,
            "data": {"id": joint_id, "angle": angle_deg, "delay_ms": delay_ms},
        }
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def set_all_joints(
        self, angles_deg: List[float], mode: int = 0, _correlation_id: Optional[str] = None
    ) -> bool:
        """Move all joints to the given angles (degrees).

        Args:
            angles_deg: List/array of 7 angles in degrees (joints 0–6).
            mode: Motion mode (default 0).
        """
        if len(angles_deg) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} angles, got {len(angles_deg)}")
        data: Dict[str, Any] = {"mode": mode}
        for i, a in enumerate(angles_deg):
            data[f"angle{i}"] = a
        cmd: Dict[str, Any] = {"address": 1, "funcode": 2, "data": data}
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def set_gripper(self, position_mm: float, _correlation_id: Optional[str] = None) -> bool:
        """Set gripper opening in millimetres (0–65 mm).

        The D1 gripper is joint 6.  The DDS protocol expects an angle value
        for joint 6, and the firmware interprets it as millimetres of opening.
        We use funcode 1 (single-joint move) targeting joint id 6.
        """
        if not (0.0 <= position_mm <= 65.0):
            logger.warning("Gripper position %.1f out of range [0, 65]", position_mm)
        cmd: Dict[str, Any] = {
            "address": 1,
            "funcode": 1,
            "data": {"id": 6, "angle": position_mm, "delay_ms": 0},
        }
        if _correlation_id:
            cmd["_correlation_id"] = _correlation_id
        return self.send_command(cmd)

    def get_gripper_position(self) -> float:
        """Return the latest gripper position (joint 6) in mm, or 0.0."""
        with self._cache.lock:
            if self._cache.joint_angles is None:
                return 0.0
            return float(self._cache.joint_angles.get("angle6", 0.0))

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_joint_angles(self) -> Optional[np.ndarray]:
        """Return the latest joint angles as a (7,) numpy array in degrees,
        or None if no feedback has been received yet."""
        with self._cache.lock:
            if self._cache.joint_angles is None:
                return None
            return np.array(
                [self._cache.joint_angles.get(f"angle{i}", 0.0) for i in range(NUM_JOINTS)],
                dtype=np.float64,
            )

    def get_status(self) -> Optional[Dict[str, int]]:
        """Return the latest status dict (enable_status, power_status,
        error_status) or None."""
        with self._cache.lock:
            if self._cache.status is None:
                return None
            return dict(self._cache.status)

    def get_state(self) -> Optional[D1State]:
        """Return a D1State populated from the latest feedback.

        Joint positions are in degrees (matching the DDS protocol).
        Velocities and torques are zeroed since the DDS feedback
        doesn't include them.  Gripper is joint 6.
        """
        angles = self.get_joint_angles()
        if angles is None:
            return None
        with self._cache.lock:
            ts = self._cache.last_update
        return D1State(
            joint_positions=angles,
            joint_velocities=np.zeros(NUM_JOINTS, dtype=np.float64),
            joint_torques=np.zeros(NUM_JOINTS, dtype=np.float64),
            gripper_position=float(angles[6]) if len(angles) > 6 else 0.0,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Background feedback reader
    # ------------------------------------------------------------------

    def _feedback_loop(self) -> None:
        """Continuously read feedback samples and update the cache."""
        logger.info("Feedback reader thread started")
        _fb_count = 0
        while not self._stop_event.is_set():
            if self._reader is None:
                break
            try:
                samples = self._reader.take(N=32)
                if _fb_count == 0 and samples:
                    logger.info("First take() returned %d samples", len(samples))
                elif _fb_count == 0 and not samples:
                    pass  # normal - no data yet
            except Exception as take_err:
                if _fb_count == 0:
                    logger.warning("take() exception (first): %s", take_err)
                samples = []
            for sample in samples:
                _fb_count += 1
                if _fb_count <= 3 or _fb_count % 100 == 0:
                    logger.info("Feedback sample #%d received, data_=%s", _fb_count, getattr(sample, 'data_', 'N/A')[:100] if hasattr(sample, 'data_') else 'no data_')
                try:
                    self._process_feedback(sample)
                except Exception as e:
                    logger.error("Error processing feedback: %s", e, exc_info=True)

            # Stale connection detection
            tc = self._get_collector()
            if tc is not None:
                now = time.monotonic()
                with self._cache.lock:
                    last = self._cache.last_update
                if last > 0 and (now - last) > 2.0:
                    if (now - self._last_stale_warn) > 5.0:
                        tc.emit(
                            "dds",
                            EventType.ERROR,
                            {
                                "error": "stale_connection",
                                "seconds_since_last": round(now - last, 2),
                            },
                        )
                        self._last_stale_warn = now

            self._stop_event.wait(self._POLL_INTERVAL)
        logger.debug("Feedback reader thread stopped")

    def _process_feedback(self, sample: ArmString_) -> None:
        """Parse a single feedback sample and update the cache."""
        try:
            msg = json.loads(sample.data_)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Ignoring malformed feedback: %r", getattr(sample, "data_", None))
            return

        funcode = msg.get("funcode")
        data = msg.get("data")
        seq = msg.get("seq", 0)

        tc = self._get_collector()
        if tc is not None:
            tc.log_dds_feedback(
                seq=seq,
                funcode=funcode,
                angles=data if funcode == 1 and isinstance(data, dict) else None,
                status=data if funcode == 3 and isinstance(data, dict) else None,
            )

        if data is None and funcode != 7:
            return

        now = time.monotonic()
        with self._cache.lock:
            self._cache.last_seq = seq
            self._cache.last_update = now
            if funcode == 1:
                # Joint angle feedback
                self._cache.joint_angles = data
                # Detect changes for telemetry
                if tc is not None and isinstance(data, dict):
                    if self._prev_angles is not None:
                        changed = {k: v for k, v in data.items() if self._prev_angles.get(k) != v}
                        if changed:
                            tc.emit("dds", EventType.STATE_UPDATE, {"changed": changed})
                    self._prev_angles = dict(data) if data else None
            elif funcode == 3:
                # Status feedback
                self._cache.status = data
                if tc is not None and isinstance(data, dict):
                    if tc is not None:
                        if "recv_status" in data:
                            tc.emit(
                                "dds",
                                EventType.CMD_ACK,
                                {"seq": seq, "recv_status": data["recv_status"]},
                            )
                        if "exec_status" in data:
                            tc.emit(
                                "dds",
                                EventType.CMD_EXEC,
                                {"seq": seq, "exec_status": data["exec_status"]},
                            )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def connect_d1_dds(interface_name: str = "eno1") -> D1DDSConnection:
    """Connect to the D1 arm via DDS on the given network interface."""
    conn = D1DDSConnection()
    if conn.connect(interface_name=interface_name):
        return conn
    raise ConnectionError(f"Failed to establish DDS connection on {interface_name}")
