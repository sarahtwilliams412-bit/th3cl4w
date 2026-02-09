"""
Gripper Contact Detection for Unitree D1 Arm.

Detects object contact by monitoring gripper position feedback during close.
If the gripper stabilizes above a threshold (object resistance), contact is declared.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ARM_API = "http://localhost:8080"

# Gripper physical range
GRIPPER_MIN_MM = 0.0
GRIPPER_MAX_MM = 65.0


class ContactStatus(Enum):
    CONTACT = "contact"
    NO_CONTACT = "no_contact"
    TIMEOUT = "timeout"


@dataclass
class ContactResult:
    """Result of a contact detection attempt."""

    contacted: bool
    status: ContactStatus
    final_mm: float
    stable_mm: float  # position where gripper stabilized (0 if no contact)
    time_s: float  # elapsed time
    readings: list = field(default_factory=list)


@dataclass
class GripResult:
    """Result of an adaptive grip sequence."""

    contacted: bool
    final_mm: float
    steps_taken: int
    grip_force_mm: float  # how far past contact we tightened
    contact_result: Optional[ContactResult] = None


@dataclass
class ObjectProfile:
    """Expected contact range for a known object type."""

    name: str
    min_contact_mm: float
    max_contact_mm: float

    def expected_at(self, mm: float) -> bool:
        return self.min_contact_mm <= mm <= self.max_contact_mm


# Pre-defined object profiles
OBJECT_PROFILES = {
    "redbull": ObjectProfile("Red Bull Can", 48.0, 54.0),
    "packet": ObjectProfile("Small Packet", 30.0, 45.0),
    "mouse": ObjectProfile("Mouse", 50.0, 65.0),
    "generic": ObjectProfile("Generic", 25.0, 60.0),
}


class GripperContactDetector:
    """Detects gripper contact with objects via position feedback monitoring."""

    def __init__(
        self,
        api_base: str = ARM_API,
        poll_hz: float = 10.0,
        stable_duration_s: float = 0.2,
        zero_filter: bool = True,
    ):
        self.api_base = api_base.rstrip("/")
        self.poll_interval = 1.0 / poll_hz
        self.stable_duration_s = stable_duration_s
        self.zero_filter = zero_filter
        self._last_result: Optional[ContactResult] = None

    @property
    def last_result(self) -> Optional[ContactResult]:
        return self._last_result

    async def _get_gripper_mm(self) -> Optional[float]:
        """Read current gripper position from the arm API."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                resp = await c.get(f"{self.api_base}/api/state")
                if resp.status_code == 200:
                    data = resp.json()
                    val = float(data.get("gripper", 0.0))
                    if self.zero_filter and val == 0.0:
                        return None  # DDS glitch — ignore
                    return val
        except Exception as e:
            logger.warning("Failed to read gripper state: %s", e)
        return None

    async def _send_gripper(self, position_mm: float) -> bool:
        """Send gripper position command."""
        position_mm = max(GRIPPER_MIN_MM, min(GRIPPER_MAX_MM, position_mm))
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                resp = await c.post(
                    f"{self.api_base}/api/command/set-gripper",
                    json={"position": position_mm},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("Failed to send gripper command: %s", e)
            return False

    async def close_and_detect(
        self,
        target_mm: float = 10.0,
        object_min_mm: float = 25.0,
        timeout_s: float = 3.0,
        profile: Optional[str] = None,
    ) -> ContactResult:
        """
        Close gripper and detect contact via position stabilization.

        Args:
            target_mm: Target close position (where gripper goes if no object).
            object_min_mm: Minimum position to consider as object contact.
            timeout_s: Max time to wait for stabilization.
            profile: Optional object profile name for adjusted thresholds.

        Returns:
            ContactResult with detection outcome.
        """
        if profile and profile in OBJECT_PROFILES:
            obj = OBJECT_PROFILES[profile]
            object_min_mm = obj.min_contact_mm - 5.0  # small margin

        # Send close command
        if not await self._send_gripper(target_mm):
            result = ContactResult(
                contacted=False,
                status=ContactStatus.TIMEOUT,
                final_mm=-1.0,
                stable_mm=0.0,
                time_s=0.0,
            )
            self._last_result = result
            return result

        start = time.monotonic()
        readings = []
        stable_start: Optional[float] = None
        stable_value: Optional[float] = None
        STABLE_TOLERANCE = 2.0  # mm — position must stay within this range

        while (elapsed := time.monotonic() - start) < timeout_s:
            mm = await self._get_gripper_mm()
            if mm is not None:
                readings.append((elapsed, mm))

                # Check if position is stable
                if stable_value is not None and abs(mm - stable_value) <= STABLE_TOLERANCE:
                    if stable_start and (elapsed - stable_start) >= self.stable_duration_s:
                        # Stable long enough — check if it's contact
                        if stable_value > object_min_mm:
                            result = ContactResult(
                                contacted=True,
                                status=ContactStatus.CONTACT,
                                final_mm=mm,
                                stable_mm=stable_value,
                                time_s=elapsed,
                                readings=readings,
                            )
                            self._last_result = result
                            logger.info(
                                "Contact detected at %.1fmm (stable for %.2fs)",
                                stable_value,
                                elapsed - stable_start,
                            )
                            return result
                else:
                    # Position changed — reset stability tracking
                    stable_value = mm
                    stable_start = elapsed

                # Check if we've reached target (no contact)
                if mm <= target_mm:
                    result = ContactResult(
                        contacted=False,
                        status=ContactStatus.NO_CONTACT,
                        final_mm=mm,
                        stable_mm=0.0,
                        time_s=elapsed,
                        readings=readings,
                    )
                    self._last_result = result
                    logger.info("No contact — gripper reached %.1fmm", mm)
                    return result

            await asyncio.sleep(self.poll_interval)

        # Timeout
        final = readings[-1][1] if readings else -1.0
        contacted = final > object_min_mm
        result = ContactResult(
            contacted=contacted,
            status=ContactStatus.CONTACT if contacted else ContactStatus.TIMEOUT,
            final_mm=final,
            stable_mm=final if contacted else 0.0,
            time_s=timeout_s,
            readings=readings,
        )
        self._last_result = result
        return result

    async def adaptive_grip(
        self,
        initial_mm: float = 15.0,
        step_mm: float = 5.0,
        object_min_mm: float = 25.0,
        profile: Optional[str] = None,
    ) -> GripResult:
        """
        Close gripper in steps to gently grip an object.

        Closes from fully open in decreasing steps, checking for object
        resistance at each step. Stops when the gripper can't close further.

        Args:
            initial_mm: Final target if no resistance detected.
            step_mm: Not used directly — we use predefined steps.
            object_min_mm: Min position to consider contact.
            profile: Optional object profile name.

        Returns:
            GripResult with grip outcome.
        """
        steps = [50, 40, 30, 20, initial_mm]

        if profile and profile in OBJECT_PROFILES:
            obj = OBJECT_PROFILES[profile]
            object_min_mm = obj.min_contact_mm - 5.0

        # Start fully open
        await self._send_gripper(GRIPPER_MAX_MM)
        await asyncio.sleep(0.5)

        prev_mm: Optional[float] = None
        steps_taken = 0
        contact_result = None

        for target in steps:
            steps_taken += 1
            logger.info("Adaptive grip step %d: closing to %.0fmm", steps_taken, target)

            if not await self._send_gripper(target):
                break

            # Wait for movement + settle
            await asyncio.sleep(0.6)

            # Read actual position
            actual = await self._get_gripper_mm()
            if actual is None:
                # Try once more
                await asyncio.sleep(0.2)
                actual = await self._get_gripper_mm()

            if actual is None:
                continue

            logger.info("  Commanded %.0fmm, actual %.1fmm", target, actual)

            # Check for resistance: gripper couldn't close to target
            delta = actual - target
            if delta > 2.0 and actual > object_min_mm:
                logger.info("  Object resistance detected at %.1fmm (delta=%.1f)", actual, delta)
                contact_result = ContactResult(
                    contacted=True,
                    status=ContactStatus.CONTACT,
                    final_mm=actual,
                    stable_mm=actual,
                    time_s=0.0,
                )
                self._last_result = contact_result
                return GripResult(
                    contacted=True,
                    final_mm=actual,
                    steps_taken=steps_taken,
                    grip_force_mm=delta,
                    contact_result=contact_result,
                )

            # Check if gripper barely moved from previous step (stalled)
            if prev_mm is not None and abs(actual - prev_mm) < 2.0 and actual > object_min_mm:
                logger.info("  Gripper stalled at %.1fmm", actual)
                contact_result = ContactResult(
                    contacted=True,
                    status=ContactStatus.CONTACT,
                    final_mm=actual,
                    stable_mm=actual,
                    time_s=0.0,
                )
                self._last_result = contact_result
                return GripResult(
                    contacted=True,
                    final_mm=actual,
                    steps_taken=steps_taken,
                    grip_force_mm=0.0,
                    contact_result=contact_result,
                )

            prev_mm = actual

        # No contact detected through all steps
        final = prev_mm if prev_mm is not None else 0.0
        result = GripResult(
            contacted=False,
            final_mm=final,
            steps_taken=steps_taken,
            grip_force_mm=0.0,
        )
        return result

    async def open_gripper(self, position_mm: float = GRIPPER_MAX_MM) -> bool:
        """Open gripper to specified position."""
        return await self._send_gripper(position_mm)
