"""
High-level arm operations — codified from hard-won operational experience.

These are patterns that were discovered through trial and error during
Days 1-4 of D1 arm operation. Each function documents WHY it exists
and what failure mode it prevents.

Usage:
    ops = ArmOps("http://localhost:8080")
    await ops.staged_reach([10, 30, 20, 0, 85, 0])  # safe multi-joint move
    await ops.full_recovery()                         # power-on + reset + enable
    await ops.retreat_home()                          # safe fallback when stuck
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("th3cl4w.control.arm_ops")


@dataclass
class MoveResult:
    success: bool
    final_joints: list[float]
    error: str = ""
    steps_taken: int = 0


class ArmOps:
    """Reusable arm operation primitives built from operational experience."""

    def __init__(self, server_url: str = "http://localhost:8080", timeout: float = 5.0):
        self.server_url = server_url
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core: move-and-verify
    # ------------------------------------------------------------------

    async def move_joint_verified(
        self,
        joint_id: int,
        target_deg: float,
        tolerance_deg: float = 5.0,
        settle_time: float = 1.0,
        max_retries: int = 3,
    ) -> bool:
        """Send a joint command and verify it reached the target via feedback.

        WHY: DDS feedback is unreliable (returns 0.0° intermittently, or stale
        values). Blind fire-and-forget commands led to the arm being in unknown
        states. This pattern catches stalls, overcurrent trips, and feedback
        glitches.

        IMPROVEMENT NEEDED: Currently uses /api/state polling. Could use
        WebSocket state stream for faster verification.
        """
        for attempt in range(max_retries):
            await self._set_joint(joint_id, target_deg)
            await asyncio.sleep(settle_time)

            current = await self._get_joints()
            if current is None:
                logger.warning("move_joint_verified: couldn't read feedback (attempt %d)", attempt)
                continue

            actual = current[joint_id]
            error = abs(actual - target_deg)
            if error <= tolerance_deg:
                return True

            logger.warning(
                "J%d target=%.1f° actual=%.1f° (error=%.1f°, attempt %d/%d)",
                joint_id,
                target_deg,
                actual,
                error,
                attempt + 1,
                max_retries,
            )

        logger.error(
            "J%d failed to reach %.1f° after %d attempts", joint_id, target_deg, max_retries
        )
        return False

    # ------------------------------------------------------------------
    # Staged reach: prevents overcurrent by sequencing joints properly
    # ------------------------------------------------------------------

    async def staged_reach(
        self,
        target_joints: list[float],
        step_deg: float = 10.0,
        step_delay: float = 0.3,
    ) -> MoveResult:
        """Move to target pose using safe joint sequencing.

        WHY: Simultaneously commanding large moves on J1 (shoulder) + J2 (elbow)
        causes overcurrent trips — the combined torque exceeds firmware limits.
        Discovered Day 2 evening when arm powered off mid-reach (J1=50° + J2=60°).

        STRATEGY (from Day 2-3 operational experience):
        1. Base yaw (J0) first — low torque, no gravity load
        2. Wrist/forearm rolls (J3, J5) — also low torque
        3. Shoulder (J1) in small increments — primary gravity load
        4. Elbow (J2) in small increments — secondary gravity load
        5. Wrist pitch (J4) last — depends on J1/J2 being settled

        IMPROVEMENT NEEDED: Step size could be adaptive based on current
        torque estimates. Also needs feedback-based stall detection to
        abort early if a joint is stuck.
        """
        current = await self._get_joints()
        if current is None:
            return MoveResult(False, [], "Can't read current joints")

        steps = 0

        # Phase 1: Low-torque joints — move directly
        low_torque = [0, 3, 5]  # J0=yaw, J3=forearm roll, J5=gripper roll
        for j in low_torque:
            if abs(current[j] - target_joints[j]) > 2.0:
                await self._set_joint(j, target_joints[j])
                steps += 1
        if steps > 0:
            await asyncio.sleep(0.5)

        # Phase 2: High-torque pitch joints — ramp in small increments
        # Order: shoulder first (needs to lift before elbow extends)
        high_torque = [1, 2, 4]  # J1=shoulder, J2=elbow, J4=wrist pitch
        for j in high_torque:
            current_angle = current[j]
            target_angle = target_joints[j]
            remaining = target_angle - current_angle

            while abs(remaining) > 2.0:
                increment = min(step_deg, abs(remaining))
                if remaining < 0:
                    increment = -increment
                next_angle = current_angle + increment
                await self._set_joint(j, next_angle)
                steps += 1
                await asyncio.sleep(step_delay)
                current_angle = next_angle
                remaining = target_angle - current_angle

            # Final target
            await self._set_joint(j, target_angle)
            steps += 1
            await asyncio.sleep(step_delay)

        # Verify final position
        await asyncio.sleep(0.5)
        final = await self._get_joints()
        if final is None:
            return MoveResult(False, [], "Can't verify final position", steps)

        # Check all joints within tolerance
        max_error = max(abs(final[i] - target_joints[i]) for i in range(6))
        ok = max_error < 10.0  # generous tolerance for now

        return MoveResult(ok, final, "" if ok else f"Max error: {max_error:.1f}°", steps)

    # ------------------------------------------------------------------
    # Staged retract: reverse of reach — distal joints first
    # ------------------------------------------------------------------

    async def staged_retract(
        self,
        target_joints: Optional[list[float]] = None,
        step_deg: float = 10.0,
        step_delay: float = 0.3,
    ) -> MoveResult:
        """Retract arm safely — distal joints first, then proximal.

        WHY: When retracting, you want to fold the arm IN before lifting
        the shoulder. Retracting shoulder while elbow is extended creates
        the same overcurrent risk as reaching. Think of it like a human
        pulling their arm back: bend elbow first, then lift shoulder.

        Order: J4 (wrist) → J2 (elbow) → J1 (shoulder) → rolls last
        """
        if target_joints is None:
            target_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # home

        current = await self._get_joints()
        if current is None:
            return MoveResult(False, [], "Can't read current joints")

        steps = 0

        # Phase 1: Distal pitch joints first (reverse of reach order)
        retract_order = [4, 2, 1]  # wrist → elbow → shoulder
        for j in retract_order:
            current_angle = current[j]
            target_angle = target_joints[j]
            remaining = target_angle - current_angle

            while abs(remaining) > 2.0:
                increment = min(step_deg, abs(remaining))
                if remaining < 0:
                    increment = -increment
                next_angle = current_angle + increment
                await self._set_joint(j, next_angle)
                steps += 1
                await asyncio.sleep(step_delay)
                current_angle = next_angle
                remaining = target_angle - current_angle

            await self._set_joint(j, target_angle)
            steps += 1

        await asyncio.sleep(0.3)

        # Phase 2: Low-torque joints
        for j in [0, 3, 5]:
            if abs(current[j] - target_joints[j]) > 2.0:
                await self._set_joint(j, target_joints[j])
                steps += 1

        await asyncio.sleep(0.5)
        final = await self._get_joints()
        return MoveResult(
            final is not None,
            final or [],
            "",
            steps,
        )

    # ------------------------------------------------------------------
    # Full recovery: power-on → reset → enable (all three required!)
    # ------------------------------------------------------------------

    async def full_recovery(self) -> bool:
        """Execute full power recovery sequence.

        WHY: After overcurrent trip or unexpected power loss, you MUST do
        all three steps in order. Enable alone doesn't work. Reset alone
        doesn't work. Discovered Day 2 evening after arm sagged to
        J1=-90°, J4=-95° following overcurrent.

        The sequence:
        1. power-on  — restores power to motors
        2. reset     — clears fault state (but DON'T use reset-to-zero,
                       which commands all joints to 0° simultaneously
                       and causes ANOTHER overcurrent trip!)
        3. enable    — activates motor control (use enable-here to hold
                       current position, not snap to 0°)

        IMPROVEMENT NEEDED: Should verify each step succeeded before
        proceeding. Currently fire-and-forget with delays.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info("Recovery: power-on")
            r = await client.post(f"{self.server_url}/api/command/power-on")
            if r.status_code != 200:
                logger.error("Power-on failed: %s", r.text)
                return False
            await asyncio.sleep(1.0)

            logger.info("Recovery: reset")
            r = await client.post(f"{self.server_url}/api/command/reset")
            if r.status_code != 200:
                logger.error("Reset failed: %s", r.text)
                return False
            await asyncio.sleep(1.0)

            # enable-here holds current position (safe)
            # enable alone snaps to 0° (dangerous if arm is extended)
            logger.info("Recovery: enable-here")
            r = await client.post(f"{self.server_url}/api/command/enable-here")
            if r.status_code != 200:
                logger.error("Enable-here failed: %s", r.text)
                return False
            await asyncio.sleep(0.5)

        logger.info("Recovery complete")
        return True

    # ------------------------------------------------------------------
    # Retreat home: safe fallback when stuck
    # ------------------------------------------------------------------

    async def retreat_home(self) -> MoveResult:
        """Safely return to home position.

        WHY: "Go home when stuck" is the #1 recovery strategy. Home position
        is [0, 0, 0, 0, 0, 0] — arm straight up, minimum torque, no risk of
        hitting table. Uses staged_retract which folds distal joints first.

        Call this when:
        - Pick attempt fails mid-sequence
        - Joints are in unknown state
        - Overcurrent was detected
        - Visual servo diverged
        - Any "I don't know what to do" situation
        """
        return await self.staged_retract([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Approach from above: the strategy that actually works for picking
    # ------------------------------------------------------------------

    async def approach_from_above(
        self,
        target_joints: list[float],
        approach_height_offset: float = 15.0,
    ) -> MoveResult:
        """Two-phase approach: position above target, then lower.

        WHY: Direct moves to a pick position risk colliding with objects
        on the table. The reference pose grab (Day 4) succeeded because
        the can was directly below — the wrist was at 90° pointing
        straight down. The approach-from-above pattern:

        1. Move to "hover" position: same XY but J1 reduced (arm higher)
           and J4 at ~70° (wrist partially tilted, not yet straight down)
        2. Lower to target: increase J1 and J4 to final values

        This mimics how a human grabs something: reach over it, then
        lower your hand onto it.

        IMPROVEMENT NEEDED: The approach offset is empirical (15° less on J1).
        Should be calculated from actual FK to maintain a specific height
        above the target. Also needs camera verification between phases.
        """
        # Phase 1: Hover above
        hover = list(target_joints)
        hover[1] = target_joints[1] - approach_height_offset  # less forward lean = higher
        hover[4] = min(target_joints[4], 70.0)  # partial wrist tilt

        result = await self.staged_reach(hover)
        if not result.success:
            logger.warning("Approach hover failed, retreating")
            await self.retreat_home()
            return result

        await asyncio.sleep(1.0)

        # Phase 2: Lower to target
        result = await self.staged_reach(target_joints, step_deg=5.0, step_delay=0.5)
        if not result.success:
            logger.warning("Approach lower failed, retreating")
            await self.retreat_home()

        return result

    # ------------------------------------------------------------------
    # Grip and verify: close gripper + check if we got something
    # ------------------------------------------------------------------

    async def grip_and_verify(
        self,
        grip_mm: float = 32.5,
        verify_with_camera: bool = True,
    ) -> bool:
        """Close gripper and optionally verify with camera.

        WHY: Blind gripping fails silently — the gripper closes to the
        target width whether or not an object is there. Camera verification
        after lifting is the only reliable way to confirm a successful pick
        (no torque/force sensing available on D1 DDS).

        Grip width 32.5mm is calibrated for a Red Bull can (~66mm diameter,
        gripper compresses ~half). Different objects need different widths.

        IMPROVEMENT NEEDED: Should use virtual grip detector in sim mode.
        Could also check gripper feedback position — if it closed PAST
        the target, nothing was grabbed.
        """
        await self._set_gripper(grip_mm)
        await asyncio.sleep(1.0)

        if verify_with_camera:
            # TODO: integrate with camera verification endpoint
            # For now just log
            logger.info("Grip complete at %.1fmm — camera verification not yet integrated", grip_mm)

        return True  # optimistic for now

    # ------------------------------------------------------------------
    # Lift after grip: raise arm while holding object
    # ------------------------------------------------------------------

    async def lift_from_pick(
        self,
        current_joints: list[float],
        lift_deg: float = 20.0,
    ) -> MoveResult:
        """Lift arm after gripping — reduce J1 to raise shoulder.

        WHY: After gripping, lift straight up before any lateral movement.
        Moving laterally while low risks dragging the object or losing grip.
        J1 reduction (less forward lean) raises the arm.

        IMPROVEMENT NEEDED: Lift amount should be adaptive — just enough
        to clear the table surface. Could use side camera to verify clearance.
        """
        lift = list(current_joints)
        lift[1] = max(0.0, current_joints[1] - lift_deg)
        return await self.staged_reach(lift, step_deg=5.0, step_delay=0.5)

    # ------------------------------------------------------------------
    # Place: lower to a position and release
    # ------------------------------------------------------------------

    async def place_at(
        self,
        target_joints: list[float],
        release_width_mm: float = 60.0,
    ) -> MoveResult:
        """Place held object at target — approach from above, release, retract.

        WHY: Dropping from height damages objects and is imprecise.
        Lower to just above the surface, open gripper, then retract.
        The approach-from-above pattern is reused here.

        IMPROVEMENT NEEDED: Needs contact/proximity detection to know
        when the object has touched the surface. Currently just trusts
        the planned joint angles.
        """
        # Approach from above
        result = await self.approach_from_above(target_joints)
        if not result.success:
            return result

        # Open gripper to release
        await self._set_gripper(release_width_mm)
        await asyncio.sleep(0.5)

        # Retract upward
        return await self.lift_from_pick(target_joints, lift_deg=20.0)

    # ------------------------------------------------------------------
    # Full pick sequence: detect → plan → approach → grip → lift
    # ------------------------------------------------------------------

    async def pick_sequence(
        self,
        target_joints: list[float],
        grip_mm: float = 32.5,
        open_mm: float = 60.0,
    ) -> MoveResult:
        """Complete pick sequence with all safety patterns applied.

        Combines all operational learnings:
        1. Open gripper first (don't approach with gripper closed)
        2. Staged reach to hover position
        3. Lower to target with small steps
        4. Grip and verify
        5. Lift straight up
        6. On any failure → retreat home

        IMPROVEMENT NEEDED: Add camera verification between each phase.
        Add episode recording. Add timeout per phase.
        """
        try:
            # Open gripper
            await self._set_gripper(open_mm)
            await asyncio.sleep(0.5)

            # Approach from above
            result = await self.approach_from_above(target_joints)
            if not result.success:
                return result

            # Grip
            ok = await self.grip_and_verify(grip_mm)
            if not ok:
                await self.retreat_home()
                return MoveResult(False, [], "Grip failed")

            # Lift
            result = await self.lift_from_pick(target_joints)
            if not result.success:
                await self.retreat_home()

            return result

        except Exception as e:
            logger.error("Pick sequence error: %s", e)
            await self.retreat_home()
            return MoveResult(False, [], str(e))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _set_joint(self, joint_id: int, angle: float):
        """Send individual joint command. Never use set_all_joints.

        WHY: set_all_joints (funcode 2) causes arm freezes on large moves.
        Individual set_joint (funcode 1) is always safe. Discovered Day 2.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.server_url}/api/command/set-joint",
                json={"id": joint_id, "angle": angle},
            )
            if r.status_code != 200:
                logger.warning("set_joint J%d=%.1f° failed: %s", joint_id, angle, r.text)

    async def _set_gripper(self, position_mm: float):
        """Send gripper command."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.server_url}/api/command/set-gripper",
                json={"position": position_mm},
            )
            if r.status_code != 200:
                logger.warning("set_gripper %.1fmm failed: %s", position_mm, r.text)

    async def _get_joints(self) -> Optional[list[float]]:
        """Get current joint angles from server. Returns None on failure.

        WHY: DDS feedback intermittently returns 0.0° for all joints.
        We retry up to 3 times with short delays to handle this.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    r = await client.get(f"{self.server_url}/api/state")
                    if r.status_code == 200:
                        data = r.json()
                        joints = data.get("joint_angles", [])
                        if len(joints) >= 6 and not all(j == 0.0 for j in joints[:6]):
                            return joints[:6]
                        # All zeros — likely stale DDS, retry
                        logger.debug("Feedback all zeros (attempt %d), retrying", attempt)
                        await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning("_get_joints error: %s", e)
                    await asyncio.sleep(0.5)
        return None

    async def _get_state(self) -> Optional[dict]:
        """Get full arm state dict."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.get(f"{self.server_url}/api/state")
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
        return None
