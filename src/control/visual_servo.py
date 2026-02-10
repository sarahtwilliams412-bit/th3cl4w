"""Visual servoing: move arm toward a target using camera feedback.

The arm is a sensor — every move is verified by taking a photo and asking
the LLM where the gripper and target are. No blind moves.

Usage:
    from src.control.visual_servo import VisualServo
    servo = VisualServo()
    result = await servo.approach("red bull can")
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ARM_API = "http://localhost:8080"
CAM_API = "http://localhost:8081"

# Image dimensions
IMG_W, IMG_H = 1920, 1080


@dataclass
class ServoStep:
    """One step of visual servoing."""

    step: int
    joints_before: list[float]
    joints_after: list[float]
    gripper_pixel: Optional[tuple[float, float]] = None
    target_pixel: Optional[tuple[float, float]] = None
    pixel_distance: float = 0.0
    action: str = ""
    notes: str = ""


@dataclass
class ServoResult:
    """Result of a visual servo approach."""

    success: bool
    steps: list[ServoStep] = field(default_factory=list)
    total_time_s: float = 0.0
    final_distance_px: float = 999.0
    message: str = ""


class VisualServo:
    """Visual servoing controller.

    Strategy:
    1. Snap both cameras
    2. Ask LLM: where is gripper tip? where is target?
    3. Compute pixel delta
    4. Map delta to joint increments
    5. Move ONE joint a small amount
    6. Snap again, verify the move reduced the distance
    7. Repeat until close enough
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        max_steps: int = 25,
        close_enough_px: int = 80,  # pixels in overhead view
        step_deg: float = 5.0,  # degrees per step
    ):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")

        from google import genai as _genai

        self._client = _genai.Client(api_key=self.api_key)
        self._model_name = "gemini-2.0-flash"

        self.max_steps = max_steps
        self.close_enough_px = close_enough_px
        self.step_deg = step_deg

    async def get_joints(self) -> list[float]:
        """Read current joint angles."""
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{ARM_API}/api/state")
            return r.json()["joints"]

    async def set_joint(self, joint_id: int, angle: float) -> bool:
        """Command a single joint. Waits for it to settle."""
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                f"{ARM_API}/api/command/set-joint",
                json={"id": joint_id, "angle": round(angle, 1)},
            )
            if r.status_code != 200:
                return False
        await asyncio.sleep(1.5)  # settle time
        return True

    async def snap(self, cam_id: int = 1) -> bytes:
        """Capture a JPEG from camera."""
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{CAM_API}/snap/{cam_id}")
            return r.content

    async def locate_in_frame(
        self, jpeg: bytes, target: str, cam_id: int = 1
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]], str]:
        """Ask LLM to find gripper tip and target in the image.

        Returns: (gripper_pixel, target_pixel, raw_response)
        """
        b64 = base64.b64encode(jpeg).decode()

        view = "overhead (top-down)" if cam_id == 1 else "front"
        prompt = (
            f"This is a 1920x1080 {view} image of a robot arm workspace.\n"
            f"Find TWO things:\n"
            f"1. The gripper tip (very end of the robot arm / end-effector)\n"
            f"2. The {target}\n\n"
            f"Return ONLY valid JSON, no markdown:\n"
            f'{{"gripper": {{"u": <x_pixel>, "v": <y_pixel>}}, '
            f'"target": {{"u": <x_pixel>, "v": <y_pixel>}}, '
            f'"distance_estimate_inches": <your estimate of real-world distance between them>}}'
        )

        try:
            from google.genai import types as _gtypes

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    _gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            text = response.text.strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            data = json.loads(text)
            g = data.get("gripper", {})
            t = data.get("target", {})
            gripper = (float(g["u"]), float(g["v"])) if g else None
            target_px = (float(t["u"]), float(t["v"])) if t else None
            return gripper, target_px, text
        except Exception as e:
            logger.warning(
                f"LLM locate failed: {e}, raw={response.text[:200] if 'response' in dir() else '?'}"
            )
            return None, None, str(e)

    async def locate_averaged(
        self, target: str, cam_id: int = 1, n: int = 3
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        """Take N readings and return median positions to reduce LLM noise."""
        grippers, targets = [], []
        for _ in range(n):
            jpeg = await self.snap(cam_id)
            g, t, _ = await self.locate_in_frame(jpeg, target, cam_id)
            if g and t:
                grippers.append(g)
                targets.append(t)

        if len(grippers) < 2:
            return None, None

        # Median of each coordinate
        import statistics

        g_u = statistics.median(g[0] for g in grippers)
        g_v = statistics.median(g[1] for g in grippers)
        t_u = statistics.median(t[0] for t in targets)
        t_v = statistics.median(t[1] for t in targets)
        return (g_u, g_v), (t_u, t_v)

    def pixel_delta_to_joint_action(
        self, gripper_px: tuple, target_px: tuple, joints: list[float], cam_id: int = 1
    ) -> tuple[int, float, str]:
        """Convert pixel delta to a single joint move.

        For overhead camera (cam_id=1):
        - du (horizontal) → J0 (base yaw)
        - dv (vertical) → J1/J2 (reach in/out)

        For front camera (cam_id=0):
        - du (horizontal) → J0 (base yaw)
        - dv (vertical) → J1/J4 (height)

        Returns: (joint_id, delta_degrees, reason)
        """
        du = target_px[0] - gripper_px[0]  # positive = target is to the right
        dv = target_px[1] - gripper_px[1]  # positive = target is below

        step = self.step_deg

        if cam_id == 1:  # Overhead
            # Determine which axis has bigger error
            if abs(du) > abs(dv):
                # Horizontal: adjust J0
                # In overhead cam, image-right could be +J0 or -J0
                # We'll try a direction and verify with next snap
                delta = step if du > 0 else -step
                return (
                    0,
                    delta,
                    f"J0 {'CW' if delta<0 else 'CCW'} (target {'right' if du>0 else 'left'} by {abs(du):.0f}px)",
                )
            else:
                # Vertical: adjust reach (J1 for lean, J2 for extend)
                # In overhead, image-down typically = further from base
                # Use J1 (shoulder lean) as primary reach control
                if dv > 0:
                    # Target further away → lean more forward (J1 more negative)
                    return 1, -step, f"J1 forward (target further by {abs(dv):.0f}px)"
                else:
                    # Target closer → lean back (J1 more positive)
                    return 1, step, f"J1 backward (target closer by {abs(dv):.0f}px)"

        else:  # Front camera
            if abs(dv) > abs(du):
                # Vertical: target below → need to lower arm
                if dv > 0:
                    # Target is below gripper. Cycle through lowering methods:
                    # 1. Extend elbow (J2+) pushes forearm down when shoulder is raised
                    # 2. Lean shoulder forward (J1-) lowers everything
                    # 3. Wrist down (J4+) as fine adjustment
                    if joints[2] < 80:
                        return 2, step, f"J2 extend (target below by {abs(dv):.0f}px)"
                    elif joints[1] > -80:
                        return 1, -step, f"J1 forward (target below by {abs(dv):.0f}px)"
                    elif joints[4] < 85:
                        return 4, step, f"J4 down (target below by {abs(dv):.0f}px)"
                    else:
                        return 1, -step, f"J1 forward (all maxed, target below by {abs(dv):.0f}px)"
                else:
                    # Target above gripper
                    if joints[4] > -80:
                        return 4, -step, f"J4 up (target above by {abs(dv):.0f}px)"
                    else:
                        return 1, step, f"J1 back (target above by {abs(dv):.0f}px)"
            else:
                delta = step if du > 0 else -step
                return 0, delta, f"J0 (target {'right' if du>0 else 'left'} by {abs(du):.0f}px)"

    async def approach(self, target: str = "red bull can") -> ServoResult:
        """Visual servo approach to target.

        1. Use overhead cam for horizontal alignment (J0 + reach)
        2. Switch to front cam for height alignment (J4)
        3. Alternate until close
        """
        t0 = time.time()
        result = ServoResult(success=False)

        # Start from a reaching pose — SEQUENCE MATTERS for torque safety:
        # 1. Lift shoulder (elbow tucked) — low torque
        # 2. Extend elbow gradually — distribute load
        # 3. Angle wrist down — point at target
        # Like a human: raise arm → reach out → angle hand
        logger.info("Visual servo: moving to initial reach pose (human-like sequence)")
        await self.set_joint(2, 0)  # tuck elbow first
        await self.set_joint(4, 0)  # wrist neutral
        await self.set_joint(1, 30)  # lift shoulder UP (with elbow tucked = safe)
        await asyncio.sleep(1)
        await self.set_joint(2, 30)  # start extending elbow
        await self.set_joint(2, 50)  # extend more
        await self.set_joint(2, 70)  # full extension
        await asyncio.sleep(1)
        await self.set_joint(4, 50)  # angle wrist down
        await self.set_joint(4, 70)  # more down

        prev_distance = 9999.0
        stall_count = 0
        cam_id = 1  # Start with overhead for horizontal alignment
        phase = "horizontal"  # horizontal → vertical → fine

        for step_num in range(self.max_steps):
            joints_before = await self.get_joints()

            # Snap and locate (averaged over 3 readings to reduce LLM noise)
            gripper_px, target_px = await self.locate_averaged(target, cam_id, n=3)

            if gripper_px is None or target_px is None:
                logger.warning(f"Step {step_num}: LLM couldn't find gripper or target")
                # Try other camera
                cam_id = 1 - cam_id
                result.steps.append(
                    ServoStep(
                        step=step_num,
                        joints_before=joints_before,
                        joints_after=joints_before,
                        action="switch_cam",
                        notes=f"Detection failed, switching to cam {cam_id}",
                    )
                )
                continue

            # Compute distance
            import math

            dx = target_px[0] - gripper_px[0]
            dy = target_px[1] - gripper_px[1]
            distance = math.sqrt(dx * dx + dy * dy)

            logger.info(
                f"Step {step_num} cam{cam_id}: gripper=({gripper_px[0]:.0f},{gripper_px[1]:.0f}) "
                f"target=({target_px[0]:.0f},{target_px[1]:.0f}) dist={distance:.0f}px"
            )

            # Check if close enough
            if distance < self.close_enough_px:
                result.success = True
                result.final_distance_px = distance
                result.message = f"Reached target within {distance:.0f}px after {step_num+1} steps"
                logger.info(f"Visual servo: SUCCESS — {result.message}")
                break

            # Check for stalls (distance not decreasing)
            if distance >= prev_distance - 10:
                stall_count += 1
                if stall_count >= 3:
                    # Switch camera or change strategy
                    cam_id = 1 - cam_id
                    stall_count = 0
                    logger.info(f"Stall detected, switching to cam {cam_id}")
            else:
                stall_count = 0

            prev_distance = distance

            # Adaptive step size: bigger when far, smaller when close
            if distance > 300:
                self.step_deg = 8.0
            elif distance > 150:
                self.step_deg = 5.0
            else:
                self.step_deg = 3.0

            # Phase management: horizontal first, then vertical
            if phase == "horizontal" and distance < 100:
                phase = "vertical"
                cam_id = 0  # switch to front cam for height
                logger.info("Phase: horizontal aligned, switching to vertical (front cam)")

            # Decide which joint to move
            joint_id, delta, reason = self.pixel_delta_to_joint_action(
                gripper_px, target_px, joints_before, cam_id
            )

            # Apply limits
            new_angle = joints_before[joint_id] + delta
            new_angle = max(-85, min(85, new_angle))

            logger.info(f"Step {step_num}: {reason} → J{joint_id} = {new_angle:.1f}°")

            # Move
            ok = await self.set_joint(joint_id, new_angle)

            # VERIFY: check if move helped
            g2, t2 = await self.locate_averaged(target, cam_id, n=2)
            if g2 and t2:
                new_dist = math.sqrt((t2[0] - g2[0]) ** 2 + (t2[1] - g2[1]) ** 2)
                if new_dist > distance + 30:
                    # Move made things worse — reverse it!
                    logger.warning(
                        f"Step {step_num}: move WORSENED distance {distance:.0f}→{new_dist:.0f}px, reversing"
                    )
                    await self.set_joint(joint_id, joints_before[joint_id])
                    reason += " (REVERSED)"

            joints_after = await self.get_joints()

            step_data = ServoStep(
                step=step_num,
                joints_before=joints_before,
                joints_after=joints_after,
                gripper_pixel=gripper_px,
                target_pixel=target_px,
                pixel_distance=distance,
                action=f"J{joint_id} {delta:+.1f}° → {new_angle:.1f}°",
                notes=reason,
            )
            result.steps.append(step_data)

        result.total_time_s = time.time() - t0
        if not result.success:
            result.final_distance_px = prev_distance
            result.message = (
                f"Did not converge after {len(result.steps)} steps (dist={prev_distance:.0f}px)"
            )

        return result


async def main():
    """CLI entry point for testing."""
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "red bull can"

    servo = VisualServo()
    result = await servo.approach(target)

    logger.info(f"\n{'SUCCESS' if result.success else 'FAILED'}: {result.message}")
    logger.info(f"Time: {result.total_time_s:.1f}s, Steps: {len(result.steps)}")
    for s in result.steps:
        logger.info(f"  Step {s.step}: {s.action} — dist={s.pixel_distance:.0f}px — {s.notes}")


if __name__ == "__main__":
    asyncio.run(main())
