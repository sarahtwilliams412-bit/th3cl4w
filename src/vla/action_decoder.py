"""Action Decoder — converts VLA model output to safe joint commands.

Takes the raw action plan from the model and:
1. Validates each action against safety constraints
2. Clamps joint deltas to ±10° max
3. Enforces sequencing rules (no simultaneous shoulder lift + elbow extend)
4. Converts to API-ready commands
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Joint limits (degrees) — with 5° safety margin
JOINT_LIMITS = {
    0: (-130.0, 130.0),  # J0 base yaw (135 - 5)
    1: (-80.0, 80.0),  # J1 shoulder pitch (85 - 5)
    2: (-130.0, 130.0),  # J2 elbow pitch
    3: (-130.0, 130.0),  # J3 forearm roll
    4: (-80.0, 80.0),  # J4 wrist pitch (85 - 5)
    5: (-130.0, 130.0),  # J5 gripper roll
}

MAX_DELTA_DEG = 10.0  # Maximum degrees per single action
GRIPPER_RANGE = (0.0, 65.0)  # mm


class ActionType(Enum):
    JOINT = "joint"
    GRIPPER = "gripper"
    VERIFY = "verify"
    DONE = "done"


@dataclass
class ArmAction:
    """A single validated, safe action for the arm."""

    action_type: ActionType
    joint_id: Optional[int] = None  # For JOINT actions
    target_angle: Optional[float] = None  # Absolute target angle for JOINT
    delta: Optional[float] = None  # Original delta requested
    gripper_mm: Optional[float] = None  # For GRIPPER actions
    reason: str = ""
    clamped: bool = False  # True if the action was modified for safety
    rejected: bool = False  # True if the action was rejected entirely
    reject_reason: str = ""

    @property
    def is_executable(self) -> bool:
        if self.rejected:
            return False
        return self.action_type in (ActionType.JOINT, ActionType.GRIPPER)

    def describe(self) -> str:
        if self.action_type == ActionType.JOINT:
            clamped_note = " (clamped)" if self.clamped else ""
            return f"J{self.joint_id} → {self.target_angle:.1f}° (Δ{self.delta:+.1f}°){clamped_note}: {self.reason}"
        elif self.action_type == ActionType.GRIPPER:
            return f"Gripper → {self.gripper_mm:.1f}mm: {self.reason}"
        elif self.action_type == ActionType.VERIFY:
            return f"VERIFY: {self.reason}"
        elif self.action_type == ActionType.DONE:
            return f"DONE: {self.reason}"
        return f"{self.action_type.value}: {self.reason}"


class ActionDecoder:
    """Decodes and validates model action plans into safe arm commands."""

    def __init__(
        self,
        max_delta_deg: float = MAX_DELTA_DEG,
        joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        self.max_delta = max_delta_deg
        self.limits = joint_limits or JOINT_LIMITS

    def decode(
        self,
        actions: List[Dict],
        current_joints: List[float],
        current_gripper_mm: float,
    ) -> List[ArmAction]:
        """Decode a list of raw model actions into validated ArmActions.

        Args:
            actions: Raw action dicts from the model
            current_joints: Current 6 joint angles in degrees
            current_gripper_mm: Current gripper opening in mm

        Returns:
            List of ArmAction objects, validated and safe to execute
        """
        decoded = []
        # Track projected joint state as we decode sequentially
        projected_joints = list(current_joints)

        for raw in actions:
            action_type = raw.get("type", "unknown")

            if action_type == "joint":
                arm_action = self._decode_joint(raw, projected_joints)
                if arm_action.is_executable:
                    # Update projected state
                    projected_joints[arm_action.joint_id] = arm_action.target_angle
                decoded.append(arm_action)

            elif action_type == "gripper":
                arm_action = self._decode_gripper(raw, current_gripper_mm)
                decoded.append(arm_action)

            elif action_type == "verify":
                decoded.append(
                    ArmAction(
                        action_type=ActionType.VERIFY,
                        reason=raw.get("reason", "verification checkpoint"),
                    )
                )

            elif action_type == "done":
                decoded.append(
                    ArmAction(
                        action_type=ActionType.DONE,
                        reason=raw.get("reason", "task complete"),
                    )
                )

            else:
                logger.warning("Unknown action type: %s", action_type)
                decoded.append(
                    ArmAction(
                        action_type=ActionType.VERIFY,
                        reason=f"Unknown action type '{action_type}' — treating as verify",
                    )
                )

        # Safety pass: check for dangerous sequences
        decoded = self._enforce_sequencing(decoded, current_joints)

        return decoded

    def _decode_joint(self, raw: Dict, projected_joints: List[float]) -> ArmAction:
        """Decode and validate a joint action."""
        joint_id = raw.get("id")
        if joint_id is None or not (0 <= joint_id <= 5):
            return ArmAction(
                action_type=ActionType.JOINT,
                rejected=True,
                reject_reason=f"Invalid joint id: {joint_id}",
                reason=raw.get("reason", ""),
            )

        delta = raw.get("delta", 0.0)
        if delta == 0:
            # Some models output absolute angles instead of deltas
            abs_angle = raw.get("angle")
            if abs_angle is not None:
                delta = abs_angle - projected_joints[joint_id]

        # Clamp delta to max
        clamped = False
        original_delta = delta
        if abs(delta) > self.max_delta:
            delta = self.max_delta if delta > 0 else -self.max_delta
            clamped = True

        # Compute target angle
        target = projected_joints[joint_id] + delta

        # Clamp to joint limits
        lo, hi = self.limits.get(joint_id, (-135, 135))
        if target < lo:
            target = lo
            delta = target - projected_joints[joint_id]
            clamped = True
        elif target > hi:
            target = hi
            delta = target - projected_joints[joint_id]
            clamped = True

        # Skip if effectively zero movement
        if abs(delta) < 0.5:
            return ArmAction(
                action_type=ActionType.JOINT,
                joint_id=joint_id,
                target_angle=projected_joints[joint_id],
                delta=0.0,
                rejected=True,
                reject_reason=f"Delta too small ({original_delta:.1f}° → {delta:.1f}°)",
                reason=raw.get("reason", ""),
            )

        return ArmAction(
            action_type=ActionType.JOINT,
            joint_id=joint_id,
            target_angle=round(target, 1),
            delta=round(delta, 1),
            clamped=clamped,
            reason=raw.get("reason", ""),
        )

    def _decode_gripper(self, raw: Dict, current_mm: float) -> ArmAction:
        """Decode and validate a gripper action."""
        position = raw.get("position_mm", raw.get("position", 0.0))
        position = max(GRIPPER_RANGE[0], min(GRIPPER_RANGE[1], float(position)))

        return ArmAction(
            action_type=ActionType.GRIPPER,
            gripper_mm=round(position, 1),
            reason=raw.get("reason", ""),
        )

    def _enforce_sequencing(
        self,
        actions: List[ArmAction],
        current_joints: List[float],
    ) -> List[ArmAction]:
        """Enforce safety sequencing rules.

        Key rule: NEVER extend elbow (J2+) while lifting shoulder (J1+) in the same batch.
        If both are present, execute shoulder first, then add a verify, then elbow.
        """
        j1_actions = [a for a in actions if a.action_type == ActionType.JOINT and a.joint_id == 1]
        j2_actions = [a for a in actions if a.action_type == ActionType.JOINT and a.joint_id == 2]

        # Check for dangerous combination: J1 going positive (up) AND J2 going positive (extend)
        j1_lifting = any(a.delta and a.delta > 0 for a in j1_actions)
        j2_extending = any(a.delta and a.delta > 0 for a in j2_actions)

        if j1_lifting and j2_extending:
            logger.warning(
                "SAFETY: Detected shoulder lift + elbow extend in same batch. "
                "Inserting verify checkpoint between them."
            )
            # Reorder: all J1 actions first, then verify, then J2 actions
            reordered = []
            other_actions = []
            for a in actions:
                if a.action_type == ActionType.JOINT and a.joint_id == 1:
                    reordered.append(a)
                elif (
                    a.action_type == ActionType.JOINT
                    and a.joint_id == 2
                    and a.delta
                    and a.delta > 0
                ):
                    other_actions.append(a)
                else:
                    reordered.append(a)

            reordered.append(
                ArmAction(
                    action_type=ActionType.VERIFY,
                    reason="Safety: verify after shoulder lift before elbow extension",
                )
            )
            reordered.extend(other_actions)
            return reordered

        return actions
