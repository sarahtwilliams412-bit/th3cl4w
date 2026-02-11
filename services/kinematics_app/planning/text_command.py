"""
Text Command Parser for Unitree D1 Robotic Arm

Parses natural language commands into structured arm actions.
Supports movement, gripper, task, and pose commands.

Examples:
    "wave hello"           -> task: wave
    "go home"              -> task: home
    "open the gripper"     -> gripper: open (65mm)
    "close gripper"        -> gripper: close (0mm)
    "move joint 0 to 45"   -> set_joint: J0 = 45
    "set all joints to 0, -45, 0, 90, 0, -45"  -> set_all_joints
    "look up"              -> set_joint: shoulder pitch up
    "reach forward"        -> set_all_joints: extended pose
    "point left"           -> set_joint: base yaw left
    "stop"                 -> stop
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.config.pick_config import get_pick_config as _get_pick_config

logger = logging.getLogger(__name__)


class CommandType(Enum):
    TASK = "task"
    SET_JOINT = "set_joint"
    SET_ALL_JOINTS = "set_all_joints"
    SET_GRIPPER = "set_gripper"
    POWER = "power"
    STOP = "stop"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Result of parsing a text command."""

    command_type: CommandType
    action: str = ""
    joints: dict[int, float] = field(default_factory=dict)
    all_joints: list[float] = field(default_factory=list)
    gripper_mm: Optional[float] = None
    speed: float = 0.6
    description: str = ""
    confidence: float = 1.0


# Joint name aliases for natural language
_JOINT_ALIASES: dict[str, int] = {
    "base": 0,
    "yaw": 0,
    "base yaw": 0,
    "j0": 0,
    "shoulder": 1,
    "shoulder pitch": 1,
    "j1": 1,
    "elbow": 2,
    "elbow pitch": 2,
    "j2": 2,
    "wrist roll": 3,
    "forearm": 3,
    "forearm roll": 3,
    "j3": 3,
    "wrist pitch": 4,
    "wrist": 4,
    "j4": 4,
    "roll": 5,
    "gripper roll": 5,
    "j5": 5,
}

# Named poses (degrees)
_NAMED_POSES: dict[str, list[float]] = {
    "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ready": [0.0, -45.0, 0.0, 90.0, 0.0, -45.0],
    "neutral": [0.0, -45.0, 0.0, 90.0, 0.0, -45.0],
    "up": [0.0, -60.0, 0.0, 0.0, 0.0, 0.0],
    "forward": [0.0, -30.0, -30.0, 60.0, 0.0, 0.0],
    "left": [-60.0, -30.0, 0.0, 60.0, 0.0, 0.0],
    "right": [60.0, -30.0, 0.0, 60.0, 0.0, 0.0],
    "tucked": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def parse_command(text: str) -> ParsedCommand:
    """Parse a natural language command into a structured action.

    Parameters
    ----------
    text : str
        The user's command string.

    Returns
    -------
    ParsedCommand with the interpreted action.
    """
    raw = text.strip()
    t = raw.lower().strip()

    if not t:
        return ParsedCommand(
            command_type=CommandType.UNKNOWN,
            description="Empty command",
            confidence=0.0,
        )

    # --- Stop / E-stop ---
    if t in ("stop", "halt", "freeze", "e-stop", "estop", "emergency stop", "abort"):
        return ParsedCommand(
            command_type=CommandType.STOP,
            action="stop",
            description=f"Stop: {raw}",
        )

    # --- Power commands ---
    if t in ("power on", "turn on", "start", "power up"):
        return ParsedCommand(
            command_type=CommandType.POWER,
            action="power-on",
            description="Power on the arm",
        )
    if t in ("power off", "turn off", "shutdown", "shut down", "power down"):
        return ParsedCommand(
            command_type=CommandType.POWER,
            action="power-off",
            description="Power off the arm",
        )
    if t in ("enable", "enable motors", "motors on", "arm on"):
        return ParsedCommand(
            command_type=CommandType.POWER,
            action="enable",
            description="Enable motors",
        )
    if t in ("disable", "disable motors", "motors off", "arm off"):
        return ParsedCommand(
            command_type=CommandType.POWER,
            action="disable",
            description="Disable motors",
        )

    # --- Task commands (wave, home, ready) ---
    if re.search(r"\bwave\b", t):
        waves = 3
        m = re.search(r"(\d+)\s*(?:times?|x)", t)
        if m:
            waves = min(int(m.group(1)), 10)
        speed = _extract_speed(t)
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="wave",
            speed=speed,
            description=f"Wave {waves} times",
            joints={},
            all_joints=[],
            gripper_mm=None,
        )

    if re.search(r"\b(?:go\s+)?home\b", t) and "homework" not in t:
        speed = _extract_speed(t)
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="home",
            speed=speed,
            description="Go to home position",
        )

    if re.search(r"\b(?:go\s+)?ready\b|get\s+ready|ready\s+position", t):
        speed = _extract_speed(t)
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="ready",
            speed=speed,
            description="Go to ready position",
        )

    if re.search(r"\breset\b", t):
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="home",
            speed=0.5,
            description="Reset to home position",
        )

    # --- Gripper commands ---
    gripper_cmd = _parse_gripper(t)
    if gripper_cmd is not None:
        return gripper_cmd

    # --- Set specific joint: "move joint 0 to 45" / "set j2 to -30" ---
    joint_cmd = _parse_set_joint(t)
    if joint_cmd is not None:
        return joint_cmd

    # --- Set all joints: "set joints to 0, -45, 0, 90, 0, -45" ---
    all_joints_cmd = _parse_set_all_joints(t)
    if all_joints_cmd is not None:
        return all_joints_cmd

    # --- Directional movement: "look up", "reach forward", "point left" ---
    direction_cmd = _parse_direction(t)
    if direction_cmd is not None:
        return direction_cmd

    # --- Named poses: "go to ready position" ---
    pose_cmd = _parse_named_pose(t)
    if pose_cmd is not None:
        return pose_cmd

    # --- Fallback: unknown ---
    return ParsedCommand(
        command_type=CommandType.UNKNOWN,
        description=f"Could not understand: {raw}",
        confidence=0.0,
    )


def _extract_speed(text: str) -> float:
    """Extract speed factor from text like 'slowly', 'fast', 'at 50% speed'."""
    if re.search(r"\b(?:slow(?:ly)?|careful(?:ly)?|gentle|gently)\b", text):
        return 0.3
    if re.search(r"\b(?:fast|quick(?:ly)?|rapid(?:ly)?)\b", text):
        return 1.0
    m = re.search(r"(\d+)\s*%?\s*speed", text)
    if m:
        return max(0.1, min(1.0, int(m.group(1)) / 100.0))
    return 0.6


def _parse_gripper(text: str) -> Optional[ParsedCommand]:
    """Parse gripper open/close/set commands."""
    # "open halfway", "half open", "half close" — check before full open/close
    if re.search(r"\b(?:half|halfway|middle|50%?)\b", text) and re.search(
        r"\b(?:gripper|claw|open|close|grip|hand)\b", text
    ):
        half = _get_pick_config().get("gripper", "max_mm") / 2.0
        return ParsedCommand(
            command_type=CommandType.SET_GRIPPER,
            gripper_mm=half,
            description=f"Set gripper to halfway ({half}mm)",
        )

    # "set gripper to 30", "gripper 30mm" — check before generic open/close
    m = re.search(r"(?:gripper|claw|grip)\s*(?:to|=|at)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?", text)
    if m:
        cfg = _get_pick_config()
        pos = max(
            cfg.get("gripper", "min_mm"), min(cfg.get("gripper", "max_mm"), float(m.group(1)))
        )
        return ParsedCommand(
            command_type=CommandType.SET_GRIPPER,
            gripper_mm=pos,
            description=f"Set gripper to {pos}mm",
        )

    # "open gripper", "gripper open", "open the claw"
    if (
        re.search(r"\b(?:open|release|let\s+go|drop)\b.*\b(?:gripper|claw|grip|hand)\b", text)
        or re.search(r"\b(?:gripper|claw|grip|hand)\b.*\b(?:open|release)\b", text)
        or text in ("open", "release", "let go", "drop")
    ):
        open_mm = _get_pick_config().get("gripper", "open_mm")
        return ParsedCommand(
            command_type=CommandType.SET_GRIPPER,
            gripper_mm=open_mm,
            description=f"Open gripper fully ({open_mm}mm)",
        )

    # "close gripper", "grip", "grab", "clamp"
    if (
        re.search(
            r"\b(?:close|clamp|squeeze|grip|grab|hold|clench)\b.*\b(?:gripper|claw|grip|hand)?\b",
            text,
        )
        or re.search(r"\b(?:gripper|claw|grip|hand)\b.*\b(?:close|shut)\b", text)
        or text in ("close", "grip", "grab", "clamp")
    ):
        # Check for partial close: "close to 30mm"
        m = re.search(r"(?:to|at)\s+(\d+(?:\.\d+)?)\s*(?:mm|millimeter)?", text)
        if m:
            cfg = _get_pick_config()
            pos = max(
                cfg.get("gripper", "min_mm"), min(cfg.get("gripper", "max_mm"), float(m.group(1)))
            )
            return ParsedCommand(
                command_type=CommandType.SET_GRIPPER,
                gripper_mm=pos,
                description=f"Set gripper to {pos}mm",
            )
        close_mm = _get_pick_config().get("gripper", "close_mm")
        return ParsedCommand(
            command_type=CommandType.SET_GRIPPER,
            gripper_mm=close_mm,
            description=f"Close gripper fully ({close_mm}mm)",
        )

    return None


def _parse_set_joint(text: str) -> Optional[ParsedCommand]:
    """Parse 'move joint X to Y' style commands."""
    # "set joint 2 to -30", "move j0 to 45", "rotate base to 90"
    m = re.search(
        r"(?:set|move|rotate|turn)\s+(?:joint\s*)?(\w+(?:\s+\w+)?)\s+(?:to|=|at)\s+(-?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)?",
        text,
    )
    if m:
        name = m.group(1).strip().lower()
        angle = float(m.group(2))
        joint_id = _resolve_joint(name)
        if joint_id is not None:
            return ParsedCommand(
                command_type=CommandType.SET_JOINT,
                joints={joint_id: angle},
                description=f"Set J{joint_id} to {angle}°",
            )

    # "j3 = 45" or "j3 45"
    m = re.search(r"\bj(\d)\s*[=:to ]\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        joint_id = int(m.group(1))
        if 0 <= joint_id <= 5:
            angle = float(m.group(2))
            return ParsedCommand(
                command_type=CommandType.SET_JOINT,
                joints={joint_id: angle},
                description=f"Set J{joint_id} to {angle}°",
            )

    return None


def _parse_set_all_joints(text: str) -> Optional[ParsedCommand]:
    """Parse 'set all joints to X,Y,Z,...' commands."""
    m = re.search(
        r"(?:set\s+)?(?:all\s+)?joints?\s+(?:to|=|:)\s*\[?\s*(-?\d+(?:\.\d+)?(?:\s*[,\s]\s*-?\d+(?:\.\d+)?){5})\s*\]?",
        text,
    )
    if m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        if len(nums) == 6:
            angles = [float(n) for n in nums]
            return ParsedCommand(
                command_type=CommandType.SET_ALL_JOINTS,
                all_joints=angles,
                description=f"Set all joints to {angles}",
            )
    return None


def _parse_direction(text: str) -> Optional[ParsedCommand]:
    """Parse directional commands like 'look up', 'reach forward', 'point left'."""
    # Look up / tilt up
    if re.search(r"\b(?:look|tilt|move|go|point|aim)\s+up\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=_NAMED_POSES["up"],
            speed=_extract_speed(text),
            description="Move arm upward",
            confidence=0.8,
        )

    # Look down / tilt down
    if re.search(r"\b(?:look|tilt|move|go|point|aim)\s+down\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=[0.0, -20.0, -60.0, 80.0, 60.0, 0.0],
            speed=_extract_speed(text),
            description="Move arm downward",
            confidence=0.8,
        )

    # Reach forward / extend
    if re.search(r"\b(?:reach|extend|stretch|go|move)\s+(?:out|forward|straight)\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=_NAMED_POSES["forward"],
            speed=_extract_speed(text),
            description="Reach forward",
            confidence=0.8,
        )

    # Point left / turn left
    if re.search(r"\b(?:point|turn|face|rotate|swing|move|go)\s+left\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=_NAMED_POSES["left"],
            speed=_extract_speed(text),
            description="Point left",
            confidence=0.8,
        )

    # Point right / turn right
    if re.search(r"\b(?:point|turn|face|rotate|swing|move|go)\s+right\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=_NAMED_POSES["right"],
            speed=_extract_speed(text),
            description="Point right",
            confidence=0.8,
        )

    # Curl / tuck / retract
    if re.search(r"\b(?:curl|tuck|retract|fold|collapse)\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=_NAMED_POSES["tucked"],
            speed=_extract_speed(text),
            description="Tuck arm in",
            confidence=0.8,
        )

    # Raise / lift
    if re.search(r"\b(?:raise|lift)\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=[0.0, -70.0, 0.0, 45.0, 0.0, 0.0],
            speed=_extract_speed(text),
            description="Raise the arm",
            confidence=0.8,
        )

    # Lower / bring down
    if re.search(r"\b(?:lower|bring\s+down)\b", text):
        return ParsedCommand(
            command_type=CommandType.SET_ALL_JOINTS,
            all_joints=[0.0, -10.0, -40.0, 50.0, 40.0, 0.0],
            speed=_extract_speed(text),
            description="Lower the arm",
            confidence=0.8,
        )

    # Nod (yes gesture)
    if re.search(r"\bnod\b", text):
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="nod",
            speed=_extract_speed(text),
            description="Nod the arm",
        )

    # Shake (no gesture)
    if re.search(r"\bshake\b", text):
        return ParsedCommand(
            command_type=CommandType.TASK,
            action="shake",
            speed=_extract_speed(text),
            description="Shake the arm side to side",
        )

    return None


def _parse_named_pose(text: str) -> Optional[ParsedCommand]:
    """Parse named pose commands like 'go to ready'."""
    for name, angles in _NAMED_POSES.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            return ParsedCommand(
                command_type=CommandType.SET_ALL_JOINTS,
                all_joints=angles,
                speed=_extract_speed(text),
                description=f"Move to {name} pose",
                confidence=0.7,
            )
    return None


def _resolve_joint(name: str) -> Optional[int]:
    """Resolve a joint name/alias to a joint ID (0-5)."""
    name = name.strip().lower()
    # Direct numeric: "0", "3", etc.
    if name.isdigit() and 0 <= int(name) <= 5:
        return int(name)
    # "j0" .. "j5"
    if len(name) == 2 and name[0] == "j" and name[1].isdigit():
        idx = int(name[1])
        if 0 <= idx <= 5:
            return idx
    # Aliases
    if name in _JOINT_ALIASES:
        return _JOINT_ALIASES[name]
    return None
