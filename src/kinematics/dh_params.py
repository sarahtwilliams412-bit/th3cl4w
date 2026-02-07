"""
Denavit-Hartenberg parameters for the Unitree D1 arm.

The D1 has 6 arm joints + 1 gripper.  The DH parameters below describe
the 6-DOF arm chain (the gripper is a prismatic end-effector, not part
of the DH chain).

These are approximate values based on the D1's published dimensions
(550mm reach, ~2.2kg).  They should be refined against the actual
URDF or CAD model for your unit.

Convention: Modified DH (Craig convention)
  - a      : link length (m)
  - alpha  : link twist (rad)
  - d      : link offset (m)
  - theta  : joint angle offset (rad) — added to the variable joint angle
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DHParam:
    """A single row of the DH parameter table."""
    a: float      # link length (m)
    alpha: float  # link twist (rad)
    d: float      # link offset (m)
    theta: float  # joint angle offset (rad)


# Unitree D1 arm DH parameters (6 joints, modified DH)
# These are estimates — verify against your URDF before trusting for precision work.
D1_DH_PARAMS: List[DHParam] = [
    # Joint 0: base rotation
    DHParam(a=0.0,    alpha=0.0,           d=0.1215,  theta=0.0),
    # Joint 1: shoulder pitch
    DHParam(a=0.0,    alpha=-math.pi / 2,  d=0.0,     theta=0.0),
    # Joint 2: shoulder roll
    DHParam(a=0.2130, alpha=0.0,           d=0.0,     theta=0.0),
    # Joint 3: elbow
    DHParam(a=0.0,    alpha=-math.pi / 2,  d=0.2130,  theta=0.0),
    # Joint 4: wrist pitch
    DHParam(a=0.0,    alpha=math.pi / 2,   d=0.0,     theta=0.0),
    # Joint 5: wrist roll
    DHParam(a=0.0,    alpha=-math.pi / 2,  d=0.0870,  theta=0.0),
]

NUM_ARM_JOINTS = len(D1_DH_PARAMS)
