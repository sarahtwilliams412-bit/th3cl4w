"""
Gravity Compensation for Unitree D1 Arm

Computes gravity torques using Recursive Newton-Euler Algorithm (RNEA)
with the D1's DH parameters and approximate link masses. This feedforward
compensation reduces the corrective effort the PD servos must produce,
eliminating oscillation and reducing servo heat buildup.

Even in position-control mode (no direct torque commands), gravity
compensation helps by computing position offsets that pre-compensate
for gravitational deflection, reducing tracking error amplitude.

The D1 weighs 2.37 kg with a 670 mm reach, meaning shoulder and elbow
servos continuously fight significant gravitational torques without
compensation.

References:
    - Caltech AMBER Lab D1 report: gravity compensation is non-negotiable
    - SO-ARM100 project: Pinocchio-based gravity comp reference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from shared.kinematics.kinematics import D1Kinematics, _D1_DH, _dh_transform

logger = logging.getLogger(__name__)

# Gravity vector (pointing down in base frame)
GRAVITY = np.array([0.0, 0.0, -9.81])

# ---------------------------------------------------------------------------
# D1 Link physical parameters (approximate)
# ---------------------------------------------------------------------------
# Masses (kg) and center of mass offsets (m) along link z-axis
# Total arm mass ~2.37 kg distributed across 7 links + gripper


@dataclass(frozen=True)
class LinkDynamics:
    """Physical properties of a single link."""

    mass: float  # kg
    com_offset: np.ndarray  # (3,) center of mass in link frame
    inertia_diag: np.ndarray  # (3,) diagonal inertia approximation


# Approximate D1 link dynamics (tuned from CAD model estimates)
_D1_LINK_DYNAMICS: list[LinkDynamics] = [
    LinkDynamics(  # J0: base yaw
        mass=0.35,
        com_offset=np.array([0.0, 0.0, 0.06]),
        inertia_diag=np.array([0.001, 0.001, 0.0005]),
    ),
    LinkDynamics(  # J1: shoulder pitch
        mass=0.45,
        com_offset=np.array([0.0, 0.0, 0.0]),
        inertia_diag=np.array([0.002, 0.002, 0.001]),
    ),
    LinkDynamics(  # J2: shoulder roll / upper arm
        mass=0.40,
        com_offset=np.array([0.0, 0.0, 0.10]),
        inertia_diag=np.array([0.003, 0.003, 0.001]),
    ),
    LinkDynamics(  # J3: elbow pitch
        mass=0.30,
        com_offset=np.array([0.0, 0.0, 0.0]),
        inertia_diag=np.array([0.001, 0.001, 0.0005]),
    ),
    LinkDynamics(  # J4: forearm
        mass=0.35,
        com_offset=np.array([0.0, 0.0, 0.10]),
        inertia_diag=np.array([0.002, 0.002, 0.0005]),
    ),
    LinkDynamics(  # J5: wrist
        mass=0.22,
        com_offset=np.array([0.0, 0.0, 0.0]),
        inertia_diag=np.array([0.0005, 0.0005, 0.0003]),
    ),
    LinkDynamics(  # J6: wrist roll + gripper
        mass=0.30,
        com_offset=np.array([0.0, 0.0, 0.05]),
        inertia_diag=np.array([0.0008, 0.0008, 0.0003]),
    ),
]


class GravityCompensator:
    """Computes gravity compensation torques for the D1 arm.

    Uses a simplified RNEA (Recursive Newton-Euler) approach that
    considers only gravitational forces (no inertial or Coriolis terms).
    This is sufficient for quasi-static and slow movements.

    For the D1 in position-control mode, the computed torques can be
    converted to position offsets via the servo stiffness, allowing
    gravity pre-compensation even without torque control.
    """

    def __init__(
        self,
        kinematics: D1Kinematics | None = None,
        link_dynamics: list[LinkDynamics] | None = None,
        gravity: np.ndarray | None = None,
    ):
        self.kinematics = kinematics or D1Kinematics()
        self.link_dynamics = link_dynamics or list(_D1_LINK_DYNAMICS)
        self.gravity = gravity if gravity is not None else GRAVITY.copy()
        self.n_joints = len(self.link_dynamics)

        if len(self.link_dynamics) != self.kinematics.n_joints:
            raise ValueError(
                f"Link dynamics ({len(self.link_dynamics)}) must match "
                f"kinematics joints ({self.kinematics.n_joints})"
            )

    def compute_gravity_torques(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques for given joint configuration.

        Uses the Jacobian transpose method:
            tau_g = sum_i( J_i^T * F_g_i )
        where J_i is the Jacobian at link i's center of mass and
        F_g_i = [m_i * g, 0] is the gravitational wrench.

        Parameters
        ----------
        joint_angles : (n_joints,) joint angles in radians

        Returns
        -------
        (n_joints,) gravity compensation torques in Nm.
        Positive values counteract gravity (add these to commanded torques).
        """
        q = np.asarray(joint_angles, dtype=np.float64)
        n = self.n_joints
        dh_params = self.kinematics.dh_params

        # Compute all frame transforms
        T_base = [np.eye(4)]
        T_accum = np.eye(4)
        for i in range(n):
            T_accum = T_accum @ _dh_transform(dh_params[i], q[i])
            T_base.append(T_accum.copy())

        tau_g = np.zeros(n)

        for link_idx in range(n):
            ld = self.link_dynamics[link_idx]
            if ld.mass < 1e-9:
                continue

            # CoM position in world frame
            T_link = T_base[link_idx + 1]
            R_link = T_link[:3, :3]
            p_link = T_link[:3, 3]

            # CoM in world = link origin + R * com_offset
            p_com = p_link + R_link @ ld.com_offset

            # Gravitational force at CoM (in world frame)
            f_gravity = ld.mass * self.gravity  # (3,)

            # Accumulate torque contribution to each joint
            for j in range(link_idx + 1):
                # Joint j axis in world frame
                z_j = T_base[j][:3, 2]
                p_j = T_base[j][:3, 3]

                # Moment arm from joint j to CoM
                r = p_com - p_j

                # Torque contribution: z_j . (r x F_g)
                tau_g[j] += z_j @ np.cross(r, f_gravity)

        return tau_g

    def compute_position_offset(
        self,
        joint_angles: np.ndarray,
        servo_stiffness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute position offsets that pre-compensate gravity deflection.

        For position-controlled servos, gravity causes a steady-state
        position error proportional to tau_g / Kp. By adding this offset
        to the commanded position, the servo doesn't need to generate
        the corrective torque reactively.

        Parameters
        ----------
        joint_angles : current joint configuration (radians)
        servo_stiffness : per-joint Kp gains (Nm/rad). If None, uses
            conservative defaults.

        Returns
        -------
        (n_joints,) position offsets in radians. Add these to the
        commanded position to pre-compensate gravity.
        """
        if servo_stiffness is None:
            # Conservative default stiffness values for D1 servos
            servo_stiffness = np.array([60.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0])

        tau_g = self.compute_gravity_torques(joint_angles)

        # Position offset = gravity_torque / stiffness
        # Clamp stiffness to avoid division by zero
        k = np.maximum(servo_stiffness, 1.0)
        offset = tau_g / k

        # Clamp offsets to reasonable range (max 5 degrees)
        max_offset = np.radians(5.0)
        offset = np.clip(offset, -max_offset, max_offset)

        return offset

    def total_gravity_load(self, joint_angles: np.ndarray) -> float:
        """Compute total gravitational load magnitude (Nm).

        Useful for thermal monitoring: high sustained gravity load
        means the servos are working hard and may overheat.
        """
        tau_g = self.compute_gravity_torques(joint_angles)
        return float(np.sqrt(np.sum(tau_g**2)))


class ThermalMonitor:
    """Monitors servo thermal load from sustained gravity compensation.

    The D1 servos are documented as "burning hot" during extended operation.
    This monitor tracks the cumulative thermal load and recommends velocity
    reduction when servos are under sustained stress.

    Thermal model: temperature rises proportional to tau^2 * dt (I^2*R heating)
    and decays exponentially toward ambient.
    """

    def __init__(
        self,
        n_joints: int,
        thermal_capacity: float = 100.0,
        dissipation_rate: float = 0.02,
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
    ):
        self.n_joints = n_joints
        self.thermal_capacity = thermal_capacity
        self.dissipation_rate = dissipation_rate
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Thermal state per joint (normalized temperature units)
        self._thermal_load = np.zeros(n_joints)
        self._ambient = 25.0  # assumed ambient temperature

    def update(self, torques: np.ndarray, dt: float) -> None:
        """Update thermal model with current torque loads."""
        torques = np.asarray(torques, dtype=np.float64)

        # Heat generation proportional to torque squared
        heat_in = (torques**2) * dt / self.thermal_capacity

        # Heat dissipation (exponential decay toward ambient)
        heat_out = self.dissipation_rate * (self._thermal_load - self._ambient) * dt

        self._thermal_load = self._thermal_load + heat_in - heat_out
        self._thermal_load = np.maximum(self._thermal_load, self._ambient)

    def speed_reduction_factor(self) -> float:
        """Compute recommended speed reduction factor [0.0, 1.0].

        Returns 1.0 (no reduction) when cool, decreases as temperature
        approaches critical threshold.
        """
        max_temp = float(np.max(self._thermal_load))

        if max_temp < self.warning_threshold:
            return 1.0
        elif max_temp >= self.critical_threshold:
            return 0.2  # severe reduction
        else:
            # Linear interpolation between warning and critical
            ratio = (max_temp - self.warning_threshold) / (
                self.critical_threshold - self.warning_threshold
            )
            return max(0.2, 1.0 - 0.8 * ratio)

    @property
    def thermal_loads(self) -> np.ndarray:
        """Current thermal load per joint."""
        return self._thermal_load.copy()

    @property
    def max_temperature(self) -> float:
        """Maximum joint temperature."""
        return float(np.max(self._thermal_load))

    @property
    def is_warning(self) -> bool:
        return self.max_temperature >= self.warning_threshold

    @property
    def is_critical(self) -> bool:
        return self.max_temperature >= self.critical_threshold
