"""
Impedance/Admittance Controller for Natural Contact Behavior

Implements variable-stiffness control that mimics human arm compliance.
In admittance mode (suitable for position-controlled servos like the D1),
external forces are converted into position modifications, allowing the
arm to yield naturally upon contact.

The controller supports:
- Variable stiffness per joint (stiff for precision, compliant for contact)
- Damping to prevent oscillation
- Task-space impedance (Cartesian stiffness/damping)
- Stiffness profiles that switch based on task phase

References:
    - Variable stiffness mimics human arm compliance
    - Admittance control for position-controlled robots
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ComplianceMode(Enum):
    """Pre-defined compliance profiles."""

    STIFF = "stiff"  # High stiffness for precision tasks
    MEDIUM = "medium"  # Balanced for general manipulation
    COMPLIANT = "compliant"  # Low stiffness for safe interaction
    VERY_COMPLIANT = "very_compliant"  # Very low stiffness for handover


# Default stiffness/damping profiles per mode (7 joints)
_STIFFNESS_PROFILES: dict[ComplianceMode, np.ndarray] = {
    ComplianceMode.STIFF: np.array([80.0, 80.0, 60.0, 50.0, 40.0, 30.0, 25.0]),
    ComplianceMode.MEDIUM: np.array([40.0, 40.0, 30.0, 25.0, 20.0, 15.0, 12.0]),
    ComplianceMode.COMPLIANT: np.array([15.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0]),
    ComplianceMode.VERY_COMPLIANT: np.array([5.0, 5.0, 4.0, 3.0, 3.0, 2.0, 2.0]),
}

_DAMPING_PROFILES: dict[ComplianceMode, np.ndarray] = {
    ComplianceMode.STIFF: np.array([8.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5]),
    ComplianceMode.MEDIUM: np.array([5.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5]),
    ComplianceMode.COMPLIANT: np.array([3.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0]),
    ComplianceMode.VERY_COMPLIANT: np.array([1.5, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]),
}


@dataclass
class ImpedanceParams:
    """Impedance control parameters per joint.

    The impedance model is:
        F = K * (x_des - x) + D * (v_des - v)

    In admittance mode (position-controlled servo), this becomes:
        x_cmd = x_des + F_ext / K  (simplified steady-state)
    """

    stiffness: np.ndarray  # K: Nm/rad (or N/m in Cartesian)
    damping: np.ndarray  # D: Nm*s/rad (or N*s/m in Cartesian)

    def __post_init__(self):
        self.stiffness = np.asarray(self.stiffness, dtype=np.float64)
        self.damping = np.asarray(self.damping, dtype=np.float64)

    @classmethod
    def from_mode(cls, mode: ComplianceMode) -> ImpedanceParams:
        """Create parameters from a pre-defined compliance mode."""
        return cls(
            stiffness=_STIFFNESS_PROFILES[mode].copy(),
            damping=_DAMPING_PROFILES[mode].copy(),
        )

    @property
    def damping_ratio(self) -> np.ndarray:
        """Critical damping ratio: zeta = D / (2 * sqrt(K * m)).

        For unit mass, zeta = D / (2 * sqrt(K)).
        zeta = 1.0 is critically damped (no overshoot).
        """
        return self.damping / (2.0 * np.sqrt(np.maximum(self.stiffness, 1e-6)))


class AdmittanceController:
    """Joint-space admittance controller for position-controlled servos.

    Converts measured external torques into position modifications,
    allowing the arm to yield naturally upon contact while maintaining
    a desired trajectory.

    For the D1 arm which primarily operates in position mode, admittance
    control is preferred over impedance control because:
    1. It works with position commands (no torque mode needed)
    2. External torques are estimated from servo current feedback
    3. The position modification naturally respects joint limits

    Update loop (call at control rate, e.g., 100 Hz):
        1. Compute external torque estimate: tau_ext = tau_measured - tau_gravity
        2. Compute admittance displacement: delta_q = tau_ext / K
        3. Apply damped integration: q_comply = LPF(delta_q)
        4. Output: q_cmd = q_desired + q_comply
    """

    def __init__(
        self,
        n_joints: int,
        params: ImpedanceParams | None = None,
        mode: ComplianceMode = ComplianceMode.MEDIUM,
        dt: float = 0.01,
        max_compliance_rad: float = 0.15,  # ~8.6 degrees max compliance
    ):
        self.n_joints = n_joints
        self.dt = dt
        self.max_compliance = max_compliance_rad

        if params is not None:
            self.params = params
        else:
            self.params = ImpedanceParams.from_mode(mode)

        # Compliance state
        self._compliance_offset = np.zeros(n_joints)
        self._compliance_velocity = np.zeros(n_joints)

    def set_mode(self, mode: ComplianceMode) -> None:
        """Switch to a pre-defined compliance profile."""
        self.params = ImpedanceParams.from_mode(mode)
        logger.info("Compliance mode set to %s", mode.value)

    def set_stiffness(self, stiffness: np.ndarray) -> None:
        """Set custom stiffness values."""
        self.params.stiffness = np.asarray(stiffness, dtype=np.float64)

    def set_damping(self, damping: np.ndarray) -> None:
        """Set custom damping values."""
        self.params.damping = np.asarray(damping, dtype=np.float64)

    def reset(self) -> None:
        """Reset compliance state to zero (no displacement)."""
        self._compliance_offset = np.zeros(self.n_joints)
        self._compliance_velocity = np.zeros(self.n_joints)

    def compute_compliance(
        self,
        desired_positions: np.ndarray,
        external_torques: np.ndarray,
    ) -> np.ndarray:
        """Compute compliant position command given external torques.

        Parameters
        ----------
        desired_positions : trajectory-planned target positions (rad)
        external_torques : estimated external torques (Nm)
            Typically: tau_measured - tau_gravity - tau_inertial

        Returns
        -------
        Compliant position command (rad) = desired + compliance offset
        """
        q_des = np.asarray(desired_positions, dtype=np.float64)
        tau_ext = np.asarray(external_torques, dtype=np.float64)

        K = self.params.stiffness
        D = self.params.damping

        # Spring-damper model:
        # M * ddq_c + D * dq_c + K * q_c = tau_ext
        # For unit mass (M=1):
        # ddq_c = tau_ext - D * dq_c - K * q_c
        acc = tau_ext - D * self._compliance_velocity - K * self._compliance_offset

        # Integrate
        self._compliance_velocity += acc * self.dt
        self._compliance_offset += self._compliance_velocity * self.dt

        # Clamp compliance offset to safety limits
        self._compliance_offset = np.clip(
            self._compliance_offset,
            -self.max_compliance,
            self.max_compliance,
        )

        return q_des + self._compliance_offset

    def estimate_external_torque(
        self,
        measured_torques: np.ndarray,
        gravity_torques: np.ndarray,
    ) -> np.ndarray:
        """Estimate external torques from measurements.

        tau_ext = tau_measured - tau_gravity

        For the D1, measured torques come from servo current feedback.
        Gravity torques come from the GravityCompensator.
        """
        return np.asarray(measured_torques) - np.asarray(gravity_torques)

    @property
    def compliance_offset(self) -> np.ndarray:
        """Current compliance displacement (rad)."""
        return self._compliance_offset.copy()

    @property
    def is_in_contact(self) -> bool:
        """Rough estimate of whether the arm is in contact with something."""
        return bool(np.max(np.abs(self._compliance_offset)) > 0.01)  # ~0.6 degrees


class StiffnessScheduler:
    """Schedules stiffness changes based on task phase.

    Provides smooth transitions between compliance modes to avoid
    sudden stiffness changes that could cause jerky motion.
    """

    def __init__(
        self,
        n_joints: int,
        transition_time: float = 0.5,
        dt: float = 0.01,
    ):
        self.n_joints = n_joints
        self.transition_time = transition_time
        self.dt = dt

        self._current_stiffness = _STIFFNESS_PROFILES[ComplianceMode.MEDIUM].copy()
        self._current_damping = _DAMPING_PROFILES[ComplianceMode.MEDIUM].copy()
        self._start_stiffness = self._current_stiffness.copy()
        self._start_damping = self._current_damping.copy()
        self._target_stiffness = self._current_stiffness.copy()
        self._target_damping = self._current_damping.copy()
        self._transition_progress = 1.0  # 1.0 = complete

    def set_target_mode(self, mode: ComplianceMode) -> None:
        """Start transition to a new compliance mode."""
        self._start_stiffness = self._current_stiffness.copy()
        self._start_damping = self._current_damping.copy()
        self._target_stiffness = _STIFFNESS_PROFILES[mode].copy()
        self._target_damping = _DAMPING_PROFILES[mode].copy()
        self._transition_progress = 0.0

    def set_target_params(self, stiffness: np.ndarray, damping: np.ndarray) -> None:
        """Start transition to custom parameters."""
        self._start_stiffness = self._current_stiffness.copy()
        self._start_damping = self._current_damping.copy()
        self._target_stiffness = np.asarray(stiffness, dtype=np.float64)
        self._target_damping = np.asarray(damping, dtype=np.float64)
        self._transition_progress = 0.0

    def update(self) -> ImpedanceParams:
        """Advance transition by one timestep. Returns current parameters."""
        if self._transition_progress < 1.0:
            step = self.dt / max(self.transition_time, 1e-6)
            self._transition_progress = min(1.0, self._transition_progress + step)

            # Smooth interpolation using minimum-jerk profile
            s = self._transition_progress
            s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5

            # Interpolate from saved start to target (not from drifting current)
            self._current_stiffness = self._start_stiffness + (self._target_stiffness - self._start_stiffness) * s_smooth
            self._current_damping = self._start_damping + (self._target_damping - self._start_damping) * s_smooth

            if self._transition_progress >= 1.0:
                self._current_stiffness = self._target_stiffness.copy()
                self._current_damping = self._target_damping.copy()

        return ImpedanceParams(
            stiffness=self._current_stiffness.copy(),
            damping=self._current_damping.copy(),
        )

    @property
    def is_transitioning(self) -> bool:
        return self._transition_progress < 1.0
