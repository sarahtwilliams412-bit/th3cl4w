"""
Command Smoothing Filters for Layer 0 (Real-Time Servo)

Provides dual-EMA (Exponential Moving Average) filtering and jerk-limited
rate limiting that sit between trajectory generation and the hardware
interface. These filters ensure that every command reaching the actuators
is smooth, even if upstream layers produce discontinuities from replanning,
IK jumps, or blending artifacts.

The dual-EMA filter applies two cascaded exponential smoothing stages,
producing a critically-damped second-order response that eliminates
discrete step discontinuities without excessive phase lag.

The jerk limiter enforces maximum velocity, acceleration, and jerk bounds
per joint, preventing excitation of structural/control resonances.

References:
    - Caltech AMBER Lab: 100Hz+ control loop eliminates shakiness
    - One-Euro Filter concept for adaptive noise filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EMAState:
    """State for a single EMA filter stage."""

    value: np.ndarray
    initialized: bool = False


class DualEMAFilter:
    """Dual-stage Exponential Moving Average filter for command smoothing.

    Two cascaded EMA stages produce a critically-damped response:
        Stage 1: y1[k] = alpha * x[k] + (1 - alpha) * y1[k-1]
        Stage 2: y2[k] = alpha * y1[k] + (1 - alpha) * y2[k-1]

    The output y2 is smoother than a single EMA with the same alpha,
    with better step response characteristics (no overshoot, smooth
    acceleration profile).

    Parameters
    ----------
    n_joints : number of joint channels
    alpha : smoothing factor in (0, 1]. Higher = less smoothing.
        - 0.1 = very smooth, ~200ms settling (good for teleoperation)
        - 0.3 = moderate, ~100ms settling (good for autonomous motion)
        - 0.5 = light, ~50ms settling (good for fast tracking)
        - 1.0 = no filtering (passthrough)
    """

    def __init__(self, n_joints: int, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.n_joints = n_joints
        self.alpha = alpha
        self._stage1 = EMAState(value=np.zeros(n_joints))
        self._stage2 = EMAState(value=np.zeros(n_joints))

    def reset(self, initial_value: np.ndarray) -> None:
        """Reset filter state to a known position (e.g., from arm feedback)."""
        val = np.asarray(initial_value, dtype=np.float64)
        self._stage1 = EMAState(value=val.copy(), initialized=True)
        self._stage2 = EMAState(value=val.copy(), initialized=True)

    def filter(self, command: np.ndarray) -> np.ndarray:
        """Apply dual-EMA filtering to a command vector.

        Returns the smoothed command. First call initializes the filter
        to the input value (no lag on startup).
        """
        x = np.asarray(command, dtype=np.float64)
        a = self.alpha

        if not self._stage1.initialized:
            self._stage1 = EMAState(value=x.copy(), initialized=True)
            self._stage2 = EMAState(value=x.copy(), initialized=True)
            return x.copy()

        # Stage 1
        self._stage1.value = a * x + (1.0 - a) * self._stage1.value
        # Stage 2
        self._stage2.value = a * self._stage1.value + (1.0 - a) * self._stage2.value

        return self._stage2.value.copy()

    @property
    def current_value(self) -> np.ndarray:
        """Current filtered output."""
        return self._stage2.value.copy()


class OneEuroFilter:
    """One-Euro Filter for adaptive noise filtering on sensor readings.

    Adapts cutoff frequency based on signal speed: slow movements get
    heavy filtering (noise reduction), fast movements get light filtering
    (responsiveness). This prevents "noise amplification" where motors
    buzz reacting to sensor jitter.

    Parameters
    ----------
    n_joints : number of channels
    min_cutoff : minimum cutoff frequency (Hz), controls jitter at rest
    beta : speed coefficient, controls lag during fast movements
    d_cutoff : cutoff frequency for derivative computation (Hz)
    rate : sampling rate (Hz)
    """

    def __init__(
        self,
        n_joints: int,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        rate: float = 100.0,
    ):
        self.n_joints = n_joints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.rate = rate

        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray = np.zeros(n_joints)
        self._initialized = False

    def _alpha(self, cutoff: float | np.ndarray) -> float | np.ndarray:
        """Compute EMA alpha from cutoff frequency (scalar or array)."""
        te = 1.0 / self.rate
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self, initial_value: np.ndarray) -> None:
        """Reset filter to known state."""
        self._x_prev = np.asarray(initial_value, dtype=np.float64).copy()
        self._dx_prev = np.zeros(self.n_joints)
        self._initialized = True

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter a sensor reading. Returns filtered value."""
        x = np.asarray(x, dtype=np.float64)

        if not self._initialized or self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(self.n_joints)
            self._initialized = True
            return x.copy()

        # Compute derivative
        dx = (x - self._x_prev) * self.rate

        # Filter derivative
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Compute adaptive cutoff per joint
        cutoffs = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filter signal with adaptive cutoff (vectorized)
        a = self._alpha(cutoffs)
        result = a * x + (1.0 - a) * self._x_prev

        self._x_prev = result.copy()
        self._dx_prev = dx_hat.copy()

        return result


class JerkLimiter:
    """Jerk-limited rate limiter for Layer 0 hardware protection.

    Enforces maximum velocity, acceleration, and jerk bounds per joint.
    Even if upstream trajectory generation is perfect, this acts as a
    final safety net ensuring the hardware never receives a command that
    would excite mechanical resonance.

    The limiter uses a cascaded approach:
    1. Compute desired velocity from position error
    2. Limit velocity to max_velocity
    3. Compute desired acceleration from velocity error
    4. Limit acceleration to max_acceleration
    5. Limit jerk (rate of acceleration change) to max_jerk

    Parameters
    ----------
    n_joints : number of joints
    max_velocity : per-joint velocity limits (rad/s)
    max_acceleration : per-joint acceleration limits (rad/s^2)
    max_jerk : per-joint jerk limits (rad/s^3)
    dt : control loop period (seconds)
    """

    def __init__(
        self,
        n_joints: int,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        dt: float = 0.01,
    ):
        self.n_joints = n_joints
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64)
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64)
        self.max_jerk = np.asarray(max_jerk, dtype=np.float64)
        self.dt = dt

        # Vectorized state arrays instead of per-joint dataclasses
        self._positions = np.zeros(n_joints)
        self._velocities = np.zeros(n_joints)
        self._accelerations = np.zeros(n_joints)
        self._initialized = False

    def reset(self, positions: np.ndarray) -> None:
        """Reset limiter state to current positions with zero velocity/acceleration."""
        self._positions = np.asarray(positions, dtype=np.float64).copy()
        self._velocities = np.zeros(self.n_joints)
        self._accelerations = np.zeros(self.n_joints)
        self._initialized = True

    def limit(self, input_positions: np.ndarray) -> np.ndarray:
        """Apply jerk-limited rate limiting to an input position signal.

        Limits the velocity, acceleration, and jerk of the output
        signal relative to the input. This is a pure signal filter:
        it clips the derivatives of the input signal to enforce
        hardware bounds.

        The input is typically the output of the EMA filter, which
        is already smoothly approaching the target. The jerk limiter
        catches any residual discontinuities.

        For a constant input, the output converges via velocity-limited
        tracking. For a smoothly changing input (EMA output), the
        output closely follows with bounded derivatives.
        """
        inp = np.asarray(input_positions, dtype=np.float64)

        if not self._initialized:
            self.reset(inp)
            return inp.copy()

        dt = self.dt
        if dt <= 0:
            return self._positions.copy()

        inv_dt = 1.0 / dt

        # Vectorized: compute implied velocity of input signal
        input_vel = (inp - self._positions) * inv_dt

        # Clamp velocity
        clamped_vel = np.clip(input_vel, -self.max_velocity, self.max_velocity)

        # Clamp acceleration (rate of velocity change)
        desired_acc = (clamped_vel - self._velocities) * inv_dt
        clamped_acc = np.clip(desired_acc, -self.max_acceleration, self.max_acceleration)

        # Clamp jerk (rate of acceleration change)
        desired_jerk = (clamped_acc - self._accelerations) * inv_dt
        clamped_jerk = np.clip(desired_jerk, -self.max_jerk, self.max_jerk)

        # Apply cascaded limits: jerk → acc → vel → pos
        self._accelerations += clamped_jerk * dt
        self._velocities += self._accelerations * dt
        np.clip(self._velocities, -self.max_velocity, self.max_velocity, out=self._velocities)
        self._positions += self._velocities * dt

        return self._positions.copy()

    @property
    def current_positions(self) -> np.ndarray:
        """Current limited positions."""
        return self._positions.copy()

    @property
    def current_velocities(self) -> np.ndarray:
        """Current velocities."""
        return self._velocities.copy()

    @property
    def current_accelerations(self) -> np.ndarray:
        """Current accelerations."""
        return self._accelerations.copy()


class SmoothCommandPipeline:
    """Complete Layer 0 command smoothing pipeline.

    Chains: Input → OneEuroFilter (sensor noise) → DualEMA (command smooth)
            → Output

    The dual-EMA filter eliminates discrete step discontinuities with
    a critically-damped second-order response. The One-Euro filter
    adaptively removes sensor noise from feedback readings.

    Jerk limiting is provided upstream by the minimum-jerk trajectory
    generator, which produces inherently jerk-minimized motion profiles.
    The JerkLimiter class is available separately for advanced use cases
    requiring per-joint derivative bounds.
    """

    def __init__(
        self,
        n_joints: int,
        ema_alpha: float = 0.3,
        max_velocity: np.ndarray | None = None,
        max_acceleration: np.ndarray | None = None,
        max_jerk: np.ndarray | None = None,
        dt: float = 0.01,
        enable_sensor_filter: bool = True,
        sensor_min_cutoff: float = 1.0,
        sensor_beta: float = 0.007,
    ):
        self.n_joints = n_joints
        self.dt = dt

        self._ema = DualEMAFilter(n_joints, alpha=ema_alpha)
        self._sensor_filter: OneEuroFilter | None = None
        if enable_sensor_filter:
            self._sensor_filter = OneEuroFilter(
                n_joints,
                min_cutoff=sensor_min_cutoff,
                beta=sensor_beta,
                rate=1.0 / dt,
            )

        self._initialized = False

    def reset(self, positions: np.ndarray) -> None:
        """Initialize pipeline from current arm positions."""
        pos = np.asarray(positions, dtype=np.float64)
        self._ema.reset(pos)
        if self._sensor_filter is not None:
            self._sensor_filter.reset(pos)
        self._initialized = True

    def filter_sensor(self, reading: np.ndarray) -> np.ndarray:
        """Filter a sensor/feedback reading (optional, for feedback smoothing)."""
        if self._sensor_filter is not None:
            return self._sensor_filter.filter(reading)
        return np.asarray(reading, dtype=np.float64)

    def smooth_command(self, command: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a position command.

        Returns the smoothed command ready for the actuators.
        The dual-EMA provides a critically-damped response that
        eliminates step discontinuities without overshoot.
        """
        cmd = np.asarray(command, dtype=np.float64)

        if not self._initialized:
            self.reset(cmd)
            return cmd.copy()

        # Dual-EMA smoothing (removes step discontinuities)
        smoothed = self._ema.filter(cmd)

        return smoothed

    @property
    def is_initialized(self) -> bool:
        return self._initialized
