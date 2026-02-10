"""
Minimum-Jerk Trajectory Generation with Fitts' Law Duration Estimation

Implements the gold-standard trajectory profile for human-like robotic motion.
The minimum-jerk polynomial s(t) = 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5 produces
bell-shaped velocity profiles that match human reaching movements.

Fitts' Law (MT = a + b*log2(2D/W)) computes movement durations that match
human expectations based on distance and target precision.

References:
    - Flash & Hogan (1985): "The coordination of arm movements"
    - Fitts (1954): "The information capacity of the human motor system"
    - Caltech AMBER Lab D1 report (IEEE 2024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinJerkProfile:
    """Minimum-jerk time-scaling profile evaluated at a single instant.

    All values are normalized (dimensionless) except where noted.
    Multiply by displacement to get joint-space values.
    """

    s: float  # position scalar [0, 1]
    ds: float  # velocity scalar (1/T units)
    dds: float  # acceleration scalar (1/T^2 units)
    ddds: float  # jerk scalar (1/T^3 units)


def minimum_jerk_scalar(tau: float) -> MinJerkProfile:
    """Evaluate minimum-jerk polynomial at normalized time tau in [0, 1].

    The quintic polynomial:
        s(tau)  = 10*tau^3 - 15*tau^4 + 6*tau^5
        s'(tau) = 30*tau^2 - 60*tau^3 + 30*tau^4
        s''(tau)= 60*tau   - 180*tau^2 + 120*tau^3
        s'''    = 60       - 360*tau   + 360*tau^2

    Properties:
        s(0)=0, s(1)=1
        s'(0)=0, s'(1)=0  (zero velocity at endpoints)
        s''(0)=0, s''(1)=0 (zero acceleration at endpoints)

    Returns MinJerkProfile with all derivatives.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    ds = 30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4
    dds = 60.0 * tau - 180.0 * tau2 + 120.0 * tau3
    ddds = 60.0 - 360.0 * tau + 360.0 * tau2

    return MinJerkProfile(s=s, ds=ds, dds=dds, ddds=ddds)


def minimum_jerk_waypoint(
    q0: np.ndarray,
    qf: np.ndarray,
    v0: np.ndarray,
    vf: np.ndarray,
    duration: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-jerk interpolation between two joint configurations.

    Unlike cubic interpolation, this uses a quintic polynomial that ensures
    zero acceleration at both endpoints, eliminating jerk spikes at
    trajectory start/end.

    When v0 and vf are both zero (the common case), this reduces to the
    standard minimum-jerk profile. When they are nonzero, a modified
    quintic polynomial is used that satisfies all 6 boundary conditions:
        q(0)=q0, q(T)=qf, v(0)=v0, v(T)=vf, a(0)=0, a(T)=0

    Parameters
    ----------
    q0 : start joint positions
    qf : end joint positions
    v0 : start joint velocities
    vf : end joint velocities
    duration : total trajectory time (seconds)
    t : current time (seconds)

    Returns
    -------
    (position, velocity, acceleration) at time t
    """
    T = duration
    if T <= 0:
        return qf.copy(), vf.copy(), np.zeros_like(qf)

    t = float(np.clip(t, 0.0, T))

    if np.allclose(v0, 0.0) and np.allclose(vf, 0.0):
        # Pure minimum-jerk (most common case)
        profile = minimum_jerk_scalar(t / T)
        delta = qf - q0
        pos = q0 + delta * profile.s
        vel = delta * profile.ds / T
        acc = delta * profile.dds / (T * T)
        return pos, vel, acc

    # General quintic with boundary conditions:
    # q(0)=q0, q(T)=qf, v(0)=v0, v(T)=vf, a(0)=0, a(T)=0
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    a0 = q0
    a1 = v0
    a2 = np.zeros_like(q0)  # a(0) = 0
    # Solve for a3, a4, a5 from remaining 3 boundary conditions
    delta = qf - q0
    a3 = (10.0 * delta - 6.0 * v0 * T - 4.0 * vf * T) / T3
    a4 = (-15.0 * delta + 8.0 * v0 * T + 7.0 * vf * T) / T4
    a5 = (6.0 * delta - 3.0 * (v0 + vf) * T) / T5

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pos = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
    vel = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
    acc = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Fitts' Law duration estimation
# ---------------------------------------------------------------------------

# Fitts' Law parameters calibrated for the D1 arm workspace (670mm reach).
# MT = a + b * log2(2D/W)
# a = reaction/startup overhead (~0.15s for servo response)
# b = movement time slope (~0.15s/bit for D1-scale movements)
# These yield 0.6-1.1s for typical workspace reaches.
_FITTS_A = 0.15  # seconds (intercept)
_FITTS_B = 0.15  # seconds/bit (slope)
_FITTS_DEFAULT_WIDTH = 0.02  # 20mm default target width (radians equivalent)
_FITTS_MIN_DURATION = 0.3  # minimum movement time
_FITTS_MAX_DURATION = 3.0  # maximum movement time


def fitts_law_duration(
    distance: float,
    target_width: float = _FITTS_DEFAULT_WIDTH,
    a: float = _FITTS_A,
    b: float = _FITTS_B,
) -> float:
    """Compute human-like movement duration using Fitts' Law.

    MT = a + b * log2(2D/W)

    This produces movements at speeds humans intuitively expect, avoiding
    the common mistake of arbitrary timing or constant-velocity moves.

    Parameters
    ----------
    distance : movement distance (in joint-space radians or meters)
    target_width : precision requirement (smaller = slower, more precise)
    a : intercept (startup overhead)
    b : slope (seconds per bit of difficulty)

    Returns
    -------
    Duration in seconds, clamped to [MIN_DURATION, MAX_DURATION].
    """
    if distance < 1e-6:
        return _FITTS_MIN_DURATION

    # Index of difficulty
    id_bits = math.log2(2.0 * distance / max(target_width, 1e-6))
    id_bits = max(id_bits, 0.0)  # can't have negative difficulty

    mt = a + b * id_bits
    return float(np.clip(mt, _FITTS_MIN_DURATION, _FITTS_MAX_DURATION))


def compute_movement_duration(
    q_start: np.ndarray,
    q_end: np.ndarray,
    target_precision: float = _FITTS_DEFAULT_WIDTH,
    speed_factor: float = 1.0,
) -> float:
    """Compute natural movement duration for a joint-space move.

    Uses maximum joint displacement as the distance metric, then
    applies Fitts' Law to get a human-like duration.

    Parameters
    ----------
    q_start : start joint angles (radians)
    q_end : end joint angles (radians)
    target_precision : how precisely we need to hit the target
    speed_factor : multiplier (>1 = faster, <1 = slower)

    Returns
    -------
    Duration in seconds.
    """
    delta = np.abs(q_end - q_start)
    max_displacement = float(np.max(delta))

    duration = fitts_law_duration(max_displacement, target_precision)

    if speed_factor > 0:
        duration /= speed_factor

    return float(np.clip(duration, _FITTS_MIN_DURATION, _FITTS_MAX_DURATION))


# ---------------------------------------------------------------------------
# Multi-point minimum-jerk trajectory
# ---------------------------------------------------------------------------


def generate_minimum_jerk_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    dt: float = 0.01,
    v_start: np.ndarray | None = None,
    v_end: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a complete minimum-jerk trajectory as arrays.

    Returns
    -------
    (times, positions, velocities, accelerations)
        times: (N,) array
        positions: (N, n_joints) array
        velocities: (N, n_joints) array
        accelerations: (N, n_joints) array
    """
    q_start = np.asarray(q_start, dtype=np.float64)
    q_end = np.asarray(q_end, dtype=np.float64)
    n_joints = len(q_start)

    if v_start is None:
        v_start = np.zeros(n_joints)
    if v_end is None:
        v_end = np.zeros(n_joints)

    n_points = max(2, int(math.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, n_points)

    positions = np.zeros((n_points, n_joints))
    velocities = np.zeros((n_points, n_joints))
    accelerations = np.zeros((n_points, n_joints))

    for i, t in enumerate(times):
        pos, vel, acc = minimum_jerk_waypoint(q_start, q_end, v_start, v_end, duration, t)
        positions[i] = pos
        velocities[i] = vel
        accelerations[i] = acc

    return times, positions, velocities, accelerations
