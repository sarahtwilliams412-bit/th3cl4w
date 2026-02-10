"""Tests for command smoothing filters (dual-EMA, One-Euro, jerk limiter)."""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.command_filter import (
    DualEMAFilter,
    OneEuroFilter,
    JerkLimiter,
    SmoothCommandPipeline,
)


class TestDualEMAFilter:
    """Test dual-stage exponential moving average filter."""

    def test_first_call_passthrough(self):
        """First call should pass through without lag."""
        f = DualEMAFilter(n_joints=3, alpha=0.3)
        result = f.filter(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_step_response_smoothed(self):
        """Step input should be smoothed (not passed through immediately)."""
        f = DualEMAFilter(n_joints=1, alpha=0.3)
        f.filter(np.array([0.0]))  # initialize
        result = f.filter(np.array([1.0]))  # step
        assert 0.0 < result[0] < 1.0, "Step should be smoothed"

    def test_converges_to_constant(self):
        """After many steps with same input, output converges."""
        f = DualEMAFilter(n_joints=2, alpha=0.3)
        target = np.array([5.0, 10.0])
        for _ in range(200):
            result = f.filter(target)
        np.testing.assert_allclose(result, target, atol=0.01)

    def test_alpha_one_passthrough(self):
        """alpha=1.0 should pass through immediately."""
        f = DualEMAFilter(n_joints=2, alpha=1.0)
        f.filter(np.array([0.0, 0.0]))  # init
        result = f.filter(np.array([10.0, 20.0]))
        np.testing.assert_allclose(result, [10.0, 20.0])

    def test_smaller_alpha_smoother(self):
        """Smaller alpha produces smoother (slower) response."""
        f_smooth = DualEMAFilter(n_joints=1, alpha=0.1)
        f_fast = DualEMAFilter(n_joints=1, alpha=0.5)

        f_smooth.filter(np.array([0.0]))
        f_fast.filter(np.array([0.0]))

        r_smooth = f_smooth.filter(np.array([1.0]))
        r_fast = f_fast.filter(np.array([1.0]))

        assert r_smooth[0] < r_fast[0], "Smaller alpha should respond slower"

    def test_reset(self):
        """Reset should re-initialize filter state."""
        f = DualEMAFilter(n_joints=2, alpha=0.3)
        f.filter(np.array([0.0, 0.0]))
        f.filter(np.array([10.0, 20.0]))
        f.reset(np.array([5.0, 5.0]))
        np.testing.assert_allclose(f.current_value, [5.0, 5.0])

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            DualEMAFilter(n_joints=1, alpha=0.0)
        with pytest.raises(ValueError):
            DualEMAFilter(n_joints=1, alpha=1.5)

    def test_no_overshoot(self):
        """Dual-EMA should not overshoot a step input."""
        f = DualEMAFilter(n_joints=1, alpha=0.3)
        f.filter(np.array([0.0]))

        for _ in range(100):
            result = f.filter(np.array([1.0]))
            assert result[0] <= 1.0 + 1e-10, "Should not overshoot"


class TestOneEuroFilter:
    """Test adaptive noise filter."""

    def test_first_call_passthrough(self):
        f = OneEuroFilter(n_joints=2, rate=100.0)
        result = f.filter(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_filters_noise(self):
        """Noisy signal around constant should be smoothed."""
        f = OneEuroFilter(n_joints=1, min_cutoff=1.0, beta=0.007, rate=100.0)
        rng = np.random.default_rng(42)

        values = []
        for _ in range(200):
            noisy = np.array([5.0 + rng.normal(0, 0.5)])
            result = f.filter(noisy)
            values.append(result[0])

        # Last 50 values should be closer to 5.0 than raw noise
        last_50 = np.array(values[-50:])
        assert np.std(last_50) < 0.3, "Filter should reduce noise variance"

    def test_tracks_fast_movement(self):
        """Fast ramp should be tracked with low lag."""
        f = OneEuroFilter(n_joints=1, min_cutoff=1.0, beta=0.5, rate=100.0)

        # Ramp from 0 to 10 quickly
        for i in range(100):
            result = f.filter(np.array([i * 0.1]))

        # Should be close to final value
        assert result[0] > 8.0, "Should track fast ramp"

    def test_reset(self):
        f = OneEuroFilter(n_joints=2, rate=100.0)
        f.filter(np.array([1.0, 2.0]))
        f.reset(np.array([5.0, 5.0]))
        result = f.filter(np.array([5.1, 5.1]))
        # After reset, should be near reset value
        np.testing.assert_allclose(result, [5.1, 5.1], atol=0.5)


class TestJerkLimiter:
    """Test jerk-limited rate limiter."""

    def test_first_call_passthrough(self):
        """First call initializes to target."""
        jl = JerkLimiter(
            n_joints=2,
            max_velocity=np.array([5.0, 5.0]),
            max_acceleration=np.array([10.0, 10.0]),
            max_jerk=np.array([50.0, 50.0]),
            dt=0.01,
        )
        result = jl.limit(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_velocity_limited(self):
        """Large step should be velocity-limited."""
        jl = JerkLimiter(
            n_joints=1,
            max_velocity=np.array([1.0]),
            max_acceleration=np.array([100.0]),
            max_jerk=np.array([10000.0]),
            dt=0.01,
        )
        jl.reset(np.array([0.0]))

        # Try to jump to 100 — should be limited
        result = jl.limit(np.array([100.0]))
        assert result[0] <= 0.01 + 1e-6, "Should be velocity limited"

    def test_bounds_velocity(self):
        """Output velocity should never exceed max_velocity."""
        jl = JerkLimiter(
            n_joints=1,
            max_velocity=np.array([2.0]),
            max_acceleration=np.array([20.0]),
            max_jerk=np.array([100.0]),
            dt=0.01,
        )
        jl.reset(np.array([0.0]))

        # Feed in linearly increasing input (constant velocity of 5.0)
        for i in range(100):
            inp = np.array([i * 0.05])  # 5.0 rad/s input rate
            jl.limit(inp)
            # Output velocity should be bounded
            assert abs(jl.current_velocities[0]) <= 2.0 + 0.1

    def test_smooth_acceleration(self):
        """Acceleration should change smoothly (no jumps)."""
        jl = JerkLimiter(
            n_joints=1,
            max_velocity=np.array([5.0]),
            max_acceleration=np.array([10.0]),
            max_jerk=np.array([30.0]),
            dt=0.01,
        )
        jl.reset(np.array([0.0]))

        accelerations = []
        target = np.array([2.0])
        for _ in range(100):
            jl.limit(target)
            accelerations.append(jl.current_accelerations[0])

        # Check jerk (rate of acceleration change) is bounded
        for i in range(1, len(accelerations)):
            jerk = abs(accelerations[i] - accelerations[i - 1]) / 0.01
            assert jerk <= 30.0 + 1.0, f"Jerk {jerk} exceeds limit"

    def test_reset_clears_velocity(self):
        jl = JerkLimiter(
            n_joints=2,
            max_velocity=np.array([5.0, 5.0]),
            max_acceleration=np.array([10.0, 10.0]),
            max_jerk=np.array([50.0, 50.0]),
            dt=0.01,
        )
        jl.reset(np.array([1.0, 2.0]))
        np.testing.assert_allclose(jl.current_velocities, [0.0, 0.0])
        np.testing.assert_allclose(jl.current_accelerations, [0.0, 0.0])


class TestSmoothCommandPipeline:
    """Test the full command smoothing pipeline."""

    def test_first_call_passthrough(self):
        pipeline = SmoothCommandPipeline(n_joints=3, dt=0.01)
        result = pipeline.smooth_command(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_smooths_step(self):
        """Step input should be smoothed."""
        pipeline = SmoothCommandPipeline(n_joints=1, ema_alpha=0.3, dt=0.01)
        pipeline.reset(np.array([0.0]))

        result = pipeline.smooth_command(np.array([10.0]))
        assert 0.0 < result[0] < 10.0, "Step should be smoothed"

    def test_converges(self):
        """Pipeline output converges to constant input via dual-EMA."""
        pipeline = SmoothCommandPipeline(
            n_joints=2,
            ema_alpha=0.3,
            dt=0.01,
        )
        pipeline.reset(np.array([0.0, 0.0]))

        target = np.array([5.0, 10.0])
        for _ in range(500):
            result = pipeline.smooth_command(target)

        np.testing.assert_allclose(result, target, atol=0.1)

    def test_initialized_flag(self):
        pipeline = SmoothCommandPipeline(n_joints=2)
        assert not pipeline.is_initialized
        pipeline.smooth_command(np.array([1.0, 2.0]))
        assert pipeline.is_initialized

    def test_sensor_filter(self):
        pipeline = SmoothCommandPipeline(n_joints=2, enable_sensor_filter=True, dt=0.01)
        pipeline.reset(np.array([0.0, 0.0]))

        result = pipeline.filter_sensor(np.array([1.0, 2.0]))
        assert result.shape == (2,)

    def test_no_sensor_filter(self):
        pipeline = SmoothCommandPipeline(n_joints=2, enable_sensor_filter=False, dt=0.01)
        result = pipeline.filter_sensor(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [1.0, 2.0])
