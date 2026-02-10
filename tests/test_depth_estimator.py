"""Tests for monocular depth estimation module."""

import numpy as np
import pytest


class TestDepthEstimator:
    """Tests for depth_estimator — graceful degradation and API contract."""

    def test_import(self):
        from src.vision.depth_estimator import estimate_depth, is_available, get_backend
        assert callable(estimate_depth)
        assert callable(is_available)
        assert callable(get_backend)

    def test_estimate_depth_returns_none_or_array(self):
        """estimate_depth should return None (no model) or HxW float32."""
        from src.vision.depth_estimator import estimate_depth

        fake_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = estimate_depth(fake_frame)

        if result is not None:
            assert result.dtype == np.float32
            assert result.shape == (240, 320)
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_estimate_metric_depth(self):
        from src.vision.depth_estimator import estimate_metric_depth

        fake_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = estimate_metric_depth(fake_frame, known_distance_m=0.5)

        if result is not None:
            assert result.dtype == np.float32
            assert result.shape == (240, 320)


class TestDepthEstimatorWithModel:
    """Tests that require the model to be loaded (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def check_model(self):
        from src.vision.depth_estimator import is_available
        if not is_available():
            pytest.skip("Depth model not available")

    def test_depth_on_sample_image(self):
        from src.vision.depth_estimator import estimate_depth

        # Create a simple gradient image (simulates depth variation)
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            img[y, :] = [int(255 * y / h)] * 3

        depth = estimate_depth(img)
        assert depth is not None
        assert depth.shape == (h, w)
        assert depth.dtype == np.float32
        # Should have some variation (not all zeros)
        assert depth.max() - depth.min() > 0.1
