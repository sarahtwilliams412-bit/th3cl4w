"""
Synthetic Tests for Visual Hull Reconstruction

Validates the core algorithm with known geometric shapes:
1. Sphere → Steinmetz solid (intersection of two cylinders)
2. Rectangular box → exact reconstruction
3. Empty scene → all voxels below threshold
4. Performance benchmark

Run with: pytest visual_hull/test_synthetic.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from calibration.ascii_to_grayscale import build_density_lut
from visual_hull.hull_reconstructor import VisualHullReconstructor
from visual_hull.temporal_filter import TemporalFilter

N = 128  # Grid resolution

# Build a simple density LUT for testing
_TEST_LUT = build_density_lut()

# Define two ASCII codepoints for convenience
EMPTY_CHAR = ord(" ")  # codepoint 32, density ≈ 0.0
SOLID_CHAR = ord("@")  # codepoint 64, density ≈ 0.9


def make_circle_grid(cx: int, cy: int, radius: int) -> np.ndarray:
    """Create a 128×128 uint8 grid with a filled circle (ASCII '@')."""
    grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)
    yy, xx = np.mgrid[0:N, 0:N]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    grid[mask] = SOLID_CHAR
    return grid


def make_rect_grid(
    col_start: int, row_start: int, width: int, height: int
) -> np.ndarray:
    """Create a 128×128 uint8 grid with a filled rectangle (ASCII '@').

    Parameters
    ----------
    col_start : column (x) start
    row_start : row (y) start
    width : number of columns
    height : number of rows
    """
    grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)
    grid[row_start : row_start + height, col_start : col_start + width] = SOLID_CHAR
    return grid


@pytest.fixture
def reconstructor() -> VisualHullReconstructor:
    return VisualHullReconstructor(density_lut=_TEST_LUT, shared_axis="x")


class TestSphereReconstruction:
    """Test 1: Sphere → Steinmetz solid.

    A sphere of radius R projects to a circle in both XY and XZ views.
    The visual hull (intersection of two extruded cylinders) produces a
    Steinmetz solid with volume = 16/3 × R³.
    """

    def test_steinmetz_volume(self, reconstructor: VisualHullReconstructor):
        radius = 20
        center = N // 2  # = 64

        # Top-down: circle in XY plane
        top_grid = make_circle_grid(center, center, radius)
        # Profile: circle in XZ plane (same center and radius)
        prof_grid = make_circle_grid(center, center, radius)

        grid = reconstructor.reconstruct(top_grid, prof_grid)

        # Get the density value of '@' from the LUT
        solid_density = _TEST_LUT[SOLID_CHAR]

        # Count occupied voxels (those with density close to solid)
        occupied = np.count_nonzero(grid > solid_density * 0.5)

        # Analytical Steinmetz solid volume (in voxels)
        steinmetz_volume = 16.0 / 3.0 * radius**3

        # Allow 10% tolerance due to discrete grid sampling
        error_pct = abs(occupied - steinmetz_volume) / steinmetz_volume * 100
        assert error_pct < 10.0, (
            f"Steinmetz volume error {error_pct:.1f}% "
            f"(got {occupied}, expected {steinmetz_volume:.0f})"
        )

    def test_symmetry(self, reconstructor: VisualHullReconstructor):
        """The Steinmetz solid should be approximately symmetric.

        Note: with center at N//2=64 on a 128-wide grid, pixel-level
        symmetry isn't exact (center is at 64, not 63.5), so we check
        that the occupied voxel count is close between original and flipped.
        """
        radius = 15
        center = N // 2

        top_grid = make_circle_grid(center, center, radius)
        prof_grid = make_circle_grid(center, center, radius)
        grid = reconstructor.reconstruct(top_grid, prof_grid)

        solid_density = _TEST_LUT[SOLID_CHAR]
        threshold = solid_density * 0.5

        original_count = np.count_nonzero(grid > threshold)

        # Check approximate symmetry by comparing occupied voxel counts
        for axis, label in [(0, "X"), (1, "Y"), (2, "Z")]:
            flipped = np.flip(grid, axis=axis)
            flipped_count = np.count_nonzero(flipped > threshold)
            # Counts should match exactly (same data, just reordered)
            assert original_count == flipped_count

            # Check that most occupied voxels overlap
            overlap = np.count_nonzero((grid > threshold) & (flipped > threshold))
            overlap_pct = overlap / original_count * 100 if original_count > 0 else 100
            assert overlap_pct > 90, (
                f"{label}-axis symmetry overlap only {overlap_pct:.0f}%"
            )


class TestBoxReconstruction:
    """Test 2: Rectangular box — should be exactly reconstructed.

    A box projects to rectangles in both views. The intersection of two
    extruded rectangles perfectly reconstructs the original box.

    Reconstruction axis mapping:
      grid[i, j, k] = min(top_d[i, j], prof_d[i, k])
    where:
      axis 0 (i) = shared axis (rows in both views)
      axis 1 (j) = columns in top-down view
      axis 2 (k) = columns in profile view
    """

    def test_exact_box(self, reconstructor: VisualHullReconstructor):
        # Place a box using the grid's axis convention
        # Top-down: rect at rows [50:60), cols [30:60) → 10 rows × 30 cols
        # Profile:  rect at rows [50:60), cols [40:60) → 10 rows × 20 cols
        # Shared axis 0 intersection = [50:60) → 10 cells
        shared_row_start, shared_height = 50, 10
        top_col_start, top_width = 30, 30
        prof_col_start, prof_width = 40, 20

        top_grid = make_rect_grid(top_col_start, shared_row_start, top_width, shared_height)
        prof_grid = make_rect_grid(prof_col_start, shared_row_start, prof_width, shared_height)

        grid = reconstructor.reconstruct(top_grid, prof_grid)

        solid_density = _TEST_LUT[SOLID_CHAR]
        occupied = np.count_nonzero(grid > solid_density * 0.5)
        expected_volume = shared_height * top_width * prof_width  # 10 × 30 × 20 = 6000

        error_pct = abs(occupied - expected_volume) / expected_volume * 100
        assert error_pct < 1.0, (
            f"Box volume error {error_pct:.1f}% "
            f"(got {occupied}, expected {expected_volume})"
        )

    def test_box_bounds(self, reconstructor: VisualHullReconstructor):
        """Verify the box occupancy is in the correct grid region."""
        shared_row_start, shared_height = 50, 15
        top_col_start, top_width = 40, 20
        prof_col_start, prof_width = 30, 25

        top_grid = make_rect_grid(top_col_start, shared_row_start, top_width, shared_height)
        prof_grid = make_rect_grid(prof_col_start, shared_row_start, prof_width, shared_height)

        grid = reconstructor.reconstruct(top_grid, prof_grid)
        solid_density = _TEST_LUT[SOLID_CHAR]

        # Occupied region should be at:
        # axis 0: [shared_row_start : shared_row_start + shared_height]
        # axis 1: [top_col_start : top_col_start + top_width]
        # axis 2: [prof_col_start : prof_col_start + prof_width]
        outside = grid.copy()
        outside[
            shared_row_start : shared_row_start + shared_height,
            top_col_start : top_col_start + top_width,
            prof_col_start : prof_col_start + prof_width,
        ] = 0.0
        assert np.all(outside < solid_density * 0.5), "Occupancy found outside box bounds"


class TestEmptyScene:
    """Test 3: Empty scene — all voxels below occupancy threshold."""

    def test_empty_grids(self, reconstructor: VisualHullReconstructor):
        top_grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)
        prof_grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)

        grid = reconstructor.reconstruct(top_grid, prof_grid)

        max_val = grid.max()
        assert max_val < 0.15, f"Empty scene has max occupancy {max_val:.3f}"

    def test_one_view_empty(self, reconstructor: VisualHullReconstructor):
        """If one view is empty, the hull should be empty."""
        top_grid = make_circle_grid(64, 64, 30)
        prof_grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)

        grid = reconstructor.reconstruct(top_grid, prof_grid)
        max_val = grid.max()
        assert max_val < 0.15, f"One-view-empty scene has max occupancy {max_val:.3f}"


class TestCoarseToFine:
    """Test the two-pass coarse-to-fine reconstruction."""

    def test_preserves_majority(self, reconstructor: VisualHullReconstructor):
        """Coarse-to-fine should preserve the majority of occupied voxels."""
        radius = 20
        center = N // 2

        top_grid = make_circle_grid(center, center, radius)
        prof_grid = make_circle_grid(center, center, radius)

        full = reconstructor.reconstruct(top_grid, prof_grid)
        ctf = reconstructor.reconstruct_coarse_to_fine(top_grid, prof_grid)

        solid_density = _TEST_LUT[SOLID_CHAR]
        threshold = solid_density * 0.5

        full_occupied = np.count_nonzero(full > threshold)
        ctf_occupied = np.count_nonzero(ctf > threshold)

        # Coarse-to-fine may lose some border voxels due to downsampling
        # but should preserve at least 85% of occupied voxels
        preservation_pct = ctf_occupied / full_occupied * 100 if full_occupied > 0 else 100
        assert preservation_pct > 85, (
            f"Coarse-to-fine preserved only {preservation_pct:.0f}% of occupied voxels "
            f"({ctf_occupied} vs {full_occupied})"
        )

    def test_no_false_positives(self, reconstructor: VisualHullReconstructor):
        """Coarse-to-fine should not add occupancy where full resolution has none."""
        radius = 20
        center = N // 2

        top_grid = make_circle_grid(center, center, radius)
        prof_grid = make_circle_grid(center, center, radius)

        full = reconstructor.reconstruct(top_grid, prof_grid)
        ctf = reconstructor.reconstruct_coarse_to_fine(top_grid, prof_grid)

        # Where full resolution is empty, CTF should also be empty
        empty_mask = full < 0.01
        assert np.all(ctf[empty_mask] < 0.01), "CTF has false positives in empty regions"

    def test_empty_coarse(self, reconstructor: VisualHullReconstructor):
        """Empty scene should produce all-zero grid in coarse-to-fine."""
        top_grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)
        prof_grid = np.full((N, N), EMPTY_CHAR, dtype=np.uint8)

        grid = reconstructor.reconstruct_coarse_to_fine(top_grid, prof_grid)
        assert grid.max() == 0.0


class TestTemporalFilter:
    """Test temporal filtering."""

    def test_first_frame_passthrough(self):
        filt = TemporalFilter(alpha=0.7, shape=(4, 4, 4))
        grid = np.ones((4, 4, 4), dtype=np.float32)
        result = filt.update(grid)
        np.testing.assert_array_equal(result, grid)

    def test_ema_blending(self):
        """Test EMA blending with values that don't trigger space-carving decay."""
        filt = TemporalFilter(
            alpha=0.7,
            shape=(4, 4, 4),
            high_threshold=0.3,
            low_threshold=0.1,
        )

        # First frame: moderate occupancy (below high_threshold to avoid decay)
        g1 = np.full((4, 4, 4), 0.25, dtype=np.float32)
        filt.update(g1)

        # Second frame: slightly different
        g2 = np.full((4, 4, 4), 0.15, dtype=np.float32)
        result = filt.update(g2)

        # EMA: 0.7 * 0.15 + 0.3 * 0.25 = 0.105 + 0.075 = 0.18
        expected = 0.18
        np.testing.assert_allclose(result, expected, atol=0.01)

    def test_space_carving_decay(self):
        """Rapid disappearance should decay gradually, not jump to zero."""
        filt = TemporalFilter(
            alpha=0.7,
            shape=(4, 4, 4),
            decay_rate=0.2,
            high_threshold=0.3,
            low_threshold=0.1,
        )

        # First frame: solid
        g1 = np.full((4, 4, 4), 0.8, dtype=np.float32)
        filt.update(g1)

        # Second frame: suddenly empty
        g2 = np.zeros((4, 4, 4), dtype=np.float32)
        result = filt.update(g2)

        # Should decay by 0.2 from 0.8 → 0.6, not jump to EMA of 0.24
        assert result[0, 0, 0] > 0.5, (
            f"Space-carving decay failed: got {result[0, 0, 0]:.2f}, expected >0.5"
        )

    def test_reset(self):
        filt = TemporalFilter(alpha=0.7, shape=(4, 4, 4))
        filt.update(np.ones((4, 4, 4), dtype=np.float32))
        filt.reset()
        assert filt.frame_count == 0
        assert filt.prev_grid.max() == 0.0


class TestPerformance:
    """Test 4: Performance benchmark."""

    def test_reconstruct_speed(self, reconstructor: VisualHullReconstructor):
        """100 frames through reconstruct() — mean time check."""
        top_grid = make_circle_grid(64, 64, 25)
        prof_grid = make_circle_grid(64, 64, 25)

        # Warm up
        for _ in range(5):
            reconstructor.reconstruct(top_grid, prof_grid)

        # Benchmark
        n_frames = 100
        t0 = time.monotonic()
        for _ in range(n_frames):
            reconstructor.reconstruct(top_grid, prof_grid)
        elapsed = time.monotonic() - t0

        mean_ms = (elapsed / n_frames) * 1000.0
        # Generous limit for CI environments (spec says <50ms on M3)
        assert mean_ms < 500, f"Mean reconstruct time {mean_ms:.1f}ms exceeds 500ms limit"
        print(f"\nReconstruct: {mean_ms:.1f}ms/frame ({n_frames/elapsed:.0f} fps)")

    def test_coarse_to_fine_speed(self, reconstructor: VisualHullReconstructor):
        """Coarse-to-fine should be comparable speed to full resolution."""
        top_grid = make_circle_grid(64, 64, 25)
        prof_grid = make_circle_grid(64, 64, 25)

        # Warm up
        for _ in range(5):
            reconstructor.reconstruct_coarse_to_fine(top_grid, prof_grid)

        n_frames = 100
        t0 = time.monotonic()
        for _ in range(n_frames):
            reconstructor.reconstruct_coarse_to_fine(top_grid, prof_grid)
        elapsed = time.monotonic() - t0

        mean_ms = (elapsed / n_frames) * 1000.0
        assert mean_ms < 500, f"Mean coarse-to-fine time {mean_ms:.1f}ms exceeds 500ms limit"
        print(f"\nCoarse-to-fine: {mean_ms:.1f}ms/frame ({n_frames/elapsed:.0f} fps)")
