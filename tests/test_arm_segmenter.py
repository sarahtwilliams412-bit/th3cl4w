"""Tests for the arm segmentation pipeline."""

import numpy as np
import pytest
import cv2

from src.vision.arm_segmenter import ArmSegmenter, ArmSegmentation, GOLD_HSV_LOWER, GOLD_HSV_UPPER


def _make_blank(h=480, w=640, color=(0, 0, 0)):
    """Create a solid BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _make_background(h=480, w=640):
    """Plain gray background."""
    return _make_blank(h, w, color=(128, 128, 128))


def _add_rect(img, x, y, w, h, color):
    """Draw a filled rectangle on img (mutates)."""
    img[y : y + h, x : x + w] = color
    return img


class TestBackgroundSubtraction:
    def test_no_background_returns_empty(self):
        seg = ArmSegmenter()
        frame = _make_blank()
        mask = seg.subtract_background(frame)
        assert mask.shape == (480, 640)
        assert mask.sum() == 0

    def test_capture_background(self):
        seg = ArmSegmenter()
        bg = _make_background()
        seg.capture_background([bg, bg.copy(), bg.copy()])
        assert seg.has_background

    def test_identical_frame_no_foreground(self):
        seg = ArmSegmenter()
        bg = _make_background()
        seg.capture_background([bg])
        mask = seg.subtract_background(bg.copy())
        assert mask.sum() == 0

    def test_different_region_detected(self):
        seg = ArmSegmenter(bg_threshold=20, min_contour_area=100)
        bg = _make_background()
        seg.capture_background([bg])

        # Add a white rectangle (simulating arm)
        frame = bg.copy()
        _add_rect(frame, 200, 150, 100, 200, (255, 255, 255))

        mask = seg.subtract_background(frame)
        assert mask.sum() > 0
        # The foreground region should be roughly where we drew
        assert mask[250, 250] == 255

    def test_empty_frames_list(self):
        seg = ArmSegmenter()
        seg.capture_background([])
        assert not seg.has_background


class TestGoldDetection:
    def _gold_bgr(self):
        """Return a BGR color that falls within the gold HSV range."""
        # HSV (30, 200, 200) -> should be in range [20-40, 100-255, 100-255]
        hsv_img = np.array([[[30, 200, 200]]], dtype=np.uint8)
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr_img[0, 0])

    def test_detect_gold_patch(self):
        seg = ArmSegmenter(gold_min_area=10)
        frame = _make_blank(color=(50, 50, 50))
        gold = self._gold_bgr()
        _add_rect(frame, 300, 200, 40, 40, gold)

        centroids = seg.detect_gold_segments(frame)
        assert len(centroids) >= 1
        cx, cy = centroids[0]
        assert 300 <= cx <= 340
        assert 200 <= cy <= 240

    def test_no_gold_in_blank(self):
        seg = ArmSegmenter()
        frame = _make_blank(color=(50, 50, 50))
        centroids = seg.detect_gold_segments(frame)
        assert centroids == []

    def test_small_gold_filtered(self):
        """Gold regions smaller than gold_min_area should be filtered."""
        seg = ArmSegmenter(gold_min_area=500)
        frame = _make_blank(color=(50, 50, 50))
        gold = self._gold_bgr()
        # Tiny 3x3 patch
        _add_rect(frame, 100, 100, 3, 3, gold)
        centroids = seg.detect_gold_segments(frame)
        assert centroids == []


class TestMorphologicalCleanup:
    def test_small_noise_removed(self):
        """Scattered single-pixel noise should be cleaned by morphological ops."""
        seg = ArmSegmenter(bg_threshold=20, morph_open_iter=2, morph_close_iter=2)
        bg = _make_background()
        seg.capture_background([bg])

        frame = bg.copy()
        # Scatter random noise pixels
        rng = np.random.RandomState(42)
        noise_coords = rng.randint(0, min(480, 640), size=(50, 2))
        for y, x in noise_coords:
            if y < 480 and x < 640:
                frame[y, x] = [255, 255, 255]

        mask = seg.subtract_background(frame)
        # Morphological opening should remove isolated pixels
        assert mask.sum() == 0 or np.count_nonzero(mask) < 20


class TestSegmentArm:
    def test_full_segmentation(self):
        seg = ArmSegmenter(bg_threshold=20, min_contour_area=100, gold_min_area=10)
        bg = _make_background()
        seg.capture_background([bg])

        frame = bg.copy()
        # Arm body (white block)
        _add_rect(frame, 200, 100, 80, 300, (255, 255, 255))
        # Gold accent
        hsv_img = np.array([[[30, 200, 200]]], dtype=np.uint8)
        gold = tuple(int(c) for c in cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)[0, 0])
        _add_rect(frame, 220, 200, 30, 30, gold)

        result = seg.segment_arm(frame)

        assert isinstance(result, ArmSegmentation)
        assert result.silhouette_mask.shape == (480, 640)
        assert result.silhouette_mask.sum() > 0
        assert result.contour is not None
        assert result.bounding_box is not None
        x, y, w, h = result.bounding_box
        assert 180 <= x <= 220
        assert len(result.gold_centroids) >= 1

    def test_dataclass_fields(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        seg = ArmSegmentation(
            silhouette_mask=mask,
            gold_centroids=[(5, 5)],
            contour=None,
            bounding_box=(0, 0, 10, 10),
        )
        assert seg.silhouette_mask is mask
        assert seg.gold_centroids == [(5, 5)]
        assert seg.bounding_box == (0, 0, 10, 10)

    def test_no_arm_present(self):
        seg = ArmSegmenter(bg_threshold=20)
        bg = _make_background()
        seg.capture_background([bg])

        result = seg.segment_arm(bg.copy())
        assert result.contour is None
        assert result.bounding_box is None
        assert result.gold_centroids == []


class TestROI:
    def test_set_and_clear_roi(self):
        seg = ArmSegmenter()
        seg.set_roi(100, 100, 200, 200)
        assert seg._roi == (100, 100, 200, 200)
        seg.clear_roi()
        assert seg._roi is None

    def test_roi_restricts_processing(self):
        seg = ArmSegmenter(bg_threshold=20, min_contour_area=50, gold_min_area=10)
        bg = _make_background()
        seg.capture_background([bg])

        frame = bg.copy()
        # Object outside ROI
        _add_rect(frame, 10, 10, 50, 50, (255, 255, 255))
        # Object inside ROI
        _add_rect(frame, 300, 300, 50, 50, (255, 255, 255))

        seg.set_roi(250, 250, 200, 200)
        result = seg.segment_arm(frame)

        # Should only detect the object inside the ROI
        if result.bounding_box is not None:
            x, y, w, h = result.bounding_box
            assert x >= 250
