"""Tests for realtime_detector.py — HSV color-based object detection."""

import numpy as np
import cv2
import pytest

from src.vision.realtime_detector import detect_object, detect_gripper, Detection


def _make_red_circle_frame(w=640, h=480, cx=320, cy=240, radius=40):
    """Create a synthetic frame with a red circle on black background."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Red in BGR
    cv2.circle(frame, (cx, cy), radius, (0, 0, 220), -1)
    return frame


def _make_redbull_frame(w=640, h=480, cx=300, cy=250):
    """Create a synthetic frame with red + blue regions (mimicking Red Bull can)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Red body
    cv2.rectangle(frame, (cx - 20, cy - 50), (cx + 20, cy + 50), (0, 0, 200), -1)
    # Blue band
    cv2.rectangle(frame, (cx - 20, cy - 10), (cx + 20, cy + 10), (200, 80, 0), -1)
    return frame


class TestDetectObject:
    def test_red_object_detected(self):
        frame = _make_red_circle_frame(cx=320, cy=240, radius=40)
        result = detect_object(frame, target="red")
        assert result.found
        assert abs(result.centroid_px[0] - 320) < 5
        assert abs(result.centroid_px[1] - 240) < 5
        assert result.confidence > 0.4
        assert result.bbox[2] > 0 and result.bbox[3] > 0

    def test_redbull_detected(self):
        frame = _make_redbull_frame(cx=300, cy=250)
        result = detect_object(frame, target="redbull")
        assert result.found
        assert result.label == "redbull"
        # Centroid should be roughly at the red region
        assert abs(result.centroid_px[0] - 300) < 35
        assert abs(result.centroid_px[1] - 250) < 35

    def test_nothing_detected_on_empty(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_object(frame, target="red")
        assert not result.found

    def test_none_frame(self):
        result = detect_object(None, target="red")
        assert not result.found

    def test_empty_frame(self):
        result = detect_object(np.array([]), target="red")
        assert not result.found

    def test_unknown_target_falls_back(self):
        frame = _make_red_circle_frame()
        result = detect_object(frame, target="nonexistent")
        # Falls back to red detection
        assert result.found

    def test_min_area_filter(self):
        frame = _make_red_circle_frame(radius=3)  # tiny circle
        result = detect_object(frame, target="red", min_area=500)
        assert not result.found

    def test_detection_speed(self):
        """Detection should complete in <30ms."""
        import time

        frame = _make_red_circle_frame(w=1920, h=1080, cx=960, cy=540, radius=80)
        t0 = time.monotonic()
        for _ in range(10):
            detect_object(frame, target="redbull")
        elapsed = (time.monotonic() - t0) / 10
        assert elapsed < 0.030, f"Detection took {elapsed*1000:.1f}ms, expected <30ms"


class TestDetectGripper:
    def test_bright_object_detected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # White/bright region
        cv2.circle(frame, (400, 300), 30, (240, 240, 240), -1)
        result = detect_gripper(frame)
        assert result.found
        assert result.label == "gripper"
        assert abs(result.centroid_px[0] - 400) < 5
