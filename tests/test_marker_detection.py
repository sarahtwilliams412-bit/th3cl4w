"""Tests for HSV marker-based joint detection and arm segmentation."""

import numpy as np
import cv2
import pytest


def _make_frame_with_markers(
    width: int = 640,
    height: int = 480,
    markers: list[tuple[int, int, tuple[int, int, int]]] | None = None,
    marker_radius: int = 15,
) -> np.ndarray:
    """Create a test BGR frame with colored circle markers.

    markers: list of (x, y, bgr_color)
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # dark gray background
    if markers:
        for x, y, bgr in markers:
            cv2.circle(frame, (x, y), marker_radius, bgr, -1)
    return frame


# BGR colors that map to the HSV ranges
NEON_GREEN_BGR = (0, 255, 0)
NEON_ORANGE_BGR = (0, 140, 255)
HOT_PINK_BGR = (180, 0, 255)


class TestJointDetectorMarkers:
    def test_detect_markers_neon_green(self):
        from src.vision.joint_detector import JointDetector

        frame = _make_frame_with_markers(markers=[(100, 200, NEON_GREEN_BGR)])
        det = JointDetector()
        markers = det.detect_markers(frame)
        assert len(markers) >= 1
        assert markers[0].color_name == "neon_green"
        assert abs(markers[0].centroid[0] - 100) < 5
        assert abs(markers[0].centroid[1] - 200) < 5

    def test_detect_markers_neon_orange(self):
        from src.vision.joint_detector import JointDetector

        frame = _make_frame_with_markers(markers=[(300, 100, NEON_ORANGE_BGR)])
        det = JointDetector()
        markers = det.detect_markers(frame)
        assert len(markers) >= 1
        assert markers[0].color_name == "neon_orange"

    def test_detect_markers_hot_pink(self):
        from src.vision.joint_detector import JointDetector

        frame = _make_frame_with_markers(markers=[(200, 300, HOT_PINK_BGR)])
        det = JointDetector()
        markers = det.detect_markers(frame)
        assert len(markers) >= 1
        assert markers[0].color_name == "hot_pink"

    def test_detect_multiple_markers(self):
        from src.vision.joint_detector import JointDetector

        frame = _make_frame_with_markers(markers=[
            (100, 200, NEON_GREEN_BGR),
            (300, 200, NEON_ORANGE_BGR),
            (500, 200, HOT_PINK_BGR),
        ])
        det = JointDetector()
        markers = det.detect_markers(frame)
        assert len(markers) == 3

    def test_no_markers_empty(self):
        from src.vision.joint_detector import JointDetector

        frame = _make_frame_with_markers()  # no markers
        det = JointDetector()
        markers = det.detect_markers(frame)
        assert len(markers) == 0

    def test_marker_detection_source_in_joint_detect(self):
        from src.vision.joint_detector import JointDetector, DetectionSource
        from src.vision.arm_segmenter import ArmSegmentation

        frame = _make_frame_with_markers(markers=[
            (100, 240, NEON_GREEN_BGR),
            (300, 240, NEON_ORANGE_BGR),
            (500, 240, HOT_PINK_BGR),
        ])
        seg = ArmSegmentation(
            silhouette_mask=np.zeros((480, 640), dtype=np.uint8),
            gold_centroids=[],
        )
        fk_pixels = [(100.0, 240.0), (300.0, 240.0), (500.0, 240.0)]
        det = JointDetector()
        joints = det.detect_joints(seg, fk_pixels, frame=frame)
        assert len(joints) == 3
        for j in joints:
            assert j.source == DetectionSource.MARKER
            assert j.confidence >= 0.7

    def test_marker_fallback_to_fk(self):
        """With no markers and no background, should fallback to FK_ONLY."""
        from src.vision.joint_detector import JointDetector, DetectionSource
        from src.vision.arm_segmenter import ArmSegmentation

        frame = _make_frame_with_markers()  # no markers
        seg = ArmSegmentation(
            silhouette_mask=np.zeros((480, 640), dtype=np.uint8),
            gold_centroids=[],
        )
        fk_pixels = [(100.0, 240.0), (300.0, 240.0)]
        det = JointDetector()
        joints = det.detect_joints(seg, fk_pixels, frame=frame)
        assert all(j.source == DetectionSource.FK_ONLY for j in joints)

    def test_custom_marker_colors(self):
        from src.vision.joint_detector import JointDetector

        custom_colors = {
            "cyan": (
                np.array([80, 100, 100], dtype=np.uint8),
                np.array([100, 255, 255], dtype=np.uint8),
            ),
        }
        det = JointDetector(marker_colors=custom_colors)
        assert "cyan" in det.marker_colors
        assert "neon_green" not in det.marker_colors


class TestArmSegmenterMarkers:
    def test_segment_arm_marker_mode(self):
        from src.vision.arm_segmenter import ArmSegmenter

        frame = _make_frame_with_markers(markers=[
            (100, 240, NEON_GREEN_BGR),
            (300, 240, NEON_ORANGE_BGR),
            (500, 240, HOT_PINK_BGR),
        ])
        seg = ArmSegmenter()
        result = seg.segment_arm(frame)
        assert len(result.marker_centroids) >= 2
        assert result.silhouette_mask.any()  # mask is non-empty
        assert result.bounding_box is not None

    def test_segment_arm_by_markers_creates_mask(self):
        from src.vision.arm_segmenter import ArmSegmenter

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        seg = ArmSegmenter()
        markers = [(100.0, 240.0), (300.0, 240.0), (500.0, 240.0)]
        mask = seg.segment_arm_by_markers(frame, markers)
        assert mask.shape == (480, 640)
        assert mask.any()
        # Check that pixels along the line are filled
        assert mask[240, 200] == 255  # between first two markers

    def test_fallback_to_bg_sub(self):
        """No markers → should use background subtraction path."""
        from src.vision.arm_segmenter import ArmSegmenter

        frame = _make_frame_with_markers()  # dark, no markers
        seg = ArmSegmenter()
        result = seg.segment_arm(frame)
        assert len(result.marker_centroids) < 2  # not enough for marker mode
