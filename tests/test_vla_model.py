"""Tests for the Video Language Action (VLA) model module."""

import time

import numpy as np
import pytest

from src.vision.vla_model import (
    VLAModel,
    VLAMode,
    VLAAnalysisResult,
    AsciiMeasurement,
    DetectedObject3D,
    ObjectShape,
)
from src.vision.camera_pipeline import AsciiFrame, StereoAsciiFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ascii_frame(text: str, camera_id: int = 0) -> AsciiFrame:
    """Create a test ASCII frame."""
    lines = text.split("\n")
    width = max(len(line) for line in lines) if lines else 0
    height = len(lines)
    return AsciiFrame(
        camera_id=camera_id,
        ascii_text=text,
        grid_width=width,
        grid_height=height,
        timestamp=time.monotonic(),
        frame_number=1,
    )


def _make_object_ascii(width=120, height=40) -> str:
    """Create ASCII art with a dense object region.

    Puts a dense block of '#' characters in the center of the frame,
    surrounded by spaces.
    """
    lines = []
    for r in range(height):
        row = []
        for c in range(width):
            # Dense block from (45,15) to (75,25)
            if 45 <= c <= 75 and 15 <= r <= 25:
                row.append("#")
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines)


def _make_multi_object_ascii(width=120, height=40) -> str:
    """Create ASCII art with two separate dense regions."""
    lines = []
    for r in range(height):
        row = []
        for c in range(width):
            # Object 1: (20,10) to (35,20)
            if 20 <= c <= 35 and 10 <= r <= 20:
                row.append("@")
            # Object 2: (80,25) to (100,35)
            elif 80 <= c <= 100 and 25 <= r <= 35:
                row.append("#")
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines)


def _make_stereo(cam0_text=None, cam1_text=None) -> StereoAsciiFrame:
    cam0 = _make_ascii_frame(cam0_text, camera_id=0) if cam0_text else None
    cam1 = _make_ascii_frame(cam1_text, camera_id=1) if cam1_text else None
    return StereoAsciiFrame(
        cam0=cam0,
        cam1=cam1,
        timestamp=time.monotonic(),
        frame_number=1,
    )


# ---------------------------------------------------------------------------
# VLAModel construction tests
# ---------------------------------------------------------------------------


class TestVLAModelInit:
    def test_default_construction(self):
        vla = VLAModel()
        assert vla.mode == VLAMode.LOCAL
        assert vla.is_running is False
        assert vla.analysis_fps == 2.0

    def test_custom_mode(self):
        vla = VLAModel(mode=VLAMode.TEXT)
        assert vla.mode == VLAMode.TEXT

    def test_custom_scale(self):
        vla = VLAModel(mm_per_col=5.0, mm_per_row=10.0)
        assert vla.mm_per_col == 5.0
        assert vla.mm_per_row == 10.0


# ---------------------------------------------------------------------------
# ASCII measurement tests
# ---------------------------------------------------------------------------


class TestAsciiMeasurement:
    def test_to_dict(self):
        m = AsciiMeasurement(
            label="test",
            grid_min_col=10,
            grid_max_col=20,
            grid_min_row=5,
            grid_max_row=15,
            occupied_cells=50,
            total_cells=110,
            width_mm=66.7,
            height_mm=125.0,
            centroid_col=15.0,
            centroid_row=10.0,
            shape=ObjectShape.RECTANGULAR,
            fill_ratio=0.45,
            confidence=0.7,
        )
        d = m.to_dict()
        assert d["label"] == "test"
        assert d["shape"] == "rectangular"
        assert d["confidence"] == 0.7


# ---------------------------------------------------------------------------
# Object detection from ASCII
# ---------------------------------------------------------------------------


class TestMeasureFromAscii:
    def test_single_object(self):
        vla = VLAModel()
        text = _make_object_ascii()
        af = _make_ascii_frame(text, camera_id=1)
        measurements = vla._measure_from_ascii(af, "cam1")

        assert len(measurements) == 1
        m = measurements[0]
        assert m.occupied_cells > 0
        assert m.grid_min_col >= 45
        assert m.grid_max_col <= 75
        assert m.grid_min_row >= 15
        assert m.grid_max_row <= 25
        assert m.width_mm > 0
        assert m.confidence > 0

    def test_multiple_objects(self):
        vla = VLAModel()
        text = _make_multi_object_ascii()
        af = _make_ascii_frame(text, camera_id=1)
        measurements = vla._measure_from_ascii(af, "cam1")

        assert len(measurements) == 2

    def test_empty_frame(self):
        vla = VLAModel()
        text = "\n".join([" " * 120] * 40)
        af = _make_ascii_frame(text, camera_id=1)
        measurements = vla._measure_from_ascii(af, "cam1")
        assert len(measurements) == 0

    def test_cam0_height_measurement(self):
        """Front camera should measure height (Z) not depth."""
        vla = VLAModel()
        text = _make_object_ascii(width=120, height=40)
        af = _make_ascii_frame(text, camera_id=0)
        measurements = vla._measure_from_ascii(af, "cam0")

        assert len(measurements) == 1
        m = measurements[0]
        assert m.height_mm > 0  # front camera provides height


# ---------------------------------------------------------------------------
# Shape classification tests
# ---------------------------------------------------------------------------


class TestShapeClassification:
    def test_rectangular(self):
        vla = VLAModel()
        shape = vla._classify_shape(0.9, 20, 10)
        assert shape == ObjectShape.RECTANGULAR

    def test_cylindrical(self):
        vla = VLAModel()
        shape = vla._classify_shape(0.75, 10, 10)
        assert shape == ObjectShape.CYLINDRICAL

    def test_spherical(self):
        vla = VLAModel()
        shape = vla._classify_shape(0.6, 10, 10)
        assert shape == ObjectShape.SPHERICAL

    def test_irregular(self):
        vla = VLAModel()
        shape = vla._classify_shape(0.3, 20, 5)
        assert shape == ObjectShape.IRREGULAR


# ---------------------------------------------------------------------------
# 3D fusion tests
# ---------------------------------------------------------------------------


class TestFusion:
    def test_cam1_only_fusion(self):
        vla = VLAModel()
        text = _make_object_ascii()
        stereo = _make_stereo(cam1_text=text)
        result = vla._analyze_frame(stereo)

        assert len(result.objects) > 0
        obj = result.objects[0]
        assert obj.position_mm is not None
        assert obj.dimensions_mm is not None
        assert obj.source_camera == "cam1"

    def test_dual_camera_fusion(self):
        vla = VLAModel()
        cam1_text = _make_object_ascii()
        cam0_text = _make_object_ascii()  # same shape in front view
        stereo = _make_stereo(cam0_text=cam0_text, cam1_text=cam1_text)
        result = vla._analyze_frame(stereo)

        assert len(result.objects) > 0
        # At least one should have height info from cam0
        has_height = any(obj.dimensions_mm[1] > 0 for obj in result.objects)
        assert has_height

    def test_reachable_classification(self):
        vla = VLAModel()
        text = _make_object_ascii()
        stereo = _make_stereo(cam1_text=text)
        result = vla._analyze_frame(stereo)

        for obj in result.objects:
            assert isinstance(obj.reachable, bool)
            assert obj.reach_distance_mm >= 0


# ---------------------------------------------------------------------------
# Mesh generation tests
# ---------------------------------------------------------------------------


class TestMeshGeneration:
    def test_box_mesh(self):
        vla = VLAModel()
        center = np.array([100.0, 200.0, 50.0])
        dims = np.array([40.0, 30.0, 40.0])
        vertices, faces = vla._generate_box_mesh(center, dims)

        assert len(vertices) == 8
        assert len(faces) == 12
        # Each face is a triangle (3 indices)
        for face in faces:
            assert len(face) == 3


# ---------------------------------------------------------------------------
# Scene description tests
# ---------------------------------------------------------------------------


class TestSceneDescription:
    def test_no_objects(self):
        vla = VLAModel()
        desc = vla._describe_scene([])
        assert "No objects" in desc

    def test_with_objects(self):
        vla = VLAModel()
        obj = DetectedObject3D(
            object_id="test_1",
            label="test",
            position_mm=np.array([200.0, 100.0, 0.0]),
            dimensions_mm=np.array([40.0, 30.0, 40.0]),
            shape=ObjectShape.RECTANGULAR,
            confidence=0.8,
            reachable=True,
            reach_distance_mm=223.6,
        )
        desc = vla._describe_scene([obj])
        assert "1 objects" in desc or "Detected 1" in desc
        assert "reach" in desc.lower()


# ---------------------------------------------------------------------------
# Continuous loop tests
# ---------------------------------------------------------------------------


class TestVLAContinuousLoop:
    def test_start_stop(self):
        vla = VLAModel(analysis_fps=20.0)
        vla.start()
        assert vla.is_running is True
        time.sleep(0.1)
        vla.stop()
        assert vla.is_running is False

    def test_feed_and_analyze(self):
        vla = VLAModel(analysis_fps=20.0)
        results = []
        vla.on_result(lambda r: results.append(r))

        vla.start()

        text = _make_object_ascii()
        stereo = _make_stereo(cam1_text=text)
        vla.feed_frame(stereo)

        time.sleep(0.5)
        vla.stop()

        assert len(results) > 0
        assert isinstance(results[0], VLAAnalysisResult)
        assert results[0].frame_number > 0

    def test_get_latest_result(self):
        vla = VLAModel(analysis_fps=20.0)
        vla.start()

        text = _make_object_ascii()
        stereo = _make_stereo(cam1_text=text)
        vla.feed_frame(stereo)

        time.sleep(0.5)
        vla.stop()

        result = vla.get_latest_result()
        assert result is not None

    def test_get_tracked_objects(self):
        vla = VLAModel(analysis_fps=20.0)
        vla.start()

        text = _make_object_ascii()
        stereo = _make_stereo(cam1_text=text)
        vla.feed_frame(stereo)

        time.sleep(0.5)
        vla.stop()

        tracked = vla.get_tracked_objects()
        assert len(tracked) > 0

    def test_get_stats(self):
        vla = VLAModel()
        stats = vla.get_stats()
        assert stats["running"] is False
        assert stats["mode"] == "local"
        assert "scale" in stats

    def test_update_scale(self):
        vla = VLAModel()
        vla.update_scale(10.0, 20.0)
        assert vla.mm_per_col == 10.0
        assert vla.mm_per_row == 20.0
