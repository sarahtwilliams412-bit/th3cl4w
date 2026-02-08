"""
Tests for the vision task planning pipeline.

Tests SceneAnalyzer (scene understanding) and VisionTaskPlanner
(instruction parsing, object matching, trajectory generation).

Uses synthetic images — no real cameras required.
"""

import sys
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python (cv2) not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.scene_analyzer import (
    SceneAnalyzer,
    SceneDescription,
    SceneObject,
    SpatialRelation,
)
from src.vision.object_detection import ObjectDetector, ColorRange
from src.planning.vision_task_planner import (
    VisionTaskPlanner,
    VisionTaskPlan,
    ActionType,
)
from src.planning.task_planner import TaskPlanner, TaskStatus
from src.planning.motion_planner import MotionPlanner, NUM_ARM_JOINTS


# ======================================================================
# Test helpers
# ======================================================================


def make_colored_circle_image(
    color_bgr: tuple = (0, 0, 255),
    center: tuple = (320, 240),
    radius: int = 50,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Create a test image with a colored circle on gray background."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    cv2.circle(img, center, radius, color_bgr, -1)
    return img


def make_multi_object_image() -> np.ndarray:
    """Create an image with a red circle on the left and blue on the right."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(img, (160, 240), 50, (0, 0, 255), -1)  # red on left
    cv2.circle(img, (480, 240), 50, (255, 0, 0), -1)  # blue on right
    return img


def make_three_object_image() -> np.ndarray:
    """Create an image with red (left), green (center), blue (right)."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.circle(img, (100, 240), 40, (0, 0, 255), -1)  # red left
    cv2.circle(img, (320, 150), 60, (0, 255, 0), -1)  # green center-top
    cv2.circle(img, (540, 350), 45, (255, 0, 0), -1)  # blue bottom-right
    return img


# ======================================================================
# SceneAnalyzer Tests
# ======================================================================


class TestSceneAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SceneAnalyzer(detector=ObjectDetector(min_area=100))

    def test_init_defaults(self):
        sa = SceneAnalyzer()
        assert sa.detector is not None
        assert sa.depth_near == 300.0
        assert sa.depth_far == 800.0

    def test_analyze_empty_scene(self, analyzer):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        scene = analyzer.analyze(img)
        assert scene.object_count == 0
        assert not scene.has_objects
        assert "No objects" in scene.summary

    def test_analyze_single_red_object(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255), center=(320, 240))
        scene = analyzer.analyze(img)
        assert scene.object_count >= 1
        assert scene.has_objects
        obj = scene.objects[0]
        assert obj.color == "red"
        assert obj.region == "center"
        assert 0.4 < obj.normalized_x < 0.6
        assert 0.4 < obj.normalized_y < 0.6

    def test_analyze_object_left_side(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255), center=(80, 240))
        scene = analyzer.analyze(img)
        assert scene.object_count >= 1
        obj = scene.objects[0]
        assert obj.normalized_x < 0.33
        assert "left" in obj.region

    def test_analyze_object_right_side(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255), center=(560, 240))
        scene = analyzer.analyze(img)
        assert scene.object_count >= 1
        obj = scene.objects[0]
        assert obj.normalized_x > 0.66
        assert "right" in obj.region

    def test_analyze_object_top(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255), center=(320, 80))
        scene = analyzer.analyze(img)
        assert scene.object_count >= 1
        obj = scene.objects[0]
        assert obj.normalized_y < 0.33
        assert "top" in obj.region

    def test_analyze_multiple_objects(self, analyzer):
        img = make_multi_object_image()
        scene = analyzer.analyze(img)
        assert scene.object_count >= 2
        colors = {obj.color for obj in scene.objects}
        assert "red" in colors
        assert "blue" in colors

    def test_spatial_relationships(self, analyzer):
        img = make_multi_object_image()
        scene = analyzer.analyze(img)
        assert len(scene.relationships) > 0
        # Red is on the left, blue on the right — should have left_of relationship
        has_left_of = any(
            r.relation == SpatialRelation.LEFT_OF for r in scene.relationships
        )
        has_right_of = any(
            r.relation == SpatialRelation.RIGHT_OF for r in scene.relationships
        )
        assert has_left_of or has_right_of

    def test_scene_objects_by_color(self, analyzer):
        img = make_multi_object_image()
        scene = analyzer.analyze(img)
        reds = scene.objects_by_color("red")
        assert len(reds) >= 1
        blues = scene.objects_by_color("blue")
        assert len(blues) >= 1

    def test_scene_largest_object(self, analyzer):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.circle(img, (200, 240), 30, (0, 0, 255), -1)  # small red
        cv2.circle(img, (400, 240), 80, (0, 255, 0), -1)  # big green
        scene = analyzer.analyze(img)
        largest = scene.largest_object()
        assert largest is not None
        assert largest.color == "green"

    def test_scene_leftmost_rightmost(self, analyzer):
        img = make_multi_object_image()
        scene = analyzer.analyze(img)
        leftmost = scene.leftmost_object()
        rightmost = scene.rightmost_object()
        assert leftmost is not None
        assert rightmost is not None
        assert leftmost.normalized_x < rightmost.normalized_x

    def test_scene_to_dict(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255))
        scene = analyzer.analyze(img)
        d = scene.to_dict()
        assert "objects" in d
        assert "relationships" in d
        assert "summary" in d
        assert "frame_size" in d
        assert d["object_count"] >= 1

    def test_scene_summary_content(self, analyzer):
        img = make_three_object_image()
        scene = analyzer.analyze(img)
        assert "red" in scene.summary.lower()
        assert "object" in scene.summary.lower()

    def test_classify_region_center(self):
        sa = SceneAnalyzer()
        assert sa._classify_region(0.5, 0.5) == "center"

    def test_classify_region_top_left(self):
        sa = SceneAnalyzer()
        assert sa._classify_region(0.1, 0.1) == "top-left"

    def test_classify_region_bottom_right(self):
        sa = SceneAnalyzer()
        assert sa._classify_region(0.9, 0.9) == "bottom-right"

    def test_classify_region_left(self):
        sa = SceneAnalyzer()
        assert sa._classify_region(0.1, 0.5) == "left"

    def test_classify_region_right(self):
        sa = SceneAnalyzer()
        assert sa._classify_region(0.9, 0.5) == "right"

    def test_annotate_frame(self, analyzer):
        img = make_colored_circle_image(color_bgr=(0, 0, 255))
        scene = analyzer.analyze(img)
        vis = analyzer.annotate_frame(img, scene)
        assert vis.shape == img.shape
        # Should be modified (has annotations)
        if scene.object_count > 0:
            assert not np.array_equal(vis, img)

    def test_size_category(self):
        obj = SceneObject(
            label="test",
            color="red",
            centroid_2d=(320, 240),
            centroid_3d=None,
            bbox=(300, 220, 40, 40),
            area=1000.0,
            depth_mm=0.0,
            confidence=0.5,
            normalized_x=0.5,
            normalized_y=0.5,
            region="center",
        )
        assert obj.size_category == "small"

        obj.area = 5000.0
        assert obj.size_category == "medium"

        obj.area = 20000.0
        assert obj.size_category == "large"


# ======================================================================
# VisionTaskPlanner — Instruction Parsing Tests
# ======================================================================


class TestInstructionParsing:
    @pytest.fixture
    def planner(self):
        return VisionTaskPlanner()

    def test_parse_pick_up(self, planner):
        action, _ = planner._parse_action("pick up the red object")
        assert action == ActionType.PICK_AND_PLACE

    def test_parse_grab(self, planner):
        action, _ = planner._parse_action("grab the blue block")
        assert action == ActionType.PICK_AND_PLACE

    def test_parse_place_on(self, planner):
        action, _ = planner._parse_action("move the red block to the green area")
        assert action == ActionType.PICK_AND_PLACE

    def test_parse_pour(self, planner):
        action, _ = planner._parse_action("pour the contents")
        assert action == ActionType.POUR

    def test_parse_wave(self, planner):
        action, _ = planner._parse_action("wave hello")
        assert action == ActionType.WAVE

    def test_parse_wave_bye(self, planner):
        action, _ = planner._parse_action("say goodbye")
        assert action == ActionType.WAVE

    def test_parse_point_at(self, planner):
        action, _ = planner._parse_action("point at the red object")
        assert action == ActionType.POINT_AT

    def test_parse_go_home(self, planner):
        action, _ = planner._parse_action("go home")
        assert action == ActionType.GO_HOME

    def test_parse_ready(self, planner):
        action, _ = planner._parse_action("go to ready position")
        assert action == ActionType.GO_READY

    def test_parse_inspect(self, planner):
        action, _ = planner._parse_action("look at the object")
        assert action == ActionType.INSPECT

    def test_parse_push(self, planner):
        action, _ = planner._parse_action("push the red block")
        assert action == ActionType.PUSH

    def test_parse_unknown(self, planner):
        action, _ = planner._parse_action("do something weird")
        assert action == ActionType.UNKNOWN

    def test_parse_case_insensitive(self, planner):
        action, _ = planner._parse_action("PICK UP the RED object")
        assert action == ActionType.PICK_AND_PLACE


# ======================================================================
# VisionTaskPlanner — Object Matching Tests
# ======================================================================


class TestObjectMatching:
    @pytest.fixture
    def planner(self):
        return VisionTaskPlanner()

    @pytest.fixture
    def scene_two_objects(self):
        """Scene with red object on left, blue on right."""
        return SceneDescription(
            objects=[
                SceneObject(
                    label="red",
                    color="red",
                    centroid_2d=(160, 240),
                    centroid_3d=None,
                    bbox=(110, 190, 100, 100),
                    area=7854.0,
                    depth_mm=500.0,
                    confidence=0.5,
                    normalized_x=0.25,
                    normalized_y=0.5,
                    region="left",
                ),
                SceneObject(
                    label="blue",
                    color="blue",
                    centroid_2d=(480, 240),
                    centroid_3d=None,
                    bbox=(430, 190, 100, 100),
                    area=7854.0,
                    depth_mm=600.0,
                    confidence=0.5,
                    normalized_x=0.75,
                    normalized_y=0.5,
                    region="right",
                ),
            ],
            frame_width=640,
            frame_height=480,
        )

    def test_match_by_color_red(self, planner, scene_two_objects):
        target, detail = planner._match_target(
            "pick up the red object", scene_two_objects
        )
        assert target is not None
        assert target.color == "red"

    def test_match_by_color_blue(self, planner, scene_two_objects):
        target, detail = planner._match_target(
            "grab the blue thing", scene_two_objects
        )
        assert target is not None
        assert target.color == "blue"

    def test_match_by_position_left(self, planner, scene_two_objects):
        target, detail = planner._match_target(
            "pick up the left object", scene_two_objects
        )
        assert target is not None
        assert target.region == "left"

    def test_match_by_position_right(self, planner, scene_two_objects):
        target, detail = planner._match_target(
            "grab the right one", scene_two_objects
        )
        assert target is not None
        assert target.region == "right"

    def test_match_nearest(self, planner, scene_two_objects):
        target, detail = planner._match_target(
            "pick up the nearest object", scene_two_objects
        )
        assert target is not None
        # Red is at 500mm, blue at 600mm — red is nearest
        assert target.color == "red"

    def test_match_default_to_largest(self, planner):
        scene = SceneDescription(
            objects=[
                SceneObject(
                    label="green",
                    color="green",
                    centroid_2d=(320, 240),
                    centroid_3d=None,
                    bbox=(270, 190, 100, 100),
                    area=5000.0,
                    depth_mm=0.0,
                    confidence=0.5,
                    normalized_x=0.5,
                    normalized_y=0.5,
                    region="center",
                ),
            ]
        )
        target, detail = planner._match_target("pick up something", scene)
        assert target is not None
        assert target.color == "green"

    def test_match_empty_scene(self, planner):
        scene = SceneDescription()
        target, detail = planner._match_target("pick up the red object", scene)
        assert target is None

    def test_match_destination_by_color(self, planner, scene_two_objects):
        dest, detail = planner._match_destination(
            "move to the blue object",
            scene_two_objects,
            scene_two_objects.objects[0],  # red is target
        )
        assert dest is not None
        assert dest.color == "blue"

    def test_match_destination_by_position(self, planner, scene_two_objects):
        dest, detail = planner._match_destination(
            "put it on the right side",
            scene_two_objects,
            scene_two_objects.objects[0],
        )
        # Should recognize "right" as a position keyword
        assert "right" in detail.lower()


# ======================================================================
# VisionTaskPlanner — Full Plan Tests
# ======================================================================


class TestVisionTaskPlan:
    @pytest.fixture
    def planner(self):
        return VisionTaskPlanner()

    @pytest.fixture
    def current_pose(self):
        return np.zeros(NUM_ARM_JOINTS)

    @pytest.fixture
    def scene_with_red(self):
        """Scene with a single red object in the center."""
        return SceneDescription(
            objects=[
                SceneObject(
                    label="red",
                    color="red",
                    centroid_2d=(320, 240),
                    centroid_3d=None,
                    bbox=(270, 190, 100, 100),
                    area=7854.0,
                    depth_mm=400.0,
                    confidence=0.7,
                    normalized_x=0.5,
                    normalized_y=0.5,
                    region="center",
                ),
            ],
            frame_width=640,
            frame_height=480,
            summary="Scene contains 1 red object.",
        )

    @pytest.fixture
    def scene_two_objects(self):
        return SceneDescription(
            objects=[
                SceneObject(
                    label="red",
                    color="red",
                    centroid_2d=(160, 240),
                    centroid_3d=None,
                    bbox=(110, 190, 100, 100),
                    area=7854.0,
                    depth_mm=500.0,
                    confidence=0.5,
                    normalized_x=0.25,
                    normalized_y=0.5,
                    region="left",
                ),
                SceneObject(
                    label="blue",
                    color="blue",
                    centroid_2d=(480, 240),
                    centroid_3d=None,
                    bbox=(430, 190, 100, 100),
                    area=7854.0,
                    depth_mm=600.0,
                    confidence=0.5,
                    normalized_x=0.75,
                    normalized_y=0.5,
                    region="right",
                ),
            ],
            frame_width=640,
            frame_height=480,
            summary="Scene contains 1 red object, 1 blue object.",
        )

    def test_plan_pick_up_red(self, planner, current_pose, scene_with_red):
        plan = planner.plan("pick up the red object", scene_with_red, current_pose)
        assert plan.action == ActionType.PICK_AND_PLACE
        assert plan.success
        assert plan.target_object is not None
        assert plan.target_object["color"] == "red"
        assert plan.trajectory is not None
        assert plan.trajectory.num_points > 0

    def test_plan_has_reasoning_steps(self, planner, current_pose, scene_with_red):
        plan = planner.plan("pick up the red object", scene_with_red, current_pose)
        assert len(plan.reasoning) >= 4
        # Check that reasoning steps are numbered
        for i, step in enumerate(plan.reasoning):
            assert step.step == i + 1
            assert step.description != ""

    def test_plan_wave(self, planner, current_pose, scene_with_red):
        plan = planner.plan("wave hello", scene_with_red, current_pose)
        assert plan.action == ActionType.WAVE
        assert plan.success
        assert plan.trajectory is not None
        assert plan.trajectory.num_points > 0

    def test_plan_go_home(self, planner, current_pose, scene_with_red):
        plan = planner.plan("go home", scene_with_red, current_pose)
        assert plan.action == ActionType.GO_HOME
        assert plan.success

    def test_plan_pour(self, planner, current_pose, scene_with_red):
        plan = planner.plan("pour it out", scene_with_red, current_pose)
        assert plan.action == ActionType.POUR
        assert plan.success
        assert plan.trajectory is not None

    def test_plan_point_at(self, planner, current_pose, scene_with_red):
        plan = planner.plan("point at the red object", scene_with_red, current_pose)
        assert plan.action == ActionType.POINT_AT
        assert plan.success
        assert plan.trajectory is not None

    def test_plan_push(self, planner, current_pose, scene_with_red):
        plan = planner.plan("push the red block", scene_with_red, current_pose)
        assert plan.action == ActionType.PUSH
        assert plan.success
        assert plan.trajectory is not None

    def test_plan_inspect(self, planner, current_pose, scene_with_red):
        plan = planner.plan("look at the red thing", scene_with_red, current_pose)
        assert plan.action == ActionType.INSPECT
        assert plan.success

    def test_plan_pick_and_place_two_objects(
        self, planner, current_pose, scene_two_objects
    ):
        plan = planner.plan(
            "pick up the red object and place it near the blue one",
            scene_two_objects,
            current_pose,
        )
        assert plan.action == ActionType.PICK_AND_PLACE
        assert plan.success
        assert plan.target_object is not None
        assert plan.target_object["color"] == "red"

    def test_plan_empty_scene_pick(self, planner, current_pose):
        empty_scene = SceneDescription()
        plan = planner.plan("pick up the red object", empty_scene, current_pose)
        assert plan.action == ActionType.PICK_AND_PLACE
        assert not plan.success

    def test_plan_empty_scene_wave(self, planner, current_pose):
        """Wave should still work even with empty scene."""
        empty_scene = SceneDescription()
        plan = planner.plan("wave hello", empty_scene, current_pose)
        assert plan.action == ActionType.WAVE
        assert plan.success

    def test_plan_unknown_action(self, planner, current_pose, scene_with_red):
        plan = planner.plan("do something impossible", scene_with_red, current_pose)
        assert plan.action == ActionType.UNKNOWN
        # Unknown defaults to go_ready, which should succeed
        assert plan.success

    def test_plan_to_dict(self, planner, current_pose, scene_with_red):
        plan = planner.plan("pick up the red object", scene_with_red, current_pose)
        d = plan.to_dict()
        assert "instruction" in d
        assert "action" in d
        assert "reasoning" in d
        assert "success" in d
        assert d["action"] == "pick_and_place"
        assert len(d["reasoning"]) >= 4

    def test_plan_trajectory_has_duration(self, planner, current_pose, scene_with_red):
        plan = planner.plan("pick up the red object", scene_with_red, current_pose)
        assert plan.success
        assert plan.trajectory.duration > 0

    def test_scene_to_joint_pose(self, planner):
        """Test that scene-to-joint mapping produces valid poses."""
        obj = SceneObject(
            label="test",
            color="red",
            centroid_2d=(320, 240),
            centroid_3d=None,
            bbox=(270, 190, 100, 100),
            area=5000.0,
            depth_mm=0.0,
            confidence=0.5,
            normalized_x=0.5,
            normalized_y=0.5,
            region="center",
        )
        pose = planner._scene_to_joint_pose(obj)
        assert pose.shape == (NUM_ARM_JOINTS,)
        # Should be within joint limits
        from src.planning.motion_planner import JOINT_LIMITS_DEG

        for i in range(NUM_ARM_JOINTS):
            assert JOINT_LIMITS_DEG[i, 0] <= pose[i] <= JOINT_LIMITS_DEG[i, 1]

    def test_scene_to_joint_pose_varies_with_position(self, planner):
        """Objects at different positions should map to different poses."""
        left_obj = SceneObject(
            label="test",
            color="red",
            centroid_2d=(100, 240),
            centroid_3d=None,
            bbox=(50, 190, 100, 100),
            area=5000.0,
            depth_mm=0.0,
            confidence=0.5,
            normalized_x=0.15,
            normalized_y=0.5,
            region="left",
        )
        right_obj = SceneObject(
            label="test",
            color="blue",
            centroid_2d=(540, 240),
            centroid_3d=None,
            bbox=(490, 190, 100, 100),
            area=5000.0,
            depth_mm=0.0,
            confidence=0.5,
            normalized_x=0.85,
            normalized_y=0.5,
            region="right",
        )
        left_pose = planner._scene_to_joint_pose(left_obj)
        right_pose = planner._scene_to_joint_pose(right_obj)
        # J0 (base yaw) should differ significantly
        assert abs(left_pose[0] - right_pose[0]) > 10.0


# ======================================================================
# Integration Tests
# ======================================================================


class TestIntegration:
    def test_full_pipeline_analyze_and_plan(self):
        """Full pipeline: image -> scene analysis -> plan generation."""
        analyzer = SceneAnalyzer(detector=ObjectDetector(min_area=100))
        planner = VisionTaskPlanner()

        img = make_multi_object_image()
        scene = analyzer.analyze(img)
        assert scene.object_count >= 2

        current = np.zeros(NUM_ARM_JOINTS)
        plan = planner.plan("pick up the red object", scene, current)

        assert plan.action == ActionType.PICK_AND_PLACE
        assert plan.success
        assert plan.target_object is not None
        assert plan.target_object["color"] == "red"
        assert len(plan.reasoning) >= 4
        assert plan.trajectory.num_points > 0
        assert plan.trajectory.duration > 0

    def test_full_pipeline_three_objects(self):
        """Pipeline with three objects and specific instruction."""
        analyzer = SceneAnalyzer(detector=ObjectDetector(min_area=100))
        planner = VisionTaskPlanner()

        img = make_three_object_image()
        scene = analyzer.analyze(img)
        assert scene.object_count >= 2  # at least red and green

        current = np.zeros(NUM_ARM_JOINTS)
        plan = planner.plan(
            "pick up the green object and place it near the blue one",
            scene,
            current,
        )
        assert plan.success

    def test_full_pipeline_serialization(self):
        """Ensure the full plan serializes cleanly to dict."""
        analyzer = SceneAnalyzer(detector=ObjectDetector(min_area=100))
        planner = VisionTaskPlanner()

        img = make_colored_circle_image(color_bgr=(0, 0, 255))
        scene = analyzer.analyze(img)
        current = np.zeros(NUM_ARM_JOINTS)
        plan = planner.plan("pick up the red object", scene, current)

        plan_dict = plan.to_dict()
        scene_dict = scene.to_dict()

        # Should be JSON-serializable
        import json

        json.dumps(plan_dict)
        json.dumps(scene_dict)

    def test_inspect_then_pick(self):
        """Two-step: inspect first, then pick."""
        analyzer = SceneAnalyzer(detector=ObjectDetector(min_area=100))
        planner = VisionTaskPlanner()

        img = make_colored_circle_image(color_bgr=(0, 255, 0), center=(320, 240))
        scene = analyzer.analyze(img)
        current = np.zeros(NUM_ARM_JOINTS)

        inspect_plan = planner.plan("look at the green object", scene, current)
        assert inspect_plan.success

        # After inspection, plan a pick from the new position
        new_pose = inspect_plan.trajectory.points[-1].positions
        pick_plan = planner.plan("pick up the green object", scene, new_pose)
        assert pick_plan.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
