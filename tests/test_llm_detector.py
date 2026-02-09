"""Tests for LLM-based joint detection via ASCII art."""

import asyncio
import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

# We need cv2 for creating test images
cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_jpeg(width=1920, height=1080) -> bytes:
    """Create a simple test JPEG image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a vertical white line to simulate an arm
    cv2.line(img, (960, 800), (960, 400), (255, 255, 255), 10)
    cv2.circle(img, (960, 400), 15, (0, 200, 255), -1)  # gold accent
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


MOCK_GEMINI_JSON = json.dumps(
    {
        "joints": [
            {"name": "base", "x": 0.5, "y": 0.9, "confidence": "high"},
            {"name": "shoulder", "x": 0.5, "y": 0.7, "confidence": "high"},
            {"name": "elbow", "x": 0.45, "y": 0.5, "confidence": "medium"},
            {"name": "wrist", "x": 0.4, "y": 0.35, "confidence": "medium"},
            {"name": "end_effector", "x": 0.35, "y": 0.2, "confidence": "low"},
        ]
    }
)


def _mock_usage():
    m = MagicMock()
    m.prompt_token_count = 1200
    m.candidates_token_count = 150
    return m


def _mock_response(text=MOCK_GEMINI_JSON):
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = _mock_usage()
    return resp


def _make_detector(**kwargs):
    """Create LLMJointDetector with mocked genai."""
    with patch("src.vision.llm_detector.genai") as mock_genai:
        mock_model_instance = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model_instance
        mock_genai.GenerationConfig = MagicMock()

        from src.vision.llm_detector import LLMJointDetector

        detector = LLMJointDetector(api_key="test-key", **kwargs)
        detector.model = mock_model_instance
        return detector, mock_model_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsciiConversion:
    def test_ascii_dimensions(self):
        """ASCII conversion at 80×35 produces correct dimensions."""
        from src.vision.ascii_converter import AsciiConverter, CHARSET_DETAILED

        conv = AsciiConverter(width=80, height=35, charset=CHARSET_DETAILED, invert=True)
        jpeg = _make_test_jpeg()
        ascii_text = conv.decode_jpeg_to_ascii(jpeg)
        lines = ascii_text.split("\n")
        assert len(lines) == 35
        assert all(len(line) == 80 for line in lines)


class TestPromptConstruction:
    def test_prompt_contains_required_elements(self):
        detector, _ = _make_detector()
        prompt = detector._build_prompt("FAKE_ASCII", camera_id=0)
        assert "robotic arm" in prompt.lower()
        assert "base" in prompt
        assert "shoulder" in prompt
        assert "elbow" in prompt
        assert "wrist" in prompt
        assert "end_effector" in prompt
        assert "0.0" in prompt and "1.0" in prompt  # normalized coords
        assert "FAKE_ASCII" in prompt
        assert "front view" in prompt

    def test_prompt_includes_joint_angles(self):
        detector, _ = _make_detector()
        angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        prompt = detector._build_prompt("ASCII", camera_id=0, joint_angles=angles)
        assert "10.0" in prompt
        assert "J0" in prompt

    def test_prompt_includes_fk_hints(self):
        detector, _ = _make_detector()
        hints = {"base": {"x": 0.5, "y": 0.9}, "elbow": {"x": 0.4, "y": 0.5}}
        prompt = detector._build_prompt("ASCII", camera_id=0, fk_hints=hints)
        assert "forward kinematics" in prompt.lower()
        assert "0.50" in prompt

    def test_overhead_camera_description(self):
        detector, _ = _make_detector()
        prompt = detector._build_prompt("ASCII", camera_id=1)
        assert "overhead" in prompt


class TestResponseParsing:
    def test_parse_valid_response(self):
        detector, _ = _make_detector()
        joints = detector._parse_response(MOCK_GEMINI_JSON, camera_id=0)
        assert len(joints) == 5
        assert joints[0].name == "base"
        assert joints[0].norm_x == 0.5
        assert joints[0].norm_y == 0.9
        assert joints[0].confidence == "high"

    def test_parse_clamps_coords(self):
        detector, _ = _make_detector()
        bad_json = json.dumps(
            {"joints": [{"name": "base", "x": -0.1, "y": 1.5, "confidence": "low"}]}
        )
        joints = detector._parse_response(bad_json, camera_id=0)
        assert joints[0].norm_x == 0.0
        assert joints[0].norm_y == 1.0

    def test_parse_skips_null_coords(self):
        detector, _ = _make_detector()
        data = json.dumps(
            {
                "joints": [
                    {"name": "base", "x": 0.5, "y": None, "confidence": "low"},
                    {"name": "elbow", "x": 0.3, "y": 0.4, "confidence": "medium"},
                ]
            }
        )
        joints = detector._parse_response(data, camera_id=0)
        assert len(joints) == 1
        assert joints[0].name == "elbow"

    def test_parse_malformed_json_raises(self):
        detector, _ = _make_detector()
        with pytest.raises(json.JSONDecodeError):
            detector._parse_response("not json at all", camera_id=0)

    def test_parse_strips_markdown_fences(self):
        detector, _ = _make_detector()
        fenced = f"```json\n{MOCK_GEMINI_JSON}\n```"
        joints = detector._parse_response(fenced, camera_id=0)
        assert len(joints) == 5

    def test_parse_empty_joints(self):
        detector, _ = _make_detector()
        joints = detector._parse_response('{"joints": []}', camera_id=0)
        assert joints == []


class TestCoordinateScaling:
    def test_default_1920x1080(self):
        detector, _ = _make_detector()
        joints = detector._parse_response(MOCK_GEMINI_JSON, camera_id=0)
        base = joints[0]  # x=0.5, y=0.9
        assert base.pixel_x == int(0.5 * 1920)  # 960
        assert base.pixel_y == int(0.9 * 1080)  # 972

    def test_custom_resolution(self):
        detector, _ = _make_detector(camera_width=1280, camera_height=720)
        data = json.dumps(
            {"joints": [{"name": "base", "x": 0.25, "y": 0.75, "confidence": "high"}]}
        )
        joints = detector._parse_response(data, camera_id=0)
        assert joints[0].pixel_x == int(0.25 * 1280)  # 320
        assert joints[0].pixel_y == int(0.75 * 720)  # 540


class TestDetectJoints:
    def test_successful_detection(self):
        detector, mock_model = _make_detector()
        mock_model.generate_content.return_value = _mock_response()
        jpeg = _make_test_jpeg()

        result = asyncio.run(detector.detect_joints(jpeg, camera_id=0))
        assert result.success is True
        assert len(result.joints) == 5
        assert result.camera_id == 0
        assert result.tokens_used == 1350
        assert result.latency_ms > 0
        assert result.error is None

    def test_api_error_returns_failure(self):
        detector, mock_model = _make_detector()
        mock_model.generate_content.side_effect = RuntimeError("API timeout")
        jpeg = _make_test_jpeg()

        result = asyncio.run(detector.detect_joints(jpeg, camera_id=0))
        assert result.success is False
        assert "API error" in result.error
        assert result.joints == []

    def test_malformed_response_returns_failure(self):
        detector, mock_model = _make_detector()
        mock_model.generate_content.return_value = _mock_response("garbage response")
        jpeg = _make_test_jpeg()

        result = asyncio.run(detector.detect_joints(jpeg, camera_id=0))
        assert result.success is False
        assert "Parse error" in result.error

    def test_bad_jpeg_returns_failure(self):
        detector, mock_model = _make_detector()

        result = asyncio.run(detector.detect_joints(b"not a jpeg", camera_id=0))
        assert result.success is False
        assert "ASCII conversion failed" in result.error

    def test_token_tracking(self):
        detector, mock_model = _make_detector()
        mock_model.generate_content.return_value = _mock_response()
        jpeg = _make_test_jpeg()

        asyncio.run(detector.detect_joints(jpeg, 0))
        asyncio.run(detector.detect_joints(jpeg, 1))
        assert detector.total_calls == 2
        assert detector.total_tokens == 2700


class TestBatchDetection:
    def test_batch_concurrent(self):
        detector, mock_model = _make_detector()
        mock_model.generate_content.return_value = _mock_response()
        jpeg = _make_test_jpeg()

        frames = [
            {"jpeg_bytes": jpeg, "camera_id": 0},
            {"jpeg_bytes": jpeg, "camera_id": 1, "joint_angles": [0.0] * 6},
        ]
        results = asyncio.run(detector.detect_joints_batch(frames))
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].camera_id == 0
        assert results[1].camera_id == 1


class TestDataclasses:
    def test_llm_detection_result_fields(self):
        from src.vision.llm_detector import LLMDetectionResult, LLMJointPosition

        jp = LLMJointPosition("base", 0.5, 0.9, 960, 972, "high")
        result = LLMDetectionResult(
            joints=[jp],
            camera_id=0,
            model="test",
            tokens_used=100,
            latency_ms=500.0,
            raw_response="{}",
            success=True,
        )
        assert result.error is None
        d = asdict(result)
        assert d["joints"][0]["name"] == "base"
