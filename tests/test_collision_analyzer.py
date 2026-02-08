"""Tests for CollisionAnalyzer — mock camera responses and vision fallback."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.vision.collision_analyzer import CollisionAnalyzer, CollisionAnalysis, DATA_DIR


class TestCollisionAnalyzer:
    def test_analyze_without_vision_key(self, tmp_path, monkeypatch):
        """Without API key, should save images and return fallback text."""
        monkeypatch.setattr("src.vision.collision_analyzer.DATA_DIR", tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        analyzer = CollisionAnalyzer(gemini_api_key=None)
        assert not analyzer.vision_available

        # Mock camera responses
        fake_jpg = b"\xff\xd8\xff\xe0fake_jpeg_data"
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = fake_jpg
            mock_client.get.return_value = mock_resp

            result = analyzer.analyze(joint_id=2, commanded_deg=45.0, actual_deg=20.0)

        assert isinstance(result, CollisionAnalysis)
        assert "manual review" in result.analysis_text
        assert not result.vision_available
        assert result.timestamp != ""

        # Check files were saved
        ts_dirs = list(tmp_path.iterdir())
        assert len(ts_dirs) == 1
        saved_dir = ts_dirs[0]
        assert (saved_dir / "cam0.jpg").exists()
        assert (saved_dir / "cam1.jpg").exists()
        assert (saved_dir / "analysis.json").exists()

        data = json.loads((saved_dir / "analysis.json").read_text())
        assert data["joint_id"] == 2
        assert data["commanded_deg"] == 45.0
        assert data["actual_deg"] == 20.0

    def test_analyze_camera_failure(self, tmp_path, monkeypatch):
        """Should handle camera failure gracefully."""
        monkeypatch.setattr("src.vision.collision_analyzer.DATA_DIR", tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        analyzer = CollisionAnalyzer(gemini_api_key=None)

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = Exception("Connection refused")

            result = analyzer.analyze(joint_id=1, commanded_deg=30.0, actual_deg=10.0)

        assert isinstance(result, CollisionAnalysis)
        assert result.cam0_path is None
        assert result.cam1_path is None

    def test_analyze_with_mock_gemini(self, tmp_path, monkeypatch):
        """Test with mocked Gemini model."""
        monkeypatch.setattr("src.vision.collision_analyzer.DATA_DIR", tmp_path)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        analyzer = CollisionAnalyzer(gemini_api_key=None)
        # Manually inject a mock model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The arm is hitting a cardboard box on the left side."
        mock_model.generate_content.return_value = mock_response
        analyzer._model = mock_model

        fake_jpg = b"\xff\xd8\xff\xe0fake_jpeg_data"
        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = fake_jpg
            mock_client.get.return_value = mock_resp

            result = analyzer.analyze(joint_id=3, commanded_deg=60.0, actual_deg=25.0)

        assert result.vision_available
        assert "cardboard box" in result.analysis_text
        mock_model.generate_content.assert_called_once()
