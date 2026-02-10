"""Test that python-dotenv loading works correctly."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv


def test_dotenv_loads_env_file():
    """Verify load_dotenv reads a .env file into os.environ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text("TEST_DOTENV_VAR=hello_from_dotenv\n")
        # Ensure not already set
        os.environ.pop("TEST_DOTENV_VAR", None)
        load_dotenv(env_path)
        assert os.environ.get("TEST_DOTENV_VAR") == "hello_from_dotenv"
        # Cleanup
        os.environ.pop("TEST_DOTENV_VAR", None)


def test_dotenv_does_not_override_existing():
    """Verify load_dotenv does NOT override already-set env vars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text("TEST_DOTENV_EXISTING=from_file\n")
        os.environ["TEST_DOTENV_EXISTING"] = "from_env"
        load_dotenv(env_path)
        assert os.environ.get("TEST_DOTENV_EXISTING") == "from_env"
        os.environ.pop("TEST_DOTENV_EXISTING", None)


def test_secrets_status_logic():
    """Test the secrets-status check logic."""

    # Simulate the check from the endpoint
    def check(val):
        return bool(val and val != "your-gemini-api-key-here")

    assert check(None) is False
    assert check("") is False
    assert check("your-gemini-api-key-here") is False
    assert check("AIzaSy_real_key") is True
