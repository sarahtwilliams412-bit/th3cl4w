"""
Shared test fixtures and configuration for th3cl4w test suite.

Handles optional dependencies (cyclonedds, cv2) by installing mock modules
before any test imports them, and provides skip markers for tests that
genuinely need the real libraries.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock optional C-library dependencies so test collection never fails.
# Individual tests that need the real library should use the skip markers below.
# ---------------------------------------------------------------------------

def _install_mock_module(name: str) -> types.ModuleType:
    """Create and register a MagicMock as a module if the real one isn't installed."""
    if name in sys.modules:
        return sys.modules[name]
    mock = MagicMock()
    mock.__name__ = name
    mock.__spec__ = None
    sys.modules[name] = mock
    return mock


def _ensure_cyclonedds_mocks():
    """Install cyclonedds mock modules if the real library is not available."""
    try:
        import cyclonedds  # noqa: F401
    except ImportError:
        for mod_name in [
            "cyclonedds",
            "cyclonedds.domain",
            "cyclonedds.idl",
            "cyclonedds.pub",
            "cyclonedds.sub",
            "cyclonedds.topic",
        ]:
            _install_mock_module(mod_name)

        # Provide stub classes used by d1_dds_connection at import time
        from dataclasses import dataclass

        @dataclass
        class _StubIdlStruct:
            """Minimal IdlStruct stub for class inheritance.

            The real IdlStruct accepts typename= in __init_subclass__.
            """
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

        sys.modules["cyclonedds.idl"].IdlStruct = _StubIdlStruct


# Run before any test collection
_ensure_cyclonedds_mocks()
# NOTE: We intentionally do NOT mock cv2. Tests that need cv2 use
# pytest.importorskip("cv2") to skip gracefully. The vision __init__.py
# already handles missing cv2 with try/except for production imports.


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_cyclonedds = pytest.mark.skipif(
    "cyclonedds" not in sys.modules or isinstance(sys.modules["cyclonedds"], MagicMock),
    reason="cyclonedds not installed",
)

requires_cv2 = pytest.mark.skipif(
    "cv2" not in sys.modules or isinstance(sys.modules["cv2"], MagicMock),
    reason="opencv-python (cv2) not installed",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry_db(tmp_path):
    """Create a temporary telemetry database path for isolated tests."""
    return str(tmp_path / "test_telemetry.db")
