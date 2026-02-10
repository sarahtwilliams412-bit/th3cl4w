"""Tests for pick_config module."""

import json
import copy
import pytest
from pathlib import Path
from unittest.mock import patch


def _fresh_config(tmp_path):
    """Create a fresh PickConfig pointing at tmp_path."""
    import src.config.pick_config as mod

    # Reset singleton
    mod.PickConfig._instance = None
    mod._CONFIG_DIR = tmp_path
    mod._CONFIG_FILE = tmp_path / "pick_config.json"
    from src.config.pick_config import get_pick_config

    return get_pick_config()


def test_defaults_loaded(tmp_path):
    cfg = _fresh_config(tmp_path)
    assert cfg.get("reach", "max_mm") == 550.0
    assert cfg.get("gripper", "open_mm") == 65.0
    assert cfg.get("safety", "torque_proxy_limit") == 150.0


def test_set_and_get(tmp_path):
    cfg = _fresh_config(tmp_path)
    cfg.set("reach", "max_mm", 600.0)
    assert cfg.get("reach", "max_mm") == 600.0
    # Verify persisted
    saved = json.loads((tmp_path / "pick_config.json").read_text())
    assert saved["reach"]["max_mm"] == 600.0


def test_update_deep_merge(tmp_path):
    cfg = _fresh_config(tmp_path)
    cfg.update({"reach": {"max_mm": 700.0}})
    assert cfg.get("reach", "max_mm") == 700.0
    # Other reach values preserved
    assert cfg.get("reach", "safe_mm") == 500.0


def test_reset(tmp_path):
    cfg = _fresh_config(tmp_path)
    cfg.set("reach", "max_mm", 999.0)
    cfg.reset()
    assert cfg.get("reach", "max_mm") == 550.0


def test_diff(tmp_path):
    cfg = _fresh_config(tmp_path)
    assert cfg.diff() == {}
    cfg.set("servo", "kp", 0.99)
    d = cfg.diff()
    assert d["servo"]["kp"] == 0.99


def test_get_section(tmp_path):
    cfg = _fresh_config(tmp_path)
    reach = cfg.get("reach")
    assert isinstance(reach, dict)
    assert "max_mm" in reach


def test_get_all(tmp_path):
    cfg = _fresh_config(tmp_path)
    all_cfg = cfg.get_all()
    assert "detection" in all_cfg
    assert "reach" in all_cfg


def test_load_from_existing_file(tmp_path):
    # Write a partial config
    (tmp_path / "pick_config.json").write_text(json.dumps({"reach": {"max_mm": 123.0}}))
    cfg = _fresh_config(tmp_path)
    assert cfg.get("reach", "max_mm") == 123.0
    # Defaults for unspecified values preserved
    assert cfg.get("reach", "safe_mm") == 500.0
