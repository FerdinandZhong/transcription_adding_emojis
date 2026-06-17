"""YAML config loading with a deep-merge over built-in defaults."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.yaml")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = None) -> Dict[str, Any]:
    default_path = os.path.normpath(_DEFAULT_PATH)
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)
    if path and os.path.normpath(path) != default_path and os.path.exists(path):
        with open(path, "r") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
