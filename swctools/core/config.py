"""Feature config JSON helpers.

Each feature keeps a JSON file under:
  swctools/tools/<tool>/configs/<feature>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def feature_config_path(tool: str, feature: str) -> Path:
    return TOOLS_DIR / tool / "configs" / f"{feature}.json"


def load_feature_config(tool: str, feature: str, default: dict | None = None) -> dict[str, Any]:
    path = feature_config_path(tool, feature)
    if not path.exists():
        return dict(default or {})
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default or {})


def save_feature_config(tool: str, feature: str, config: dict[str, Any]) -> Path:
    path = feature_config_path(tool, feature)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def merge_config(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating inputs."""
    if not overrides:
        return dict(base)

    out: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = merge_config(current, value)
        else:
            out[key] = value
    return out
