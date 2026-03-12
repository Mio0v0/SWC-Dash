"""Atlas registration placeholder backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.models import FeatureResult
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "atlas_registration"
FEATURE = "registration"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "default",
    "atlas": {
        "name": "",
        "space": "",
    },
}


def _builtin_register(path: str, atlas_name: str | None, config: dict[str, Any]) -> FeatureResult:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)
    return FeatureResult(
        ok=False,
        message="Atlas registration is not implemented yet.",
        payload={
            "input_path": str(fp),
            "atlas_name": atlas_name or config.get("atlas", {}).get("name", ""),
            "config": config,
        },
    )


register_builtin_method(FEATURE_KEY, "default", _builtin_register)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def register_to_atlas(
    path: str,
    *,
    atlas_name: str | None = None,
    config_overrides: dict | None = None,
) -> FeatureResult:
    cfg = merge_config(get_config(), config_overrides)

    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(path, atlas_name, cfg)
