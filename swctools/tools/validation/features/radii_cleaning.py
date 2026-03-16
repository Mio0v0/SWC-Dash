"""Validation radii-cleaning feature.

This is a thin wrapper over the shared Batch Processing radii-clean backend so
GUI/CLI/Validation all use exactly the same cleaning implementation.
"""

from __future__ import annotations

from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.plugins.registry import register_builtin_method, resolve_method
from swctools.tools.batch_processing.features.radii_cleaning import (
    clean_file as shared_clean_file,
    clean_folder as shared_clean_folder,
    clean_path as shared_clean_path,
)

TOOL = "validation"
FEATURE = "radii_cleaning"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "shared",
    "config_overrides": {},
}


def _builtin_shared_clean(path: str, config: dict[str, Any]) -> dict[str, Any]:
    cfg_overrides = dict(config.get("config_overrides", {}))
    return shared_clean_path(path, config_overrides=cfg_overrides)


register_builtin_method(FEATURE_KEY, "shared", _builtin_shared_clean)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def clean_path(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)
    method = str(cfg.get("method", "shared"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(path, cfg)


def clean_file(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    # Keep helper for callers that require explicit file-only contract.
    cfg = dict(config_overrides or {})
    return shared_clean_file(path, config_overrides=cfg)


def clean_folder(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = dict(config_overrides or {})
    return shared_clean_folder(path, config_overrides=cfg)
