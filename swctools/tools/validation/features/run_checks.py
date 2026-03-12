"""Structured validation runner feature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.validation_engine import run_validation_text
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "validation"
FEATURE = "run_checks"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
}


def _builtin_run(swc_text: str, config: dict[str, Any]):
    overrides = config.get("config_overrides")
    return run_validation_text(swc_text, config_overrides=overrides)


register_builtin_method(FEATURE_KEY, "default", _builtin_run)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def validate_text(
    swc_text: str,
    *,
    config_overrides: dict | None = None,
    feature_overrides: dict | None = None,
):
    cfg = merge_config(get_config(), feature_overrides)
    if config_overrides:
        cfg["config_overrides"] = dict(config_overrides)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def validate_file(
    path: str,
    *,
    config_overrides: dict | None = None,
    feature_overrides: dict | None = None,
):
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    report = validate_text(
        text,
        config_overrides=config_overrides,
        feature_overrides=feature_overrides,
    )
    return report
