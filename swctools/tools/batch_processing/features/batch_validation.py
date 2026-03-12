"""Batch validation feature for Batch Processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.validation import run_per_tree_validation
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "batch_processing"
FEATURE = "batch_validation"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "include": {"extensions": [".swc"]},
}


def _builtin_validate_text(swc_text: str, config: dict[str, Any]):
    _ = config
    return run_per_tree_validation(swc_text)


register_builtin_method(FEATURE_KEY, "default", _builtin_validate_text)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def validate_swc_text(swc_text: str, *, config_overrides: dict | None = None):
    cfg = merge_config(get_config(), config_overrides)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def validate_folder(folder: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    in_dir = Path(folder)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(folder)

    exts = {e.lower() for e in cfg.get("include", {}).get("extensions", [".swc"])}
    swc_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    rows = []
    failures = []
    for fp in swc_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            check_names, tree_results = validate_swc_text(text, config_overrides=cfg)
            rows.append({
                "file": fp.name,
                "checks": check_names,
                "tree_results": tree_results,
                "tree_count": len(tree_results),
            })
        except Exception as e:  # noqa: BLE001
            failures.append(f"{fp.name}: {e}")

    return {
        "folder": str(in_dir),
        "files_total": len(swc_files),
        "files_validated": len(rows),
        "files_failed": len(failures),
        "results": rows,
        "failures": failures,
    }
