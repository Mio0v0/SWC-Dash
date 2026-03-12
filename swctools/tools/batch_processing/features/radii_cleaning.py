"""Radii cleaning feature for Batch Processing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from swctools.core.config import load_feature_config, merge_config
from swctools.core.swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "batch_processing"
FEATURE = "radii_cleaning"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "rules": {
        "replace_non_positive": True,
        "replace_nan": True,
        "copy_parent_radius": True,
        "min_radius": 0.1,
    },
    "output": {
        "suffix": "_radii_cleaned",
    },
}


def _valid_radius(v: float) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(float(v)) and float(v) > 0.0


def _builtin_clean_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    rules = config.get("rules", {})
    do_non_positive = bool(rules.get("replace_non_positive", True))
    do_nan = bool(rules.get("replace_nan", True))
    copy_parent = bool(rules.get("copy_parent_radius", True))
    min_radius = float(rules.get("min_radius", 0.1))

    id_to_idx = {int(out.iloc[i]["id"]): i for i in range(len(out))}
    changes = 0

    for i in range(len(out)):
        cur = float(out.iloc[i]["radius"])
        bad = False
        if do_nan and math.isnan(cur):
            bad = True
        if do_non_positive and cur <= 0:
            bad = True
        if not bad:
            continue

        replacement = min_radius
        if copy_parent:
            pid = int(out.iloc[i]["parent"])
            pidx = id_to_idx.get(pid)
            if pidx is not None:
                pr = float(out.iloc[pidx]["radius"])
                if _valid_radius(pr):
                    replacement = pr

        out.at[out.index[i], "radius"] = replacement
        if "radius_str" in out.columns:
            out.at[out.index[i], "radius_str"] = str(replacement)
        changes += 1

    return out, changes


register_builtin_method(FEATURE_KEY, "default", _builtin_clean_dataframe)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def clean_swc_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    df = parse_swc_text_preserve_tokens(swc_text)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    out_df, changes = fn(df, cfg)
    out_bytes = write_swc_to_bytes_preserve_tokens(out_df)
    return {"changes": int(changes), "bytes": out_bytes, "dataframe": out_df}


def clean_folder(folder: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    in_dir = Path(folder)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(folder)

    suffix = str(cfg.get("output", {}).get("suffix", "_radii_cleaned"))
    out_dir = in_dir / f"{in_dir.name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    swc_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".swc")
    failures: list[str] = []
    per_file: list[dict[str, Any]] = []
    total_changes = 0

    for fp in swc_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            out = clean_swc_text(text, config_overrides=cfg)
            out_path = out_dir / fp.name
            out_path.write_bytes(out["bytes"])
            c = int(out["changes"])
            total_changes += c
            per_file.append({"file": fp.name, "radius_changes": c, "out_file": str(out_path)})
        except Exception as e:  # noqa: BLE001
            failures.append(f"{fp.name}: {e}")

    return {
        "folder": str(in_dir),
        "out_dir": str(out_dir),
        "files_total": len(swc_files),
        "files_processed": len(per_file),
        "files_failed": len(failures),
        "total_radius_changes": total_changes,
        "per_file": per_file,
        "failures": failures,
    }
