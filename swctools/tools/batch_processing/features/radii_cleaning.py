"""Radii cleaning feature for Batch Processing.

This module is the shared backend used by:
- Batch GUI radii cleaning tab
- Validation GUI radii cleaning tab
- CLI batch/validation radii-clean commands
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from swctools.core.config import load_feature_config, merge_config
from swctools.core.radii_cleaning import clean_radii_dataframe
from swctools.core.reporting import format_radii_cleaning_report_text, write_text_report
from swctools.core.swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "batch_processing"
FEATURE = "radii_cleaning"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "rules": {
        "preserve_soma": True,
        "small_radius_zero_only": True,
        "threshold_mode": "percentile",
        "global_percentile_bounds": {
            "min": 1.0,
            "max": 99.5,
        },
        "global_absolute_bounds": {
            "min": 0.05,
            "max": 30.0,
        },
        "type_thresholds": {
            "2": {"enabled": True, "min_percentile": 1.0, "max_percentile": 99.5, "min_abs": 0.05, "max_abs": 30.0},
            "3": {"enabled": True, "min_percentile": 1.0, "max_percentile": 99.5, "min_abs": 0.05, "max_abs": 30.0},
            "4": {"enabled": True, "min_percentile": 1.0, "max_percentile": 99.5, "min_abs": 0.05, "max_abs": 30.0},
        },
        "replace_non_positive": True,
        "replace_non_finite": True,
        "detect_spikes": True,
        "detect_dips": True,
        "spike_ratio_threshold": 2.8,
        "dip_ratio_threshold": 0.35,
        "min_neighbor_count": 1,
        "iterations": 4,
        "max_descendant_search_depth": 32,
        "replacement": {
            "clamp_min": 0.05,
            "clamp_max": 30.0,
        },
    },
    "output": {
        "folder_suffix": "_radii_cleaned",
        "file_suffix": "_radii_cleaned",
        "report_name": "radii_cleaning_report.txt",
    },
}


def _builtin_clean_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return clean_radii_dataframe(df, rules=dict(config.get("rules", {})))


register_builtin_method(FEATURE_KEY, "default", _builtin_clean_dataframe)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def _normalize_method_output(method_out: Any, input_df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    # Backward compatibility with older plugin signatures.
    if isinstance(method_out, tuple) and len(method_out) >= 2:
        out_df = method_out[0]
        changes = int(method_out[1])
        details = []
        if isinstance(out_df, pd.DataFrame):
            in_r = input_df["radius"].to_numpy(dtype=float)
            out_r = out_df["radius"].to_numpy(dtype=float)
            n = min(len(in_r), len(out_r), len(input_df))
            for i in range(n):
                if float(in_r[i]) == float(out_r[i]):
                    continue
                details.append(
                    {
                        "node_id": int(input_df.iloc[i]["id"]),
                        "old_radius": float(in_r[i]),
                        "new_radius": float(out_r[i]),
                        "reasons": ["plugin_change"],
                    }
                )
        return out_df, changes, details

    if isinstance(method_out, dict):
        out_df = method_out.get("dataframe")
        if not isinstance(out_df, pd.DataFrame):
            raise TypeError("radii-clean method returned dict without DataFrame 'dataframe'")
        details = list(method_out.get("change_details", []))
        changes = int(method_out.get("total_changes", len(details)))
        return out_df, changes, details

    raise TypeError("radii-clean method output must be tuple(DataFrame, int) or dict")


def clean_swc_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    df = parse_swc_text_preserve_tokens(swc_text)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    method_out = fn(df, cfg)
    out_df, changes, details = _normalize_method_output(method_out, df)
    stats_by_type = {}
    if isinstance(method_out, dict):
        stats_by_type = dict(method_out.get("stats_by_type", {}))

    out_bytes = write_swc_to_bytes_preserve_tokens(out_df)
    return {
        "changes": int(changes),
        "change_details": details,
        "bytes": out_bytes,
        "dataframe": out_df,
        "stats_by_type": stats_by_type,
        "config_used": cfg,
    }


def _format_change_lines(change_details: list[dict[str, Any]], *, limit: int | None = None) -> list[str]:
    rows = change_details if limit is None else change_details[: max(0, int(limit))]
    out: list[str] = []
    for row in rows:
        reasons = ",".join(str(r) for r in row.get("reasons", [])) or "unknown"
        out.append(
            f"node_id={row.get('node_id')} old={row.get('old_radius')} "
            f"new={row.get('new_radius')} reasons={reasons}"
        )
    if limit is not None and len(change_details) > limit:
        out.append(f"... ({len(change_details) - int(limit)} more node changes)")
    return out


def clean_file(
    path: str,
    *,
    out_path: str | None = None,
    write_output: bool = True,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    in_path = Path(path)
    if not in_path.exists() or not in_path.is_file():
        raise FileNotFoundError(path)

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    out = clean_swc_text(text, config_overrides=cfg)

    output_cfg = dict(cfg.get("output", {}))
    file_suffix = str(output_cfg.get("file_suffix", "_radii_cleaned"))

    output_path: Path | None = None
    if write_output:
        if out_path:
            output_path = Path(out_path)
        else:
            output_path = in_path.with_name(f"{in_path.stem}{file_suffix}{in_path.suffix}")
        output_path.write_bytes(out["bytes"])

    report = {
        "mode": "file",
        "input_path": str(in_path),
        "output_path": str(output_path) if output_path else None,
        "radius_changes": int(out["changes"]),
        "change_count": int(len(out.get("change_details", []))),
        "change_details": list(out.get("change_details", [])),
        "change_lines": _format_change_lines(list(out.get("change_details", []))),
        "config_used": cfg,
    }

    report_base = output_path if output_path is not None else in_path
    report_path = report_base.with_name(f"{report_base.stem}_radii_cleaning_report.txt")
    report["log_path"] = write_text_report(report_path, format_radii_cleaning_report_text(report))
    return report


def clean_folder(folder: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    in_dir = Path(folder)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(folder)

    output_cfg = dict(cfg.get("output", {}))
    folder_suffix = str(output_cfg.get("folder_suffix", "_radii_cleaned"))
    report_name = str(output_cfg.get("report_name", "radii_cleaning_report.txt"))

    out_dir = in_dir / f"{in_dir.name}{folder_suffix}"
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
            details = list(out.get("change_details", []))
            total_changes += c
            per_file.append(
                {
                    "file": fp.name,
                    "radius_changes": c,
                    "out_file": str(out_path),
                    "change_count": len(details),
                    "change_lines": _format_change_lines(details),
                }
            )
        except Exception as e:  # noqa: BLE001
            failures.append(f"{fp.name}: {e}")

    out_report = {
        "mode": "folder",
        "folder": str(in_dir),
        "out_dir": str(out_dir),
        "files_total": len(swc_files),
        "files_processed": len(per_file),
        "files_failed": len(failures),
        "total_radius_changes": total_changes,
        "per_file": per_file,
        "failures": failures,
        "config_used": cfg,
    }

    out_report["log_path"] = write_text_report(out_dir / report_name, format_radii_cleaning_report_text(out_report))
    return out_report


def clean_path(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        return clean_folder(str(p), config_overrides=config_overrides)
    if p.is_file():
        return clean_file(str(p), config_overrides=config_overrides)
    raise FileNotFoundError(path)
