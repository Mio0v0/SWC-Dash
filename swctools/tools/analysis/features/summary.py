"""Basic morphology summary analysis feature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.swc_io import parse_swc_text_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "analysis"
FEATURE = "summary"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "default",
    "metrics": {
        "count_nodes_by_type": True,
        "count_roots": True,
        "count_branch_points": True,
    },
}


def _builtin_summary_from_text(swc_text: str, config: dict[str, Any]) -> dict[str, Any]:
    df = parse_swc_text_preserve_tokens(swc_text)
    if df.empty:
        return {
            "nodes": 0,
            "roots": 0,
            "branch_points": 0,
            "types": {},
        }

    id_to_idx = {int(df.iloc[i]["id"]): i for i in range(len(df))}
    child_counts = [0] * len(df)
    for i in range(len(df)):
        pid = int(df.iloc[i]["parent"])
        pidx = id_to_idx.get(pid)
        if pidx is not None:
            child_counts[pidx] += 1

    metrics = config.get("metrics", {})
    out: dict[str, Any] = {"nodes": int(len(df))}
    if bool(metrics.get("count_roots", True)):
        out["roots"] = int((df["parent"] == -1).sum())
    if bool(metrics.get("count_branch_points", True)):
        out["branch_points"] = int(sum(1 for c in child_counts if c > 1))
    if bool(metrics.get("count_nodes_by_type", True)):
        known = {str(i): int((df["type"] == i).sum()) for i in range(1, 8)}
        out["types"] = known
        out["types_unknown"] = int((~df["type"].isin(range(1, 8))).sum())
    return out


register_builtin_method(FEATURE_KEY, "default", _builtin_summary_from_text)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def analyze_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def analyze_file(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)

    text = fp.read_text(encoding="utf-8", errors="ignore")
    out = analyze_text(text, config_overrides=config_overrides)
    out["input_path"] = str(fp)
    return out
