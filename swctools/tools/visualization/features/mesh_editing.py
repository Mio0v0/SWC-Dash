"""Mesh editing feature backend.

This module provides a reusable backend payload that both GUI and CLI can use
for mesh-related operations without duplicating parsing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from swctools.core.config import load_feature_config, merge_config
from swctools.core.swc_io import parse_swc_text_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "visualization"
FEATURE = "mesh_editing"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "output": {
        "include_edges": False,
    },
}


def _bbox(df) -> dict[str, list[float]]:
    if df.empty:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    mins = [float(df[c].min()) for c in ("x", "y", "z")]
    maxs = [float(df[c].max()) for c in ("x", "y", "z")]
    return {"min": mins, "max": maxs}


def _builtin_build_mesh_payload(swc_text: str, config: dict[str, Any]) -> dict[str, Any]:
    df = parse_swc_text_preserve_tokens(swc_text)
    if df.empty:
        return {
            "nodes": 0,
            "segments": 0,
            "bbox": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            "mean_radius": 0.0,
            "edges": [],
        }

    id_to_idx = {int(df.iloc[i]["id"]): i for i in range(len(df))}
    edges: list[tuple[int, int]] = []
    for i in range(len(df)):
        pid = int(df.iloc[i]["parent"])
        if pid == -1:
            continue
        pidx = id_to_idx.get(pid)
        if pidx is None:
            continue
        edges.append((pidx, i))

    include_edges = bool(config.get("output", {}).get("include_edges", False))
    payload = {
        "nodes": int(len(df)),
        "segments": int(len(edges)),
        "bbox": _bbox(df),
        "mean_radius": float(np.nanmean(df["radius"].to_numpy(dtype=float))),
    }
    if include_edges:
        payload["edges"] = edges
    return payload


register_builtin_method(FEATURE_KEY, "default", _builtin_build_mesh_payload)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def build_mesh_from_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def build_mesh_from_file(path: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    out = build_mesh_from_text(text, config_overrides=config_overrides)
    out["input_path"] = str(fp)
    return out
