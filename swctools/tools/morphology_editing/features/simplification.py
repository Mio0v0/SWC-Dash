"""Smart decimation (RDP-based) for morphology editing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swctools.core.config import load_feature_config, merge_config
from swctools.core.reporting import format_simplification_report_text, write_text_report
from swctools.core.swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "morphology_editing"
FEATURE = "simplification"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "thresholds": {
        "epsilon": 2.0,
        "radius_tolerance": 0.5,
    },
    "flags": {
        "keep_tips": True,
        "keep_bifurcations": True,
        "keep_roots": True,
    },
    "output": {
        "suffix": "_simplified",
    },
}


def _point_line_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _rdp_indices(points: np.ndarray, epsilon: float) -> list[int]:
    n = int(points.shape[0])
    if n <= 2:
        return [0, max(0, n - 1)]

    start = points[0]
    end = points[-1]

    max_dist = -1.0
    max_idx = -1
    for i in range(1, n - 1):
        d = _point_line_dist(points[i], start, end)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > float(epsilon) and max_idx > 0:
        left = _rdp_indices(points[: max_idx + 1], epsilon)
        right = _rdp_indices(points[max_idx:], epsilon)
        return left[:-1] + [max_idx + r for r in right]

    return [0, n - 1]


def _build_graph(df: pd.DataFrame) -> tuple[dict[int, int], list[list[int]], np.ndarray, np.ndarray]:
    ids = df["id"].to_numpy(dtype=int)
    parents = df["parent"].to_numpy(dtype=int)
    id_to_idx = {int(ids[i]): int(i) for i in range(len(ids))}
    children: list[list[int]] = [[] for _ in range(len(ids))]
    for i, pid in enumerate(parents):
        pidx = id_to_idx.get(int(pid))
        if pidx is not None:
            children[pidx].append(i)
    child_counts = np.asarray([len(c) for c in children], dtype=int)
    return id_to_idx, children, ids, child_counts


def _extract_anchor_paths(children: list[list[int]], anchors: set[int]) -> list[list[int]]:
    paths: list[list[int]] = []
    for start in sorted(anchors):
        for c in children[start]:
            path = [start, c]
            cur = c
            # Walk down linear chain until next anchor.
            while cur not in anchors:
                nxts = children[cur]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                path.append(nxt)
                cur = nxt
            paths.append(path)
    return paths


def _nearest_kept_parent(
    idx: int,
    parents: np.ndarray,
    ids: np.ndarray,
    id_to_idx: dict[int, int],
    keep_mask: np.ndarray,
) -> int:
    cur_pid = int(parents[idx])
    while cur_pid != -1:
        pidx = id_to_idx.get(cur_pid)
        if pidx is None:
            return -1
        if bool(keep_mask[pidx]):
            return int(ids[pidx])
        cur_pid = int(parents[pidx])
    return -1


def _builtin_simplify_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    out_df = df.copy()
    if out_df.empty:
        return {
            "dataframe": out_df,
            "kept_node_ids": [],
            "removed_node_ids": [],
            "original_node_count": 0,
            "new_node_count": 0,
            "reduction_percent": 0.0,
            "params_used": {
                "epsilon": float(config.get("thresholds", {}).get("epsilon", 2.0)),
                "radius_tolerance": float(config.get("thresholds", {}).get("radius_tolerance", 0.5)),
                "keep_tips": bool(config.get("flags", {}).get("keep_tips", True)),
                "keep_bifurcations": bool(config.get("flags", {}).get("keep_bifurcations", True)),
            },
            "protected_counts": {"roots": 0, "tips": 0, "bifurcations": 0, "radius_sensitive": 0},
        }

    thresholds = dict(config.get("thresholds", {}))
    flags = dict(config.get("flags", {}))

    epsilon = max(0.0, float(thresholds.get("epsilon", 2.0)))
    radius_tol = max(0.0, float(thresholds.get("radius_tolerance", 0.5)))
    keep_tips = bool(flags.get("keep_tips", True))
    keep_bifs = bool(flags.get("keep_bifurcations", True))
    keep_roots = bool(flags.get("keep_roots", True))

    id_to_idx, children, ids, child_counts = _build_graph(out_df)
    parents = out_df["parent"].to_numpy(dtype=int)
    xyz = out_df[["x", "y", "z"]].to_numpy(dtype=float)
    radii = out_df["radius"].to_numpy(dtype=float)

    root_idx = {i for i in range(len(out_df)) if int(parents[i]) == -1}
    tip_idx = {i for i in range(len(out_df)) if int(child_counts[i]) == 0}
    bif_idx = {i for i in range(len(out_df)) if int(child_counts[i]) > 1}

    protected: set[int] = set(root_idx if keep_roots else [])
    if keep_tips:
        protected.update(tip_idx)
    if keep_bifs:
        protected.update(bif_idx)

    # Structural anchors define linear segments for RDP traversal.
    anchors: set[int] = set(root_idx)
    anchors.update(i for i in range(len(out_df)) if int(child_counts[i]) != 1)

    paths = _extract_anchor_paths(children, anchors)

    rdp_keep: set[int] = set()
    radius_sensitive: set[int] = set()

    for path in paths:
        if len(path) < 3:
            continue

        pts = xyz[path, :]
        keep_local = set(_rdp_indices(pts, epsilon))
        for li in keep_local:
            if li <= 0 or li >= len(path) - 1:
                continue
            rdp_keep.add(path[li])

        valid_rs = [float(radii[i]) for i in path if math.isfinite(float(radii[i])) and float(radii[i]) > 0]
        if not valid_rs:
            continue
        seg_mean = float(np.mean(valid_rs))
        if seg_mean <= 0:
            continue

        for i in path[1:-1]:
            rv = float(radii[i])
            if not math.isfinite(rv) or rv <= 0:
                radius_sensitive.add(i)
                continue
            dev = abs(rv - seg_mean) / seg_mean
            if dev > radius_tol:
                radius_sensitive.add(i)

    keep_idx = set(protected)
    keep_idx.update(rdp_keep)
    keep_idx.update(radius_sensitive)
    keep_idx.update(root_idx)

    if not keep_idx:
        keep_idx.add(0)

    keep_indices = sorted(keep_idx)
    keep_mask = np.zeros(len(out_df), dtype=bool)
    keep_mask[keep_indices] = True

    new_parents: list[int] = []
    for idx in keep_indices:
        if int(parents[idx]) == -1:
            new_parents.append(-1)
            continue
        new_parents.append(_nearest_kept_parent(idx, parents, ids, id_to_idx, keep_mask))

    simplified = out_df.iloc[keep_indices].copy().reset_index(drop=True)
    simplified["parent"] = np.asarray(new_parents, dtype=int)
    if "parent_str" in simplified.columns:
        for i in range(len(simplified)):
            simplified.at[simplified.index[i], "parent_str"] = str(int(simplified.iloc[i]["parent"]))

    kept_ids = [int(ids[i]) for i in keep_indices]
    kept_id_set = set(kept_ids)
    removed_ids = [int(v) for v in ids.tolist() if int(v) not in kept_id_set]

    original_count = int(len(out_df))
    new_count = int(len(simplified))
    reduction = 0.0 if original_count <= 0 else max(0.0, 100.0 * (original_count - new_count) / original_count)

    return {
        "dataframe": simplified,
        "kept_node_ids": kept_ids,
        "removed_node_ids": removed_ids,
        "original_node_count": original_count,
        "new_node_count": new_count,
        "reduction_percent": reduction,
        "params_used": {
            "epsilon": epsilon,
            "radius_tolerance": radius_tol,
            "keep_tips": keep_tips,
            "keep_bifurcations": keep_bifs,
            "keep_roots": keep_roots,
        },
        "protected_counts": {
            "roots": len(root_idx),
            "tips": len(tip_idx) if keep_tips else 0,
            "bifurcations": len(bif_idx) if keep_bifs else 0,
            "radius_sensitive": len(radius_sensitive),
        },
    }


register_builtin_method(FEATURE_KEY, "default", _builtin_simplify_dataframe)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def simplify_dataframe(df: pd.DataFrame, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    out = fn(df.copy(), cfg)
    if not isinstance(out, dict) or "dataframe" not in out:
        raise TypeError("simplification method must return dict with 'dataframe'")
    out["config_used"] = cfg
    return out


def simplify_swc_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    df = parse_swc_text_preserve_tokens(swc_text)
    out = simplify_dataframe(df, config_overrides=config_overrides)
    out_df = out["dataframe"]
    out["bytes"] = write_swc_to_bytes_preserve_tokens(out_df)
    return out


def simplify_file(
    path: str,
    *,
    out_path: str | None = None,
    write_output: bool = False,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)

    text = fp.read_text(encoding="utf-8", errors="ignore")
    out = simplify_swc_text(text, config_overrides=config_overrides)

    cfg = dict(out.get("config_used", {}))
    suffix = str(cfg.get("output", {}).get("suffix", "_simplified"))

    output_path: Path | None = None
    if write_output:
        output_path = Path(out_path) if out_path else fp.with_name(f"{fp.stem}{suffix}{fp.suffix}")
        output_path.write_bytes(out["bytes"])

    payload = {
        "mode": "file",
        "input_path": str(fp),
        "output_path": str(output_path) if output_path else None,
        "original_node_count": int(out.get("original_node_count", 0)),
        "new_node_count": int(out.get("new_node_count", 0)),
        "reduction_percent": float(out.get("reduction_percent", 0.0)),
        "params_used": dict(out.get("params_used", {})),
        "protected_counts": dict(out.get("protected_counts", {})),
        "removed_node_ids": list(out.get("removed_node_ids", [])),
    }

    report_base = output_path if output_path is not None else fp
    report_path = report_base.with_name(f"{report_base.stem}_simplification_log.txt")
    payload["log_path"] = write_text_report(report_path, format_simplification_report_text(payload))
    out["summary"] = payload
    out["input_path"] = str(fp)
    out["output_path"] = str(output_path) if output_path else None
    out["log_path"] = payload["log_path"]
    return out
