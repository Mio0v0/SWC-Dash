"""Shared radii-cleaning logic used by CLI + GUI features."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

TYPE_NAMES = {
    1: "soma",
    2: "axon",
    3: "basal",
    4: "apical",
}

DEFAULT_RULES: dict[str, Any] = {
    # Safety: never alter soma radii by default.
    "preserve_soma": True,
    # User-requested default: lower-end outlier is only treated as abnormal when <= 0.
    "small_radius_zero_only": True,
    # Thresholding mode: "percentile" or "absolute".
    "threshold_mode": "percentile",
    "global_percentile_bounds": {
        "min": 1.0,
        "max": 99.5,
    },
    "global_absolute_bounds": {
        "min": 0.05,
        "max": 30.0,
    },
    # Per-type overrides (keys are SWC type integers encoded as strings).
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
}


def _is_valid_radius(v: float) -> bool:
    return isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) and float(v) > 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, v))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if not isinstance(overrides, dict):
        return out
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _build_topology(ids: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    id_to_idx = {int(ids[i]): int(i) for i in range(len(ids))}
    parent_idx = np.full(len(ids), -1, dtype=int)
    children: list[list[int]] = [[] for _ in range(len(ids))]
    for i, pid in enumerate(parents):
        pidx = id_to_idx.get(int(pid))
        if pidx is not None:
            parent_idx[i] = pidx
            children[pidx].append(i)
    return parent_idx, children


def _depths_from_roots(parent_idx: np.ndarray, children: list[list[int]]) -> np.ndarray:
    n = len(parent_idx)
    depths = np.full(n, -1, dtype=int)
    roots = [i for i in range(n) if int(parent_idx[i]) < 0]
    q: deque[int] = deque(roots)
    for r in roots:
        depths[r] = 0
    while q:
        i = q.popleft()
        for c in children[i]:
            if depths[c] >= 0:
                continue
            depths[c] = depths[i] + 1
            q.append(c)
    for i in range(n):
        if depths[i] < 0:
            depths[i] = 0
    return depths


def radii_stats_by_type(df: pd.DataFrame, *, bins: int = 12) -> dict[str, Any]:
    """Compute per-type distribution stats + histogram bins for radii."""
    if df.empty or "type" not in df.columns or "radius" not in df.columns:
        return {"type_stats": {}}

    types = np.array(df["type"], dtype=int, copy=False)
    radii = np.array(df["radius"], dtype=float, copy=False)
    type_stats: dict[str, Any] = {}
    for t in sorted({int(v) for v in types.tolist()}):
        mask_t = types == int(t)
        all_vals = radii[mask_t]
        valid = all_vals[np.isfinite(all_vals) & (all_vals > 0.0)]
        row: dict[str, Any] = {
            "type_id": int(t),
            "type_name": TYPE_NAMES.get(int(t), f"type_{int(t)}"),
            "count_total": int(mask_t.sum()),
            "count_valid_positive": int(valid.size),
        }
        if valid.size > 0:
            q1 = float(np.percentile(valid, 25))
            med = float(np.percentile(valid, 50))
            q3 = float(np.percentile(valid, 75))
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            mean = float(np.mean(valid))
            b = max(4, int(bins))
            if vmin == vmax:
                edges = [vmin, vmax]
                counts = [int(valid.size)]
            else:
                counts_np, edges_np = np.histogram(valid, bins=b, range=(vmin, vmax))
                counts = [int(x) for x in counts_np.tolist()]
                edges = [float(x) for x in edges_np.tolist()]
            row.update(
                {
                    "mean": mean,
                    "median": med,
                    "q1": q1,
                    "q3": q3,
                    "min": vmin,
                    "max": vmax,
                    "hist_counts": counts,
                    "hist_edges": edges,
                }
            )
        else:
            row.update(
                {
                    "mean": None,
                    "median": None,
                    "q1": None,
                    "q3": None,
                    "min": None,
                    "max": None,
                    "hist_counts": [],
                    "hist_edges": [],
                }
            )
        type_stats[str(int(t))] = row
    return {"type_stats": type_stats}


def _bounds_for_type(
    t: int,
    cfg: dict[str, Any],
    stats_map: dict[str, Any],
) -> tuple[bool, float, float]:
    preserve_soma = bool(cfg.get("preserve_soma", True))
    if preserve_soma and int(t) == 1:
        return False, 0.0, float("inf")

    per_type = dict(cfg.get("type_thresholds", {})).get(str(int(t)), {})
    enabled = bool(per_type.get("enabled", True))
    if not enabled:
        return False, 0.0, float("inf")

    mode = str(cfg.get("threshold_mode", "percentile")).strip().lower()
    g_pct = dict(cfg.get("global_percentile_bounds", {}))
    g_abs = dict(cfg.get("global_absolute_bounds", {}))
    zero_only_small = bool(cfg.get("small_radius_zero_only", True))

    if mode == "absolute":
        lo = float(per_type.get("min_abs", g_abs.get("min", 0.05)))
        hi = float(per_type.get("max_abs", g_abs.get("max", 30.0)))
    else:
        # Percentile mode is computed from per-type valid-positive distribution.
        row = dict(stats_map.get(str(int(t)), {}))
        lo_p = float(per_type.get("min_percentile", g_pct.get("min", 1.0)))
        hi_p = float(per_type.get("max_percentile", g_pct.get("max", 99.5)))
        edges = row.get("hist_edges") or []
        # Use raw values when available, else fall back to global absolute bounds.
        if row.get("count_valid_positive", 0) and row.get("min") is not None and row.get("max") is not None:
            vals_min = float(row["min"])
            vals_max = float(row["max"])
            if vals_min == vals_max:
                lo = vals_min
                hi = vals_max
            else:
                # Rebuild percentiles from raw dataframe stats is not stored; approximate with min/max
                # bounds if distribution summary only is available.
                # Better accuracy is handled in clean_radii_dataframe via direct per-type arrays.
                lo = vals_min
                hi = vals_max
        else:
            lo = float(g_abs.get("min", 0.05))
            hi = float(g_abs.get("max", 30.0))
        # Keep explicit percentile numbers available to caller for true computation.
        _ = lo_p, hi_p, edges

    if zero_only_small:
        lo = 0.0
    if hi < lo:
        lo, hi = hi, lo
    return True, float(lo), float(hi)


def _nearest_ancestor_index(
    i: int,
    parent_idx: np.ndarray,
    normal_mask: np.ndarray,
    radii: np.ndarray,
) -> int | None:
    p = int(parent_idx[i])
    while p >= 0:
        if bool(normal_mask[p]) and _is_valid_radius(float(radii[p])):
            return int(p)
        p = int(parent_idx[p])
    return None


def _nearest_descendant_indices(
    i: int,
    children: list[list[int]],
    normal_mask: np.ndarray,
    radii: np.ndarray,
    max_depth: int,
) -> list[int]:
    q: deque[tuple[int, int]] = deque((c, 1) for c in children[i])
    found: list[int] = []
    found_depth = -1
    while q:
        n, d = q.popleft()
        if d > int(max_depth):
            continue
        if found_depth > 0 and d > found_depth:
            break
        if bool(normal_mask[n]) and _is_valid_radius(float(radii[n])):
            found.append(int(n))
            found_depth = d
            continue
        for c in children[n]:
            q.append((c, d + 1))
    return found


def clean_radii_dataframe(df: pd.DataFrame, *, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Clean abnormal radii and return structured change details."""

    cfg = _deep_merge(DEFAULT_RULES, rules)
    out = df.copy()
    if out.empty or "id" not in out.columns or "radius" not in out.columns or "parent" not in out.columns:
        return {"dataframe": out, "total_changes": 0, "change_details": [], "stats_by_type": {"type_stats": {}}}

    ids = np.array(out["id"], dtype=int, copy=False)
    types = np.array(out["type"], dtype=int, copy=False) if "type" in out.columns else np.zeros(len(out), dtype=int)
    parents = np.array(out["parent"], dtype=int, copy=False)
    radii = np.array(out["radius"], dtype=float, copy=True)
    original = np.array(radii, dtype=float, copy=True)

    parent_idx, children = _build_topology(ids, parents)
    depths = _depths_from_roots(parent_idx, children)

    stats = radii_stats_by_type(out)
    stats_map = dict(stats.get("type_stats", {}))

    # Build true percentile bounds directly from data arrays.
    g_pct = dict(cfg.get("global_percentile_bounds", {}))
    g_abs = dict(cfg.get("global_absolute_bounds", {}))
    mode = str(cfg.get("threshold_mode", "percentile")).strip().lower()
    zero_only_small = bool(cfg.get("small_radius_zero_only", True))
    preserve_soma = bool(cfg.get("preserve_soma", True))
    per_type_cfg = dict(cfg.get("type_thresholds", {}))
    bounds: dict[int, tuple[bool, float, float]] = {}
    for t in sorted({int(v) for v in types.tolist()}):
        enabled, lo, hi = _bounds_for_type(int(t), cfg, stats_map)
        if mode != "absolute":
            t_cfg = dict(per_type_cfg.get(str(int(t)), {}))
            lp = float(t_cfg.get("min_percentile", g_pct.get("min", 1.0)))
            hp = float(t_cfg.get("max_percentile", g_pct.get("max", 99.5)))
            vals = radii[(types == int(t)) & np.isfinite(radii) & (radii > 0.0)]
            if vals.size > 0:
                lo = float(np.percentile(vals, lp))
                hi = float(np.percentile(vals, hp))
            else:
                lo = float(t_cfg.get("min_abs", g_abs.get("min", 0.05)))
                hi = float(t_cfg.get("max_abs", g_abs.get("max", 30.0)))
            if zero_only_small:
                lo = 0.0
        if hi < lo:
            lo, hi = hi, lo
        bounds[int(t)] = (bool(enabled), float(lo), float(hi))

    replacement_cfg = dict(cfg.get("replacement", {}))
    clamp_min = float(replacement_cfg.get("clamp_min", g_abs.get("min", 0.05)))
    clamp_max = float(replacement_cfg.get("clamp_max", g_abs.get("max", 30.0)))
    replace_non_positive = bool(cfg.get("replace_non_positive", True))
    replace_non_finite = bool(cfg.get("replace_non_finite", True))
    detect_spikes = bool(cfg.get("detect_spikes", True))
    detect_dips = bool(cfg.get("detect_dips", True))
    spike_ratio = float(cfg.get("spike_ratio_threshold", 2.8))
    dip_ratio = float(cfg.get("dip_ratio_threshold", 0.35))
    min_neighbor_count = int(cfg.get("min_neighbor_count", 1))
    iterations = max(1, int(cfg.get("iterations", 4)))
    max_desc_depth = max(1, int(cfg.get("max_descendant_search_depth", 32)))

    valid_global = radii[np.isfinite(radii) & (radii > 0.0)]
    global_median = float(np.median(valid_global)) if valid_global.size else max(0.05, clamp_min)
    type_medians: dict[int, float] = {}
    for t in sorted({int(v) for v in types.tolist()}):
        vals = radii[(types == int(t)) & np.isfinite(radii) & (radii > 0.0)]
        if vals.size > 0:
            type_medians[int(t)] = float(np.median(vals))

    reasons_by_idx: dict[int, set[str]] = {}

    def _is_out_of_range(i: int, cur: float) -> list[str]:
        t = int(types[i])
        if preserve_soma and t == 1:
            return []
        enabled, lo, hi = bounds.get(t, (True, 0.0, float("inf")))
        if not enabled:
            return []
        rr: list[str] = []
        if not zero_only_small and cur < lo:
            rr.append("below_type_min")
        if cur > hi:
            rr.append("above_type_max")
        return rr

    for _ in range(iterations):
        reasons_cur: list[list[str]] = [[] for _ in range(len(radii))]
        abnormal: list[int] = []

        for i in range(len(radii)):
            t = int(types[i])
            if preserve_soma and t == 1:
                continue
            cur = float(radii[i])
            r: list[str] = []

            if replace_non_finite and not math.isfinite(cur):
                r.append("non_finite")
            if replace_non_positive and cur <= 0.0:
                r.append("non_positive")
            if math.isfinite(cur) and cur > 0.0:
                r.extend(_is_out_of_range(i, cur))

                neigh_vals: list[float] = []
                p = int(parent_idx[i])
                if p >= 0 and _is_valid_radius(float(radii[p])):
                    neigh_vals.append(float(radii[p]))
                for c in children[i]:
                    if _is_valid_radius(float(radii[c])):
                        neigh_vals.append(float(radii[c]))
                if len(neigh_vals) >= min_neighbor_count:
                    navg = float(np.mean(neigh_vals))
                    if detect_spikes and navg > 0 and cur > navg * spike_ratio:
                        r.append("local_spike")
                    if detect_dips and navg > 0 and cur < navg * dip_ratio:
                        r.append("local_dip")

            if r:
                reasons_cur[i] = sorted(set(r))
                abnormal.append(i)

        if not abnormal:
            break

        normal_mask = np.ones(len(radii), dtype=bool)
        for i in abnormal:
            normal_mask[i] = False

        changed = False
        for i in sorted(abnormal, key=lambda j: int(depths[j])):
            t = int(types[i])
            if preserve_soma and t == 1:
                continue

            cur = float(radii[i])
            anc = _nearest_ancestor_index(i, parent_idx, normal_mask, radii)
            desc = _nearest_descendant_indices(i, children, normal_mask, radii, max_desc_depth)
            vals: list[float] = []
            if anc is not None:
                vals.append(float(radii[anc]))
            if desc:
                vals.append(float(np.mean([float(radii[d]) for d in desc])))

            if vals:
                rep = float(np.mean(vals))
            else:
                rep = float(type_medians.get(t, global_median))

            enabled, lo, hi = bounds.get(t, (True, 0.0, float("inf")))
            rep = _clamp(rep, clamp_min, clamp_max)
            if enabled and math.isfinite(rep):
                if not zero_only_small:
                    rep = max(rep, lo)
                rep = min(rep, hi)
            if not _is_valid_radius(rep):
                rep = max(clamp_min, float(type_medians.get(t, global_median)))

            if float(rep) != float(cur):
                radii[i] = float(rep)
                changed = True
            normal_mask[i] = True
            reasons_by_idx.setdefault(i, set()).update(reasons_cur[i])

        if not changed:
            break

    # Final hard enforcement: no non-soma node remains outside configured bounds.
    final_normal = np.ones(len(radii), dtype=bool)
    for i in range(len(radii)):
        t = int(types[i])
        if preserve_soma and t == 1:
            continue
        cur = float(radii[i])
        bad = []
        if replace_non_finite and not math.isfinite(cur):
            bad.append("non_finite")
        if replace_non_positive and cur <= 0.0:
            bad.append("non_positive")
        if math.isfinite(cur) and cur > 0.0:
            bad.extend(_is_out_of_range(i, cur))
        if bad:
            final_normal[i] = False
            reasons_by_idx.setdefault(i, set()).update(bad)

    for i in np.flatnonzero(~final_normal):
        t = int(types[i])
        if preserve_soma and t == 1:
            continue
        anc = _nearest_ancestor_index(int(i), parent_idx, final_normal, radii)
        desc = _nearest_descendant_indices(int(i), children, final_normal, radii, max_desc_depth)
        vals: list[float] = []
        if anc is not None:
            vals.append(float(radii[anc]))
        if desc:
            vals.append(float(np.mean([float(radii[d]) for d in desc])))
        rep = float(np.mean(vals)) if vals else float(type_medians.get(t, global_median))
        enabled, lo, hi = bounds.get(t, (True, 0.0, float("inf")))
        rep = _clamp(rep, clamp_min, clamp_max)
        if enabled and math.isfinite(rep):
            if not zero_only_small:
                rep = max(rep, lo)
            rep = min(rep, hi)
        if not _is_valid_radius(rep):
            rep = max(clamp_min, float(type_medians.get(t, global_median)))
        radii[int(i)] = float(rep)
        final_normal[int(i)] = True
        reasons_by_idx.setdefault(int(i), set()).add("final_enforce")

    # Strict guarantee: preserve soma radii.
    if preserve_soma and len(types) == len(radii):
        soma_mask = types == 1
        radii[soma_mask] = original[soma_mask]

    out["radius"] = radii
    if "radius_str" in out.columns:
        for i in range(len(out)):
            out.at[out.index[i], "radius_str"] = str(float(radii[i]))

    change_details: list[dict[str, Any]] = []
    for i in range(len(radii)):
        old_r = float(original[i])
        new_r = float(radii[i])
        if old_r == new_r:
            continue
        change_details.append(
            {
                "node_id": int(ids[i]),
                "old_radius": old_r,
                "new_radius": new_r,
                "reasons": sorted(reasons_by_idx.get(i, set())),
            }
        )

    return {
        "dataframe": out,
        "total_changes": len(change_details),
        "change_details": change_details,
        "stats_by_type": stats,
    }
