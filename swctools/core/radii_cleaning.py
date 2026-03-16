"""Shared radii-cleaning logic used by CLI + GUI features.

Algorithm summary:
- Detect invalid radii (NaN/non-finite/non-positive/out-of-range).
- Detect local spikes/dips relative to parent+children neighborhood.
- Replace abnormal radii with mean(parent, children) when available;
  otherwise use global median fallback.
- Iterate for a small number of passes to smooth abrupt artifacts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RULES: dict[str, Any] = {
    "replace_non_positive": True,
    "replace_non_finite": True,
    "detect_spikes": True,
    "detect_dips": True,
    "spike_ratio_threshold": 3.0,
    "dip_ratio_threshold": 0.33,
    "abs_min_radius": 0.05,
    "abs_max_radius": 30.0,
    "min_neighbor_count": 1,
    "iterations": 2,
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


def clean_radii_dataframe(df: pd.DataFrame, *, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Clean abnormal radii and return structured change details.

    Returns dict with keys:
      - dataframe: cleaned dataframe
      - total_changes: int
      - change_details: list[dict]
    """

    cfg = dict(DEFAULT_RULES)
    if isinstance(rules, dict):
        for k, v in rules.items():
            if k == "replacement" and isinstance(v, dict):
                rep = dict(cfg.get("replacement", {}))
                rep.update(v)
                cfg["replacement"] = rep
            else:
                cfg[k] = v

    out = df.copy()
    if out.empty or "id" not in out.columns or "radius" not in out.columns or "parent" not in out.columns:
        return {"dataframe": out, "total_changes": 0, "change_details": []}

    ids = np.array(out["id"], dtype=int, copy=False)
    parents = np.array(out["parent"], dtype=int, copy=False)
    # Use an explicit writable copy; pandas may return a read-only view.
    radii = np.array(out["radius"], dtype=float, copy=True)
    original = np.array(radii, dtype=float, copy=True)

    id_to_idx = {int(ids[i]): int(i) for i in range(len(ids))}
    children: list[list[int]] = [[] for _ in range(len(ids))]
    for i, pid in enumerate(parents):
        pidx = id_to_idx.get(int(pid))
        if pidx is not None:
            children[pidx].append(i)

    valid_global = [float(r) for r in radii if _is_valid_radius(float(r))]
    global_median = float(np.median(valid_global)) if valid_global else float(cfg["abs_min_radius"])

    replace_non_positive = bool(cfg.get("replace_non_positive", True))
    replace_non_finite = bool(cfg.get("replace_non_finite", True))
    detect_spikes = bool(cfg.get("detect_spikes", True))
    detect_dips = bool(cfg.get("detect_dips", True))
    spike_ratio_threshold = float(cfg.get("spike_ratio_threshold", 3.0))
    dip_ratio_threshold = float(cfg.get("dip_ratio_threshold", 0.33))
    abs_min_radius = float(cfg.get("abs_min_radius", 0.05))
    abs_max_radius = float(cfg.get("abs_max_radius", 30.0))
    min_neighbor_count = int(cfg.get("min_neighbor_count", 1))
    iterations = max(1, int(cfg.get("iterations", 2)))

    replacement_cfg = dict(cfg.get("replacement", {}))
    clamp_min = float(replacement_cfg.get("clamp_min", abs_min_radius))
    clamp_max = float(replacement_cfg.get("clamp_max", abs_max_radius))

    reasons_by_idx: dict[int, set[str]] = {}

    for _ in range(iterations):
        changed_in_pass = False
        for i in range(len(radii)):
            cur = float(radii[i])
            pidx = id_to_idx.get(int(parents[i]))
            neigh_vals: list[float] = []
            if pidx is not None:
                pr = float(radii[pidx])
                if _is_valid_radius(pr):
                    neigh_vals.append(pr)
            for cidx in children[i]:
                cr = float(radii[cidx])
                if _is_valid_radius(cr):
                    neigh_vals.append(cr)

            neighbor_avg = float(np.mean(neigh_vals)) if neigh_vals else global_median
            bad_reasons: list[str] = []

            if replace_non_finite and not math.isfinite(cur):
                bad_reasons.append("non_finite")
            if replace_non_positive and cur <= 0.0:
                bad_reasons.append("non_positive")
            if math.isfinite(cur):
                if cur < abs_min_radius:
                    bad_reasons.append("below_abs_min")
                if cur > abs_max_radius:
                    bad_reasons.append("above_abs_max")

            if len(neigh_vals) >= min_neighbor_count and _is_valid_radius(neighbor_avg):
                if detect_spikes and math.isfinite(cur) and neighbor_avg > 0 and cur > neighbor_avg * spike_ratio_threshold:
                    bad_reasons.append("local_spike")
                if detect_dips and math.isfinite(cur) and neighbor_avg > 0 and cur < neighbor_avg * dip_ratio_threshold:
                    bad_reasons.append("local_dip")

            if not bad_reasons:
                continue

            replacement_vals: list[float] = []
            if pidx is not None:
                pr = float(radii[pidx])
                if _is_valid_radius(pr):
                    replacement_vals.append(pr)
            for cidx in children[i]:
                cr = float(radii[cidx])
                if _is_valid_radius(cr):
                    replacement_vals.append(cr)

            replacement = float(np.mean(replacement_vals)) if replacement_vals else global_median
            replacement = _clamp(float(replacement), clamp_min, clamp_max)
            if not math.isfinite(replacement) or replacement <= 0:
                replacement = max(abs_min_radius, clamp_min)

            if float(replacement) != float(cur):
                radii[i] = float(replacement)
                changed_in_pass = True
                reasons_by_idx.setdefault(i, set()).update(bad_reasons)

        if not changed_in_pass:
            break

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
    }
