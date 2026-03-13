"""Auto-typing / rule-batch implementation moved to core.

This module contains the rule-based auto-typing logic formerly located in
`swctools.gui.rule_batch_processor`. It is kept in `swctools.core` so both GUI
and CLI can use the same implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile
import math
from swctools.core.config import load_feature_config, merge_config, save_feature_config
from swctools.core.reporting import format_auto_typing_report_text, write_text_report


TOOL = "batch_processing"
FEATURE = "auto_typing"
_DEFAULT_CFG: dict[str, Any] | None = None


def _default_rules_config() -> dict[str, Any]:
    return {
        "class_labels": {"1": "soma", "2": "axon", "3": "basal", "4": "apical"},
        "branch_score_weights": {
            "axon": {"path": 0.32, "radial": 0.24, "radius": 0.20, "branch": 0.14, "prior": 0.10},
            "apical": {"z": 0.30, "path": 0.22, "radius": 0.18, "branch": 0.15, "prior": 0.15},
            "basal": {"z": 0.30, "branch": 0.22, "radius": 0.18, "path": 0.15, "prior": 0.15},
        },
        "ml_blend": 0.28,
        "ml_base_weight": 0.72,
        "seed_prior_threshold": 0.55,
        "assign_missing": {"min_score": 0.58, "min_gain": -0.06},
        "smoothing": {"maj_fraction": 0.67, "flip_margin": 0.10},
        "propagation_weights": {
            "self": 0.35,
            "parent": 0.35,
            "children": 0.20,
            "branch_prior": 0.30,
            "iterations": 4,
        },
        "radius": {"copy_parent_if_zero": True},
        "notes": (
            "This JSON controls the auto-labeling behavior "
            "(weights, thresholds, and options). Edit carefully."
        ),
    }


def _load_cfg() -> dict[str, Any]:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is not None:
        return dict(_DEFAULT_CFG)

    feature_cfg = load_feature_config(TOOL, FEATURE, default={})
    rules_cfg = feature_cfg.get("rules", feature_cfg if "feature" not in feature_cfg else {})
    _DEFAULT_CFG = merge_config(_default_rules_config(), rules_cfg)
    return dict(_DEFAULT_CFG)


def get_config() -> dict:
    """Return the active configuration dict (loaded from JSON if available)."""
    return _load_cfg()


def save_config(cfg: dict) -> None:
    """Save rule settings into the batch auto-typing feature config."""
    global _DEFAULT_CFG
    feature_cfg = load_feature_config(TOOL, FEATURE, default={})
    updated_cfg = merge_config(feature_cfg, {"rules": cfg})
    save_feature_config(TOOL, FEATURE, updated_cfg)
    _DEFAULT_CFG = merge_config(_default_rules_config(), cfg)


@dataclass
class RuleBatchOptions:
    soma: bool = False
    axon: bool = False
    apic: bool = False
    basal: bool = False
    rad: bool = False
    zip_output: bool = False


@dataclass
class RuleBatchResult:
    folder: str
    out_dir: str
    zip_path: str | None
    files_total: int
    files_processed: int
    files_failed: int
    total_nodes: int
    total_type_changes: int
    total_radius_changes: int
    failures: list[str]
    per_file: list[str]
    log_path: str | None


def _parse_swc(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    headers: list[str] = []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                headers.append(line.rstrip("\n"))
                continue

            parts = s.split()
            if len(parts) < 7:
                continue

            try:
                rid = int(float(parts[0]))
                rtype = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                radius = float(parts[5])
                parent = int(float(parts[6]))
            except Exception:
                continue

            rows.append(
                {
                    "id": rid,
                    "type": rtype,
                    "x": x,
                    "y": y,
                    "z": z,
                    "radius": radius,
                    "parent": parent,
                }
            )
    return headers, rows


def _build_topology(rows: list[dict[str, Any]]) -> tuple[list[int | None], list[list[int]], list[int]]:
    n = len(rows)
    id_to_idx = {int(row["id"]): i for i, row in enumerate(rows)}
    parent_idx: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]

    for i, row in enumerate(rows):
        pidx = id_to_idx.get(int(row["parent"]))
        parent_idx[i] = pidx
        if pidx is not None:
            children[pidx].append(i)

    roots = [i for i, row in enumerate(rows) if int(row["parent"]) == -1 or parent_idx[i] is None]
    roots.sort(key=lambda idx: int(rows[idx]["id"]))

    order: list[int] = []
    seen = set()
    queue = list(roots)
    while queue:
        idx = queue.pop(0)
        if idx in seen:
            continue
        seen.add(idx)
        order.append(idx)
        kids = sorted(children[idx], key=lambda k: int(rows[k]["id"]))
        queue.extend(kids)

    for i in sorted(range(n), key=lambda idx: int(rows[idx]["id"])):
        if i not in seen:
            order.append(i)

    return parent_idx, children, order


def _normalize_map(vals: dict[int, float]) -> dict[int, float]:
    if not vals:
        return {}
    lo = min(vals.values())
    hi = max(vals.values())
    if hi <= lo:
        return {k: 0.5 for k in vals}
    scale = hi - lo
    return {k: (v - lo) / scale for k, v in vals.items()}


def _iter_subtree(start: int, children: list[list[int]]) -> list[int]:
    out: list[int] = []
    stack = [start]
    seen = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
        stack.extend(children[i])
    return out


def _branch_partition(
    rows: list[dict[str, Any]],
    parent_idx: list[int | None],
    children: list[list[int]],
    types: list[int],
) -> tuple[dict[int, list[int]], dict[int, int], list[int]]:
    n = len(rows)
    roots = [i for i, p in enumerate(parent_idx) if p is None]
    soma_roots = [i for i in roots if int(types[i]) == 1]
    anchors = soma_roots if soma_roots else roots

    node_branch = [-1] * n
    branch_nodes: dict[int, list[int]] = {}
    branch_anchor: dict[int, int] = {}
    bid = 0

    for a in anchors:
        ch = sorted(children[a], key=lambda i: int(rows[i]["id"]))
        if not ch and int(types[a]) != 1:
            nodes = _iter_subtree(a, children)
            branch_nodes[bid] = nodes
            branch_anchor[bid] = a
            for x in nodes:
                node_branch[x] = bid
            bid += 1
            continue

        for c in ch:
            nodes = _iter_subtree(c, children)
            branch_nodes[bid] = nodes
            branch_anchor[bid] = a
            for x in nodes:
                node_branch[x] = bid
            bid += 1

    # Include disconnected leftovers as standalone branches.
    for i in range(n):
        if node_branch[i] != -1:
            continue
        if int(types[i]) == 1:
            continue
        nodes = _iter_subtree(i, children)
        branch_nodes[bid] = nodes
        branch_anchor[bid] = parent_idx[i] if parent_idx[i] is not None else i
        for x in nodes:
            if node_branch[x] == -1:
                node_branch[x] = bid
        bid += 1

    return branch_nodes, branch_anchor, node_branch


def _branch_scores(
    rows: list[dict[str, Any]],
    parent_idx: list[int | None],
    children: list[list[int]],
    types: list[int],
    branch_nodes: dict[int, list[int]],
    branch_anchor: dict[int, int],
    enabled_neurites: set[int],
) -> tuple[dict[int, dict[int, float]], dict[int, tuple[float, float, float, float, float]], dict[tuple[int, int], float]]:
    x = [float(r["x"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    z = [float(r["z"]) for r in rows]
    rad = [float(r["radius"]) for r in rows]

    path_len: dict[int, float] = {}
    radial_extent: dict[int, float] = {}
    mean_radius: dict[int, float] = {}
    branchiness: dict[int, float] = {}
    z_mean_rel: dict[int, float] = {}
    existing_ratio: dict[tuple[int, int], float] = {}

    for bid, nodes in branch_nodes.items():
        a = branch_anchor[bid]
        ax, ay, az = x[a], y[a], z[a]
        plen = 0.0
        max_r = 0.0
        bif = 0
        for i in nodes:
            p = parent_idx[i]
            if p is not None:
                dx = x[i] - x[p]
                dy = y[i] - y[p]
                dz = z[i] - z[p]
                plen += math.sqrt(dx * dx + dy * dy + dz * dz)
            dxa = x[i] - ax
            dya = y[i] - ay
            dza = z[i] - az
            max_r = max(max_r, math.sqrt(dxa * dxa + dya * dya + dza * dza))
            if len(children[i]) > 1:
                bif += 1

        path_len[bid] = plen
        radial_extent[bid] = max_r
        mean_radius[bid] = sum(rad[i] for i in nodes) / max(1, len(nodes))
        branchiness[bid] = bif / max(1, len(nodes))
        z_mean_rel[bid] = sum((z[i] - az) for i in nodes) / max(1, len(nodes))

        for cls in enabled_neurites:
            c = sum(1 for i in nodes if int(types[i]) == cls)
            existing_ratio[(bid, cls)] = c / max(1, len(nodes))

    n_path = _normalize_map(path_len)
    n_radial = _normalize_map(radial_extent)
    n_radius = _normalize_map(mean_radius)
    n_branch = _normalize_map(branchiness)
    n_z = _normalize_map(z_mean_rel)

    scores: dict[int, dict[int, float]] = {}
    features: dict[int, tuple[float, float, float, float, float]] = {}
    cfg = _load_cfg()
    weights = cfg.get("branch_score_weights", {})
    for bid in branch_nodes:
        features[bid] = (
            n_path.get(bid, 0.5),
            n_radial.get(bid, 0.5),
            n_radius.get(bid, 0.5),
            n_branch.get(bid, 0.5),
            n_z.get(bid, 0.5),
        )
        br_scores: dict[int, float] = {}
        for cls in enabled_neurites:
            prior = existing_ratio.get((bid, cls), 0.0)
            if cls == 2:  # axon
                w = weights.get("axon", {})
                s = (
                    w.get("path", 0.32) * n_path.get(bid, 0.5)
                    + w.get("radial", 0.24) * n_radial.get(bid, 0.5)
                    + w.get("radius", 0.20) * (1.0 - n_radius.get(bid, 0.5))
                    + w.get("branch", 0.14) * (1.0 - n_branch.get(bid, 0.5))
                    + w.get("prior", 0.10) * prior
                )
            elif cls == 4:  # apical
                w = weights.get("apical", {})
                s = (
                    w.get("z", 0.30) * n_z.get(bid, 0.5)
                    + w.get("path", 0.22) * n_path.get(bid, 0.5)
                    + w.get("radius", 0.18) * n_radius.get(bid, 0.5)
                    + w.get("branch", 0.15) * n_branch.get(bid, 0.5)
                    + w.get("prior", 0.15) * prior
                )
            else:  # basal
                w = weights.get("basal", {})
                s = (
                    w.get("z", 0.30) * (1.0 - n_z.get(bid, 0.5))
                    + w.get("branch", 0.22) * n_branch.get(bid, 0.5)
                    + w.get("radius", 0.18) * n_radius.get(bid, 0.5)
                    + w.get("path", 0.15) * n_path.get(bid, 0.5)
                    + w.get("prior", 0.15) * prior
                )
            br_scores[cls] = s
        scores[bid] = br_scores
    return scores, features, existing_ratio


def _euclid_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    d2 = 0.0
    for x, y in zip(a, b):
        d = x - y
        d2 += d * d
    dist = math.sqrt(d2)
    max_dist = math.sqrt(float(len(a)))
    if max_dist <= 0:
        return 0.5
    sim = 1.0 - (dist / max_dist)
    return max(0.0, min(1.0, sim))


def _ml_refine_scores(
    scores: dict[int, dict[int, float]],
    features: dict[int, tuple[float, float, float, float, float]],
    existing_ratio: dict[tuple[int, int], float],
    enabled_neurites: set[int],
) -> dict[int, dict[int, float]]:
    if not scores or not enabled_neurites:
        return scores

    classes = sorted(enabled_neurites)
    branch_ids = sorted(features.keys())
    if not branch_ids:
        return scores

    cfg = _load_cfg()
    seed_map: dict[int, list[int]] = {c: [] for c in classes}
    seed_prior_threshold = float(cfg.get("seed_prior_threshold", 0.55))
    for bid in branch_ids:
        priors = {c: existing_ratio.get((bid, c), 0.0) for c in classes}
        best_c = max(classes, key=lambda c: priors[c])
        if priors[best_c] >= seed_prior_threshold:
            seed_map[best_c].append(bid)

    for c in classes:
        if seed_map[c]:
            continue
        best_bid = max(branch_ids, key=lambda b: scores.get(b, {}).get(c, -1e9))
        seed_map[c].append(best_bid)

    prototypes: dict[int, tuple[float, float, float, float, float]] = {}
    for c in classes:
        seeds = seed_map[c]
        if not seeds:
            continue
        acc = [0.0] * 5
        for b in seeds:
            fv = features[b]
            for i in range(5):
                acc[i] += fv[i]
        n = float(len(seeds))
        prototypes[c] = tuple(v / n for v in acc)

    out: dict[int, dict[int, float]] = {}
    for bid in branch_ids:
        out[bid] = {}
        fv = features[bid]
        for c in classes:
            base = scores.get(bid, {}).get(c, 0.0)
            proto = prototypes.get(c)
            if proto is None:
                out[bid][c] = base
                continue
            sim = _euclid_similarity(fv, proto)
            ml_blend = float(cfg.get("ml_blend", 0.28))
            ml_base = float(cfg.get("ml_base_weight", 0.72))
            out[bid][c] = ml_base * base + ml_blend * sim
    return out


def _assign_branches(
    branch_nodes: dict[int, list[int]],
    scores: dict[int, dict[int, float]],
    enabled_neurites: set[int],
) -> dict[int, int]:
    if not branch_nodes or not enabled_neurites:
        return {}

    selected = sorted(enabled_neurites)
    assign: dict[int, int] = {}
    if len(selected) == 1:
        only = selected[0]
        for bid in branch_nodes:
            assign[bid] = only
        return assign

    for bid in branch_nodes:
        b_scores = scores.get(bid, {})
        cls = max(selected, key=lambda c: b_scores.get(c, -1e9))
        assign[bid] = cls

    missing = [c for c in selected if c not in set(assign.values())]
    if missing and len(branch_nodes) >= len(selected):
        for need in missing:
            best_bid = None
            best_gain = -1e9
            best_need_score = -1e9
            for bid in branch_nodes:
                cur = assign[bid]
                cur_s = scores.get(bid, {}).get(cur, 0.0)
                need_s = scores.get(bid, {}).get(need, 0.0)
                gain = need_s - cur_s
                if gain > best_gain:
                    best_gain = gain
                    best_need_score = need_s
                    best_bid = bid
            assign_cfg = _load_cfg().get("assign_missing", {})
            min_score = float(assign_cfg.get("min_score", 0.58))
            min_gain = float(assign_cfg.get("min_gain", -0.06))
            if best_bid is not None and (best_need_score >= min_score and best_gain >= min_gain):
                assign[best_bid] = need
    return assign


def _smooth_branch_labels(
    branch_class: dict[int, int],
    scores: dict[int, dict[int, float]],
    branch_anchor: dict[int, int],
) -> dict[int, int]:
    if not branch_class:
        return branch_class

    out = dict(branch_class)
    anchor_to_branches: dict[int, list[int]] = {}
    for bid, a in branch_anchor.items():
        anchor_to_branches.setdefault(a, []).append(bid)

    for bid, cur_cls in list(out.items()):
        anchor = branch_anchor.get(bid)
        sibs = [s for s in anchor_to_branches.get(anchor, []) if s != bid]
        if len(sibs) < 2:
            continue

        counts: dict[int, int] = {}
        for s in sibs:
            c = out.get(s)
            if c is None:
                continue
            counts[c] = counts.get(c, 0) + 1
        if not counts:
            continue

        maj_cls, maj_count = max(counts.items(), key=lambda kv: kv[1])
        if maj_cls == cur_cls:
            continue
        smooth_cfg = _load_cfg().get("smoothing", {})
        maj_frac = float(smooth_cfg.get("maj_fraction", 0.67))
        flip_margin = float(smooth_cfg.get("flip_margin", 0.10))
        if maj_count / max(1, len(sibs)) < maj_frac:
            continue

        cur_score = scores.get(bid, {}).get(cur_cls, 0.0)
        maj_score = scores.get(bid, {}).get(maj_cls, 0.0)
        if cur_score - maj_score < flip_margin:
            out[bid] = maj_cls

    return out


def _apply_rules(rows: list[dict[str, Any]], opts: RuleBatchOptions) -> tuple[list[int], list[float], int, int]:
    orig_types = [int(row["type"]) for row in rows]
    types = list(orig_types)
    orig_radii = [float(row["radius"]) for row in rows]
    radii = list(orig_radii)
    parent_idx, children, order = _build_topology(rows)

    if opts.soma:
        for i, row in enumerate(rows):
            if int(row["parent"]) == -1 and types[i] != 1:
                types[i] = 1

    enabled_neurites: set[int] = set()
    if opts.axon:
        enabled_neurites.add(2)
    if opts.basal:
        enabled_neurites.add(3)
    if opts.apic:
        enabled_neurites.add(4)

    if enabled_neurites:
        branch_nodes, branch_anchor, node_branch = _branch_partition(rows, parent_idx, children, types)
        scores, features, existing_ratio = _branch_scores(
            rows, parent_idx, children, types, branch_nodes, branch_anchor, enabled_neurites
        )
        scores = _ml_refine_scores(scores, features, existing_ratio, enabled_neurites)
        branch_class = _assign_branches(branch_nodes, scores, enabled_neurites)
        branch_class = _smooth_branch_labels(branch_class, scores, branch_anchor)

        for bid, nodes in branch_nodes.items():
            cls = branch_class.get(bid)
            if cls is None:
                continue
            for i in nodes:
                if opts.soma and int(types[i]) == 1:
                    continue
                types[i] = cls

        prop_cfg = _load_cfg().get("propagation_weights", {})
        w_self = float(prop_cfg.get("self", 0.35))
        w_parent = float(prop_cfg.get("parent", 0.35))
        w_children_total = float(prop_cfg.get("children", 0.20))
        w_branch_prior = float(prop_cfg.get("branch_prior", 0.30))
        prop_iters = int(prop_cfg.get("iterations", 4))

        for _ in range(prop_iters):
            new_types = list(types)
            for idx in order:
                if opts.soma and int(types[idx]) == 1:
                    continue
                if idx < len(node_branch) and node_branch[idx] == -1:
                    continue
                votes: dict[int, float] = {c: 0.0 for c in enabled_neurites}
                cur = int(types[idx])
                if cur in votes:
                    votes[cur] += w_self

                pidx = parent_idx[idx]
                if pidx is not None:
                    pt = int(types[pidx])
                    if pt in votes:
                        votes[pt] += w_parent

                ch = children[idx]
                if ch:
                    w = (w_children_total / len(ch)) if len(ch) > 0 else 0.0
                    for c in ch:
                        ct = int(types[c])
                        if ct in votes:
                            votes[ct] += w

                bid = node_branch[idx] if idx < len(node_branch) else -1
                if bid in branch_class:
                    votes[branch_class[bid]] = votes.get(branch_class[bid], 0.0) + w_branch_prior

                if votes:
                    best = max(votes.items(), key=lambda kv: kv[1])[0]
                    new_types[idx] = best
            types = new_types

        if opts.soma:
            for i, row in enumerate(rows):
                if int(row["parent"]) == -1:
                    types[i] = 1

    if opts.rad:
        radius_cfg = _load_cfg().get("radius", {})
        copy_parent = bool(radius_cfg.get("copy_parent_if_zero", True))
        if copy_parent:
            for idx in order:
                pidx = parent_idx[idx]
                if pidx is None:
                    continue
                if radii[idx] <= 0 and radii[pidx] > 0:
                    radii[idx] = radii[pidx]

    type_changes = sum(1 for old, new in zip(orig_types, types) if int(old) != int(new))
    radius_changes = sum(1 for old, new in zip(orig_radii, radii) if float(old) != float(new))
    return types, radii, type_changes, radius_changes


def _write_swc(path: Path, headers: list[str], rows: list[dict[str, Any]], types: list[int], radii: list[float]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for h in headers:
            fh.write(f"{h}\n")
        for i, row in enumerate(rows):
            fh.write(
                f"{int(row['id'])} {int(types[i])} "
                f"{float(row['x']):.10g} {float(row['y']):.10g} {float(row['z']):.10g} "
                f"{float(radii[i]):.10g} {int(row['parent'])}\n"
            )


def run_rule_batch(folder: str, opts: RuleBatchOptions) -> RuleBatchResult:
    in_dir = Path(folder)
    swc_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".swc"])

    out_dir = in_dir / f"{in_dir.name}_auto_typing"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    per_file: list[str] = []
    change_details: list[str] = []

    processed = 0
    total_nodes = 0
    total_type_changes = 0
    total_radius_changes = 0

    for swc_path in swc_files:
        try:
            headers, rows = _parse_swc(swc_path)
            if not rows:
                failures.append(f"{swc_path.name}: no valid SWC rows")
                continue

            orig_types = [int(r["type"]) for r in rows]
            orig_radii = [float(r["radius"]) for r in rows]
            types, radii, type_changes, radius_changes = _apply_rules(rows, opts)
            out_path = out_dir / swc_path.name
            _write_swc(out_path, headers, rows, types, radii)

            processed += 1
            total_nodes += len(rows)
            total_type_changes += type_changes
            total_radius_changes += radius_changes
            out_counts = {
                1: sum(1 for t in types if int(t) == 1),
                2: sum(1 for t in types if int(t) == 2),
                3: sum(1 for t in types if int(t) == 3),
                4: sum(1 for t in types if int(t) == 4),
            }
            per_file.append(
                f"{swc_path.name}: nodes={len(rows)}, type_changes={type_changes}, "
                f"radius_changes={radius_changes}, out_types(soma/axon/basal/apic)="
                f"{out_counts[1]}/{out_counts[2]}/{out_counts[3]}/{out_counts[4]}"
            )

            if type_changes > 0 or radius_changes > 0:
                change_details.append(f"[{swc_path.name}]")
            if type_changes > 0:
                change_details.append("type_changes:")
                for row, old_t, new_t in zip(rows, orig_types, types):
                    if int(old_t) != int(new_t):
                        change_details.append(
                            f"  node_id={int(row['id'])}: old_type={int(old_t)} -> new_type={int(new_t)}"
                        )
            if radius_changes > 0:
                change_details.append("radius_changes:")
                for row, old_r, new_r in zip(rows, orig_radii, radii):
                    if float(old_r) != float(new_r):
                        change_details.append(
                            f"  node_id={int(row['id'])}: old_radius={float(old_r):.10g} -> new_radius={float(new_r):.10g}"
                        )
            if type_changes > 0 or radius_changes > 0:
                change_details.append("")
        except Exception as e:
            failures.append(f"{swc_path.name}: {e}")

    zip_path: str | None = None
    if opts.zip_output and processed > 0:
        zip_target = in_dir / f"{in_dir.name}_auto_typing.zip"
        with zipfile.ZipFile(zip_target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.glob("*.swc")):
                zf.write(f, arcname=f"{out_dir.name}/{f.name}")
        zip_path = str(zip_target)

    payload = {
        "folder": str(in_dir),
        "out_dir": str(out_dir),
        "zip_path": zip_path,
        "files_total": len(swc_files),
        "files_processed": processed,
        "files_failed": len(failures),
        "total_nodes": total_nodes,
        "total_type_changes": total_type_changes,
        "total_radius_changes": total_radius_changes,
        "failures": failures,
        "per_file": per_file,
        "change_details": change_details,
    }
    log_path = write_text_report(out_dir / "auto_typing_report.txt", format_auto_typing_report_text(payload))

    return RuleBatchResult(
        folder=str(in_dir),
        out_dir=str(out_dir),
        zip_path=zip_path,
        files_total=len(swc_files),
        files_processed=processed,
        files_failed=len(failures),
        total_nodes=total_nodes,
        total_type_changes=total_type_changes,
        total_radius_changes=total_radius_changes,
        failures=failures,
        per_file=per_file,
        log_path=log_path,
    )
