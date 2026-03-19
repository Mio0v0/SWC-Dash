"""Shared auto-typing guide text for CLI and GUI."""

from __future__ import annotations

from typing import Any

from swctools.core.auto_typing import get_auto_rules_config


def _f(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float(default)


def _i(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return int(default)


def _b(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    if v is None:
        return bool(default)
    return bool(v)


def get_auto_typing_guide(config: dict[str, Any] | None = None) -> dict[str, str]:
    cfg = config if isinstance(config, dict) else get_auto_rules_config()

    w_axon = cfg.get("branch_score_weights", {}).get("axon", {})
    w_apical = cfg.get("branch_score_weights", {}).get("apical", {})
    w_basal = cfg.get("branch_score_weights", {}).get("basal", {})

    ax_path = _f(w_axon.get("path", 0.32), 0.32)
    ax_radial = _f(w_axon.get("radial", 0.24), 0.24)
    ax_radius = _f(w_axon.get("radius", 0.20), 0.20)
    ax_branch = _f(w_axon.get("branch", 0.14), 0.14)
    ax_prior = _f(w_axon.get("prior", 0.10), 0.10)

    ap_z = _f(w_apical.get("z", 0.30), 0.30)
    ap_path = _f(w_apical.get("path", 0.22), 0.22)
    ap_radius = _f(w_apical.get("radius", 0.18), 0.18)
    ap_branch = _f(w_apical.get("branch", 0.15), 0.15)
    ap_prior = _f(w_apical.get("prior", 0.15), 0.15)

    ba_z = _f(w_basal.get("z", 0.30), 0.30)
    ba_branch = _f(w_basal.get("branch", 0.22), 0.22)
    ba_radius = _f(w_basal.get("radius", 0.18), 0.18)
    ba_path = _f(w_basal.get("path", 0.15), 0.15)
    ba_prior = _f(w_basal.get("prior", 0.15), 0.15)

    seed_prior_threshold = _f(cfg.get("seed_prior_threshold", 0.55), 0.55)
    ml_blend = _f(cfg.get("ml_blend", 0.28), 0.28)
    ml_base_weight = _f(cfg.get("ml_base_weight", 0.72), 0.72)

    assign_cfg = cfg.get("assign_missing", {})
    min_score = _f(assign_cfg.get("min_score", 0.58), 0.58)
    min_gain = _f(assign_cfg.get("min_gain", -0.06), -0.06)

    smooth_cfg = cfg.get("smoothing", {})
    maj_fraction = _f(smooth_cfg.get("maj_fraction", 0.67), 0.67)
    flip_margin = _f(smooth_cfg.get("flip_margin", 0.10), 0.10)

    prop_cfg = cfg.get("propagation_weights", {})
    w_self = _f(prop_cfg.get("self", 0.35), 0.35)
    w_parent = _f(prop_cfg.get("parent", 0.35), 0.35)
    w_children = _f(prop_cfg.get("children", 0.20), 0.20)
    w_branch_prior = _f(prop_cfg.get("branch_prior", 0.30), 0.30)
    prop_iters = _i(prop_cfg.get("iterations", 4), 4)

    radius_cfg = cfg.get("radius", {})
    copy_parent_if_zero = _b(radius_cfg.get("copy_parent_if_zero", True), True)

    body = (
        "This panel shows the JSON configuration that controls the auto-labeling\n"
        "algorithm.\n"
        "You can edit thresholds and weights and save to change behavior.\n\n"
        "Decision summary:\n"
        "1) Partition branches anchored at soma/roots.\n"
        "2) Compute branch features (path length, radial extent, mean radius,\n"
        "   branchiness, z-mean).\n"
        "3) Score each branch for axon/apical/basal using weighted features + prior\n"
        "   from existing labels.\n"
        "4) Optionally refine scores via a nearest-centroid (ML) step seeded by\n"
        "   confident branches.\n"
        "5) Assign branch-level classes, smooth locally among siblings, then\n"
        "   propagate labels to nodes using neighborhood votes.\n"
        "6) Radius rule: copy parent radius into zero/invalid radii when enabled.\n\n"
        "Type decision boundaries:\n"
        "- Soma (type 1): if --soma is enabled, root nodes (parent == -1) are\n"
        "  forced to soma before and after propagation.\n"
        "- Dendrite note: --dendrite means enabling both basal (type 3) and\n"
        "  apical (type 4). There is no separate single \"dendrite\" class.\n"
        "- Axon score:\n"
        f"  score_axon = {ax_path:.3f}*path + {ax_radial:.3f}*radial + "
        f"{ax_radius:.3f}*(1-radius) + {ax_branch:.3f}*(1-branchiness) + {ax_prior:.3f}*prior\n"
        "- Apical score:\n"
        f"  score_apical = {ap_z:.3f}*z + {ap_path:.3f}*path + {ap_radius:.3f}*radius + "
        f"{ap_branch:.3f}*branchiness + {ap_prior:.3f}*prior\n"
        "- Basal score:\n"
        f"  score_basal = {ba_z:.3f}*(1-z) + {ba_branch:.3f}*branchiness + {ba_radius:.3f}*radius + "
        f"{ba_path:.3f}*path + {ba_prior:.3f}*prior\n"
        "- Branch class assignment: choose highest score among enabled classes.\n\n"
        "Global thresholds:\n"
        f"- Seed prior threshold: prior >= {seed_prior_threshold:.3f}\n"
        f"- Missing-class reassignment: score >= {min_score:.3f} and gain >= {min_gain:.3f}\n"
        f"- Sibling smoothing: majority >= {maj_fraction:.3f} and margin < {flip_margin:.3f}\n"
        f"- ML blend: final = ({ml_base_weight:.3f} * base_score) + ({ml_blend:.3f} * similarity)\n"
        f"- Node propagation votes: self={w_self:.3f}, parent={w_parent:.3f}, "
        f"children_total={w_children:.3f}, branch_prior={w_branch_prior:.3f}, iterations={prop_iters}\n"
        f"- Radius boundary: copy_parent_if_zero = {copy_parent_if_zero}"
    )
    return {"title": "Decision engine — auto-label rules", "body": body}


def format_auto_typing_guide_text(config: dict[str, Any] | None = None) -> str:
    guide = get_auto_typing_guide(config=config)
    title = str(guide.get("title", "Auto Typing Rule Guide"))
    body = str(guide.get("body", "")).rstrip()
    return f"{title}\n{'-' * len(title)}\n{body}".rstrip()
