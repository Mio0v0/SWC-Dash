"""Auto typing feature for Batch Processing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swctools.core.auto_typing import RuleBatchOptions, run_rule_batch
from swctools.core.config import load_feature_config, merge_config
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "batch_processing"
FEATURE = "auto_typing"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "options": {
        "soma": True,
        "axon": True,
        "apic": False,
        "basal": True,
        "rad": False,
        "zip_output": False,
    },
    "rules": {
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
    },
}


def _builtin_run(folder: str, options: RuleBatchOptions, config: dict[str, Any]):
    _ = config
    return run_rule_batch(folder, options)


register_builtin_method(FEATURE_KEY, "default", _builtin_run)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def _options_from_config(cfg: dict[str, Any]) -> RuleBatchOptions:
    opts = cfg.get("options", {})
    return RuleBatchOptions(
        soma=bool(opts.get("soma", True)),
        axon=bool(opts.get("axon", True)),
        apic=bool(opts.get("apic", False)),
        basal=bool(opts.get("basal", True)),
        rad=bool(opts.get("rad", False)),
        zip_output=bool(opts.get("zip_output", False)),
    )


def run_folder(
    folder: str,
    *,
    options: RuleBatchOptions | None = None,
    config_overrides: dict | None = None,
):
    cfg = merge_config(get_config(), config_overrides)

    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    opts = options if options is not None else _options_from_config(cfg)
    return fn(folder, opts, cfg)


def options_to_dict(opts: RuleBatchOptions) -> dict[str, Any]:
    return asdict(opts)
