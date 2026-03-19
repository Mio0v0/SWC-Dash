"""Wrapped neuron_morphology (NeuroM) checks."""

from __future__ import annotations

from typing import Any, Callable

from neurom.check import morphology_checks as nm_checks

from swctools.core.validation_registry import register_check
from swctools.core.validation_results import CheckResult


_REGISTERED = False


def _run_neurom_bool_check(
    *,
    key: str,
    label: str,
    func: Callable,
    ctx,
    params: dict[str, Any],
) -> CheckResult:
    _ = params
    morph = ctx.get_morphology()
    if morph is None:
        return CheckResult.from_pass_fail(
            key=key,
            label=label,
            passed=False,
            severity="error",
            message=f"Unable to build morphology for NeuroM check: {ctx.morphology_error}",
            source="neuron_morphology",
            error=True,
        )

    try:
        code = getattr(func, "__code__", None)
        if code is not None and "neurite_filter" in code.co_varnames:
            passed = bool(func(morph, neurite_filter=None))
        else:
            passed = bool(func(morph))
        msg = f"{label}: {'pass' if passed else 'fail'}."
        return CheckResult.from_pass_fail(
            key=key,
            label=label,
            passed=passed,
            severity="error",
            message=msg,
            source="neuron_morphology",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult.from_pass_fail(
            key=key,
            label=label,
            passed=False,
            severity="error",
            message=f"NeuroM check error: {e}",
            source="neuron_morphology",
            error=True,
        )


def _wrapper(key: str, label: str, nm_name: str):
    func = getattr(nm_checks, nm_name)

    def _run(ctx, params: dict[str, Any]) -> CheckResult:
        return _run_neurom_bool_check(key=key, label=label, func=func, ctx=ctx, params=params)

    return _run


def register_neuron_morphology_checks() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    mapping = {
        "has_multifurcation": ("Contains multifurcation", "has_multifurcation"),
        "no_back_tracking": ("No geometric backtracking", "has_no_back_tracking"),
        "no_fat_terminal_ends": ("No oversized terminal ends", "has_no_fat_ends"),
        "no_flat_neurites": ("No flattened neurites", "has_no_flat_neurites"),
        "no_section_index_jumps": ("No section index gaps", "has_no_jumps"),
        "no_ultranarrow_sections": ("No extremely narrow sections", "has_no_narrow_neurite_section"),
        "no_ultranarrow_starts": ("No extremely narrow branch starts", "has_no_narrow_start"),
        "no_root_index_jumps": ("No root index gaps", "has_no_root_node_jumps"),
        "no_single_child_chains": ("No single-child chains", "has_no_single_children"),
        "soma_radius_nonzero": ("Soma radius is positive", "has_nonzero_soma_radius"),
        "has_unifurcation": ("Contains unifurcation", "has_unifurcation"),
    }

    for key, (label, nm_name) in mapping.items():
        register_check(
            key=key,
            label=label,
            source="neuron_morphology",
            runner=_wrapper(key, label, nm_name),
        )

    _REGISTERED = True
