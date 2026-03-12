"""Native validation checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from swctools.validation.registry import register_check
from swctools.validation.results import CheckResult


_REGISTERED = False


def _native_cache(ctx) -> dict[str, Any]:
    cache = getattr(ctx, "_native_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_native_cache", cache)
    return cache


def _segment_length_stats(ctx) -> dict[str, Any]:
    cache = _native_cache(ctx)
    key = "segment_length_stats"
    if key in cache:
        return cache[key]

    ids = np.asarray(ctx.ids, dtype=np.int64)
    parents = np.asarray(ctx.parents, dtype=np.int64)
    xyz = np.asarray(ctx.xyz, dtype=np.float64)

    child_mask = parents >= 0
    child_idx = np.flatnonzero(child_mask)
    if child_idx.size == 0:
        out = {"segment_count": 0, "invalid_ids": []}
        cache[key] = out
        return out

    parent_ids = parents[child_idx]
    sort_idx = np.argsort(ids)
    ids_sorted = ids[sort_idx]
    pos = np.searchsorted(ids_sorted, parent_ids)
    valid = (pos < ids_sorted.size) & (ids_sorted[pos] == parent_ids)
    child_idx = child_idx[valid]
    if child_idx.size == 0:
        out = {"segment_count": 0, "invalid_ids": []}
        cache[key] = out
        return out

    parent_idx = sort_idx[pos[valid]]
    lengths = np.linalg.norm(xyz[child_idx] - xyz[parent_idx], axis=1)
    bad_mask = (~np.isfinite(lengths)) | (lengths <= 0.0)
    bad_ids = ids[child_idx[bad_mask]].astype(np.int64).tolist()
    out = {
        "segment_count": int(lengths.size),
        "invalid_ids": bad_ids,
    }
    cache[key] = out
    return out


def _check_has_soma(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    ids = np.asarray(ctx.ids, dtype=np.int64)
    soma_mask = np.asarray(ctx.types, dtype=np.int64) == 1
    soma_ids = ids[soma_mask].astype(np.int64).tolist()
    passed = len(soma_ids) > 0
    msg = "Soma present." if passed else "No soma node found."
    return CheckResult.from_pass_fail(
        key="has_soma",
        label="Soma present",
        passed=passed,
        severity="warning",
        message=msg,
        source="native",
        failing_node_ids=[] if passed else list(soma_ids),
        metrics={"soma_count": len(soma_ids)},
    )


def _check_has_axon(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    count = int(np.sum(ctx.types == 2)) if len(ctx.types) else 0
    passed = count > 0
    return CheckResult.from_pass_fail(
        key="has_axon",
        label="Axon present",
        passed=passed,
        severity="warning",
        message="Axon present." if passed else "No axon node found.",
        source="native",
        metrics={"axon_node_count": count},
    )


def _check_has_basal_dendrite(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    count = int(np.sum(ctx.types == 3)) if len(ctx.types) else 0
    passed = count > 0
    return CheckResult.from_pass_fail(
        key="has_basal_dendrite",
        label="Basal dendrite present",
        passed=passed,
        severity="warning",
        message="Basal dendrite present." if passed else "No basal dendrite node found.",
        source="native",
        metrics={"basal_node_count": count},
    )


def _check_has_apical_dendrite(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    count = int(np.sum(ctx.types == 4)) if len(ctx.types) else 0
    passed = count > 0
    return CheckResult.from_pass_fail(
        key="has_apical_dendrite",
        label="Apical dendrite present",
        passed=passed,
        severity="warning",
        message="Apical dendrite present." if passed else "No apical dendrite node found.",
        source="native",
        metrics={"apical_node_count": count},
    )


def _check_all_neurite_radii_nonzero(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    ids = np.asarray(ctx.ids, dtype=np.int64)
    types = np.asarray(ctx.types, dtype=np.int64)
    radii = np.asarray(ctx.radii, dtype=np.float64)
    bad_mask = (types != 1) & ((~np.isfinite(radii)) | (radii <= 0.0))
    bad_ids = ids[bad_mask].astype(np.int64).tolist()
    passed = len(bad_ids) == 0
    msg = "All neurite radii are positive." if passed else f"Found {len(bad_ids)} neurite nodes with non-positive/NaN radius."
    return CheckResult.from_pass_fail(
        key="all_neurite_radii_nonzero",
        label="All neurite radii are positive",
        passed=passed,
        severity="error",
        message=msg,
        source="native",
        failing_node_ids=bad_ids,
        metrics={"invalid_radius_count": len(bad_ids)},
    )


def _check_all_segment_lengths_nonzero(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    stats = _segment_length_stats(ctx)
    bad_ids = list(stats["invalid_ids"])
    passed = len(bad_ids) == 0
    msg = "All segment lengths are positive." if passed else f"Found {len(bad_ids)} zero-length/invalid segments."
    return CheckResult.from_pass_fail(
        key="all_segment_lengths_nonzero",
        label="All segment lengths are positive",
        passed=passed,
        severity="error",
        message=msg,
        source="native",
        failing_node_ids=bad_ids,
        failing_section_ids=bad_ids,
        metrics={"segment_count": int(stats["segment_count"]), "invalid_segment_count": len(bad_ids)},
    )


def _check_all_section_lengths_nonzero(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    stats = _segment_length_stats(ctx)
    bad_ids = list(stats["invalid_ids"])
    passed = len(bad_ids) == 0
    msg = (
        "All section lengths are positive."
        if passed
        else "Detected section(s) with non-positive total length (segment-based approximation)."
    )
    return CheckResult.from_pass_fail(
        key="all_section_lengths_nonzero",
        label="All section lengths are positive",
        passed=passed,
        severity="error",
        message=msg,
        source="native",
        failing_node_ids=bad_ids,
        failing_section_ids=bad_ids,
        metrics={"section_count_approx": int(stats["segment_count"]), "invalid_section_count": len(bad_ids)},
    )


def _check_no_dangling_branches(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    ids = np.asarray(ctx.ids, dtype=np.int64)
    parents = np.asarray(ctx.parents, dtype=np.int64)
    root_ids = ids[parents < 0].astype(np.int64).tolist()
    passed = len(root_ids) <= 1
    msg = "No dangling branches." if passed else f"Detected {len(root_ids)} roots; expected one connected tree."
    return CheckResult.from_pass_fail(
        key="no_dangling_branches",
        label="No dangling branches",
        passed=passed,
        severity="error",
        message=msg,
        source="native",
        failing_node_ids=[] if passed else root_ids,
        metrics={"root_count": len(root_ids)},
    )


def _check_no_duplicate_3d_points(ctx, params: dict[str, Any]) -> CheckResult:
    _ = params
    ids = np.asarray(ctx.ids, dtype=np.int64)
    xyz = np.ascontiguousarray(np.asarray(ctx.xyz, dtype=np.float64))
    if xyz.shape[0] == 0:
        dup_ids: list[int] = []
    else:
        row_view = xyz.view(np.dtype((np.void, xyz.dtype.itemsize * xyz.shape[1]))).ravel()
        _, first_idx, inverse = np.unique(row_view, return_index=True, return_inverse=True)
        counts = np.bincount(inverse)
        repeated = counts[inverse] > 1
        first_mask = np.zeros(row_view.shape[0], dtype=bool)
        first_mask[first_idx] = True
        dup_mask = repeated & (~first_mask)
        dup_ids = ids[dup_mask].astype(np.int64).tolist()
    passed = len(dup_ids) == 0
    msg = "No duplicate 3D points." if passed else f"Found {len(dup_ids)} duplicated 3D points."
    return CheckResult.from_pass_fail(
        key="no_duplicate_3d_points",
        label="No duplicate 3D points",
        passed=passed,
        severity="error",
        message=msg,
        source="native",
        failing_node_ids=dup_ids,
        metrics={"duplicate_point_count": len(dup_ids)},
    )


def _check_radius_upper_bound(ctx, params: dict[str, Any]) -> CheckResult:
    max_radius = float(params.get("max_radius", 20.0))
    ids = np.asarray(ctx.ids, dtype=np.int64)
    radii = np.asarray(ctx.radii, dtype=np.float64)
    bad_ids = ids[radii > max_radius].astype(np.int64).tolist()
    passed = len(bad_ids) == 0
    msg = (
        f"All radii are <= {max_radius:g}."
        if passed
        else f"Found {len(bad_ids)} nodes with radius > {max_radius:g}."
    )
    return CheckResult.from_pass_fail(
        key="radius_upper_bound",
        label="Radius upper bound",
        passed=passed,
        severity="warning",
        message=msg,
        source="native",
        failing_node_ids=bad_ids,
        params_used={"max_radius": max_radius},
        metrics={"max_radius_observed": float(np.max(radii)) if radii.size else 0.0},
    )


def register_native_checks() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    register_check(key="has_soma", label="Soma present", source="native", runner=_check_has_soma)
    register_check(key="has_axon", label="Axon present", source="native", runner=_check_has_axon)
    register_check(
        key="has_basal_dendrite",
        label="Basal dendrite present",
        source="native",
        runner=_check_has_basal_dendrite,
    )
    register_check(
        key="has_apical_dendrite",
        label="Apical dendrite present",
        source="native",
        runner=_check_has_apical_dendrite,
    )
    register_check(
        key="all_neurite_radii_nonzero",
        label="All neurite radii are positive",
        source="native",
        runner=_check_all_neurite_radii_nonzero,
    )
    register_check(
        key="all_section_lengths_nonzero",
        label="All section lengths are positive",
        source="native",
        runner=_check_all_section_lengths_nonzero,
    )
    register_check(
        key="all_segment_lengths_nonzero",
        label="All segment lengths are positive",
        source="native",
        runner=_check_all_segment_lengths_nonzero,
    )
    register_check(
        key="no_dangling_branches",
        label="No dangling branches",
        source="native",
        runner=_check_no_dangling_branches,
    )
    register_check(
        key="no_duplicate_3d_points",
        label="No duplicate 3D points",
        source="native",
        runner=_check_no_duplicate_3d_points,
    )
    register_check(
        key="radius_upper_bound",
        label="Radius upper bound",
        source="native",
        runner=_check_radius_upper_bound,
    )
    _REGISTERED = True
