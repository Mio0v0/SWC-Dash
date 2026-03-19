"""Core validation adapters.

This module keeps backward-compatible function signatures while routing check
execution through the shared validation backend in :mod:`swctools.core`.
"""

from __future__ import annotations

import os
from typing import Any

from swctools.core import validation_impl as _legacy


_PASS = "pass"
_WARN = "warning"
_FAIL = "fail"
_ERROR = "error"


def _to_legacy_status(status: str, message: str) -> bool | str:
    s = str(status or "").lower()
    if s == _PASS:
        return True
    if s == _ERROR:
        return f"ERROR: {message}"
    # Keep backward-compatible bool behavior for warn/fail.
    return False


def _sanitize_swc_text(swc_text: str) -> tuple[str, bytes]:
    arr = _legacy._load_swc_to_array(swc_text)
    if arr.size:
        _legacy._sanitize_types_inplace(arr)

    tmp_path = _legacy._write_array_to_tmp_swc(arr)
    try:
        with open(tmp_path, "rb") as f:
            out_bytes = f.read()
        out_text = out_bytes.decode("utf-8", errors="ignore")
        return out_text, out_bytes
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def run_format_validation_from_text(swc_text: str):
    """Run unified validation backend and return legacy-compatible shape.

    Returns:
      results_dict: {check_key: bool|str("ERROR: ...")}
      sanitized_swc_bytes: bytes
      table_rows: [{"check": label, "status": bool|str}, ...]
    """
    from swctools.core.validation_engine import run_validation_text

    sanitized_text, sanitized_bytes = _sanitize_swc_text(swc_text)
    report = run_validation_text(sanitized_text, profile="default")

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for item in report.results:
        legacy_status = _to_legacy_status(item.status, item.message)
        results[item.key] = legacy_status
        rows.append({"check": item.label, "status": legacy_status})
    return results, sanitized_bytes, rows


def run_per_tree_validation(swc_text: str):
    """Run unified validation backend for each soma-split tree."""
    from swctools.core.validation_engine import run_validation_text

    trees = _legacy._split_swc_by_soma_roots(swc_text)
    if not trees:
        trees = _legacy._split_swc_by_trees(swc_text)
    if not trees:
        return [], []

    labels: dict[str, str] = {}
    tree_results = []

    for root_id, sub_text, node_count in trees:
        report = run_validation_text(sub_text, profile="default")
        row_map: dict[str, Any] = {}
        for item in report.results:
            labels.setdefault(item.key, item.label)
            row_map[item.key] = _to_legacy_status(item.status, item.message)
        tree_results.append((root_id, node_count, row_map))

    check_names = sorted(labels.items(), key=lambda x: x[1].lower())
    return check_names, tree_results


def clear_cache() -> None:
    # Legacy cache no longer drives validation checks, but keep the API.
    _legacy.clear_cache()


def _split_swc_by_soma_roots(swc_text: str):
    return _legacy._split_swc_by_soma_roots(swc_text)


__all__ = [
    "run_format_validation_from_text",
    "run_per_tree_validation",
    "_split_swc_by_soma_roots",
    "clear_cache",
]


def validate_text(swc_text: str) -> Any:
    """Validate SWC text and return full result (wrapper)."""
    return run_format_validation_from_text(swc_text)


def per_tree(swc_text: str) -> Any:
    return run_per_tree_validation(swc_text)
