# validation_core.py
import os
import io
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Iterable

import numpy as np
import morphio
from neurom.core import Morphology
from neurom.check import morphology_checks as checks

# Keep MorphIO quiet
morphio.set_maximum_warnings(0)

# --------- HUMAN-READABLE LABELS ----------
FRIENDLY_LABELS = {
    "has_all_nonzero_neurite_radii":    "Neurite radii are non-zero",
    "has_all_nonzero_section_lengths":  "All section lengths > 0",
    "has_all_nonzero_segment_lengths":  "All segment lengths > 0",
    "has_apical_dendrite":              "Has apical dendrite",
    "has_axon":                         "Has axon",
    "has_basal_dendrite":               "Has basal dendrite",
    "has_multifurcation":               "Contains any multifurcation",
    "has_no_back_tracking":             "No geometric back-tracking",
    "has_no_dangling_branch":           "No dangling branches",
    "has_no_fat_ends":                  "No “fat” terminal ends",
    "has_no_flat_neurites":             "No flat neurites",
    "has_no_jumps":                     "No section index jumps",
    "has_no_narrow_neurite_section":    "No ultra-narrow sections",
    "has_no_narrow_start":              "No ultra-narrow starts",
    "has_no_overlapping_point":         "No duplicate 3D points",
    "has_no_root_node_jumps":           "No root index jumps",
    "has_no_single_children":           "No single-child chains",
    "has_nonzero_soma_radius":          "Soma radius > 0 (if present)",
    "has_unifurcation":                 "Has any unifurcation",
    # custom:
    "has_soma":                         "Has soma",
}

def _friendly_label(name: str) -> str:
    if name in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[name]
    base = name[4:] if name.startswith("has_") else name
    return base.replace("_", " ").capitalize()

# --------- PRE-RESOLVE CHECKS (avoid per-call reflection) ----------
_CHECK_FUNCS: List[Tuple[str, Any, bool]] = []
for _name in dir(checks):
    if not _name.startswith("has_"):
        continue
    _func = getattr(checks, _name)
    if callable(_func):
        co = getattr(_func, "__code__", None)
        co_vars = getattr(co, "co_varnames", ())
        _CHECK_FUNCS.append((_name, _func, "neurite_filter" in co_vars))

# Fixed-set selection:
#   - include everything EXCEPT the very slow "has_no_back_tracking"
#   - explicitly keep "has_no_overlapping_point"
_EXCLUDE = {"has_no_back_tracking"}

def _selected_checks() -> Iterable[Tuple[str, Any, bool]]:
    for n, f, nf in _CHECK_FUNCS:
        if n in _EXCLUDE:
            continue
        yield (n, f, nf)

def _run_one_check(name, func, needs_nf, morph):
    try:
        if needs_nf:
            return name, bool(func(morph, neurite_filter=None))
        return name, bool(func(morph))
    except Exception as e:
        return name, f"ERROR: {e}"

# --------- FAST I/O HELPERS ----------
_SWCTYPE = np.dtype([
    ("id",     np.int64),
    ("type",   np.int64),
    ("x",      np.float64),
    ("y",      np.float64),
    ("z",      np.float64),
    ("radius", np.float64),
    ("parent", np.int64),
])

def _load_swc_to_array(swc_text: str) -> np.ndarray:
    """Fast parser using NumPy; ignores lines starting with '#'."""
    buf = io.StringIO(swc_text)
    arr = np.genfromtxt(
        buf,
        comments="#",
        dtype=_SWCTYPE,
        invalid_raise=False,
        autostrip=True
    )
    if arr.ndim == 0:  # single-line files
        arr = arr.reshape(1)
    return arr

def _sanitize_types_inplace(arr: np.ndarray) -> None:
    """type==0 or type>7 -> 7 (in-place, vectorized)."""
    t = arr["type"]
    bad = (t == 0) | (t > 7)
    if np.any(bad):
        t[bad] = 7

def _write_array_to_tmp_swc(arr: np.ndarray) -> str:
    """Write array back to a temp .swc path (fast)."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".swc")
    os.close(tmp_fd)
    stacked = np.column_stack([
        arr["id"], arr["type"], arr["x"], arr["y"], arr["z"], arr["radius"], arr["parent"]
    ])
    np.savetxt(tmp_path, stacked, fmt=["%d","%d","%.10g","%.10g","%.10g","%.10g","%d"], delimiter=" ")
    return tmp_path

def _sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

# --------- IN-MEMORY CACHE (by sanitized bytes) ----------
_CACHE: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
# key -> (results_dict, rows)

# --------- CUSTOM ORDERING FOR TABLE ROWS ----------
# Put these first (exact order), then all others follow in their usual alpha order.
_FIRST_SIX = [
    "has_soma",
    "has_nonzero_soma_radius",
    "has_axon",
    "has_basal_dendrite",
    "has_apical_dendrite",
    "has_no_dangling_branch",
]
_PRIORITY_MAP = {name: idx for idx, name in enumerate(_FIRST_SIX)}

def _row_sort_key(code_name: str, friendly: str) -> tuple:
    """Primary: our custom priority; Secondary: friendly name for stable ordering."""
    pri = _PRIORITY_MAP.get(code_name, 1_000_000)
    return (pri, friendly.lower())

# --------- MAIN ENTRY ----------
def run_format_validation_from_text(swc_text: str):
    """
    Input:
      swc_text: raw SWC string
    Returns:
      results_dict: { check_name: bool or "ERROR: ..." }
      sanitized_swc_bytes: bytes (the exact file used for checks)
      table_rows: [ { "check": <friendly>, "status": <bool or 'ERROR: ...'> }, ... ]
    """
    # Parse + sanitize
    arr = _load_swc_to_array(swc_text)
    _sanitize_types_inplace(arr)

    # Serialize sanitized file once
    tmp_path = _write_array_to_tmp_swc(arr)
    try:
        with open(tmp_path, "rb") as f:
            sanitized_bytes = f.read()
        cache_key = _sha1(sanitized_bytes)

        # Cache hit?
        hit = _CACHE.get(cache_key)
        if hit is not None:
            results, rows = hit
            return results, sanitized_bytes, rows

        # Build morphology
        raw = morphio.Morphology(
            tmp_path,
            options=morphio.Option.allow_unifurcated_section_change
        )
        morph = Morphology(raw)

        # Run the fixed check set in parallel
        results: Dict[str, Any] = {}
        max_workers = min(8, (os.cpu_count() or 2))
        selected = list(_selected_checks())
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run_one_check, n, f, nf, morph) for (n, f, nf) in selected]
            for fut in as_completed(futs):
                name, value = fut.result()
                results[name] = value

        # Custom: has_soma (quick)
        try:
            has_soma_points = hasattr(raw, "soma") and getattr(raw.soma, "points", None) is not None \
                              and len(raw.soma.points) > 0
        except Exception:
            has_soma_points = False
        has_soma_by_type = bool(np.any(arr["type"] == 1))
        results["has_soma"] = bool(has_soma_points or has_soma_by_type)

        # Build human-readable rows with the new custom ordering
        rows_unsorted = [(code, _friendly_label(code), status) for code, status in results.items()]
        rows_unsorted.sort(key=lambda t: _row_sort_key(t[0], t[1]))  # sort by our priority, then friendly label
        rows = [{"check": friendly, "status": status} for (code, friendly, status) in rows_unsorted]

        # Save to cache and return
        _CACHE[cache_key] = (results, rows)
        return results, sanitized_bytes, rows

    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
