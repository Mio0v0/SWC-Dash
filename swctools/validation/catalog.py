"""Shared validation label/category catalog used by GUI and CLI."""

from __future__ import annotations

from typing import Any


CHECK_CATALOG: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Structural presence",
        [
            ("has_soma", "Soma present", "At least one node is labeled as soma (type 1)."),
            ("has_axon", "Axon present", "At least one node is labeled as axon (type 2)."),
            (
                "has_basal_dendrite",
                "Basal dendrite present",
                "At least one node is labeled as basal dendrite (type 3).",
            ),
            (
                "has_apical_dendrite",
                "Apical dendrite present",
                "At least one node is labeled as apical dendrite (type 4).",
            ),
        ],
    ),
    (
        "Radius and size",
        [
            (
                "all_neurite_radii_nonzero",
                "All neurite radii are positive",
                "Every non-soma node must have radius > 0.",
            ),
            (
                "soma_radius_nonzero",
                "Soma radius is positive",
                "Soma radius must be greater than zero when soma exists.",
            ),
            (
                "no_ultranarrow_sections",
                "No extremely narrow sections",
                "Flags sections containing radii below the narrow-section threshold.",
            ),
            (
                "no_ultranarrow_starts",
                "No extremely narrow branch starts",
                "Flags branches that start with very small radius values.",
            ),
            (
                "no_fat_terminal_ends",
                "No oversized terminal ends",
                "Terminal points should not be abnormally thicker than the branch trend.",
            ),
        ],
    ),
    (
        "Length and geometry",
        [
            (
                "all_section_lengths_nonzero",
                "All section lengths are positive",
                "Each section must have total geometric length > 0.",
            ),
            (
                "all_segment_lengths_nonzero",
                "All segment lengths are positive",
                "Every parent-child segment must have positive 3D length.",
            ),
            (
                "no_back_tracking",
                "No geometric backtracking",
                "Branches should progress without folding back on themselves excessively.",
            ),
            (
                "no_flat_neurites",
                "No flattened neurites",
                "Neurites should maintain realistic 3D spread rather than collapsing into a plane.",
            ),
            (
                "no_duplicate_3d_points",
                "No duplicate 3D points",
                "Two different nodes should not share identical XYZ coordinates.",
            ),
        ],
    ),
    (
        "Topology",
        [
            (
                "no_dangling_branches",
                "No dangling branches",
                "Morphology should be one connected tree without extra disconnected roots.",
            ),
            (
                "no_single_child_chains",
                "No single-child chains",
                "Avoid long runs of trivial one-child sections that suggest indexing artifacts.",
            ),
            (
                "has_unifurcation",
                "Contains unifurcation",
                "Reports whether unifurcation patterns are present in the morphology.",
            ),
            (
                "has_multifurcation",
                "Contains multifurcation",
                "Reports whether branch points with more than two children are present.",
            ),
        ],
    ),
    (
        "Index consistency",
        [
            (
                "no_section_index_jumps",
                "No section index gaps",
                "Section indexing should be continuous without invalid jumps.",
            ),
            (
                "no_root_index_jumps",
                "No root index gaps",
                "Root-level indexing should remain continuous and consistent.",
            ),
        ],
    ),
]


CHECK_ORDER: dict[str, int] = {}
CHECK_CATEGORY: dict[str, str] = {}
CHECK_LABEL: dict[str, str] = {}
CHECK_RULE: dict[str, str] = {}
for _idx, (_category, _items) in enumerate(CHECK_CATALOG):
    for _j, (_key, _label, _rule) in enumerate(_items):
        CHECK_ORDER[_key] = _idx * 100 + _j
        CHECK_CATEGORY[_key] = _category
        CHECK_LABEL[_key] = _label
        CHECK_RULE[_key] = _rule


def sort_key_for_check(key: str, label: str) -> tuple[int, str]:
    return (CHECK_ORDER.get(key, 10_000), str(label).lower())


def rule_for_key(key: str) -> str:
    return str(CHECK_RULE.get(str(key), ""))


def group_rows_by_category(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {cat: [] for cat, _ in CHECK_CATALOG}
    other: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)
        key = str(item.get("key", ""))
        label = str(item.get("label") or CHECK_LABEL.get(key) or key)
        item["label"] = label
        category = CHECK_CATEGORY.get(key, "Other")
        if category == "Other":
            other.append(item)
        else:
            groups[category].append(item)

    out: list[tuple[str, list[dict[str, Any]]]] = []
    for category, _checks in CHECK_CATALOG:
        items = groups.get(category, [])
        if items:
            items.sort(key=lambda r: sort_key_for_check(str(r.get("key", "")), str(r.get("label", ""))))
            out.append((category, items))

    if other:
        other.sort(key=lambda r: str(r.get("label", "")).lower())
        out.append(("Other", other))
    return out
