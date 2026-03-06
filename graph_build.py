from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import DEFAULT_COLORS, label_for_type, TREE_COLORS
from graph_utils import (
    build_tree_cache,
    find_all_roots,
    cumlens_from_root_cache,
    layout_y_positions_cache,
    children_payload,
    compute_levels,
    TreeCache,
)


def _build_tree_figure(
    tree: TreeCache,
    root: int,
    cum_raw: np.ndarray,
    y: np.ndarray,
    levels: np.ndarray,
    tree_mask: np.ndarray,
    compress: bool,
    tree_idx: int,
    n_trees: int,
) -> go.Figure:
    """Build a single dendrogram figure for one tree."""
    LINE_W_VERT = 1.5
    LINE_W_HORIZ = 1.5
    LINE_W_ROOT_CONN = 1
    ROOT_DOT_SIZE = 6

    cum = np.sqrt(cum_raw) if compress else cum_raw

    offsets = tree.child_offsets
    child_indices = tree.child_indices
    child_counts = np.diff(offsets)
    labels_arr = np.array([label_for_type(int(t)) for t in tree.types.tolist()], dtype=object)

    # All edges: parent -> child
    parent_indices_all = np.repeat(np.arange(tree.size, dtype=np.int32), child_counts)
    edge_children_all = child_indices

    # Filter to edges within this tree (child must belong to this tree)
    edge_in_tree = tree_mask[edge_children_all]
    parent_indices = parent_indices_all[edge_in_tree]
    edge_children = edge_children_all[edge_in_tree]

    # ---- Per-type colored vertical connectors ----
    vert_traces = []
    if edge_children.size > 0:
        edge_labels = labels_arr[edge_children]

        for label in DEFAULT_COLORS.keys():
            mask = edge_labels == label
            if not np.any(mask):
                continue

            p_idx = parent_indices[mask]
            c_idx = edge_children[mask]
            m = int(mask.sum())

            vx = np.empty(m * 3, dtype=np.float32)
            vy = np.empty(m * 3, dtype=np.float32)

            vx[0::3] = cum[p_idx]
            vx[1::3] = cum[p_idx]
            vx[2::3] = np.nan

            vy[0::3] = y[c_idx]
            vy[1::3] = y[p_idx]
            vy[2::3] = np.nan

            vert_traces.append(
                go.Scattergl(
                    x=vx, y=vy, mode="lines",
                    line=dict(color=DEFAULT_COLORS[label], width=LINE_W_VERT),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=label,
                )
            )

    # ---- Horizontal edges grouped by type (with level in hover) ----
    groups = {
        "undefined": {"hx": None, "hy": None, "hc": None, "text": None},
        "soma": {"hx": None, "hy": None, "hc": None, "text": None},
        "axon": {"hx": None, "hy": None, "hc": None, "text": None},
        "basal dendrite": {"hx": None, "hy": None, "hc": None, "text": None},
        "apical dendrite": {"hx": None, "hy": None, "hc": None, "text": None},
        "custom": {"hx": None, "hy": None, "hc": None, "text": None},
    }

    if edge_children.size > 0:
        edge_parent_x = cum[parent_indices]
        edge_child_x = cum[edge_children]
        edge_child_y = y[edge_children]
        edge_child_ids = tree.ids[edge_children].astype(np.int64, copy=False)
        edge_child_types = tree.types[edge_children].astype(np.int32, copy=False)
        edge_child_levels = levels[edge_children]
        edge_labels_all = labels_arr[edge_children]

        for label in groups.keys():
            mask = edge_labels_all == label
            if not np.any(mask):
                continue

            m = int(mask.sum())
            hx = np.empty(m * 3, dtype=np.float32)
            hy = np.empty(m * 3, dtype=np.float32)

            child_x = edge_child_x[mask]
            parent_x = edge_parent_x[mask]
            child_y = edge_child_y[mask]

            hx[0::3] = parent_x
            hx[1::3] = child_x
            hx[2::3] = np.nan

            hy[0::3] = child_y
            hy[1::3] = child_y
            hy[2::3] = np.nan

            child_idx = edge_children[mask]
            swc_ids = edge_child_ids[mask]
            t_vals = edge_child_types[mask]
            labels_sel = labels_arr[child_idx]
            lvl_vals = edge_child_levels[mask]

            hover = np.array(
                [f"id={swc}, type={lbl} ({t}), level={lv}"
                 for swc, lbl, t, lv in zip(swc_ids, labels_sel, t_vals, lvl_vals)],
                dtype=object,
            )
            text = np.empty(m * 3, dtype=object)
            text[0::3] = hover
            text[1::3] = hover
            text[2::3] = None

            customdata = np.empty(m * 3, dtype=object)
            child_list = child_idx.tolist()
            repeated = [[c] for c in child_list]
            customdata[0::3] = repeated
            customdata[1::3] = repeated
            customdata[2::3] = [[None] for _ in range(m)]

            groups[label]["hx"] = hx
            groups[label]["hy"] = hy
            groups[label]["hc"] = customdata.tolist()
            groups[label]["text"] = text.tolist()

    horiz_traces = []
    for label, g in groups.items():
        hx = g["hx"]
        if hx is None or len(hx) == 0:
            continue
        horiz_traces.append(
            go.Scattergl(
                x=hx, y=g["hy"], mode="lines",
                line=dict(width=LINE_W_HORIZ, color=DEFAULT_COLORS[label]),
                name=f"{label}" if label != "custom" else "custom (5+)",
                hoverinfo="text", text=g["text"],
                customdata=g["hc"], showlegend=True,
                legendgroup=label,
            )
        )

    # Ensure legend always shows all types
    ALL_LABELS = ["undefined", "soma", "axon", "basal dendrite", "apical dendrite", "custom"]
    present_labels = {tr.legendgroup for tr in horiz_traces if getattr(tr, "legendgroup", None)}
    for label in ALL_LABELS:
        if label not in present_labels:
            horiz_traces.append(
                go.Scattergl(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=LINE_W_HORIZ, color=DEFAULT_COLORS[label]),
                    name=f"{label}" if label != "custom" else "custom (5+)",
                    hoverinfo="skip", showlegend=True, legendgroup=label,
                    visible="legendonly",
                )
            )

    fig = go.Figure(data=[*vert_traces, *horiz_traces])

    # Root dot + connector
    root_type = int(tree.types[root])
    root_label = label_for_type(root_type)
    root_id = int(tree.ids[root])
    root_color = DEFAULT_COLORS.get(root_label, "#666")

    root_x = float(cum[root])
    y_root = float(y[root])

    tree_cum = cum[tree_mask]
    finite_mask = np.isfinite(tree_cum)
    valid_x = tree_cum[finite_mask]
    x_range = float(valid_x.max() - valid_x.min()) if valid_x.size >= 2 else 1.0
    offset = max(1e-9, 0.002 * x_range)
    dot_x = root_x - offset

    fig.add_trace(
        go.Scattergl(
            x=[dot_x], y=[y_root], mode="markers",
            marker=dict(size=ROOT_DOT_SIZE, color=root_color, line=dict(width=1, color="#333")),
            hoverinfo="text",
            text=[f"id={root_id} (root), type={root_label}, level=0"],
            customdata=[[int(root)]],
            showlegend=False,
            legendgroup=root_label,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[dot_x, root_x], y=[y_root, y_root], mode="lines",
            line=dict(width=LINE_W_ROOT_CONN, color=root_color),
            hoverinfo="skip", customdata=[None, None], showlegend=False,
        )
    )

    # Figure layout
    tree_node_count = int(tree_mask.sum())
    x_label = "Path length from soma (√µm, compressed)" if compress else "Path length from soma (µm)"
    tree_title = f"Tree {tree_idx + 1} — root id={root_id} ({root_label}), {tree_node_count} nodes"
    if n_trees == 1:
        tree_title = "Dendrogram (click a horizontal edge to select its subtree root)"

    # Scale height by node count (min 250, max 900)
    height = max(250, min(900, int(tree_node_count * 0.5)))

    fig.update_layout(
        title=tree_title,
        xaxis_title=x_label,
        yaxis=dict(showticklabels=False),
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        legend_title_text="Type",
        hovermode="closest",
    )

    return fig


def make_dendrogram_figures(
    df: pd.DataFrame, compress: bool = True
) -> Tuple[List[go.Figure], Dict[str, Any]]:
    """
    Build one dendrogram figure per *soma* tree. Returns (list_of_figs, combined_info).
    Non-soma (dangling) roots are grouped with the nearest soma tree and
    rendered in the same figure.
    """
    if df.empty:
        empty_fig = go.Figure(layout=dict(template="plotly_white", height=300, title="Empty SWC"))
        return [empty_fig], {}

    tree = build_tree_cache(df)
    roots = find_all_roots(tree)

    if not roots:
        empty_fig = go.Figure(layout=dict(template="plotly_white", height=300, title="No roots found"))
        return [empty_fig], {}

    # ----- Classify roots into soma vs dangling -----
    soma_roots = []
    dangling_roots = []
    for root in roots:
        levels = compute_levels(tree, root)
        members = np.flatnonzero(levels >= 0)
        has_soma = np.any(tree.types[members] == 1)
        if has_soma:
            soma_roots.append(root)
        else:
            dangling_roots.append(root)

    # If no soma roots, fall back to treating all roots independently
    if not soma_roots:
        soma_roots = roots
        dangling_roots = []

    # ----- Assign dangling roots to nearest soma root -----
    # dangling_root -> soma_root_index (index into soma_roots list)
    dangling_assignment: Dict[int, int] = {}
    if dangling_roots and soma_roots:
        # Collect all soma node indices for distance lookup
        soma_node_mask = tree.types == 1
        soma_node_indices = np.flatnonzero(soma_node_mask)
        soma_xyz = tree.xyz[soma_node_indices]  # (S, 3)

        # Map soma node index -> soma_roots list index
        soma_node_to_tree = {}
        for si, sroot in enumerate(soma_roots):
            slvls = compute_levels(tree, sroot)
            for idx in np.flatnonzero(slvls >= 0):
                if tree.types[idx] == 1:
                    soma_node_to_tree[idx] = si

        for droot in dangling_roots:
            dxyz = tree.xyz[droot].astype(np.float64)
            dists = np.linalg.norm(soma_xyz.astype(np.float64) - dxyz, axis=1)
            nearest = int(np.argmin(dists))
            nearest_node_idx = int(soma_node_indices[nearest])
            target_tree = soma_node_to_tree.get(nearest_node_idx, 0)
            dangling_assignment[droot] = target_tree

    # ----- Build per-soma-tree groups: list of all roots for each figure -----
    tree_root_groups: List[List[int]] = [[] for _ in soma_roots]
    for i, sroot in enumerate(soma_roots):
        tree_root_groups[i].append(sroot)
    for droot, tidx in dangling_assignment.items():
        tree_root_groups[tidx].append(droot)

    # ----- Compute layouts and build figures -----
    combined_cum = np.zeros(tree.size, dtype=np.float32)
    combined_y = np.zeros(tree.size, dtype=np.float32)
    combined_levels = np.full(tree.size, -1, dtype=np.int32)
    tree_membership = np.full(tree.size, -1, dtype=np.int32)
    tree_meta: List[Dict[str, Any]] = []
    figs: List[go.Figure] = []

    n_trees = len(soma_roots)

    for tree_idx, root_group in enumerate(tree_root_groups):
        primary_root = root_group[0]  # the soma root

        # Compute layout for primary root
        cum_raw = cumlens_from_root_cache(tree, primary_root)
        y = layout_y_positions_cache(tree, primary_root)
        levels = compute_levels(tree, primary_root)
        tree_mask = levels >= 0

        # Also compute layout for each dangling root in this group
        # and merge into the same arrays
        for extra_root in root_group[1:]:
            ecum = cumlens_from_root_cache(tree, extra_root)
            ey = layout_y_positions_cache(tree, extra_root)
            elvls = compute_levels(tree, extra_root)
            emask = elvls >= 0

            # Offset y so dangling nodes appear below the main tree
            if np.any(tree_mask):
                y_min = float(y[tree_mask].min())
            else:
                y_min = 0.0
            ey_offset = y_min - 2.0  # small gap below
            ey[emask] += ey_offset

            cum_raw[emask] = ecum[emask]
            y[emask] = ey[emask]
            levels[emask] = elvls[emask]
            tree_mask = tree_mask | emask

        tree_node_count = int(tree_mask.sum())

        # Write into combined arrays
        combined_cum[tree_mask] = cum_raw[tree_mask]
        combined_y[tree_mask] = y[tree_mask]
        combined_levels[tree_mask] = levels[tree_mask]
        tree_membership[tree_mask] = tree_idx

        fig = _build_tree_figure(
            tree, primary_root, cum_raw, y, levels, tree_mask,
            compress, tree_idx, n_trees,
        )
        figs.append(fig)

        root_type = int(tree.types[primary_root])
        tree_meta.append({
            "root_index": int(primary_root),
            "root_id": int(tree.ids[primary_root]),
            "root_type": root_type,
            "root_label": label_for_type(root_type),
            "node_count": tree_node_count,
        })

    info = {
        "kids": children_payload(tree),
        "parent_index": tree.parent_index.astype(int, copy=False).tolist(),
        "swc_ids": tree.ids.astype(int, copy=False).tolist(),
        "types": tree.types.astype(int, copy=False).tolist(),
        "root": int(soma_roots[0]),
        "roots": [int(r) for r in soma_roots],
        "cum": combined_cum.astype(float, copy=False).tolist(),
        "y": combined_y.astype(float, copy=False).tolist(),
        "levels": combined_levels.astype(int, copy=False).tolist(),
        "tree_membership": tree_membership.astype(int, copy=False).tolist(),
        "tree_meta": tree_meta,
    }
    return figs, info



# Backward-compatible wrapper (returns first figure only)
def make_dendrogram_figure(
    df: pd.DataFrame, compress: bool = True
) -> Tuple[go.Figure, Dict[str, Any]]:
    figs, info = make_dendrogram_figures(df, compress=compress)
    return figs[0] if figs else go.Figure(), info
