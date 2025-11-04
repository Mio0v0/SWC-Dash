from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import DEFAULT_COLORS, label_for_type
from graph_utils import (
    build_tree_cache,
    pick_root_from_cache,
    cumlens_from_root_cache,
    layout_y_positions_cache,
    children_payload,
)

def make_dendrogram_figure(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
    # styling knobs
    LINE_W_VERT = 2
    LINE_W_HORIZ = 2
    LINE_W_ROOT_CONN = 1
    ROOT_DOT_SIZE = 6

    if df.empty:
        return go.Figure(layout=dict(template="plotly_white", height=900, title="Empty SWC")), {}

    tree = build_tree_cache(df)
    root = pick_root_from_cache(tree)
    cum = cumlens_from_root_cache(tree, root)
    y = layout_y_positions_cache(tree, root)

    offsets = tree.child_offsets
    child_indices = tree.child_indices
    child_counts = np.diff(offsets)
    labels_arr = np.array([label_for_type(int(t)) for t in tree.types.tolist()], dtype=object)
    root_type = int(tree.types[root]) if tree.size > 0 else None
    root_label = label_for_type(root_type) if root_type is not None else None

    # vertical connectors (vectorised)
    parents = np.flatnonzero(child_counts > 0)
    if parents.size:
        vx = np.empty(parents.size * 3, dtype=np.float32)
        vy = np.empty(parents.size * 3, dtype=np.float32)

        vx[0::3] = cum[parents]
        vx[1::3] = cum[parents]
        vx[2::3] = np.nan

        child_y = y[child_indices]
        min_y = np.minimum.reduceat(child_y, offsets[parents])
        max_y = np.maximum.reduceat(child_y, offsets[parents])

        vy[0::3] = min_y
        vy[1::3] = max_y
        vy[2::3] = np.nan
    else:
        vx = np.empty(0, dtype=np.float32)
        vy = np.empty(0, dtype=np.float32)

    vert = go.Scattergl(
        x=vx, y=vy, mode="lines",
        line=dict(color="#999", width=LINE_W_VERT),
        hoverinfo="skip", showlegend=False,
    )

    # horizontal edges grouped by type
    groups = {
        "undefined": {"hx": None, "hy": None, "hc": None, "text": None},
        "soma": {"hx": None, "hy": None, "hc": None, "text": None},
        "axon": {"hx": None, "hy": None, "hc": None, "text": None},
        "basal dendrite": {"hx": None, "hy": None, "hc": None, "text": None},
        "apical dendrite": {"hx": None, "hy": None, "hc": None, "text": None},
        "custom": {"hx": None, "hy": None, "hc": None, "text": None},  # types >= 5
    }

    parent_indices = np.repeat(np.arange(tree.size, dtype=np.int32), child_counts)
    edge_children = child_indices
    edge_labels = labels_arr[edge_children]
    edge_parent_x = cum[parent_indices]
    edge_child_x = cum[edge_children]
    edge_child_y = y[edge_children]
    edge_child_ids = tree.ids[edge_children].astype(np.int64, copy=False)
    edge_child_types = tree.types[edge_children].astype(np.int32, copy=False)

    for label in groups.keys():
        mask = edge_labels == label
        if not np.any(mask):
            continue

        p_idx = parent_indices[mask]
        c_idx = edge_children[mask]

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

        hover = np.array([f"id={swc}, type={lbl} ({t})" for swc, lbl, t in zip(swc_ids, labels_sel, t_vals)], dtype=object)
        text = np.empty(m * 3, dtype=object)
        text[0::3] = hover
        text[1::3] = hover
        text[2::3] = None

        customdata = np.empty(m * 3, dtype=object)
        child_list = child_idx.tolist()
        repeated = [[c] for c in child_list]
        # populate using slicing to avoid Python loop per element
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

    # ensure legend always shows all types
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

    fig = go.Figure(data=[vert, *horiz_traces])

    # root dot + tiny connector
    if tree.size > 0:
        root_x = float(cum[root])
        y_root = float(y[root])

        finite_mask = np.isfinite(cum)
        valid_x = cum[finite_mask]
        x_range = float(valid_x.max() - valid_x.min()) if valid_x.size >= 2 else 1.0
        offset = max(1e-9, 0.002 * x_range)
        dot_x = root_x - offset

        root_color = DEFAULT_COLORS.get(root_label, "#666")
        root_id = int(tree.ids[root])
        fig.add_trace(
            go.Scattergl(
                x=[dot_x], y=[y_root], mode="markers",
                marker=dict(size=ROOT_DOT_SIZE, color=root_color, line=dict(width=1, color="#333")),
                hoverinfo="text", text=[f"id={root_id} (root)"],
                customdata=[[int(root)]],
                showlegend=False,
                legendgroup=root_label if root_label is not None else "root",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=[dot_x, root_x], y=[y_root, y_root], mode="lines",
                line=dict(width=LINE_W_ROOT_CONN, color=root_color),
                hoverinfo="skip", customdata=[None, None], showlegend=False,
            )
        )

    fig.update_layout(
        title="Dendrogram (click a horizontal edge to select its subtree root)",
        xaxis_title="Path length from soma (µm)",
        yaxis=dict(showticklabels=False),
        height=900, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        legend_title_text="Type",
        hovermode="closest",
    )

    info = {
        "kids": children_payload(tree),
        "root": int(root),
        "cum": cum.astype(float, copy=False).tolist(),
        "y": y.astype(float, copy=False).tolist(),
    }
    return fig, info
