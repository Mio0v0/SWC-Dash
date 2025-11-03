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
    labels_arr = np.array([label_for_type(int(t)) for t in tree.types.tolist()], dtype=object)
    root_type = int(tree.types[root]) if tree.size > 0 else None
    root_label = label_for_type(root_type) if root_type is not None else None

    # vertical connectors
    vxs, vys = [], []
    for u in range(tree.size):
        start = int(offsets[u])
        end = int(offsets[u + 1])
        if start == end:
            continue
        children = child_indices[start:end]
        ys_children = y[children]
        x = float(cum[u])
        vxs += [x, x, np.nan]
        vys += [float(np.min(ys_children)), float(np.max(ys_children)), np.nan]
    vert = go.Scattergl(
        x=vxs, y=vys, mode="lines",
        line=dict(color="#999", width=LINE_W_VERT),
        hoverinfo="skip", showlegend=False,
    )

    # horizontal edges grouped by type
    groups = {
        "undefined": {"hx": [], "hy": [], "hc": [], "text": []},
        "soma": {"hx": [], "hy": [], "hc": [], "text": []},
        "axon": {"hx": [], "hy": [], "hc": [], "text": []},
        "basal dendrite": {"hx": [], "hy": [], "hc": [], "text": []},
        "apical dendrite": {"hx": [], "hy": [], "hc": [], "text": []},
        "custom": {"hx": [], "hy": [], "hc": [], "text": []},  # types >= 5
    }

    parent_indices = np.repeat(np.arange(tree.size, dtype=np.int32), np.diff(offsets))
    edge_children = child_indices
    edge_labels = labels_arr[edge_children]

    for label in groups.keys():
        mask = edge_labels == label
        if not np.any(mask):
            continue

        p_idx = parent_indices[mask]
        c_idx = edge_children[mask]

        hx = np.empty(mask.sum() * 3, dtype=np.float32)
        hy = np.empty(mask.sum() * 3, dtype=np.float32)

        hx[0::3] = cum[p_idx]
        hx[1::3] = cum[c_idx]
        hx[2::3] = np.nan

        hy[0::3] = y[c_idx]
        hy[1::3] = y[c_idx]
        hy[2::3] = np.nan

        text = []
        customdata = []
        for child in c_idx.tolist():
            t_val = int(tree.types[child])
            swc_id = int(tree.ids[child])
            label_str = labels_arr[child]
            hover = f"id={swc_id}, type={label_str} ({t_val})"
            text.extend([hover, hover, None])
            customdata.extend([[child], [child], [None]])

        groups[label]["hx"] = hx.tolist()
        groups[label]["hy"] = hy.tolist()
        groups[label]["hc"] = customdata
        groups[label]["text"] = text

    horiz_traces = []
    for label, g in groups.items():
        if not g["hx"]:
            continue
        horiz_traces.append(
            go.Scattergl(
                x=g["hx"], y=g["hy"], mode="lines",
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
    root_in_traces = root_label in present_labels if root_label is not None else False
    if root_label is not None:
        present_labels.add(root_label)
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
        root_showlegend = not root_in_traces
        legend_name = "custom (5+)" if root_label == "custom" else (root_label or "root")

        fig.add_trace(
            go.Scattergl(
                x=[dot_x], y=[y_root], mode="markers",
                marker=dict(size=ROOT_DOT_SIZE, color=root_color, line=dict(width=1, color="#333")),
                hoverinfo="text", text=[f"id={root_id} (root)"],
                customdata=[[int(root)]],
                showlegend=root_showlegend,
                legendgroup=root_label if root_label is not None else "root",
                name=legend_name if root_showlegend else None,
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
