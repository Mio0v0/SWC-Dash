from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import SWC_COLS, DEFAULT_COLORS, label_for_type, color_for_type
from graph_utils import children_lists, pick_root, cumlens_from_root, layout_y_positions

def make_dendrogram_figure(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
    # styling knobs
    LINE_W_VERT = 2
    LINE_W_HORIZ = 2
    LINE_W_ROOT_CONN = 1
    ROOT_DOT_SIZE = 6

    if df.empty:
        return go.Figure(layout=dict(template="plotly_white", height=900, title="Empty SWC")), {}

    arr = df[SWC_COLS].to_records(index=False)
    kids = children_lists(arr)
    root = pick_root(arr, kids)
    cum = cumlens_from_root(arr, kids, root)
    y = layout_y_positions(kids, root)

    # vertical connectors
    vxs, vys = [], []
    for u in range(len(arr)):
        if not kids[u]:
            continue
        ys_children = [y[v] for v in kids[u]]
        x = float(cum[u])
        vxs += [x, x, np.nan]
        vys += [min(ys_children), max(ys_children), np.nan]
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

    for u in range(len(arr)):
        for v in kids[u]:
            xv0, xv1 = float(cum[u]), float(cum[v])
            yv = y[v]
            t_val = int(arr[v][1])
            label = label_for_type(t_val)

            groups[label]["hx"] += [xv0, xv1, np.nan]
            groups[label]["hy"] += [yv, yv, np.nan]
            groups[label]["hc"] += [[v], [v], [None]]
            swc_id = int(arr[v][0])
            groups[label]["text"] += [f"id={swc_id}, type={label} ({t_val})",
                                      f"id={swc_id}, type={label} ({t_val})", None]

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
    if len(arr) > 0:
        root_x = float(cum[root])
        y_root = y[root]

        valid_x = [c for c in cum if c is not None and not np.isnan(c)]
        x_range = (max(valid_x) - min(valid_x)) if len(valid_x) >= 2 else 1.0
        offset = max(1e-9, 0.002 * x_range)
        dot_x = root_x - offset

        root_type = int(arr[root][1])
        root_color = DEFAULT_COLORS[label_for_type(root_type)]
        root_id = int(arr[root][0])

        fig.add_trace(
            go.Scattergl(
                x=[dot_x], y=[y_root], mode="markers",
                marker=dict(size=ROOT_DOT_SIZE, color=root_color, line=dict(width=1, color="#333")),
                hoverinfo="text", text=[f"id={root_id} (root)"], customdata=[None], showlegend=False,
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

    info = {"kids": kids, "root": root, "cum": cum, "y": y}
    return fig, info
