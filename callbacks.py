import base64
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, dcc

from swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from graph_build import make_dendrogram_figure
from graph_utils import subtree_nodes, children_lists
from constants import color_for_type, label_for_type, SWC_COLS, DEFAULT_COLORS

from layout import _dendrogram_tab, _validation_tab, _viewer_tab


# ---------- helpers ----------
def _decode_uploaded_text(contents: str) -> str:
    """Accepts data URL or raw base64; returns decoded UTF-8 text."""
    if contents is None:
        raise ValueError("No uploaded content")
    if isinstance(contents, list):
        if not contents:
            raise ValueError("Empty upload list")
        contents = contents[0]
    if not isinstance(contents, str):
        raise TypeError("Unexpected upload contents type")
    if contents.startswith("data:"):
        parts = contents.split(",", 1)
        b64 = parts[1] if len(parts) == 2 else parts[0]
    else:
        b64 = contents
    raw = base64.b64decode(b64, validate=False)
    return raw.decode("utf-8", errors="ignore")


def _make_3d_edges_figure(df: pd.DataFrame) -> go.Figure:
    """
    Build a simple 3D figure with edges only (no nodes),
    color-coded by child's type/label.
    """
    if df is None or df.empty:
        return go.Figure(layout=dict(template="plotly_white", height=900))  # <-- bigger

    arr = df[SWC_COLS].to_records(index=False)
    kids = children_lists(arr)

    traces = []
    for lbl, color in DEFAULT_COLORS.items():
        xs, ys, zs = [], [], []
        for u in range(len(arr)):
            for v in kids[u]:
                # color by child's label
                tval = int(arr[v][1])
                if label_for_type(tval) != lbl:
                    continue
                x0, y0, z0 = float(arr[u][2]), float(arr[u][3]), float(arr[u][4])
                x1, y1, z1 = float(arr[v][2]), float(arr[v][3]), float(arr[v][4])
                xs += [x0, x1, None]
                ys += [y0, y1, None]
                zs += [z0, z1, None]
        if xs:
            traces.append(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(width=2.2, color=color),
                    hoverinfo="skip", showlegend=True, name=lbl, legendgroup=lbl
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        height=900,  # <-- bigger
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Type",
    )
    return fig


def register_callbacks(app):

    # ---------------- Tabs renderer ----------------
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        prevent_initial_call=False,
    )
    def render_tab(which):
        if which == "tab-dendro":
            return _dendrogram_tab()
        elif which == "tab-validate":
            return _validation_tab()
        elif which == "tab-viewer":
            return _viewer_tab()
        return _dendrogram_tab()

    # ---------------- DENDROGRAM: load file ----------------
    @app.callback(
        Output("edit-file-info", "children"),
        Output("store-working-df", "data", allow_duplicate=True),
        Output("fig-dendro", "figure", allow_duplicate=True),
        Output("fig-dendro-3d", "figure", allow_duplicate=True),   # NEW
        Output("store-dendro-info", "data", allow_duplicate=True),
        Output("table-changes", "data", allow_duplicate=True),
        Output("apply-msg", "children", allow_duplicate=True),
        Output("store-filename", "data", allow_duplicate=True),
        Input("upload-edit", "contents"),
        State("upload-edit", "filename"),
        prevent_initial_call=True,
    )
    def load_file(contents, filename):
        if not contents:
            return "No file.", None, go.Figure(), go.Figure(), None, [], "", None
        try:
            text = _decode_uploaded_text(contents)
            df = parse_swc_text_preserve_tokens(text)
            if df.empty:
                return f"Loaded {filename}: 0 rows.", None, go.Figure(), go.Figure(), None, [], "", filename
            dendro_fig, info = make_dendrogram_figure(df)
            fig3d = _make_3d_edges_figure(df)  # NEW
            msg = f"Loaded {filename} with {len(df)} nodes."
            return msg, df.to_dict("records"), dendro_fig, fig3d, info, [], "", filename
        except Exception as e:
            return f"Failed to load: {e}", None, go.Figure(), go.Figure(), None, [], "", filename

    # ---------------- DENDROGRAM: click select ----------------
    @app.callback(
        Output("selected-node-id", "children"),
        Output("selected-type", "children"),
        Output("type-chip", "style"),
        Input("fig-dendro", "clickData"),
        State("store-dendro-info", "data"),
        State("store-working-df", "data"),
    )
    def on_click_edge(clickData, info, df_records):
        base_chip_style = {
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "marginRight": "6px",
            "backgroundColor": "#ccc", "verticalAlign": "middle",
        }
        if not clickData or not info or not df_records:
            return "-", "-", base_chip_style
        try:
            pt = clickData["points"][0]
            cd = pt.get("customdata")
            if not (cd and cd[0] is not None):
                return "-", "-", base_chip_style
            v = int(cd[0])
            df = pd.DataFrame(df_records)
            swc_id = int(df.iloc[v]["id"])
            tval = int(df.iloc[v]["type"])
            label = label_for_type(tval)
            chip_style = dict(base_chip_style, backgroundColor=color_for_type(tval))
            return str(swc_id), f"{label} ({tval})", chip_style
        except Exception:
            return "-", "-", base_chip_style

    # ---------------- DENDROGRAM: apply type change ----------------
    @app.callback(
        Output("store-working-df", "data", allow_duplicate=True),
        Output("apply-msg", "children", allow_duplicate=True),
        Output("table-changes", "data", allow_duplicate=True),
        Output("table-changes", "page_current", allow_duplicate=True),
        Output("fig-dendro", "figure", allow_duplicate=True),
        Output("fig-dendro-3d", "figure", allow_duplicate=True),   # NEW
        Output("store-dendro-info", "data", allow_duplicate=True),
        Output("selected-type", "children", allow_duplicate=True),
        Output("type-chip", "style", allow_duplicate=True),
        Input("btn-apply", "n_clicks"),
        State("selected-node-id", "children"),
        State("new-type", "value"),
        State("store-working-df", "data"),
        State("store-dendro-info", "data"),
        State("table-changes", "data"),
        prevent_initial_call=True,
    )
    def apply_type_change(n, selected_swc_id, new_type, df_records, info, table):
        base_chip_style = {
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "marginRight": "6px",
            "backgroundColor": "#ccc", "verticalAlign": "middle",
        }

        if not df_records or not info:
            return (
                dash.no_update, "Upload a file first.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            )

        try:
            sel_id = int(selected_swc_id)
        except Exception:
            return (
                dash.no_update, "Click a branch in the dendrogram first.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            )

        if new_type is None or int(new_type) < 0:
            return (
                dash.no_update, "Enter a valid non-negative SWC type.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            )

        df = pd.DataFrame(df_records)
        matches = df.index[df["id"] == sel_id]
        if len(matches) == 0:
            return (
                dash.no_update, f"Could not find SWC id {sel_id}.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            )
        v = int(matches[0])

        kids = info["kids"]
        subtree = subtree_nodes(kids, v)

        new_t = int(new_type)
        old_types = df.loc[subtree, "type"].to_numpy().tolist()
        df.loc[subtree, "type"] = new_t

        changes_batch = []
        node_ids = df.loc[subtree, "id"].to_numpy()
        for nid, old in zip(node_ids, old_types):
            changes_batch.append(
                {"node_id": int(nid), "old_type": int(old), "new_type": new_t, "scope": "subtree"}
            )
        table = changes_batch + list(table or [])

        dendro_fig, info2 = make_dendrogram_figure(df)
        fig3d = _make_3d_edges_figure(df)  # NEW
        msg = f"Applied type {new_t} to subtree rooted at SWC id {sel_id} ({len(subtree)} nodes)."

        label = label_for_type(new_t)
        chip_style = dict(base_chip_style, backgroundColor=color_for_type(new_t))

        return (
            df.to_dict("records"), msg, table, 0, dendro_fig, fig3d, info2, f"{label} ({new_t})", chip_style,
        )

    # ---------------- DENDROGRAM: downloads ----------------
    @app.callback(
        Output("download-edited-swc", "data"),
        Input("btn-dl-swc", "n_clicks"),
        State("store-working-df", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def download_swc(n, df_records, filename):
        if not df_records:
            return dash.no_update
        df = pd.DataFrame(df_records)
        content = write_swc_to_bytes_preserve_tokens(df)
        base = os.path.splitext(filename or "edited")[0]
        out_name = f"{base}_edit.swc"

        def writer(f):
            f.write(content)

        return dcc.send_bytes(writer, out_name)

    @app.callback(
        Output("download-changelog", "data"),
        Input("btn-dl-log", "n_clicks"),
        State("table-changes", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def download_log(n, table, filename):
        if not table:
            return dash.no_update
        df = pd.DataFrame(table)
        csv_text = df.to_csv(index=False)
        base = os.path.splitext(filename or "change_log")[0]
        out_name = f"{base}_log.csv"
        return dcc.send_string(csv_text, out_name)

    # =====================================================================
    #                          2D / 3D VIEWER (unchanged)
    # =====================================================================

    @app.callback(
        Output("viewer-file-info", "children"),
        Output("store-viewer-df", "data", allow_duplicate=True),
        Input("upload-viewer", "contents"),
        State("upload-viewer", "filename"),
        prevent_initial_call=True,
    )
    def viewer_load(contents, filename):
        if not contents:
            return "No file.", None
        try:
            text = _decode_uploaded_text(contents)
            df = parse_swc_text_preserve_tokens(text)
            if df.empty:
                return f"Loaded {filename}: 0 rows.", None
            msg = f"Loaded {filename} with {len(df)} nodes."
            return msg, df.to_dict("records")
        except Exception as e:
            return f"Failed to load: {e}", None

    @app.callback(
        Output("fig-view-2d", "figure"),
        Output("fig-view-3d", "figure"),
        Input("store-viewer-df", "data"),
        Input("viewer-2d-view", "value"),
        Input("viewer-type-select", "value"),
        Input("viewer-range-undefined", "value"),
        Input("viewer-range-soma", "value"),
        Input("viewer-range-axon", "value"),
        Input("viewer-range-basal", "value"),
        Input("viewer-range-apical", "value"),
        Input("viewer-range-custom", "value"),
        Input("viewer-performance", "value"),
        Input("viewer-width-bins", "value"),
        prevent_initial_call=True,
    )
    def viewer_draw(df_records, view, type_selected,
                    rng_u, rng_s, rng_a, rng_b, rng_ap, rng_c,
                    perf_flags, bins_per_type):
        # unchanged
        if not df_records:
            return go.Figure(), go.Figure()

        quantize = "quantize" in (perf_flags or [])
        hide_thin = "hide_thin" in (perf_flags or [])

        df = pd.DataFrame(df_records)
        arr = df[SWC_COLS].to_records(index=False)
        kids = children_lists(arr)

        df = df.copy()
        df["label"] = [label_for_type(int(t)) for t in df["type"].tolist()]
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce").fillna(0.0)

        perc = np.zeros(len(df), dtype=float)
        for lbl, sub in df.groupby("label"):
            idx = sub.index
            r = sub["radius"].to_numpy()
            if len(r) > 1:
                order = np.argsort(r)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, len(r) + 1)
                p = 100.0 * (ranks - 1) / max(1, len(r) - 1)
            else:
                p = np.array([100.0 if r[0] > 0 else 0.0])
            perc[idx] = p
        df["perc"] = perc

        label_windows = {
            "undefined": (rng_u or [70, 100]),
            "soma": (rng_s or [70, 100]),
            "axon": (rng_a or [70, 100]),
            "basal dendrite": (rng_b or [70, 100]),
            "apical dendrite": (rng_ap or [70, 100]),
            "custom": (rng_c or [70, 100]),
        }
        selected_labels = set(type_selected or [])

        THIN_2D, THIN_3D = 0.8, 1.6
        MIN_2D, MIN_3D = 0.2, 0.4
        SCALE_2D, SCALE_3D = 1.0, 1.0

        def w2d(r): return max(MIN_2D, float(r) * SCALE_2D)
        def w3d(r): return max(MIN_3D, float(r) * SCALE_3D)

        if view == "xz":
            a1, a2, xlab, ylab = "x", "z", "X", "Z"
        elif view == "yz":
            a1, a2, xlab, ylab = "y", "z", "Y", "Z"
        else:
            a1, a2, xlab, ylab = "x", "y", "X", "Y"

        traces2d = []
        legend_done2d = set()
        for lbl, color in DEFAULT_COLORS.items():
            pmin, pmax = map(float, label_windows[lbl])

            if not hide_thin:
                thin_x, thin_y = [], []
            bins2d = {}

            for u in range(len(arr)):
                for v in kids[u]:
                    if df.iloc[v]["label"] != lbl:
                        continue
                    p = float(df.iloc[v]["perc"])
                    r = float(df.iloc[v]["radius"])

                    x0, y0 = float(df.iloc[u][a1]), float(df.iloc[u][a2])
                    x1, y1 = float(df.iloc[v][a1]), float(df.iloc[v][a2])

                    in_window = (lbl in selected_labels) and (pmin <= p <= pmax)

                    if quantize and in_window:
                        span = max(1e-9, (pmax - pmin))
                        rel = (p - pmin) / span
                        bi = int(np.clip(np.floor(rel * bins_per_type), 0, bins_per_type - 1))
                        b = bins2d.setdefault(bi, {"x": [], "y": [], "rsum": 0.0, "n": 0})
                        b["x"] += [x0, x1, np.nan]
                        b["y"] += [y0, y1, np.nan]
                        b["rsum"] += r
                        b["n"] += 1
                    elif (not quantize) and in_window:
                        traces2d.append(
                            go.Scattergl(
                                x=[x0, x1], y=[y0, y1], mode="lines",
                                line=dict(width=w2d(r), color=color),
                                hoverinfo="skip",
                                showlegend=(lbl not in legend_done2d),
                                name=lbl, legendgroup=lbl,
                            )
                        )
                        legend_done2d.add(lbl)
                    else:
                        if not hide_thin:
                            thin_x += [x0, x1, np.nan]
                            thin_y += [y0, y1, np.nan]

            if not hide_thin and thin_x:
                traces2d.append(
                    go.Scattergl(
                        x=thin_x, y=thin_y, mode="lines",
                        line=dict(width=THIN_2D, color=color),
                        hoverinfo="skip",
                        showlegend=(lbl not in legend_done2d),
                        name=lbl, legendgroup=lbl,
                    )
                )
                legend_done2d.add(lbl)

            if quantize:
                for bi in sorted(bins2d.keys()):
                    b = bins2d[bi]
                    avg_r = (b["rsum"] / max(1, b["n"]))
                    traces2d.append(
                        go.Scattergl(
                            x=b["x"], y=b["y"], mode="lines",
                            line=dict(width=w2d(avg_r), color=color),
                            hoverinfo="skip",
                            showlegend=(lbl not in legend_done2d),
                            name=lbl, legendgroup=lbl,
                        )
                    )
                    legend_done2d.add(lbl)

        fig2d = go.Figure(data=traces2d)
        fig2d.update_layout(
            xaxis_title=xlab, yaxis_title=ylab,
            template="plotly_white",
            height=600, margin=dict(l=10, r=10, t=30, b=10),
            legend_title_text="Type",
        )
        fig2d.update_yaxes(scaleanchor="x", scaleratio=1.0)

        traces3d = []
        legend_done3d = set()
        for lbl, color in DEFAULT_COLORS.items():
            pmin, pmax = map(float, label_windows[lbl])

            if not hide_thin:
                thin_x, thin_y, thin_z = [], [], []
            bins3d = {}

            for u in range(len(arr)):
                for v in kids[u]:
                    if df.iloc[v]["label"] != lbl:
                        continue
                    p = float(df.iloc[v]["perc"])
                    r = float(df.iloc[v]["radius"])

                    x0, y0, z0 = float(df.iloc[u]["x"]), float(df.iloc[u]["y"]), float(df.iloc[u]["z"])
                    x1, y1, z1 = float(df.iloc[v]["x"]), float(df.iloc[v]["y"]), float(df.iloc[v]["z"])

                    in_window = (lbl in selected_labels) and (pmin <= p <= pmax)

                    if quantize and in_window:
                        span = max(1e-9, (pmax - pmin))
                        rel = (p - pmin) / span
                        bi = int(np.clip(np.floor(rel * bins_per_type), 0, bins_per_type - 1))
                        b = bins3d.setdefault(bi, {"x": [], "y": [], "z": [], "rsum": 0.0, "n": 0})
                        b["x"] += [x0, x1, None]
                        b["y"] += [y0, y1, None]
                        b["z"] += [z0, z1, None]
                        b["rsum"] += r
                        b["n"] += 1
                    elif (not quantize) and in_window:
                        traces3d.append(
                            go.Scatter3d(
                                x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                                line=dict(width=w3d(r), color=color),
                                hoverinfo="skip",
                                showlegend=(lbl not in legend_done3d),
                                name=lbl, legendgroup=lbl,
                            )
                        )
                        legend_done3d.add(lbl)
                    else:
                        if not hide_thin:
                            thin_x += [x0, x1, None]
                            thin_y += [y0, y1, None]
                            thin_z += [z0, z1, None]

            if not hide_thin and thin_x:
                traces3d.append(
                    go.Scatter3d(
                        x=thin_x, y=thin_y, z=thin_z, mode="lines",
                        line=dict(width=THIN_3D, color=color),
                        hoverinfo="skip",
                        showlegend=(lbl not in legend_done3d),
                        name=lbl, legendgroup=lbl,
                    )
                )
                legend_done3d.add(lbl)

            if quantize:
                for bi in sorted(bins3d.keys()):
                    b = bins3d[bi]
                    avg_r = (b["rsum"] / max(1, b["n"]))
                    traces3d.append(
                        go.Scatter3d(
                            x=b["x"], y=b["y"], z=b["z"], mode="lines",
                            line=dict(width=w3d(avg_r), color=color),
                            hoverinfo="skip",
                            showlegend=(lbl not in legend_done3d),
                            name=lbl, legendgroup=lbl,
                        )
                    )
                    legend_done3d.add(lbl)

        fig3d = go.Figure(data=traces3d)
        fig3d.update_layout(
            template="plotly_white",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            height=600, margin=dict(l=10, r=10, t=30, b=10),
            legend_title_text="Type",
        )

        return fig2d, fig3d

