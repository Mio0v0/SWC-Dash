import base64
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import json
from dash import Input, Output, State, dcc

from swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from graph_build import make_dendrogram_figure
from graph_utils import subtree_nodes, children_lists
from constants import color_for_type, label_for_type, SWC_COLS, DEFAULT_COLORS
from validation_core import run_format_validation_from_text

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

from datetime import datetime

# ---- Optional: your long free-text header goes here (commented in the output) ----
# If you want the *entire* document you pasted to appear before the XML header,
# put it in this string. If you only want the first line/title, just leave that.
SYNOPSIS_SWC_PLUS = """*** working document ***
SWC plus (SWC+) format specification
"""

def _strip_swc_header(body_text: str) -> str:
    """Remove any existing SWC '#' comment header from SWC text."""
    lines = body_text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith('#')):
        i += 1
    return "\n".join(lines[i:]) + ("\n" if lines and not lines[-1].endswith("\n") else "")

def _build_swc_plus_header_xml(filename: str | None) -> str:
    """
    Minimal SWC+ XML header (commented lines).
    Lines start with '# ' as required by SWC/SWC+.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fname = filename or "edited.swc"

    xml = f"""# <SWCplus version="0.12">
#   <MetaData>
#     <FileHistory originalName="{fname}" originalFormat="SWC">
#       <Modification date="{today}" software="SWC Tools – Dendrogram Editor" command="type-edit" summary="Edited types via dendrogram"/>
#     </FileHistory>
#   </MetaData>
#   <CustomTypes version="0.12">
#     <!-- Place customized Types here if you use TypeIDs ≥16 -->
#   </CustomTypes>
# </SWCplus>
"""
    return xml

def make_swc_plus_bytes_from_df(df: pd.DataFrame, filename: str | None) -> bytes:
    """
    Compose a valid SWC+ file:
      [optional free-text synopsis as commented lines]
      [commented XML <SWCplus> header]
      [plain SWC matrix (no original # header)]
    """
    # 1) base body produced by your existing SWC writer
    body_bytes = write_swc_to_bytes_preserve_tokens(df)
    body_text = body_bytes.decode("utf-8", errors="replace")
    body_text = _strip_swc_header(body_text)

    # 2) optional free-text synopsis (commented)
    synopsis_block = ""
    if SYNOPSIS_SWC_PLUS.strip():
        synopsis_block = "".join(f"# {ln}\n" for ln in SYNOPSIS_SWC_PLUS.strip().splitlines())

    # 3) SWC+ XML header (commented)
    xml_block = _build_swc_plus_header_xml(filename)

    # 4) glue together
    out_text = synopsis_block + xml_block + body_text
    return out_text.encode("utf-8")

def _make_3d_edges_figure(df: pd.DataFrame) -> go.Figure:
    """
    Build a 3D figure with:
      • edges color-coded by child's type/label (as before)
      • soma nodes (type==1) explicitly rendered as spheres (markers)
    """
    if df is None or df.empty:
        return go.Figure(layout=dict(template="plotly_white", height=900))

    arr = df[SWC_COLS].to_records(index=False)
    kids = children_lists(arr)

    traces = []

    # ---- Edges (unchanged logic): color by CHILD's label ----
    for lbl, color in DEFAULT_COLORS.items():
        xs, ys, zs = [], [], []
        for u in range(len(arr)):
            for v in kids[u]:
                tval = int(arr[v][1])  # child's type
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

    # ---- Soma spheres (type==1): show even if there are no edges) ----
    soma_x, soma_y, soma_z, soma_sizes = [], [], [], []
    soma_color = DEFAULT_COLORS.get("soma", "#d62728")  # fallback color if needed

    for i in range(len(arr)):
        try:
            if int(arr[i][1]) == 1:  # soma type
                soma_x.append(float(arr[i][2]))
                soma_y.append(float(arr[i][3]))
                soma_z.append(float(arr[i][4]))
                # If radius present (SWC column 6 / index 5), scale a bit for visibility
                try:
                    r = float(arr[i][5])
                    soma_sizes.append(max(5.0, 6.0 * r))  # tweakable scale
                except Exception:
                    soma_sizes.append(8.0)  # default size if radius missing
        except Exception:
            pass

    if soma_x:
        traces.append(
            go.Scatter3d(
                x=soma_x, y=soma_y, z=soma_z,
                mode="markers",
                marker=dict(size=soma_sizes, color=soma_color, opacity=0.95),
                name="soma",
                legendgroup="soma",
                hoverinfo="text",
                text=[f"soma id={int(arr[i][0])}" for i in range(len(arr)) if int(arr[i][1]) == 1],
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        height=900,
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

        # Build SWC+ bytes (commented SWC+ header + SWC body without original header)
        content = make_swc_plus_bytes_from_df(df, filename)

        base = os.path.splitext(filename or "edited")[0]
        out_name = f"{base}_edit.swc"  # SWC+ keeps .swc extension

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
    #                          2D / 3D VIEWER
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

        # percentile per type (child node rank percent)
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

        # line width helpers
        THIN_2D, THIN_3D = 0.8, 1.6
        MIN_2D, MIN_3D = 0.2, 0.4
        SCALE_2D, SCALE_3D = 1.0, 1.0

        def w2d(r):
            return max(MIN_2D, float(r) * SCALE_2D)

        def w3d(r):
            return max(MIN_3D, float(r) * SCALE_3D)

        # view axes
        if view == "xz":
            a1, a2, xlab, ylab = "x", "z", "X", "Z"
        elif view == "yz":
            a1, a2, xlab, ylab = "y", "z", "Y", "Z"
        else:
            a1, a2, xlab, ylab = "x", "y", "X", "Y"

        # ---------------- 2D with hover showing radius (4 decimals) ----------------
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
                        # aggregate into bins; hover will show AVG radius per bin
                        span = max(1e-9, (pmax - pmin))
                        rel = (p - pmin) / span
                        bi = int(np.clip(np.floor(rel * bins_per_type), 0, bins_per_type - 1))
                        b = bins2d.setdefault(bi, {"x": [], "y": [], "rsum": 0.0, "n": 0})
                        b["x"] += [x0, x1, np.nan]
                        b["y"] += [y0, y1, np.nan]
                        b["rsum"] += r
                        b["n"] += 1

                    elif (not quantize) and in_window:
                        # segment inside window: enable hover with this child radius
                        customdata = [r, r]  # one per vertex
                        traces2d.append(
                            go.Scattergl(
                                x=[x0, x1], y=[y0, y1], mode="lines",
                                line=dict(width=w2d(r), color=color),
                                hovertemplate="radius = %{customdata:.4f}<extra></extra>",
                                customdata=customdata,
                                showlegend=(lbl not in legend_done2d),
                                name=lbl, legendgroup=lbl,
                            )
                        )
                        legend_done2d.add(lbl)

                    else:
                        if not hide_thin:
                            thin_x += [x0, x1, np.nan]
                            thin_y += [y0, y1, np.nan]

            # background thin edges: hover off
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

            # quantized bins: hover shows average radius
            if quantize:
                for bi in sorted(bins2d.keys()):
                    b = bins2d[bi]
                    avg_r = (b["rsum"] / max(1, b["n"]))
                    cd = [avg_r] * len(b["x"])
                    traces2d.append(
                        go.Scattergl(
                            x=b["x"], y=b["y"], mode="lines",
                            line=dict(width=w2d(avg_r), color=color),
                            hovertemplate="avg radius = %{customdata:.4f}<extra></extra>",
                            customdata=cd,
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
            hovermode="closest",  # pointer-style hover near a line segment
            hoverdistance=15,  # pixels; tweak for stickier/looser hover
        )
        fig2d.update_yaxes(scaleanchor="x", scaleratio=1.0)

        # ---------------- 3D (unchanged; hover disabled to keep it light) ----------------
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
                                hoverinfo="skip",  # keep 3D quiet per original behavior
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

    # =====================================================================
    #                          VALIDATION PAGE
    # =====================================================================
    @app.callback(
        Output("validate-file-info", "children"),
        Output("table-validate-results", "data", allow_duplicate=True),
        Output("store-validate-table", "data", allow_duplicate=True),
        Input("upload-validate", "contents"),
        State("upload-validate", "filename"),
        prevent_initial_call=True,
    )
    def run_validation(contents, filename):
        if not contents:
            return "No file.", [], None
        try:
            header, b64 = contents.split(",", 1)
            text = base64.b64decode(b64).decode("utf-8", errors="ignore")

            _, _sanitized_bytes, table_rows = run_format_validation_from_text(text)
            msg = f"Validated {filename} • {len(table_rows)} checks"

            return (msg, table_rows, table_rows)
        except Exception as e:
            return f"Validation failed: {e}", [], None

    @app.callback(
        Output("download-validate-json", "data"),
        Input("btn-dl-validate-json", "n_clicks"),
        State("store-validate-table", "data"),
        State("upload-validate", "filename"),
        prevent_initial_call=True,
    )
    def download_validate_json(n, table_rows, filename):
        if not table_rows:
            return dash.no_update
        base = os.path.splitext(filename or "validation")[0]
        out_name = f"{base}_validation.json"
        return dcc.send_string(json.dumps(table_rows, indent=2), out_name)
