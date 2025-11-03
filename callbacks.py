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
from graph_utils import subtree_nodes, build_tree_cache
from constants import color_for_type, label_for_type, DEFAULT_COLORS
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
    Build a 3D figure with edges color-coded by the child node's label and
    soma nodes rendered as spheres. Uses the shared TreeCache to minimise
    per-callback work.
    """
    if df is None or df.empty:
        return go.Figure(layout=dict(template="plotly_white", height=900))

    tree = build_tree_cache(df)
    if tree.size == 0:
        return go.Figure(layout=dict(template="plotly_white", height=900))

    coords = tree.xyz
    offsets = tree.child_offsets
    children = tree.child_indices
    parent_idx = np.repeat(np.arange(tree.size, dtype=np.int32), np.diff(offsets))
    labels_arr = np.array([label_for_type(int(t)) for t in tree.types.tolist()], dtype=object)
    edge_labels = labels_arr[children]

    traces = []
    present_labels = set()

    for lbl, color in DEFAULT_COLORS.items():
        mask = edge_labels == lbl
        if not np.any(mask):
            continue

        present_labels.add(lbl)
        p_idx = parent_idx[mask]
        c_idx = children[mask]
        m = int(mask.sum())

        xs = np.empty(m * 3, dtype=np.float32)
        ys = np.empty(m * 3, dtype=np.float32)
        zs = np.empty(m * 3, dtype=np.float32)

        xs[0::3] = coords[p_idx, 0]
        xs[1::3] = coords[c_idx, 0]
        xs[2::3] = np.nan

        ys[0::3] = coords[p_idx, 1]
        ys[1::3] = coords[c_idx, 1]
        ys[2::3] = np.nan

        zs[0::3] = coords[p_idx, 2]
        zs[1::3] = coords[c_idx, 2]
        zs[2::3] = np.nan

        traces.append(
            go.Scatter3d(
                x=xs.tolist(), y=ys.tolist(), z=zs.tolist(),
                mode="lines",
                line=dict(width=2.2, color=color),
                hoverinfo="skip",
                showlegend=True,
                name=lbl if lbl != "custom" else "custom (5+)",
                legendgroup=lbl,
            )
        )

    soma_idx = np.flatnonzero(tree.types == 1)
    if soma_idx.size:
        present_labels.add("soma")
        radii = np.nan_to_num(tree.radius[soma_idx], nan=0.0, posinf=0.0, neginf=0.0)
        soma_sizes = np.maximum(5.0, 6.0 * radii).astype(np.float32)
        soma_sizes = soma_sizes.tolist()
        soma_color = DEFAULT_COLORS.get("soma", "#d62728")
        traces.append(
            go.Scatter3d(
                x=coords[soma_idx, 0].astype(float).tolist(),
                y=coords[soma_idx, 1].astype(float).tolist(),
                z=coords[soma_idx, 2].astype(float).tolist(),
                mode="markers",
                marker=dict(size=soma_sizes, color=soma_color, opacity=0.95),
                hoverinfo="text",
                text=[f"soma id={int(tree.ids[i])}" for i in soma_idx.tolist()],
                legendgroup="soma",
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="lines",
                line=dict(width=2.2, color=soma_color),
                hoverinfo="skip",
                legendgroup="soma",
                name="soma",
                showlegend=True,
                visible="legendonly",
            )
        )

    for lbl, color in DEFAULT_COLORS.items():
        if lbl in present_labels:
            continue
        traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="lines",
                line=dict(width=2.2, color=color),
                hoverinfo="skip",
                showlegend=True,
                name=lbl if lbl != "custom" else "custom (5+)",
                legendgroup=lbl,
                visible="legendonly",
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
        State("apply-scope", "value"),
        State("store-working-df", "data"),
        State("store-dendro-info", "data"),
        State("table-changes", "data"),
        prevent_initial_call=True,
    )
    def apply_type_change(n, selected_swc_id, new_type, scope_mode, df_records, info, table):
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
        scope_mode = (scope_mode or "subtree").lower()
        if scope_mode not in {"subtree", "node"}:
            scope_mode = "subtree"

        if scope_mode == "node":
            subtree = [int(v)]
        else:
            subtree = subtree_nodes(kids, v)

        new_t = int(new_type)
        old_types = df.loc[subtree, "type"].to_numpy().tolist()
        df.loc[subtree, "type"] = new_t

        changes_batch = []
        node_ids = df.loc[subtree, "id"].to_numpy()
        for nid, old in zip(node_ids, old_types):
            changes_batch.append(
                {
                    "node_id": int(nid),
                    "old_type": int(old),
                    "new_type": new_t,
                    "scope": "node" if scope_mode == "node" else "subtree",
                }
            )
        table = changes_batch + list(table or [])

        dendro_fig, info2 = make_dendrogram_figure(df)
        fig3d = _make_3d_edges_figure(df)  # NEW
        if scope_mode == "node":
            msg = f"Applied type {new_t} to SWC id {sel_id}."
        else:
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
    # ---------------- Viewer: load file ----------------
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

    # ---------------- Helper ----------------
    def _clamp_round(v):
        try:
            v = float(v)
        except Exception:
            return None
        v = max(0.01, min(10.0, v))
        return float(round(v, 2))

    # ---------------- A) Controls -> Store (single callback, no cycles) ----------------
    @app.callback(
        Output("viewer-topk-store", "data"),
        # sliders
        Input("viewer-topk-undefined", "value"),
        Input("viewer-topk-soma", "value"),
        Input("viewer-topk-axon", "value"),
        Input("viewer-topk-basal", "value"),
        Input("viewer-topk-apical", "value"),
        Input("viewer-topk-custom", "value"),
        # numeric inputs
        Input("viewer-topk-undefined-input", "value"),
        Input("viewer-topk-soma-input", "value"),
        Input("viewer-topk-axon-input", "value"),
        Input("viewer-topk-basal-input", "value"),
        Input("viewer-topk-apical-input", "value"),
        Input("viewer-topk-custom-input", "value"),
        # current store
        State("viewer-topk-store", "data"),
        prevent_initial_call=True,
    )
    def any_control_updates_store(
            s_undef, s_soma, s_axon, s_basal, s_apical, s_custom,
            n_undef, n_soma, n_axon, n_basal, n_apical, n_custom,
            store,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        src = ctx.triggered[0]["prop_id"].split(".")[0]
        data = dict(store or {})

        mapping = {
            "viewer-topk-undefined": ("undefined", s_undef),
            "viewer-topk-soma": ("soma", s_soma),
            "viewer-topk-axon": ("axon", s_axon),
            "viewer-topk-basal": ("basal dendrite", s_basal),
            "viewer-topk-apical": ("apical dendrite", s_apical),
            "viewer-topk-custom": ("custom", s_custom),

            "viewer-topk-undefined-input": ("undefined", n_undef),
            "viewer-topk-soma-input": ("soma", n_soma),
            "viewer-topk-axon-input": ("axon", n_axon),
            "viewer-topk-basal-input": ("basal dendrite", n_basal),
            "viewer-topk-apical-input": ("apical dendrite", n_apical),
            "viewer-topk-custom-input": ("custom", n_custom),
        }

        if src in mapping:
            lbl, raw = mapping[src]
            val = _clamp_round(raw)
            if val is not None:
                data[lbl] = val
                return data

        return dash.no_update

    # ---------------- B) Draw figures (read store) ----------------
    @app.callback(
        Output("fig-view-2d", "figure"),
        Output("fig-view-3d", "figure"),
        Input("store-viewer-df", "data"),
        Input("viewer-2d-view", "value"),
        Input("viewer-type-select", "value"),
        Input("viewer-performance", "value"),
        Input("viewer-topk-store", "data"),
        prevent_initial_call=True,
    )
    def viewer_draw(df_records, view, type_selected, perf_flags, topk_store):
        if not df_records:
            return go.Figure(), go.Figure()

        hide_thin = "hide_thin" in (perf_flags or [])
        THIN_2D, THIN_3D = 0.8, 1.6
        DOT_COLOR, DOT_EDGE_COLOR = "rgba(170,0,255,0.95)", "white"
        DOT_SCALE, DOT_MIN, DOT_MAX = 4.0, 3.0, 24.0

        def fmt_k(k: float) -> str:
            if abs(k - round(k)) < 1e-6:
                return f"{int(round(k))}%"
            s = f"{k:.2f}".rstrip("0").rstrip(".")
            return f"{s}%"

        df = pd.DataFrame(df_records).copy()
        for col in ("x", "y", "z", "radius", "type"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["label"] = [label_for_type(int(t)) for t in df["type"].tolist()]

        # percentiles by type
        perc = np.zeros(len(df), dtype=np.float32)
        for lbl, sub in df.groupby("label"):
            idx = sub.index.to_numpy()
            rvals = sub["radius"].to_numpy(np.float32)
            if rvals.size > 1:
                order = np.argsort(rvals)
                ranks = np.empty_like(order, dtype=np.float32)
                ranks[order] = np.arange(1, rvals.size + 1, dtype=np.float32)
                p = 100.0 * (ranks - 1.0) / float(max(1, rvals.size - 1))
            else:
                p = np.array([100.0 if (rvals.size == 1 and rvals[0] > 0.0) else 0.0], dtype=np.float32)
            perc[idx] = p
        df["perc"] = perc

        def clamp_k(v):
            return max(0.01, min(10.0, float(v))) if v is not None else 10.0

        d = topk_store or {}
        topk_by_label = {
            "undefined": clamp_k(d.get("undefined", 10.0)),
            "soma": clamp_k(d.get("soma", 10.0)),
            "axon": clamp_k(d.get("axon", 10.0)),
            "basal dendrite": clamp_k(d.get("basal dendrite", 10.0)),
            "apical dendrite": clamp_k(d.get("apical dendrite", 10.0)),
            "custom": clamp_k(d.get("custom", 10.0)),
        }

        # choose axes
        if view == "xz":
            a1, a2, xlab, ylab = "x", "z", "X", "Z"
        elif view == "yz":
            a1, a2, xlab, ylab = "y", "z", "Y", "Z"
        else:
            a1, a2, xlab, ylab = "x", "y", "X", "Y"

        X = df[a1].to_numpy(np.float32)
        Y = df[a2].to_numpy(np.float32)
        Z = df["z"].to_numpy(np.float32)
        R = df["radius"].to_numpy(np.float32)
        L = np.array(df["label"].tolist())
        P = df["perc"].to_numpy(np.float32)
        has_id = "id" in df.columns
        ID = df["id"].to_numpy(np.int64) if has_id else None

        tree = build_tree_cache(df)
        if tree.child_indices.size == 0:
            return go.Figure(), go.Figure()
        e_u = np.repeat(np.arange(tree.size, dtype=np.int32), np.diff(tree.child_offsets))
        e_v = tree.child_indices.astype(np.int32, copy=False)

        def segments_2d(x0, y0, x1, y1):
            m = x0.shape[0]
            Xs = np.empty(m * 3, dtype=np.float32)
            Ys = np.empty(m * 3, dtype=np.float32)
            Xs[0::3] = x0;
            Xs[1::3] = x1;
            Xs[2::3] = np.nan
            Ys[0::3] = y0;
            Ys[1::3] = y1;
            Ys[2::3] = np.nan
            return Xs, Ys

        # 2D base
        traces2d, legend2d = [], set()
        for lbl, color in DEFAULT_COLORS.items():
            mask_lbl = (L[e_v] == lbl)
            if not np.any(mask_lbl):
                traces2d.append(go.Scattergl(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=THIN_2D, color=color),
                    hoverinfo="skip", name=lbl, legendgroup=lbl,
                    showlegend=True, visible="legendonly",
                ))
                continue
            uu = e_u[mask_lbl];
            vv = e_v[mask_lbl]
            x0, y0 = X[uu], Y[uu]
            x1, y1 = X[vv], Y[vv]
            Xs, Ys = segments_2d(x0, y0, x1, y1)
            cd = np.repeat(R[vv], 2).astype(np.float32)

            if "hide_thin" not in (perf_flags or []):
                traces2d.append(go.Scattergl(
                    x=Xs, y=Ys, mode="lines",
                    line=dict(width=THIN_2D, color=color),
                    hovertemplate="radius = %{customdata:.4f}<extra></extra>",
                    customdata=cd,
                    name=lbl, legendgroup=lbl, showlegend=(lbl not in legend2d),
                ))
                legend2d.add(lbl)
            else:
                traces2d.append(go.Scattergl(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=THIN_2D, color=color),
                    hoverinfo="skip", name=lbl, legendgroup=lbl,
                    showlegend=True, visible="legendonly",
                ))

        # 2D Top-K% dots
        selected_labels = set(type_selected or [])
        for lbl, _ in DEFAULT_COLORS.items():
            if lbl not in selected_labels:
                continue
            K = float(topk_by_label.get(lbl, 10.0))
            thresh = 100.0 - K
            cand = np.where((L[e_v] == lbl) & (P[e_v] >= thresh))[0]
            if cand.size == 0:
                continue
            vv = e_v[cand]
            dot_x, dot_y = X[vv], Y[vv]
            sz = np.clip((DOT_SCALE * R[vv]).astype(np.float32), DOT_MIN, DOT_MAX)
            label_text = f"Top-{fmt_k(K)}"
            if has_id:
                hover = [f"{label_text} • id={int(ID[v])} • radius={float(R[v]):.4f} • {lbl}" for v in vv]
            else:
                hover = [f"{label_text} • radius={float(R[v]):.4f} • {lbl}" for v in vv]

            traces2d.append(go.Scatter(
                x=dot_x, y=dot_y, mode="markers",
                marker=dict(size=sz, color=DOT_COLOR, line=dict(color=DOT_EDGE_COLOR, width=0.8), opacity=0.95),
                hovertemplate="%{text}<extra></extra>", text=hover,
                name=f"{lbl} ({label_text})", legendgroup=f"{lbl}-topk", showlegend=False,
            ))

        fig2d = go.Figure(traces2d)
        fig2d.update_layout(
            xaxis_title=xlab, yaxis_title=ylab, template="plotly_white",
            height=600, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Type",
            hovermode="closest", hoverdistance=15,
        )
        fig2d.update_yaxes(scaleanchor="x", scaleratio=1.0)

        # 3D base
        traces3d, legend3d = [], set()
        for lbl, color in DEFAULT_COLORS.items():
            lv = (L[e_v] == lbl)
            if not np.any(lv):
                traces3d.append(go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="lines",
                    line=dict(width=THIN_3D, color=color),
                    hoverinfo="skip", name=lbl, legendgroup=lbl,
                    showlegend=True, visible="legendonly",
                ))
                continue
            uu = e_u[lv];
            vv = e_v[lv]
            x0, y0, z0 = X[uu], Y[uu], Z[uu]
            x1, y1, z1 = X[vv], Y[vv], Z[vv]
            m = uu.size
            X3 = np.empty(m * 3, dtype=np.float32);
            Y3 = np.empty(m * 3, dtype=np.float32);
            Z3 = np.empty(m * 3, dtype=np.float32)
            X3[0::3] = x0;
            X3[1::3] = x1;
            X3[2::3] = np.nan
            Y3[0::3] = y0;
            Y3[1::3] = y1;
            Y3[2::3] = np.nan
            Z3[0::3] = z0;
            Z3[1::3] = z1;
            Z3[2::3] = np.nan

            traces3d.append(go.Scatter3d(
                x=X3, y=Y3, z=Z3, mode="lines",
                line=dict(width=THIN_3D, color=color),
                hoverinfo="skip", name=lbl, legendgroup=lbl, showlegend=(lbl not in legend3d),
            ))
            legend3d.add(lbl)

        fig3d = go.Figure(traces3d)
        fig3d.update_layout(
            template="plotly_white",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            height=600, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Type",
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
