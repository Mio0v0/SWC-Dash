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
        return go.Figure(layout=dict(template="plotly_white", height=520))

    tree = build_tree_cache(df)
    if tree.size == 0:
        return go.Figure(layout=dict(template="plotly_white", height=520))

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
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Type",
    )
    return fig


def _validation_rows_from_df(df: pd.DataFrame) -> list[dict]:
    """
    Run the NeuroM validation checks against the provided DataFrame
    representation of the current SWC.
    """
    swc_bytes = write_swc_to_bytes_preserve_tokens(df)
    swc_text = swc_bytes.decode("utf-8", errors="ignore")
    _, _sanitized_bytes, table_rows = run_format_validation_from_text(swc_text)
    return table_rows



def register_callbacks(app):

    # ---------------- Tabs visibility ----------------
    @app.callback(
        Output("tab-pane-dendro", "style"),
        Output("tab-pane-validate", "style"),
        Output("tab-pane-viewer", "style"),
        Input("tabs", "value"),
        prevent_initial_call=False,
    )
    def set_tab_visibility(which):
        show = {"display": "block"}
        hide = {"display": "none"}
        if which == "tab-validate":
            return hide, show, hide
        if which == "tab-viewer":
            return hide, hide, show
        return show, hide, hide

    # ---------------- DENDROGRAM: render from shared store ----------------
    @app.callback(
        Output("edit-file-info", "children"),
        Output("fig-dendro", "figure"),
        Output("fig-dendro-3d", "figure"),
        Output("store-dendro-info", "data"),
        Input("store-working-df", "data"),
        State("store-filename", "data"),
        prevent_initial_call=False,
    )
    def render_dendrogram(shared_records, filename):
        if not shared_records:
            return (
                "No file loaded. Upload on the Format Validation tab.",
                go.Figure(),
                go.Figure(),
                None,
            )
        try:
            df = pd.DataFrame(shared_records)
            if df.empty:
                msg = f"Loaded {filename or 'current file'}: 0 rows."
                return msg, go.Figure(), go.Figure(), None

            dendro_fig, info = make_dendrogram_figure(df)
            fig3d = _make_3d_edges_figure(df)
            msg = f"Loaded {filename or 'current file'} with {len(df)} nodes."
            return msg, dendro_fig, fig3d, info
        except Exception as e:
            return (
                f"Failed to render dendrogram: {e}",
                go.Figure(),
                go.Figure(),
                None,
            )

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
        Output("validate-file-info", "children", allow_duplicate=True),
        Output("table-validate-results", "data", allow_duplicate=True),
        Output("store-validate-table", "data", allow_duplicate=True),
        Input("btn-apply", "n_clicks"),
        State("selected-node-id", "children"),
        State("new-type", "value"),
        State("apply-scope", "value"),
        State("store-working-df", "data"),
        State("store-dendro-info", "data"),
        State("table-changes", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def apply_type_change(n, selected_swc_id, new_type, scope_mode, df_records, info, table, filename):
        base_chip_style = {
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "marginRight": "6px",
            "backgroundColor": "#ccc", "verticalAlign": "middle",
        }

        if not df_records or not info:
            return (
                dash.no_update, "Upload a file first.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
            )

        try:
            sel_id = int(selected_swc_id)
        except Exception:
            return (
                dash.no_update, "Click a branch in the dendrogram first.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
            )

        if new_type is None or int(new_type) < 0:
            return (
                dash.no_update, "Enter a valid non-negative SWC type.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
            )

        df = pd.DataFrame(df_records)
        matches = df.index[df["id"] == sel_id]
        if len(matches) == 0:
            return (
                dash.no_update, f"Could not find SWC id {sel_id}.", dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
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

        validation_msg = dash.no_update
        validate_rows = dash.no_update
        validate_store = dash.no_update
        try:
            rows = _validation_rows_from_df(df)
            validation_msg = f"Validated {filename or 'current file'} • {len(rows)} checks (auto-updated)"
            validate_rows = rows
            validate_store = rows
        except Exception as e:
            validation_msg = f"Validation failed after edit: {e}"

        return (
            df.to_dict("records"), msg, table, 0, dendro_fig, fig3d, info2, f"{label} ({new_t})", chip_style,
            validation_msg, validate_rows, validate_store,
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
    # ---------------- Viewer: info --------------------------------------
    @app.callback(
        Output("viewer-file-info", "children"),
        Input("store-working-df", "data"),
        State("store-filename", "data"),
        prevent_initial_call=False,
    )
    def viewer_info(shared_records, filename):
        if not shared_records:
            return "No file loaded. Upload on the Format Validation tab."
        try:
            df = pd.DataFrame(shared_records)
            if df.empty:
                return f"Loaded {filename or 'current file'}: 0 rows."
            return f"Loaded {filename or 'current file'} with {len(df)} nodes."
        except Exception as e:
            return f"Failed to display viewer info: {e}"


    # ---------------- Helpers ----------------
    def _clamp_percent(v):
        try:
            v = float(v)
        except Exception:
            return None
        return float(round(max(0.01, min(10.0, v)), 2))

    def _clamp_abs(v):
        if v in (None, ""):
            return None
        try:
            v = float(v)
        except Exception:
            return None
        return float(round(max(0.0, v), 4))


    # ---------------- Bi-directional sync: slider <-> numeric ----------------
    @app.callback(
        # numeric inputs
        Output("viewer-topk-undefined-input", "value"),
        Output("viewer-topk-soma-input", "value"),
        Output("viewer-topk-axon-input", "value"),
        Output("viewer-topk-basal-input", "value"),
        Output("viewer-topk-apical-input", "value"),
        Output("viewer-topk-custom-input", "value"),
        # sliders
        Output("viewer-topk-undefined", "value"),
        Output("viewer-topk-soma", "value"),
        Output("viewer-topk-axon", "value"),
        Output("viewer-topk-basal", "value"),
        Output("viewer-topk-apical", "value"),
        Output("viewer-topk-custom", "value"),
        # inputs
        Input("viewer-topk-undefined", "value"),
        Input("viewer-topk-soma", "value"),
        Input("viewer-topk-axon", "value"),
        Input("viewer-topk-basal", "value"),
        Input("viewer-topk-apical", "value"),
        Input("viewer-topk-custom", "value"),
        Input("viewer-topk-undefined-input", "value"),
        Input("viewer-topk-soma-input", "value"),
        Input("viewer-topk-axon-input", "value"),
        Input("viewer-topk-basal-input", "value"),
        Input("viewer-topk-apical-input", "value"),
        Input("viewer-topk-custom-input", "value"),
        prevent_initial_call=True,
    )
    def sync_slider_numeric(s_undef, s_soma, s_axon, s_basal, s_apical, s_custom,
                            n_undef, n_soma, n_axon, n_basal, n_apical, n_custom):
        ctx = dash.callback_context
        no = dash.no_update
        out_numeric = [no]*6
        out_sliders = [no]*6
        if not ctx.triggered:
            return (*out_numeric, *out_sliders)

        src = ctx.triggered[0]["prop_id"].split(".")[0]

        slider_map = {
            "viewer-topk-undefined": (0, s_undef),
            "viewer-topk-soma": (1, s_soma),
            "viewer-topk-axon": (2, s_axon),
            "viewer-topk-basal": (3, s_basal),
            "viewer-topk-apical": (4, s_apical),
            "viewer-topk-custom": (5, s_custom),
        }
        numeric_map = {
            "viewer-topk-undefined-input": (0, n_undef),
            "viewer-topk-soma-input": (1, n_soma),
            "viewer-topk-axon-input": (2, n_axon),
            "viewer-topk-basal-input": (3, n_basal),
            "viewer-topk-apical-input": (4, n_apical),
            "viewer-topk-custom-input": (5, n_custom),
        }

        if src in slider_map:
            i, val = slider_map[src]
            v = _clamp_percent(val)
            tmp = list(out_numeric); tmp[i] = v
            return (*tmp, *out_sliders)

        if src in numeric_map:
            i, val = numeric_map[src]
            v = _clamp_percent(val)
            tmp = list(out_sliders); tmp[i] = v
            return (*out_numeric, *tmp)

        return (*out_numeric, *out_sliders)


    # ---------------- K% -> store ----------------
    @app.callback(
        Output("viewer-topk-store", "data"),
        Input("viewer-topk-undefined", "value"),
        Input("viewer-topk-soma", "value"),
        Input("viewer-topk-axon", "value"),
        Input("viewer-topk-basal", "value"),
        Input("viewer-topk-apical", "value"),
        Input("viewer-topk-custom", "value"),
        Input("viewer-topk-undefined-input", "value"),
        Input("viewer-topk-soma-input", "value"),
        Input("viewer-topk-axon-input", "value"),
        Input("viewer-topk-basal-input", "value"),
        Input("viewer-topk-apical-input", "value"),
        Input("viewer-topk-custom-input", "value"),
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
            val = _clamp_percent(raw)
            if val is not None:
                data[lbl] = val
                return data

        return dash.no_update


    # ---------------- ABS -> store ----------------
    @app.callback(
        Output("viewer-abs-store", "data"),
        Input("viewer-abs-undefined", "value"),
        Input("viewer-abs-soma", "value"),
        Input("viewer-abs-axon", "value"),
        Input("viewer-abs-basal", "value"),
        Input("viewer-abs-apical", "value"),
        Input("viewer-abs-custom", "value"),
        State("viewer-abs-store", "data"),
        prevent_initial_call=True,
    )
    def abs_inputs_to_store(a_undef, a_soma, a_axon, a_basal, a_apical, a_custom, store):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        src = ctx.triggered[0]["prop_id"].split(".")[0]
        data = dict(store or {
            "undefined": None, "soma": None, "axon": None,
            "basal dendrite": None, "apical dendrite": None, "custom": None
        })

        mapping = {
            "viewer-abs-undefined": ("undefined", a_undef),
            "viewer-abs-soma": ("soma", a_soma),
            "viewer-abs-axon": ("axon", a_axon),
            "viewer-abs-basal": ("basal dendrite", a_basal),
            "viewer-abs-apical": ("apical dendrite", a_apical),
            "viewer-abs-custom": ("custom", a_custom),
        }
        if src in mapping:
            lbl, raw = mapping[src]
            data[lbl] = _clamp_abs(raw)
            return data

        return dash.no_update


    # ---------------- Clean Radii ----------------
    @app.callback(
        Output("store-working-df", "data", allow_duplicate=True),
        Output("table-viewer-clean-log", "data", allow_duplicate=True),
        Output("viewer-clean-msg", "children", allow_duplicate=True),
        Output("validate-file-info", "children", allow_duplicate=True),
        Output("table-validate-results", "data", allow_duplicate=True),
        Output("store-validate-table", "data", allow_duplicate=True),
        Input("btn-viewer-clean", "n_clicks"),
        State("store-working-df", "data"),
        State("viewer-type-select", "value"),
        State("viewer-clean-mode", "value"),
        State("viewer-topk-store", "data"),
        State("viewer-abs-store", "data"),
        State("table-viewer-clean-log", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def viewer_clean(n, df_records, type_selected, mode, topk_store, abs_store, log_rows, filename):
        if not df_records:
            return (
                dash.no_update,
                dash.no_update,
                "Upload a file first.",
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        try:
            df = pd.DataFrame(df_records).copy()
            for col in ("x", "y", "z", "radius", "type"):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            # Determine labels to operate on
            selected_labels = set(type_selected or [])
            # Always exclude soma from cleaning
            if "soma" in selected_labels:
                selected_labels.discard("soma")
            if not selected_labels:
                return (
                    dash.no_update,
                    dash.no_update,
                    "Select at least one Type.",
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            # Prepare tree cache
            cache = build_tree_cache(df)
            if cache.size == 0:
                return (
                    dash.no_update,
                    dash.no_update,
                    "No nodes to clean.",
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            # Build label per node
            labels = np.array([label_for_type(int(t)) for t in df["type"].to_numpy()], dtype=object)
            radii = df["radius"].to_numpy(dtype=float)

            # For percent mode, compute threshold per label from viewer-topk-store; for absolute, per label from viewer-abs-store
            cut_by_label = {}
            if mode == "percent":
                for lbl in selected_labels:
                    mask = labels == lbl
                    vals = radii[mask]
                    if vals.size == 0:
                        cut_by_label[lbl] = np.inf  # nothing will pass
                    else:
                        # Top-K% per label from store (clamped to [0.01, 10])
                        K = topk_store.get(lbl, 10.0) if isinstance(topk_store, dict) else 10.0
                        try:
                            K = float(K)
                        except Exception:
                            K = 10.0
                        K = max(0.01, min(10.0, K))
                        pct_cut = float(np.percentile(vals, 100.0 - K))
                        cut_by_label[lbl] = pct_cut
            else:
                # Absolute value per label from store; if None, skip that label
                for lbl in selected_labels:
                    thr = None
                    if isinstance(abs_store, dict):
                        thr = abs_store.get(lbl, None)
                    if thr is None:
                        # mark as skip by setting +inf so mask will be empty
                        cut_by_label[lbl] = np.inf
                    else:
                        try:
                            cut_by_label[lbl] = float(max(0.0, float(thr)))
                        except Exception:
                            cut_by_label[lbl] = np.inf

            # Compute which nodes to clean (build a boolean mask across all labels)
            n_nodes = len(df)
            to_clean_mask = np.zeros(n_nodes, dtype=bool)
            for lbl in selected_labels:
                thr = cut_by_label.get(lbl, np.inf)
                if not np.isfinite(thr):
                    continue
                mask_lbl = (labels == lbl)
                if not np.any(mask_lbl):
                    continue
                to_clean_mask |= (mask_lbl & (radii >= thr))

            to_clean_idx = np.flatnonzero(to_clean_mask)
            if to_clean_idx.size == 0:
                return (
                    dash.no_update,
                    log_rows or [],
                    "No nodes met the per-type cutoffs.",
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            to_clean_lbl = labels[to_clean_idx].tolist()

            # Pre-compute per-type fallback means
            #  - mean_outside: mean of nodes OUTSIDE the cleaning range per label
            #  - mean_all: mean of all nodes per label (used if outside mean unavailable)
            mean_by_label = {}
            mean_all_by_label = {}
            for lbl in selected_labels:
                mask_lbl = (labels == lbl)
                # outside-range mean
                mask_ok = mask_lbl & (~to_clean_mask)
                vals_ok = radii[mask_ok]
                mean_by_label[lbl] = float(np.mean(vals_ok)) if vals_ok.size else None
                # overall label mean
                vals_all = radii[mask_lbl]
                mean_all_by_label[lbl] = float(np.mean(vals_all)) if vals_all.size else None

            # For reproducibility: sort by DataFrame index order
            order = np.argsort(to_clean_idx)
            to_clean_idx = to_clean_idx[order]
            to_clean_lbl = [to_clean_lbl[i] for i in order]

            # Apply neighbor averaging
            parent_index = cache.parent_index
            child_offsets = cache.child_offsets
            child_indices = cache.child_indices
            new_radii = radii.copy()
            changes = []

            for pos, lbl in zip(to_clean_idx.tolist(), to_clean_lbl):
                old_r = float(radii[pos])

                neighbors = []
                # parent
                p = int(parent_index[pos])
                parent_ok = (p >= 0) and (not to_clean_mask[p])
                if parent_ok:
                    neighbors.append(float(radii[p]))
                # children: up to N children (2 normally, 3 if no parent)
                start = int(child_offsets[pos])
                end = int(child_offsets[pos + 1])
                kids = child_indices[start:end]

                # If parent unusable (missing or flagged), allow up to 3 children; else up to 2
                max_children = 2 if parent_ok else 3
                if kids.size:
                    # iterate children in order, picking only those not flagged, up to max_children
                    picked = 0
                    for k in kids.tolist():
                        if picked >= max_children:
                            break
                        if to_clean_mask[int(k)]:
                            continue
                        neighbors.append(float(radii[int(k)]))
                        picked += 1

                # Use neighbor average only if we have at least 2 valid neighbors (all are < cutoff by construction)
                if len(neighbors) >= 2:
                    new_r = float(np.mean(neighbors))
                else:
                    # Fallback to per-type mean (outside range); if unavailable, fallback to overall type mean
                    fallback = mean_by_label.get(lbl, None)
                    if fallback is None or not np.isfinite(fallback):
                        fallback = mean_all_by_label.get(lbl, None)
                    if fallback is None or not np.isfinite(fallback):
                        continue
                    new_r = float(fallback)
                # Enforce new radius strictly below the cutoff for this node's label
                thr = float(cut_by_label.get(lbl, np.inf))
                if np.isfinite(thr):
                    if thr > 0.0:
                        # push just below threshold if needed
                        eps_below = np.nextafter(thr, -np.inf)
                        new_r = min(new_r, eps_below)
                    else:
                        # non-positive cutoff: clamp to non-negative domain
                        new_r = max(0.0, min(new_r, thr))
                new_radii[pos] = new_r

                changes.append({
                    "node_id": int(df.iloc[pos]["id"]),
                    "label": lbl,
                    "old_radius": round(old_r, 6),
                    "new_radius": round(new_r, 6),
                    "mode": mode,
                    "cutoff": round(float(cut_by_label.get(lbl, np.nan)), 6),
                })

            if not changes:
                return (
                    dash.no_update,
                    log_rows or [],
                    "No eligible neighbors to average.",
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            # Write back radii (+ keep token string in sync for saving)
            df.loc[:, "radius"] = new_radii
            # Update radius_str where present to reflect new value for changed nodes
            if "radius_str" in df.columns:
                for ch in changes:
                    node_id = ch["node_id"]
                    # find by id equals (unique IDs in SWC)
                    row_idx = df.index[df["id"] == int(node_id)]
                    if len(row_idx) > 0:
                        df.at[row_idx[0], "radius_str"] = f"{ch['new_radius']:.6f}"

            # Append logs (prepend like dendrogram)
            new_log = (changes + list(log_rows or []))

            msg = f"Cleaned {len(changes)} node(s). Mode: {'Top-K%' if mode=='percent' else 'Absolute'}."
            validation_msg = dash.no_update
            validate_rows = dash.no_update
            validate_store = dash.no_update
            try:
                rows = _validation_rows_from_df(df)
                validation_msg = f"Validated {filename or 'current file'} • {len(rows)} checks (auto-updated)"
                validate_rows = rows
                validate_store = rows
            except Exception as e:
                validation_msg = f"Validation failed after cleaning: {e}"

            return df.to_dict("records"), new_log, msg, validation_msg, validate_rows, validate_store
        except Exception as e:
            return (
                dash.no_update,
                dash.no_update,
                f"Clean failed: {e}",
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

    # ---------------- Draw figures (edges unfiltered; overlay uses MAX rule) ----------------
    @app.callback(
        Output("fig-view-2d", "figure"),
        Output("fig-view-3d", "figure"),
        Input("store-working-df", "data"),
        Input("viewer-2d-view", "value"),
        Input("viewer-type-select", "value"),
        Input("viewer-performance", "value"),
        Input("viewer-topk-store", "data"),
        Input("viewer-abs-store", "data"),
        prevent_initial_call=True,
    )
    def viewer_draw(df_records, view, type_selected, perf_flags, topk_store, abs_store):
        if not df_records:
            return go.Figure(), go.Figure()

        hide_thin = "hide_thin" in (perf_flags or [])
        THIN_2D, THIN_3D = 0.8, 1.6
        DOT_COLOR, DOT_EDGE_COLOR = "rgba(170,0,255,0.95)", "white"
        DOT_SCALE, DOT_MIN, DOT_MAX = 4.0, 3.0, 24.0

        LABELS = ["undefined", "soma", "axon", "basal dendrite", "apical dendrite", "custom"]

        df = pd.DataFrame(df_records).copy()
        for col in ("x", "y", "z", "radius", "type"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["label"] = [label_for_type(int(t)) for t in df["type"].tolist()]

        def clamp_k(v):
            return max(0.01, min(10.0, float(v))) if v is not None else 10.0

        topk = {lbl: clamp_k((topk_store or {}).get(lbl, 10.0)) for lbl in LABELS}
        abs_thr = {lbl: (abs_store or {}).get(lbl, None) for lbl in LABELS}

        # Axes
        if view == "xz":
            a1, a2, xlab, ylab = "x", "z", "X", "Z"
        elif view == "yz":
            a1, a2, xlab, ylab = "y", "z", "Y", "Z"
        else:
            a1, a2, xlab, ylab = "x", "y", "X", "Y"

        X2 = df[a1].to_numpy(np.float32)
        Y2 = df[a2].to_numpy(np.float32)
        coords_x = df["x"].to_numpy(np.float32)
        coords_y = df["y"].to_numpy(np.float32)
        coords_z = df["z"].to_numpy(np.float32)
        R = df["radius"].to_numpy(np.float32)
        L = np.array(df["label"].tolist())
        has_id = "id" in df.columns
        ID = df["id"].to_numpy(np.int64) if has_id else None

        # Tree for edges
        tree = build_tree_cache(df)
        if tree.child_indices.size == 0:
            fig2d = go.Figure(layout=dict(template="plotly_white", xaxis_title=xlab, yaxis_title=ylab))
            fig3d = go.Figure(layout=dict(template="plotly_white", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")))
            return fig2d, fig3d

        e_u = np.repeat(np.arange(tree.size, dtype=np.int32), np.diff(tree.child_offsets))
        e_v = tree.child_indices.astype(np.int32, copy=False)

        def segments_2d(x0, y0, x1, y1):
            m = x0.shape[0]
            Xs = np.empty(m * 3, dtype=np.float32)
            Ys = np.empty(m * 3, dtype=np.float32)
            Xs[0::3] = x0; Xs[1::3] = x1; Xs[2::3] = np.nan
            Ys[0::3] = y0; Ys[1::3] = y1; Ys[2::3] = np.nan
            return Xs, Ys

        # Effective per-label cutoff (MAX of percentile & absolute)
        eff_cut_by_label = {}
        for lbl in DEFAULT_COLORS.keys():
            mask_lbl = (L[e_v] == lbl)
            r_lbl = R[e_v][mask_lbl]
            if r_lbl.size == 0:
                eff_cut_by_label[lbl] = -np.inf
                continue
            K = float(topk.get(lbl, 10.0))
            pct_cut = np.percentile(r_lbl, 100.0 - K)
            abs_cut = abs_thr.get(lbl, None)
            abs_cut = 0.0 if (abs_cut is None) else float(abs_cut)
            eff_cut_by_label[lbl] = max(pct_cut, abs_cut)

        # ---- 2D base edges (UNFILTERED — only hide_thin affects visibility)
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
            uu = e_u[mask_lbl]; vv = e_v[mask_lbl]
            x0, y0 = X2[uu], Y2[uu]
            x1, y1 = X2[vv], Y2[vv]
            Xs, Ys = segments_2d(x0, y0, x1, y1)
            cd = np.repeat(R[vv], 2).astype(np.float32)

            if not hide_thin:
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

        # ---- 2D overlay dots (filtered by MAX rule)
        selected_labels = set(type_selected or [])
        for lbl, color in DEFAULT_COLORS.items():
            if lbl not in selected_labels:
                continue

            mask_lbl = (L[e_v] == lbl) & (R[e_v] >= eff_cut_by_label[lbl])
            if not np.any(mask_lbl):
                continue

            vv = e_v[mask_lbl]
            dot_x, dot_y = X2[vv], Y2[vv]
            sz = np.clip((DOT_SCALE * R[vv]).astype(np.float32), DOT_MIN, DOT_MAX)

            if has_id:
                hover = [f"id={int(ID[v])} • radius={float(R[v]):.4f} • {lbl}" for v in vv]
            else:
                hover = [f"radius={float(R[v]):.4f} • {lbl}" for v in vv]

            traces2d.append(go.Scatter(
                x=dot_x, y=dot_y, mode="markers",
                marker=dict(size=sz, color=DOT_COLOR, line=dict(color=DOT_EDGE_COLOR, width=0.8), opacity=0.95),
                hovertemplate="%{text}<extra></extra>", text=hover,
                name=f"{lbl} (overlay: max % & abs)", legendgroup=f"{lbl}-overlay", showlegend=False,
            ))

        fig2d = go.Figure(traces2d)
        fig2d.update_layout(
            xaxis_title=xlab, yaxis_title=ylab, template="plotly_white",
            height=600, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Type",
            hovermode="closest", hoverdistance=15,
        )
        fig2d.update_yaxes(scaleanchor="x", scaleratio=1.0)

        # ---- 3D base (edges only, unfiltered)
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
            uu = e_u[lv]; vv = e_v[lv]
            x0, y0, z0 = coords_x[uu], coords_y[uu], coords_z[uu]
            x1, y1, z1 = coords_x[vv], coords_y[vv], coords_z[vv]
            m = uu.size
            X3 = np.empty(m * 3, dtype=np.float32); Y3 = np.empty(m * 3, dtype=np.float32); Z3 = np.empty(m * 3, dtype=np.float32)
            X3[0::3] = x0; X3[1::3] = x1; X3[2::3] = np.nan
            Y3[0::3] = y0; Y3[1::3] = y1; Y3[2::3] = np.nan
            Z3[0::3] = z0; Z3[1::3] = z1; Z3[2::3] = np.nan

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



    # ---------------- Viewer: downloads (cleaned SWC & clean log) ----------------
    @app.callback(
        Output("download-viewer-clean-swc", "data"),
        Input("btn-dl-viewer-clean-swc", "n_clicks"),
        State("store-working-df", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def viewer_download_cleaned_swc(n, df_records, filename):
        if not df_records:
            return dash.no_update
        df = pd.DataFrame(df_records)
        content = make_swc_plus_bytes_from_df(df, filename)
        base = os.path.splitext(filename or "viewer")[0]
        out_name = f"{base}_clean.swc"

        def writer(f):
            f.write(content)

        return dcc.send_bytes(writer, out_name)

    @app.callback(
        Output("download-viewer-clean-log", "data"),
        Input("btn-dl-viewer-clean-log", "n_clicks"),
        State("table-viewer-clean-log", "data"),
        State("store-filename", "data"),
        prevent_initial_call=True,
    )
    def viewer_download_clean_log(n, table_rows, filename):
        if not table_rows:
            return dash.no_update
        df = pd.DataFrame(table_rows)
        csv_text = df.to_csv(index=False)
        base = os.path.splitext(filename or "viewer_clean_log")[0]
        out_name = f"{base}_clean_log.csv"
        return dcc.send_string(csv_text, out_name)

    # =====================================================================
    #                          VALIDATION PAGE
    # =====================================================================
    @app.callback(
        Output("validate-file-info", "children", allow_duplicate=True),
        Output("table-validate-results", "data", allow_duplicate=True),
        Output("store-validate-table", "data", allow_duplicate=True),
        Output("store-working-df", "data", allow_duplicate=True),
        Output("store-filename", "data", allow_duplicate=True),
        Output("table-changes", "data", allow_duplicate=True),
        Output("table-changes", "page_current", allow_duplicate=True),
        Output("apply-msg", "children", allow_duplicate=True),
        Output("table-viewer-clean-log", "data", allow_duplicate=True),
        Output("viewer-clean-msg", "children", allow_duplicate=True),
        Input("upload-validate", "contents"),
        State("upload-validate", "filename"),
        prevent_initial_call=True,
    )
    def run_validation_upload(contents, filename):
        if not contents:
            return (
                "No file.",
                [],
                None,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        try:
            text = _decode_uploaded_text(contents)
            df = parse_swc_text_preserve_tokens(text)
            rows = []
            try:
                _, _sanitized_bytes, rows = run_format_validation_from_text(text)
            except Exception as e:
                return (
                    f"Validation failed: {e}",
                    [],
                    None,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            msg = f"Validated {filename or 'uploaded file'} • {len(rows)} checks"

            return (
                msg,
                rows,
                rows,
                df.to_dict("records") if not df.empty else [],
                filename,
                [],
                0,
                "",
                [],
                "",
            )
        except Exception as e:
            return (
                f"Validation failed: {e}",
                [],
                None,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

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
