from dash import dcc, html
from dash import dash_table
from constants import APP_TITLE


def _dendrogram_tab():
    return html.Div(
        [
            html.H2(APP_TITLE, style={"marginBottom": 6}),
            html.P(
                "Upload an SWC, see a 3D color-coded structure, then edit types in the dendrogram. "
                "Click a dendrogram branch to select a subtree, set a new SWC type, and apply. "
                "Download the edited SWC and a change log."
            ),

            html.Div(
                [
                    html.Span("File source: "),
                    html.Span(
                        "Upload on the Format Validation tab to load and share an SWC file.",
                        style={"fontWeight": 600},
                    ),
                ],
                style={
                    "padding": "12px",
                    "border": "1px solid #ddd",
                    "borderRadius": 6,
                    "backgroundColor": "#f7f7f7",
                    "marginBottom": 10,
                },
            ),
            html.Div(
                id="edit-file-info",
                children="Upload on the Format Validation tab to load a shared SWC.",
                style={"color": "#555", "marginBottom": 8},
            ),

            # 3D plot (under upload) and larger
            html.H4("3D Structure (edges only, color-coded by type)", style={"marginTop": 6}),
            dcc.Graph(
                id="fig-dendro-3d",
                style={
                    "height": 520,
                    "maxWidth": "640px",
                    "width": "100%",
                    "margin": "0 auto 8px",
                },
            ),

            # Dendrogram editor
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Dendrogram"),
                            dcc.Graph(id="fig-dendro", style={"height": 900}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Selected SWC id: "),
                                            html.Code(id="selected-node-id", children="-"),
                                        ],
                                        style={"marginTop": 4},
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Type: "),
                                            html.Span(
                                                id="type-chip",
                                                style={
                                                    "display": "inline-block",
                                                    "width": "12px",
                                                    "height": "12px",
                                                    "borderRadius": "2px",
                                                    "marginRight": "6px",
                                                    "backgroundColor": "#ccc",
                                                    "verticalAlign": "middle",
                                                },
                                            ),
                                            html.Span(id="selected-type", children="-", style={"fontWeight": 600}),
                                        ],
                                        style={"marginTop": 4},
                                    ),
                                ],
                                style={"marginTop": 6},
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="new-type", type="number",
                                        placeholder="New SWC type (0–7+ allowed)",
                                        min=0, step=1,
                                        style={"width": 240, "marginRight": 8},
                                    ),
                                    html.Button("Apply type change", id="btn-apply"),
                                ],
                                style={"marginTop": 8},
                            ),
                            dcc.RadioItems(
                                id="apply-scope",
                                options=[
                                    {"label": "Entire subtree", "value": "subtree"},
                                    {"label": "Selected node only", "value": "node"},
                                ],
                                value="subtree",
                                inline=True,
                                style={"marginTop": 6},
                            ),
                            html.Div(id="apply-msg", style={"color": "#0a7", "marginTop": 6}),
                        ],
                        style={"flex": 1, "minWidth": 0},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),

            html.Hr(),
            html.H4("Change log"),
            dash_table.DataTable(
                id="table-changes",
                columns=[
                    {"name": "node_id", "id": "node_id"},
                    {"name": "old_type", "id": "old_type"},
                    {"name": "new_type", "id": "new_type"},
                    {"name": "scope", "id": "scope"},
                ],
                data=[],
                page_size=10,
                style_table={"overflowX": "auto"},
            ),
            html.Div(
                [
                    html.Button("Download edited SWC", id="btn-dl-swc"),
                    html.Button("Download change log", id="btn-dl-log", style={"marginLeft": 10}),
                ],
                style={"marginTop": 10},
            ),
        ]
    )


def _validation_tab():
    return html.Div(
        [
            html.H2("SWC Format Validation", style={"marginBottom": 6}),
            dcc.Upload(
                id="upload-validate", multiple=False,
                children=html.Div(["Drag & drop or ", html.A("select an SWC file to validate")]),
                style={
                    "height": "64px", "lineHeight": "64px",
                    "border": "2px dashed #999", "borderRadius": 6,
                    "textAlign": "center", "marginBottom": 10,
                },
            ),

            html.Div(id="validate-file-info", style={"color": "#555", "marginBottom": 8}),
            dash_table.DataTable(
                id="table-validate-results",
                columns=[
                    {"name": "check",  "id": "check"},
                    {"name": "status", "id": "status"},
                ],
                data=[],
                page_size=15,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 14},
            ),

            html.Div(
                [
                    html.Button("Download validation JSON", id="btn-dl-validate-json"),
                ],
                style={"marginTop": 10},
            ),

            dcc.Store(id="store-validate-table"),
            dcc.Download(id="download-validate-json"),
        ]
    )


from dash import html, dcc

def _viewer_tab():
    def topk_row(
        label_txt: str,
        slider_id: str, input_id: str,
        abs_input_id: str,
        avg_id: str,
        mode_id: str,
        button_id: str,
        default_val: float = 10.0,
    ):
        """
        Per-type controls:
          - Top-K% slider + numeric input (K in [0.01, 10])
          - Absolute radius numeric input (>= 0)
          - Mode selector + clean button for that type
        """
        return html.Div(
            [
                html.Label(label_txt, style={"display": "block", "marginBottom": 6}),
                # K% controls
                html.Div(
                    [
                        html.Div(
                            dcc.Slider(
                                id=slider_id,
                                min=0.01, max=10.0, step=0.01, value=float(default_val),
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                                updatemode="mouseup",
                            ),
                            style={"flex": 1, "minWidth": 220}
                        ),
                        dcc.Input(
                            id=input_id, type="number",
                            min=0.01, max=10.0, step=0.01, value=float(default_val),
                            style={"marginLeft": 10, "width": 120},
                        ),
                        html.Span(" Top-K %", style={"marginLeft": 6, "color": "#666"}),
                        html.Span(id=avg_id, children=" Avg: -", style={"marginLeft": 14, "color": "#444"}),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                # Absolute controls
                html.Div(
                    [
                        dcc.Input(
                            id=abs_input_id, type="number",
                            placeholder="Absolute radius ≥ ...",
                            debounce=True, min=0, step=0.01,
                            style={"width": 220, "marginTop": 6},
                        ),
                        html.Span(" Absolute radius", style={"marginLeft": 8, "color": "#666"}),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(
                    [
                        dcc.RadioItems(
                            id=mode_id,
                            options=[
                                {"label": "Top-K%", "value": "percent"},
                                {"label": "Absolute", "value": "absolute"},
                            ],
                            value="percent",
                            inline=True,
                            style={"fontSize": 13},
                        ),
                        html.Button("Clean radii", id=button_id, style={"marginLeft": 12}),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginTop": 6, "gap": "8px"},
                ),
            ],
            style={"marginBottom": 18},
        )

    return html.Div(
        [
            html.H2("Radii cleaner", style={"marginBottom": 6}),
            html.P(
                "Per-type Top-K% (0.01–10) and per-type absolute radius control the overlay. "
            ),
            html.Div(
                [
                    html.Span("File source: "),
                    html.Span(
                        "Upload on the Format Validation tab. All tabs share the same working file.",
                        style={"fontWeight": 600},
                    ),
                ],
                style={
                    "padding": "12px",
                    "border": "1px solid #ddd",
                    "borderRadius": 6,
                    "backgroundColor": "#f7f7f7",
                    "marginBottom": 10,
                },
            ),
            html.Div(
                id="viewer-file-info",
                children="Upload on the Format Validation tab to load a shared SWC.",
                style={"color": "#555", "marginBottom": 12},
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Performance"),
                            dcc.Checklist(
                                id="viewer-performance",
                                options=[{"label": " Hide thin edges", "value": "hide_thin"}],
                                value=[],
                                inline=True,
                            ),
                        ],
                        style={"marginRight": 24},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "18px", "marginBottom": 10},
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H4("2D View"),
                            dcc.RadioItems(
                                id="viewer-2d-view",
                                options=[
                                    {"label": "XY (front)", "value": "xy"},
                                    {"label": "XZ (top)", "value": "xz"},
                                    {"label": "YZ (side)", "value": "yz"},
                                ],
                                value="xy",
                                inline=True,
                                style={"marginBottom": 8},
                            ),
                            dcc.Graph(id="fig-view-2d", style={"height": 600}),
                        ],
                        style={"flex": 1, "minWidth": 0},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "stretch"},
            ),

            html.Hr(),
            html.H4("Types to draw"),
            dcc.Checklist(
                id="viewer-type-select",
                options=[
                    {"label": "Undefined (0)", "value": "undefined"},
                    {"label": "Soma (1)", "value": "soma"},
                    {"label": "Axon (2)", "value": "axon"},
                    {"label": "Basal dendrite (3)", "value": "basal dendrite"},
                    {"label": "Apical dendrite (4)", "value": "apical dendrite"},
                    {"label": "Custom (5+)", "value": "custom"},
                ],
                value=["undefined", "soma", "axon", "basal dendrite", "apical dendrite", "custom"],
                inline=True,
                style={"marginBottom": 12},
            ),

            html.H4("Per-type Top-K% and Absolute Radius"),
            html.P(
                "Set thresholds to highlight large radii: K keeps the top K% (K=10 → top 10%, K=0.50 → top 0.5%) "
                "and Absolute keeps radii ≥ your value (e.g. ≥ 2.5). If both inputs exist for a type, the stricter "
                "cutoff is used. Compare overlays and averages to decide a sensible cutoff, then press that row's "
                "clean button to replace outliers with neighbor averages along the same branch.",
                style={"maxWidth": 940, "color": "#444"},
            ),
            html.Div(
                [
                    topk_row(
                        "Undefined",
                        "viewer-topk-undefined", "viewer-topk-undefined-input",
                        "viewer-abs-undefined", "viewer-avg-undefined",
                        "viewer-clean-mode-undefined", "btn-viewer-clean-undefined",
                        10.0,
                    ),
                    topk_row(
                        "Axon",
                        "viewer-topk-axon", "viewer-topk-axon-input",
                        "viewer-abs-axon", "viewer-avg-axon",
                        "viewer-clean-mode-axon", "btn-viewer-clean-axon",
                        10.0,
                    ),
                    topk_row(
                        "Basal dendrite",
                        "viewer-topk-basal", "viewer-topk-basal-input",
                        "viewer-abs-basal", "viewer-avg-basal",
                        "viewer-clean-mode-basal", "btn-viewer-clean-basal",
                        10.0,
                    ),
                    topk_row(
                        "Apical dendrite",
                        "viewer-topk-apical", "viewer-topk-apical-input",
                        "viewer-abs-apical", "viewer-avg-apical",
                        "viewer-clean-mode-apical", "btn-viewer-clean-apical",
                        10.0,
                    ),
                    topk_row(
                        "Custom (type≥5)",
                        "viewer-topk-custom", "viewer-topk-custom-input",
                        "viewer-abs-custom", "viewer-avg-custom",
                        "viewer-clean-mode-custom", "btn-viewer-clean-custom",
                        10.0,
                    ),
                ],
                style={"maxWidth": 940},
            ),
            html.Div(id="viewer-clean-msg", style={"marginTop": 6, "color": "#0a7"}),

            dash_table.DataTable(
                id="table-viewer-clean-log",
                columns=[
                    {"name": "node_id", "id": "node_id"},
                    {"name": "label", "id": "label"},
                    {"name": "old_radius", "id": "old_radius"},
                    {"name": "new_radius", "id": "new_radius"},
                    {"name": "mode", "id": "mode"},
                    {"name": "cutoff", "id": "cutoff"},
                ],
                data=[],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 14},
            ),
            html.Div(
                [
                    html.Button("Download cleaned SWC", id="btn-dl-viewer-clean-swc"),
                    html.Button("Download clean log", id="btn-dl-viewer-clean-log", style={"marginLeft": 10}),
                ],
                style={"marginTop": 10, "marginBottom": 10},
            ),

            # Stores
            dcc.Store(
                id="viewer-topk-store",
                data={"undefined": 10.0, "soma": 10.0, "axon": 10.0,
                      "basal dendrite": 10.0, "apical dendrite": 10.0, "custom": 10.0},
            ),
            dcc.Store(
                id="viewer-abs-store",
                data={"undefined": None, "soma": None, "axon": None,
                      "basal dendrite": None, "apical dendrite": None, "custom": None},
            ),

            dcc.Download(id="download-viewer-clean-swc"),
            dcc.Download(id="download-viewer-clean-log"),
        ]
    )



def build_layout():
    return html.Div(
        [
            dcc.Tabs(
                id="tabs",
                value="tab-validate",
                children=[
                    dcc.Tab(label="Format Validation", value="tab-validate"),
                    dcc.Tab(label="Dendrogram Editor", value="tab-dendro"),
                    dcc.Tab(label="Radii cleaner", value="tab-viewer"),
                ],
            ),
            html.Div(
                id="tab-content",
                children=[
                    html.Div(
                        _validation_tab(),
                        id="tab-pane-validate",
                    ),
                    html.Div(_dendrogram_tab(), id="tab-pane-dendro", style={"display": "none"}),
                    html.Div(_viewer_tab(), id="tab-pane-viewer", style={"display": "none"}),
                ],
            ),

            # global stores/downloads shared by the Dendrogram page
            dcc.Store(id="store-working-df"),
            dcc.Store(id="store-dendro-info"),
            dcc.Store(id="store-filename"),
            dcc.Download(id="download-edited-swc"),
            dcc.Download(id="download-changelog"),
        ],
        style={"fontFamily": "system-ui, Segoe UI, Roboto", "padding": 16},
    )
