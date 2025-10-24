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

            # Upload first
            dcc.Upload(
                id="upload-edit", multiple=False,
                children=html.Div(["Drag & drop or ", html.A("select an SWC file")]),
                style={
                    "height": "64px", "lineHeight": "64px",
                    "border": "2px dashed #999", "borderRadius": 6,
                    "textAlign": "center", "marginBottom": 10,
                },
            ),
            html.Div(id="edit-file-info", style={"color": "#555", "marginBottom": 8}),

            # 3D plot (under upload) and larger
            html.H4("3D Structure (edges only, color-coded by type)", style={"marginTop": 6}),
            dcc.Graph(id="fig-dendro-3d", style={"height": 900, "marginBottom": 5}),  # <-- taller box

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
                                    html.Button("Apply to subtree", id="btn-apply"),
                                ],
                                style={"marginTop": 8},
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
            html.P(
                "Drop an SWC file to run NeuroM checks. The file is sanitized (type=0 or >7 → 7) before checks."
            ),

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


def _viewer_tab():
    def percentile_slider(component_id: str):
        return dcc.RangeSlider(
            id=component_id,
            min=0, max=100, step=1, value=[70, 100],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
        )

    return html.Div(
        [
            html.H2("2D / 3D Viewer", style={"marginBottom": 6}),
            html.P(
                "Drop an SWC file. Edges are color-coded by type. "
                "Edge width follows child node radius. Use type selection and percentile windows to highlight."
            ),
            dcc.Upload(
                id="upload-viewer", multiple=False,
                children=html.Div(["Drag & drop or ", html.A("select an SWC file to view")]),
                style={
                    "height": "64px", "lineHeight": "64px",
                    "border": "2px dashed #999", "borderRadius": 6,
                    "textAlign": "center", "marginBottom": 10,
                },
            ),
            html.Div(id="viewer-file-info", style={"color": "#555", "marginBottom": 12}),

            # Performance controls
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Performance mode"),
                            dcc.Checklist(
                                id="viewer-performance",
                                options=[
                                    {"label": " Quantize widths (fast)", "value": "quantize"},
                                    {"label": " Hide thin edges", "value": "hide_thin"},
                                ],
                                value=["quantize"],  # default ON
                                inline=True,
                            ),
                        ],
                        style={"marginRight": 24},
                    ),
                    html.Div(
                        [
                            html.Label("Width bins per type (when quantized)"),
                            dcc.Slider(
                                id="viewer-width-bins",
                                min=4, max=32, step=1, value=12,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"flex": 1, "minWidth": 240},
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
                        style={"flex": 1, "minWidth": 0, "marginRight": 12},
                    ),
                    html.Div(
                        [
                            html.H4("3D View"),
                            dcc.Graph(id="fig-view-3d", style={"height": 600}),
                        ],
                        style={"flex": 1, "minWidth": 0},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "stretch"},
            ),

            html.Hr(),
            html.H4("Types to highlight (percentile windows apply per type)"),
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
                value=["undefined", "soma", "axon", "basal dendrite", "apical dendrite", "custom"],  # all selected
                inline=True,
                style={"marginBottom": 10},
            ),

            html.H4("Percentile windows per type"),
            html.P("Example: [70,100] highlights the top 30% radii of that type. Outside the window = thin edges."),
            html.Div(
                [
                    html.Div([html.Label("Undefined"), percentile_slider("viewer-range-undefined")], style={"marginBottom": 14}),
                    html.Div([html.Label("Soma"), percentile_slider("viewer-range-soma")], style={"marginBottom": 14}),
                    html.Div([html.Label("Axon"), percentile_slider("viewer-range-axon")], style={"marginBottom": 14}),
                    html.Div([html.Label("Basal dendrite"), percentile_slider("viewer-range-basal")], style={"marginBottom": 14}),
                    html.Div([html.Label("Apical dendrite"), percentile_slider("viewer-range-apical")], style={"marginBottom": 14}),
                    html.Div([html.Label("Custom (type≥5)"), percentile_slider("viewer-range-custom")], style={"marginBottom": 14}),
                ],
                style={"maxWidth": 760},
            ),

            dcc.Store(id="store-viewer-df"),
        ]
    )


def build_layout():
    return html.Div(
        [
            dcc.Tabs(
                id="tabs",
                value="tab-dendro",
                children=[
                    dcc.Tab(label="Dendrogram Editor", value="tab-dendro"),
                    dcc.Tab(label="Format Validation", value="tab-validate"),
                    dcc.Tab(label="2D/3D Viewer", value="tab-viewer"),
                ],
            ),
            html.Div(id="tab-content"),

            # global stores/downloads shared by the Dendrogram page
            dcc.Store(id="store-working-df"),
            dcc.Store(id="store-dendro-info"),
            dcc.Store(id="store-filename"),
            dcc.Download(id="download-edited-swc"),
            dcc.Download(id="download-changelog"),
        ],
        style={"fontFamily": "system-ui, Segoe UI, Roboto", "padding": 16},
    )
