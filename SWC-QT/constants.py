APP_TITLE = "SWC Tools – Dendrogram Editor"

SWC_COLS = ["id", "type", "x", "y", "z", "radius", "parent"]

TYPE_LABEL = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal dendrite",
    4: "apical dendrite",
}

DEFAULT_COLORS = {
    "undefined": "#808080",
    "soma": "#2ca02c",
    "axon": "#1f77b4",
    "basal dendrite": "#d62728",
    "apical dendrite": "#e377c2",
    "custom": "#ff7f0e",  # for types >= 5
}

# Distinct colors for identifying trees (up to 10)
TREE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

def label_for_type(t: int) -> str:
    t = int(t)
    return TYPE_LABEL.get(t, "custom") if t <= 4 else "custom"

def color_for_type(t: int) -> str:
    return DEFAULT_COLORS[label_for_type(t)]
