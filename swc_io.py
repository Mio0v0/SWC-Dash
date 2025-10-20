import io
import numpy as np
import pandas as pd

from constants import SWC_COLS

def parse_swc_text_preserve_tokens(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue

        id_tok, type_tok, x_tok, y_tok, z_tok, radius_tok, parent_tok = parts[:7]

        def _to_int(tok, default):
            try:
                return int(float(tok))
            except Exception:
                return default

        def _to_float(tok):
            try:
                return float(tok)
            except Exception:
                return np.nan

        rows.append({
            "id": _to_int(id_tok, -1),
            "type": _to_int(type_tok, 0),
            "x": _to_float(x_tok),
            "y": _to_float(y_tok),
            "z": _to_float(z_tok),
            "radius": _to_float(radius_tok),
            "parent": _to_int(parent_tok, -1),
            "x_str": x_tok,
            "y_str": y_tok,
            "z_str": z_tok,
            "radius_str": radius_tok,
            "id_str": id_tok,
            "parent_str": parent_tok,
        })

    df = pd.DataFrame(rows, columns=[
        "id","type","x","y","z","radius","parent",
        "x_str","y_str","z_str","radius_str","id_str","parent_str"
    ])
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int)
    df["type"] = pd.to_numeric(df["type"], errors="coerce").fillna(0).astype(int)
    df["parent"] = pd.to_numeric(df["parent"], errors="coerce").fillna(-1).astype(int)
    return df

def write_swc_to_bytes_preserve_tokens(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    has_tok = all(c in df.columns for c in ["x_str","y_str","z_str","radius_str","id_str","parent_str"])
    for _, row in df.iterrows():
        id_out = row["id_str"] if has_tok and isinstance(row["id_str"], str) else str(int(row["id"]))
        parent_out = row["parent_str"] if has_tok and isinstance(row["parent_str"], str) else str(int(row["parent"]))
        type_out = str(int(row["type"]))

        x_out = row["x_str"] if has_tok and isinstance(row["x_str"], str) else f"{row['x']}"
        y_out = row["y_str"] if has_tok and isinstance(row["y_str"], str) else f"{row['y']}"
        z_out = row["z_str"] if has_tok and isinstance(row["z_str"], str) else f"{row['z']}"
        radius_out = row["radius_str"] if has_tok and isinstance(row["radius_str"], str) else f"{row['radius']}"

        buf.write(f"{id_out} {type_out} {x_out} {y_out} {z_out} {radius_out} {parent_out}\n")
    return buf.getvalue().encode("utf-8")
