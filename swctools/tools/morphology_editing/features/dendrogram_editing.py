"""Dendrogram editing backend utilities.

Provides data-level operations used by interactive dendrogram editors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from swctools.core.config import load_feature_config, merge_config
from swctools.core.swc_io import parse_swc_text_preserve_tokens, write_swc_to_bytes_preserve_tokens
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "morphology_editing"
FEATURE = "dendrogram_editing"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "rules": {
        "include_selected_node": True,
        "preserve_soma_type": True,
    },
}


def _build_children(df: pd.DataFrame) -> tuple[dict[int, int], list[list[int]]]:
    id_to_idx = {int(df.iloc[i]["id"]): i for i in range(len(df))}
    children: list[list[int]] = [[] for _ in range(len(df))]
    for i in range(len(df)):
        pid = int(df.iloc[i]["parent"])
        pidx = id_to_idx.get(pid)
        if pidx is not None:
            children[pidx].append(i)
    return id_to_idx, children


def _collect_subtree(start_idx: int, children: list[list[int]]) -> list[int]:
    out: list[int] = []
    stack = [start_idx]
    seen = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        stack.extend(children[idx])
    return out


def _builtin_reassign_subtree(
    df: pd.DataFrame,
    node_id: int,
    new_type: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, int, list[int]]:
    id_to_idx, children = _build_children(df)
    start_idx = id_to_idx.get(int(node_id))
    if start_idx is None:
        raise KeyError(f"Node id {node_id} not found")

    rules = config.get("rules", {})
    include_selected = bool(rules.get("include_selected_node", True))
    preserve_soma = bool(rules.get("preserve_soma_type", True))

    target_idx = _collect_subtree(start_idx, children)
    if not include_selected:
        target_idx = [i for i in target_idx if i != start_idx]

    out = df.copy()
    changed = 0
    changed_ids: list[int] = []
    for i in target_idx:
        current_type = int(out.iloc[i]["type"])
        if preserve_soma and current_type == 1:
            continue
        if current_type == int(new_type):
            continue
        out.at[out.index[i], "type"] = int(new_type)
        changed += 1
        changed_ids.append(int(out.iloc[i]["id"]))

    return out, changed, changed_ids


register_builtin_method(FEATURE_KEY, "default", _builtin_reassign_subtree)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def reassign_subtree_types(
    swc_text: str,
    *,
    node_id: int,
    new_type: int,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    df = parse_swc_text_preserve_tokens(swc_text)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    out_df, changed, changed_ids = fn(df, int(node_id), int(new_type), cfg)

    return {
        "changes": int(changed),
        "changed_node_ids": changed_ids,
        "dataframe": out_df,
        "bytes": write_swc_to_bytes_preserve_tokens(out_df),
    }


def reassign_subtree_types_in_file(
    path: str,
    *,
    node_id: int,
    new_type: int,
    out_path: str | None = None,
    write_output: bool = False,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(path)

    text = fp.read_text(encoding="utf-8", errors="ignore")
    out = reassign_subtree_types(
        text,
        node_id=node_id,
        new_type=new_type,
        config_overrides=config_overrides,
    )

    output_path: Path | None = None
    if write_output:
        output_path = Path(out_path) if out_path else fp.with_name(f"{fp.stem}_typed{fp.suffix}")
        output_path.write_bytes(out["bytes"])

    out["input_path"] = str(fp)
    out["output_path"] = str(output_path) if output_path else None
    return out
