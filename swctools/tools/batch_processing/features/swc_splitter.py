"""SWC splitter feature for Batch Processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.validation import _split_swc_by_soma_roots
from swctools.plugins.registry import register_builtin_method, resolve_method

TOOL = "batch_processing"
FEATURE = "swc_splitter"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "naming": {
        "output_mode": "single_output_subdir",
        "output_dir_pattern": "{folder_name}_split",
        "tree_pattern": "{stem}_tree{index}.swc",
    },
}


def _builtin_split_text(swc_text: str, config: dict[str, Any]) -> list[tuple[int, str, int]]:
    _ = config
    return _split_swc_by_soma_roots(swc_text)


register_builtin_method(FEATURE_KEY, "default", _builtin_split_text)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def split_swc_text(swc_text: str, *, config_overrides: dict | None = None):
    cfg = merge_config(get_config(), config_overrides)
    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def split_folder(folder: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    in_dir = Path(folder)
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(folder)

    swc_files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".swc")
    files_split = 0
    files_skipped = 0
    trees_saved = 0
    failures: list[str] = []

    naming_cfg = dict(cfg.get("naming", {}))
    output_mode = str(naming_cfg.get("output_mode", "single_output_subdir")).lower()
    output_dir_pattern = str(naming_cfg.get("output_dir_pattern", "{folder_name}_split"))
    tree_pattern = str(naming_cfg.get("tree_pattern", "{stem}_tree{index}.swc"))

    out_dir = in_dir / output_dir_pattern.format(folder_name=in_dir.name)
    out_dir_created = False

    for fp in swc_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            trees = split_swc_text(text, config_overrides=cfg)
            if len(trees) <= 1:
                files_skipped += 1
                continue

            files_split += 1
            if output_mode == "per_file_subdir":
                file_out_dir = out_dir / fp.stem
            else:
                file_out_dir = out_dir
            if not out_dir_created or output_mode == "per_file_subdir":
                file_out_dir.mkdir(parents=True, exist_ok=True)
                out_dir_created = True

            for idx, (_root_id, sub_text, _node_count) in enumerate(trees, start=1):
                out_name = tree_pattern.format(stem=fp.stem, index=idx)
                out_path = file_out_dir / out_name
                out_path.write_text(sub_text, encoding="utf-8")
                trees_saved += 1
        except Exception as e:  # noqa: BLE001
            failures.append(f"{fp.name}: {e}")

    return {
        "folder": str(in_dir),
        "out_dir": str(out_dir),
        "files_total": len(swc_files),
        "files_split": files_split,
        "files_skipped": files_skipped,
        "trees_saved": trees_saved,
        "failures": failures,
    }
