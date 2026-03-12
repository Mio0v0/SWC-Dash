"""Public core API.

This module exposes shared backend primitives used by tool/feature modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.auto_typing import RuleBatchOptions, get_auto_rules_config, run_rule_batch
from swctools.core.swc_io import (
    SWC_COLS,
    parse_swc_text_preserve_tokens,
    write_swc_to_bytes_preserve_tokens,
)
from swctools.core.validation import (
    _split_swc_by_soma_roots,
    run_format_validation_from_text,
    run_per_tree_validation,
)


def parse_swc_text(text: str):
    return parse_swc_text_preserve_tokens(text)


def write_swc_bytes(df) -> bytes:
    return write_swc_to_bytes_preserve_tokens(df)


def validate_text(swc_text: str):
    return run_format_validation_from_text(swc_text)


def validate_file(path: Path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    return validate_text(txt)


def per_tree_validation(swc_text: str):
    return run_per_tree_validation(swc_text)


def split_by_soma_roots(swc_text: str):
    return _split_swc_by_soma_roots(swc_text)


def run_auto_typing_folder(folder: str, options: RuleBatchOptions | None = None) -> Any:
    opts = options if options is not None else RuleBatchOptions()
    return run_rule_batch(folder, opts)


__all__ = [
    "SWC_COLS",
    "RuleBatchOptions",
    "parse_swc_text",
    "write_swc_bytes",
    "validate_text",
    "validate_file",
    "per_tree_validation",
    "split_by_soma_roots",
    "run_auto_typing_folder",
    "get_auto_rules_config",
]
