"""Backward-compatible wrappers for validation feature imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .auto_fix import auto_fix_file, auto_fix_text
from swctools.core.validation import run_per_tree_validation
from .run_checks import validate_file as run_checks_file, validate_text as run_checks_text


def validate_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return auto_fix_file(path, write_output=False)


def per_tree(text: str):
    return run_per_tree_validation(text)


__all__ = [
    "validate_file",
    "per_tree",
    "auto_fix_file",
    "auto_fix_text",
    "run_checks_file",
    "run_checks_text",
]
