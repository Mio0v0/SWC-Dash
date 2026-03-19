"""Thin GUI shim for validation functions.

The implementation lives in :mod:`swctools.core.validation` so GUI and CLI
share exactly the same backend logic.
"""

from swctools.core.validation import (
    _split_swc_by_soma_roots,
    clear_cache,
    run_format_validation_from_text,
    run_per_tree_validation,
)

__all__ = [
    "run_format_validation_from_text",
    "run_per_tree_validation",
    "_split_swc_by_soma_roots",
    "clear_cache",
]
