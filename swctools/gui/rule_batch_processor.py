"""Thin GUI shim for rule-based auto-typing.

The actual implementation lives in `swctools.core.auto_typing_impl` so both GUI
and CLI import the same backend logic. This module preserves the original
import path for backwards compatibility.
"""

from swctools.core.auto_typing_impl import (
    RuleBatchOptions,
    RuleBatchResult,
    run_rule_batch,
    get_config as get_config,
    save_config as save_config,
)

__all__ = [
    "RuleBatchOptions",
    "RuleBatchResult",
    "run_rule_batch",
    "get_config",
    "save_config",
]
