"""Core auto-typing / rule-batch re-exports.

Expose the rule-based auto-typing API from a stable location under
`swctools.core` for library, CLI, and GUI callers.
"""
from typing import Any

from swctools.core.auto_typing_impl import (
    RuleBatchOptions,
    RuleBatchResult,
    RuleFileResult,
    get_config as get_auto_rules_config,
    run_rule_file,
    run_rule_batch,
    save_config as save_auto_rules_config,
)

__all__ = [
    "RuleBatchOptions",
    "RuleBatchResult",
    "RuleFileResult",
    "run_rule_batch",
    "run_rule_file",
    "get_auto_rules_config",
    "save_auto_rules_config",
]


def run_folder(folder: str, opts: Any):
    """Run the rule-based auto-typing over a folder (wrapper)."""
    return run_rule_batch(folder, opts)
