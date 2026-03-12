"""Shared validation backend package."""

from .engine import build_precheck_summary, load_validation_config, run_validation_text
from .registry import get_check, list_checks, register_check, register_plugin_check
from .results import CheckResult, PreCheckItem, ValidationReport

__all__ = [
    "CheckResult",
    "PreCheckItem",
    "ValidationReport",
    "register_check",
    "register_plugin_check",
    "get_check",
    "list_checks",
    "load_validation_config",
    "build_precheck_summary",
    "run_validation_text",
]
