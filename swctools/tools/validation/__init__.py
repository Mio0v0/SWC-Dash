"""Validation tool API.

`tools/validation` is the public entry layer. Core implementation lives in
`swctools.core` and is consumed by these tool features.
"""

from swctools.core.validation_engine import (
    build_precheck_summary,
    load_validation_config,
    run_validation_text,
)
from swctools.core.validation_registry import (
    get_check,
    list_checks,
    register_check,
    register_plugin_check,
)
from swctools.core.validation_results import CheckResult, PreCheckItem, ValidationReport

from .features import auto_fix, auto_typing, radii_cleaning, run_checks

__all__ = [
    "auto_fix",
    "auto_typing",
    "run_checks",
    "radii_cleaning",
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
