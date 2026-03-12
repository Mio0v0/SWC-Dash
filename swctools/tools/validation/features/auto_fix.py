"""Auto-fix validation feature.

This feature runs the shared validation backend and returns the sanitized SWC
content that can be written back to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swctools.core.config import load_feature_config, merge_config
from swctools.core.validation import run_format_validation_from_text
from swctools.plugins.registry import register_builtin_method, resolve_method
from swctools.tools.validation.features.run_checks import validate_text as run_structured_validation

TOOL = "validation"
FEATURE = "auto_fix"
FEATURE_KEY = f"{TOOL}.{FEATURE}"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "method": "default",
    "profile": "default",
    "fixes": {
        "sanitize_invalid_types": True,
    },
}


def _builtin_auto_fix_text(swc_text: str, config: dict[str, Any]) -> dict[str, Any]:
    profile = str(config.get("profile", "default"))
    results, sanitized_bytes, rows = run_format_validation_from_text(swc_text)
    sanitized_text = sanitized_bytes.decode("utf-8", errors="ignore")
    report = run_structured_validation(sanitized_text, profile=profile)
    return {
        "report": report.to_dict(),
        "results": results,
        "rows": rows,
        "sanitized_bytes": sanitized_bytes,
        "sanitized_text": sanitized_text,
    }


register_builtin_method(FEATURE_KEY, "default", _builtin_auto_fix_text)


def get_config() -> dict[str, Any]:
    loaded = load_feature_config(TOOL, FEATURE, default=DEFAULT_CONFIG)
    return merge_config(DEFAULT_CONFIG, loaded)


def auto_fix_text(swc_text: str, *, config_overrides: dict | None = None) -> dict[str, Any]:
    cfg = merge_config(get_config(), config_overrides)

    method = str(cfg.get("method", "default"))
    fn = resolve_method(FEATURE_KEY, method)
    return fn(swc_text, cfg)


def auto_fix_file(
    path: str,
    *,
    out_path: str | None = None,
    write_output: bool = False,
    config_overrides: dict | None = None,
) -> dict[str, Any]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(path)

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    out = auto_fix_text(text, config_overrides=config_overrides)

    output_path: Path | None = None
    if write_output:
        if out_path:
            output_path = Path(out_path)
        else:
            output_path = in_path.with_name(f"{in_path.stem}_autofix{in_path.suffix}")
        output_path.write_bytes(out["sanitized_bytes"])

    out["input_path"] = str(in_path)
    out["output_path"] = str(output_path) if output_path else None
    return out
