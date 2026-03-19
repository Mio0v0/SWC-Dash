"""Single-file auto typing for Validation tool.

Uses the same core rule engine and JSON config as Batch Processing -> Auto Typing.
"""

from __future__ import annotations

from typing import Any

from swctools.core.auto_typing import RuleBatchOptions, run_rule_file
from swctools.core.config import merge_config
from swctools.tools.batch_processing.features.auto_typing import get_config as get_batch_auto_config


def _options_from_config(cfg: dict[str, Any]) -> RuleBatchOptions:
    opts = dict(cfg.get("options", {}))
    return RuleBatchOptions(
        soma=bool(opts.get("soma", True)),
        axon=bool(opts.get("axon", True)),
        apic=bool(opts.get("apic", False)),
        basal=bool(opts.get("basal", True)),
        rad=bool(opts.get("rad", False)),
        zip_output=bool(opts.get("zip_output", False)),
    )


def run_file(
    file_path: str,
    *,
    options: RuleBatchOptions | None = None,
    config_overrides: dict | None = None,
    output_path: str | None = None,
    write_output: bool = True,
    write_log: bool = True,
):
    cfg = get_batch_auto_config()
    if isinstance(config_overrides, dict) and config_overrides:
        cfg = merge_config(cfg, config_overrides)
    opts = options if options is not None else _options_from_config(cfg)
    return run_rule_file(
        file_path,
        opts,
        output_path=output_path,
        write_output=write_output,
        write_log=write_log,
    )
