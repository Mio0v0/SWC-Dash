# Validation System and Rules

This project uses one shared validation backend for GUI and CLI.

## Architecture

Validation components:

- engine: `swctools.core.validation_engine`
- registry: `swctools.core.validation_registry`
- result models: `swctools.core.validation_results`
- check catalogs and friendly labels: `swctools.core.validation_catalog`
- wrappers/entrypoints: `swctools.tools.validation.features.run_checks`

Both GUI and CLI call this same pipeline:

1. load config (`default.json`)
2. build pre-check summary
3. run enabled checks
4. return structured `ValidationReport`
5. render in UI/terminal and optionally export report

## Config file

Main validation config:

- `swctools/tools/validation/configs/default.json`

Per-check structure follows:

```json
{
  "checks": {
    "has_soma": {
      "enabled": true,
      "severity": "warning",
      "params": {}
    }
  }
}
```

Notes:

- `enabled`: run/skip check
- `severity`: if failed, classify as `warning` or `fail`
- `params`: check-specific thresholds/options

## Rule categories and friendly labels

Validation rules are grouped into these categories:

## Structural presence

- Soma present
- Axon present
- Basal dendrite present
- Apical dendrite present

## Radius and size

- All neurite radii are positive
- Soma radius is positive
- No extremely narrow sections
- No extremely narrow branch starts
- No oversized terminal ends

## Length and geometry

- All section lengths are positive
- All segment lengths are positive
- No geometric backtracking
- No flattened neurites
- No duplicate 3D points

## Topology

- No dangling branches
- No single-child chains
- Contains unifurcation
- Contains multifurcation

## Index consistency

- No section index gaps
- No root index gaps

These labels/rules are centralized in:

- `swctools/core/validation_catalog.py`

## Output model

Each check returns a `CheckResult` with structured fields:

- `key`, `label`
- `passed`, `status` (`pass`, `warning`, `fail`)
- `message`
- `failing_node_ids`
- `failing_section_ids`
- `metrics`
- `params_used`
- `thresholds_used`

Full run returns `ValidationReport` containing:

- `precheck` list
- `results` list
- `summary` counts

## CLI behavior

For `validation run` and `batch validate`, CLI prints:

1. pre-check summary (rule guide)
2. grouped result summary
3. detailed findings for warnings/fails

## GUI behavior

Validation panels show:

- Rule Guide (manual button)
- results table (status + label)
- detail view per failed/warn row
- export/download report controls

## Why GUI and CLI stay consistent

Because both call `run_validation_text(...)` through tool wrappers, there is only one source of validation logic and one result format.
