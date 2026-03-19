# Architecture

This project follows a shared-core architecture:

- GUI and CLI are interface layers
- all real feature logic lives in Python backend modules
- plugin registry can override methods without rewriting tool interfaces

## Directory structure

```text
swctools/
  api.py
  core/
  tools/
    batch_processing/
    validation/
    visualization/
    morphology_editing/
    atlas_registration/
    analysis/
  plugins/
  cli/
  gui/
```

## Layer responsibilities

## `swctools/core`

Shared infrastructure and algorithm foundations:

- SWC parsing/writing
- validation engine + check models
- auto-typing engine
- radii cleaning primitives
- report formatting/writing
- config merge/load/save

## `swctools/tools`

Feature wrappers grouped by tool.

Each feature module typically provides:

- `TOOL`, `FEATURE`, `FEATURE_KEY`
- `DEFAULT_CONFIG`
- builtin method registration (`register_builtin_method`)
- public entrypoint(s) used by CLI/GUI/API

## `swctools/plugins`

Method registry + plugin loader contract.

- builtin methods and plugin methods share feature/method lookup
- external plugins register methods by feature key
- manifest includes versioned `api_version`

## `swctools/cli`

Thin argparse-based front-end.

- parses command args
- applies temporary config overrides
- calls shared tool feature functions
- prints structured result summaries

## `swctools/gui`

Qt desktop interface.

- renders controls, tables, plots, and logs
- uses same backend feature functions as CLI
- keeps algorithm behavior consistent with CLI

## Config design

Feature config files:

- `swctools/tools/<tool>/configs/<feature>.json`

JSON stores:

- method selection
- thresholds/weights/flags
- rule enable/disable options

Python stores:

- all actual algorithm logic

## Plugin extension points

Every feature can be addressed by `feature_key = "tool.feature"`.

Examples:

- `batch_processing.auto_typing`
- `validation.run_checks`
- `morphology_editing.simplification`
- `atlas_registration.registration`

A plugin can register one or more methods under these keys and select them through config/CLI.

## Data/report consistency

Validation and reporting are intentionally centralized:

- shared `ValidationReport` model
- shared report text formatting
- same outputs regardless of GUI or CLI caller
