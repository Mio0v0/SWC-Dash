# Logs and Reports

The project has a shared report layer so CLI and GUI produce consistent text logs.

Core module:

- `swctools.core.reporting`

## Why this matters

- one report format per feature
- same content in CLI and GUI runs
- easier audit trail for lab workflows

## Report builders

Main helper functions:

- `format_validation_report_text`
- `format_batch_validation_report_text`
- `format_split_report_text`
- `format_auto_typing_report_text`
- `format_radii_cleaning_report_text`
- `format_simplification_report_text`
- `format_morphology_session_log_text`
- `write_text_report`

## Default report naming

## Validation

- single file: `<stem>_validation_report.txt`
- batch folder: `<folder>_batch_validation_report.txt` (in target/output folder)

## Batch Split

- output folder: `<folder>/<folder>_split/`
- report file: `split_report.txt`

## Auto Typing

- output folder: `<folder>/<folder>_auto_typing/`
- report file: `auto_typing_report.txt`

## Radii Cleaning

- file mode report: `<stem>_radii_cleaning_report.txt`
- folder mode report: `<out_dir>/radii_cleaning_report.txt`

## Smart Decimation

- report: `<stem>_simplification_log.txt`

## Morphology edit session

- report: `<stem>_morphology_session_log.txt`
- records all edits in one app session for that SWC

## Typical report contents

Validation reports:

- pre-check rule summary
- grouped pass/warn/fail results
- detailed findings (node IDs, section IDs, thresholds, metrics)

Auto typing reports:

- files processed/failed
- total nodes and type changes
- per-file output type counts
- node-level change details

Radii cleaning reports:

- number of changed nodes
- per-file change summary
- node-level old/new values and reason tags

Simplification logs:

- original/new node counts
- reduction percentage
- parameters used
- protected node stats

## Programmatic use

Use `write_text_report(path, text)` in custom features/plugins to produce logs in the same style as built-in tools.
