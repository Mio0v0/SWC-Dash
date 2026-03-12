# CLI Reference

This document covers the `swctools` command-line interface.

## Install and Run

```bash
pip install -e .
swctools --help
```

Entry points from `pyproject.toml`:

- `swctools` -> CLI
- `swctools-gui` -> GUI launcher

## Command Structure

```bash
swctools <tool> <feature> [arguments] [options]
```

Use the full `swctools ...` command. Do not run `batch ...` by itself.

Current tools:

- `batch`
- `validation`
- `visualization`
- `morphology`
- `atlas`
- `analysis`
- `plugins`

## Common Option: `--config-json`

Most feature commands support inline config override:

```bash
swctools validation run file.swc --config-json '{"checks":{"no_back_tracking":{"enabled":true}}}'
```

The value must be a JSON object.

## Commands

### Batch Processing

#### `swctools batch validate <folder> [--config-json JSON]`

Runs batch validation over all SWC files in a folder using the same rule set as `validation run`.

CLI prints:

- pre-check summary (same rule text as Validation)
- per-file validation summaries/details

#### `swctools batch split <folder> [--config-json JSON]`

Splits each SWC into trees by soma roots. Output naming is controlled by `swctools/tools/batch_processing/configs/swc_splitter.json`.

#### `swctools batch auto-typing <folder> [flags] [--config-json JSON]`

Rule-based auto-typing over a folder.

Before processing, CLI prints an `Auto Typing Rule Guide` section that explains:

- how the auto-typing pipeline works
- the active decision boundaries/thresholds

Flags:

- `--soma`
- `--axon`
- `--apic`
- `--basal`
- `--rad`
- `--zip`

If no flags are provided, defaults come from `auto_typing.json`.

#### `swctools batch radii-clean <folder> [--config-json JSON]`

Cleans invalid radii in folder SWCs.

### Validation

#### `swctools validation run <file.swc> [--config-json JSON]`

Runs structured validation checks and prints:

- pre-check summary (rules + params)
- result summary
- detailed findings for warning/fail checks

#### `swctools validation auto-fix <file.swc> [--write] [--out PATH] [--config-json JSON]`

Runs auto-fix plus structured validation report.

- `--write`: writes sanitized SWC
- `--out`: optional output file path

Without `--write`, data is returned/printed but not saved.

### Visualization

#### `swctools visualization mesh-editing <file.swc> [--include-edges] [--config-json JSON]`

Builds mesh payload summary for GUI/CLI use.

### Morphology Editing

#### `swctools morphology dendrogram-edit <file.swc> --node-id N --new-type T [--write] [--out PATH] [--config-json JSON]`

Reassigns a subtree node type.

- `--node-id`: subtree root node id
- `--new-type`: SWC type to assign
- `--write`: write updated file
- `--out`: optional output file path

### Atlas Registration (Placeholder)

#### `swctools atlas register <file.swc> [--atlas NAME] [--config-json JSON]`

Placeholder command. Returns a structured "not implemented" result.

### Analysis

#### `swctools analysis summary <file.swc> [--config-json JSON]`

Computes basic morphology summary metrics.

### Plugin Inspection

#### `swctools plugins list [--feature-key TOOL.FEATURE]`

Lists builtin and plugin method names currently registered.

## Exit Codes

- `0`: success
- `1`: invalid command usage / help displayed
- `2`: runtime error

## Examples

```bash
swctools batch split /path/to/folder
swctools batch auto-typing /path/to/folder --soma --axon --basal
swctools validation run /path/to/file.swc
swctools validation auto-fix /path/to/file.swc --write
swctools visualization mesh-editing /path/to/file.swc --include-edges
swctools morphology dendrogram-edit /path/to/file.swc --node-id 42 --new-type 3 --write
swctools plugins list
```
