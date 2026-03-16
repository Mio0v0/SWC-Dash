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

## Tool -> Feature Map (Current Structure)

- `batch`
  - `validate`
  - `split`
  - `auto-typing`
  - `radii-clean`
- `validation`
  - `rule-guide`
  - `run`
  - `auto-fix`
  - `radii-clean`
- `visualization`
  - `mesh-editing`
- `morphology`
  - `dendrogram-edit`
  - `smart-decimation`
- `atlas`
  - `register` (placeholder)
- `analysis`
  - `summary` (placeholder)
- `plugins`
  - `list`

## Common Option: `--config-json`

Most feature commands support inline config override:

```bash
swctools validation run file.swc --config-json '{"checks":{"no_back_tracking":{"enabled":true}}}'
```

The value must be a JSON object.

Shared radii-clean config file:

- `swctools/tools/batch_processing/configs/radii_cleaning.json`

## Commands

### Batch Processing

#### `swctools batch validate <folder> [--config-json JSON]`

Runs batch validation over all SWC files in a folder using the same checks as `validation run`.

Special alias:

- `swctools batch validate rule-guide` prints only the validation Rule Guide (no file run).

Output:

- pre-check summary (rule guide)
- per-file validation results
- batch summary + report file path

#### `swctools batch split <folder> [--config-json JSON]`

Splits each SWC into trees by soma roots.

Default output layout:

- output folder: `<folder>/<folder_name>_split`
- output files: `<original_swc_name>_tree1.swc`, `<original_swc_name>_tree2.swc`, ...

#### `swctools batch auto-typing <folder> [flags] [--config-json JSON]`

Rule-based auto-typing over a folder.

Before processing, CLI prints an `Auto Typing Rule Guide` section.

Flags:

- `--soma`
- `--axon`
- `--apic`
- `--basal`
- `--rad`
- `--zip`

If no flags are provided, defaults come from `auto_typing.json`.

#### `swctools batch radii-clean <file_or_folder> [--config-json JSON]`

Runs radii cleaning on one SWC file or all SWCs in a folder.

Behavior:

- detects non-positive/non-finite radii
- detects local spikes/dips vs parent+children neighborhood
- replaces abnormal radii with local mean(parent + children)
- writes a text log report

### Validation

#### `swctools validation rule-guide [--config-json JSON]`

Prints only the validation Rule Guide (no file run).

#### `swctools validation run <file.swc> [--config-json JSON]`

Runs structured validation checks and prints summary + details.

#### `swctools validation auto-fix <file.swc> [--write] [--out PATH] [--config-json JSON]`

Runs auto-fix plus validation report.

- `--write`: writes sanitized SWC
- `--out`: optional output path

#### `swctools validation radii-clean <file_or_folder> [--config-json JSON]`

Runs the same shared radii-clean backend as `batch radii-clean`.

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

#### `swctools morphology smart-decimation <file.swc> [--write] [--out PATH] [--config-json JSON]`

Runs RDP-based Smart Decimation with graph-aware node protection.

Protected node rules include:

- root/soma nodes
- optional tips and bifurcations (config flags)
- radius-sensitive nodes that exceed configured tolerance

- `--write`: write simplified SWC
- `--out`: optional output file path

Output includes node-count reduction and a simplification log file path.

### Atlas Registration (Placeholder)

#### `swctools atlas register <file.swc> [--atlas NAME] [--config-json JSON]`

Placeholder command (structured response, not implemented).

### Analysis (Placeholder)

#### `swctools analysis summary <file.swc> [--config-json JSON]`

Placeholder/basic summary command.

### Plugin Inspection

#### `swctools plugins list [--feature-key TOOL.FEATURE]`

Lists builtin and plugin method names currently registered.

## Exit Codes

- `0`: success
- `1`: invalid usage / help shown
- `2`: runtime error

## Examples

```bash
swctools batch validate rule-guide
swctools batch split /path/to/folder
swctools batch auto-typing /path/to/folder --soma --axon --basal
swctools batch radii-clean /path/to/file_or_folder

swctools validation rule-guide
swctools validation run /path/to/file.swc
swctools validation auto-fix /path/to/file.swc --write
swctools validation radii-clean /path/to/file_or_folder

swctools visualization mesh-editing /path/to/file.swc --include-edges

swctools morphology dendrogram-edit /path/to/file.swc --node-id 42 --new-type 3 --write
swctools morphology smart-decimation /path/to/file.swc --write

swctools plugins list
```
