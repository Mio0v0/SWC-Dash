# CLI Reference

This is the complete command reference for the `swctools` CLI.

## Install and Verify

```bash
pip install -e .
swctools --help
```

## Command Shape

```bash
swctools <tool> <feature> [args] [options]
```

Top-level tools:

- `batch`
- `validation`
- `visualization`
- `morphology`
- `atlas`
- `analysis`
- `plugins`

## Common Option: `--config-json`

Most feature commands accept:

```bash
--config-json '{"some":"override"}'
```

This must be a JSON object and is merged into feature config for that run.

## `batch`

### `swctools batch validate <folder>`

- Purpose: run validation on all SWC files in a folder
- Special behavior: if `folder` is literal `rule-guide`, prints rules only

Options:

- `--config-json JSON`

Example:

```bash
swctools batch validate ./data
swctools batch validate rule-guide
```

### `swctools batch split <folder>`

- Purpose: split SWC files by soma-root trees

Options:

- `--config-json JSON`

Example:

```bash
swctools batch split ./data
```

### `swctools batch auto-typing <folder>`

- Purpose: rule-based auto-labeling for SWCs in folder
- CLI prints auto-typing rule guide before processing

Flags:

- `--soma`
- `--axon`
- `--apic`
- `--basal`
- `--rad`
- `--zip`
- `--config-json JSON`

Example:

```bash
swctools batch auto-typing ./data --soma --axon --basal
```

### `swctools batch radii-clean <target>`

- Purpose: clean abnormal radii on a file or folder

Arguments:

- `target`: file path or directory path

Options:

- `--threshold-mode {percentile,absolute}`
- `--fix-soma-radii`
- `--preserve-soma-radii`
- `--percentile-min FLOAT`
- `--percentile-max FLOAT`
- `--abs-min FLOAT`
- `--abs-max FLOAT`
- `--config-json JSON`

Example:

```bash
swctools batch radii-clean ./data/single-soma.swc --threshold-mode absolute --abs-min 0.05 --abs-max 20
```

## `validation`

### `swctools validation rule-guide`

- Purpose: print validation pre-check/rule guide only

Options:

- `--config-json JSON`

### `swctools validation run <file>`

- Purpose: run full structured validation for one file

Options:

- `--config-json JSON`

Example:

```bash
swctools validation run ./data/single-soma.swc
```

### `swctools validation auto-fix <file>`

- Purpose: sanitize + revalidate one file

Options:

- `--write`
- `--out PATH`
- `--config-json JSON`

Example:

```bash
swctools validation auto-fix ./data/single-soma.swc --write
```

### `swctools validation radii-clean <target>`

- Purpose: same shared radii-clean backend used by batch

Arguments:

- `target`: file path or directory path

Options:

- `--threshold-mode {percentile,absolute}`
- `--fix-soma-radii`
- `--preserve-soma-radii`
- `--percentile-min FLOAT`
- `--percentile-max FLOAT`
- `--abs-min FLOAT`
- `--abs-max FLOAT`
- `--config-json JSON`

## `visualization`

### `swctools visualization mesh-editing <file>`

- Purpose: build reusable mesh payload summary

Options:

- `--include-edges`
- `--config-json JSON`

## `morphology`

### `swctools morphology dendrogram-edit <file> --node-id N --new-type T`

- Purpose: reassign subtree node types

Options:

- `--node-id INT` (required)
- `--new-type INT` (required)
- `--write`
- `--out PATH`
- `--config-json JSON`

Example:

```bash
swctools morphology dendrogram-edit ./data/single-soma.swc --node-id 42 --new-type 3 --write
```

### `swctools morphology smart-decimation <file>`

- Purpose: graph-aware RDP simplification
- CLI prints simplification rule guide before running

Options:

- `--write`
- `--out PATH`
- `--config-json JSON`

Example:

```bash
swctools morphology smart-decimation ./data/single-soma.swc --write
```

## `atlas`

### `swctools atlas register <file>`

- Purpose: atlas registration placeholder command

Options:

- `--atlas NAME`
- `--config-json JSON`

## `analysis`

### `swctools analysis summary <file>`

- Purpose: basic morphology summary

Options:

- `--config-json JSON`

## `plugins`

### `swctools plugins list`

- Purpose: list builtin/plugin methods in registry

Options:

- `--feature-key TOOL.FEATURE`

Example:

```bash
swctools plugins list
swctools plugins list --feature-key batch_processing.auto_typing
```

### `swctools plugins load <module>`

- Purpose: load an external plugin module by Python import path
- Contract expected:
  - `PLUGIN_MANIFEST` dict (or `get_plugin_manifest()`)
  - `register_plugin(registrar)` OR `PLUGIN_METHODS`
- Scope: current CLI process only (use `SWCTOOLS_PLUGINS` for autoload in each run)

Example:

```bash
swctools plugins load my_lab_plugins.brainglobe_adapter
```

### `swctools plugins list-loaded`

- Purpose: list loaded plugin manifests (`plugin_id`, version, capabilities, API version)

Example:

```bash
swctools plugins list-loaded
```

### Environment Autoload

Plugins can autoload on every CLI run via:

```bash
export SWCTOOLS_PLUGINS="my_lab_plugins.brainglobe_adapter,my_lab_plugins.custom_auto_typing"
```

## Output and Reports

Most commands return JSON/text in terminal and also write report files where applicable.

Typical reports include:

- `*_validation_report.txt`
- `*_batch_validation_report.txt`
- `split_report.txt`
- `*_radii_cleaning_report.txt`
- `*_auto_typing_report.txt`
- `*_simplification_log.txt`

## Exit Codes

- `0`: success
- `1`: usage error / parse error
- `2`: runtime error
