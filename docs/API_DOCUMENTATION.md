# API Documentation (Library Reference)

This document describes the Python library API for `swctools`.

## Design

- CLI and GUI both call shared backend functions.
- Feature logic lives in Python modules under `swctools/tools`.
- Config values live in JSON files under each tool's `configs/` folder.
- Plugin methods can override builtin feature methods through the registry.

## Primary Import

```python
import swctools.api as swc
```

## Public API (`swctools.api`)

### Data Types

#### `RuleBatchOptions`

Dataclass used by batch auto-typing:

- `soma: bool`
- `axon: bool`
- `apic: bool`
- `basal: bool`
- `rad: bool`
- `zip_output: bool`

### Batch Processing

#### `batch_validate_folder(folder, *, config_overrides=None) -> dict`

Runs batch validation on all SWC files in a folder.

#### `batch_split_folder(folder, *, config_overrides=None) -> dict`

Splits SWC files by soma roots.

#### `batch_auto_typing(folder, *, options=None, config_overrides=None) -> RuleBatchResult`

Runs rule-based auto-typing on a folder.

#### `batch_radii_cleaning(folder, *, config_overrides=None) -> dict`

Runs radii cleaning on all SWC files in a folder.

### Validation

#### `validation_run_text(swc_text, *, profile=None, config_overrides=None, feature_overrides=None) -> ValidationReport`

Runs structured validation on SWC text.

#### `validation_run_file(path, *, profile=None, config_overrides=None, feature_overrides=None) -> ValidationReport`

Runs structured validation on one SWC file.

#### `auto_fix_text(swc_text, *, config_overrides=None) -> dict`

Runs auto-fix and returns sanitized content plus validation report.

#### `auto_fix_file(path, *, out_path=None, write_output=False, config_overrides=None) -> dict`

Runs auto-fix for a file and optionally writes output.

### Visualization

#### `build_mesh_from_text(swc_text, *, config_overrides=None) -> dict`

Builds a reusable mesh payload from SWC text.

#### `build_mesh_from_file(path, *, config_overrides=None) -> dict`

Builds a reusable mesh payload from a file.

### Morphology Editing

#### `reassign_subtree_types(swc_text, *, node_id, new_type, config_overrides=None) -> dict`

Reassigns all nodes in a selected subtree to a new SWC type.

#### `reassign_subtree_types_in_file(path, *, node_id, new_type, out_path=None, write_output=False, config_overrides=None) -> dict`

File wrapper for subtree type reassignment with optional write.

### Atlas Registration (Placeholder)

#### `register_to_atlas(path, *, atlas_name=None, config_overrides=None) -> FeatureResult`

Returns a structured placeholder response (`ok=False`) until implemented.

### Analysis

#### `analysis_summary_file(path, *, config_overrides=None) -> dict`

Computes basic summary metrics for one SWC file.

### Plugins

#### `register_method(feature_key, method_name, func) -> None`

Registers plugin method override for feature dispatch.

#### `unregister_method(feature_key, method_name) -> None`

Removes plugin method override.

#### `list_feature_methods(feature_key) -> dict`

Lists plugin and builtin methods for one feature key.

#### `list_all_feature_methods() -> dict`

Lists methods for all feature keys.

## Validation Backend Models

Structured validation output uses these core models:

- `PreCheckItem`
- `CheckResult`
- `ValidationReport`

They are available from:

```python
from swctools.tools.validation import PreCheckItem, CheckResult, ValidationReport
```

`ValidationReport.to_dict()` includes:

- `profile`
- `precheck`
- `results`
- `summary` (`pass`, `warning`, `fail`, `error`, `total`)

## Plugin Extension Pattern

Feature methods resolve by priority:

1. plugin method
2. builtin method

Example:

```python
from swctools.plugins import register_method

def my_auto_typing(folder, options, config):
    # custom implementation
    return ...

register_method("batch_processing.auto_typing", "default", my_auto_typing)
```

## Config Files

Default configs are stored per feature:

- `swctools/tools/<tool>/configs/<feature>.json`

Validation profiles:

- `swctools/tools/validation/configs/default.json`
- `swctools/tools/validation/configs/strict.json`
- `swctools/tools/validation/configs/tolerant.json`

Use runtime overrides with `config_overrides` (API) or `--config-json` (CLI).

## Minimal Usage Example

```python
import swctools.api as swc

# Validation
report = swc.validation_run_file("data/example.swc", profile="default")
print(report.summary())

# Batch auto-typing
opts = swc.RuleBatchOptions(soma=True, axon=True, basal=True)
result = swc.batch_auto_typing("data/folder", options=opts)
print(result.files_processed, result.total_type_changes)
```

