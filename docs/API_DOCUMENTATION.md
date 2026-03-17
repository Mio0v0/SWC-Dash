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

## Tool -> Feature Map (Current Structure)

Python backend modules under `swctools/tools/*/features`:

- `batch_processing`
  - `auto_typing`
  - `batch_validation`
  - `radii_cleaning`
  - `split`
  - `swc_splitter`
  - `validation`
- `validation`
  - `auto_fix`
  - `core`
  - `radii_cleaning`
  - `run_checks`
- `visualization`
  - `mesh_editing`
- `morphology_editing`
  - `dendrogram_editing`
  - `simplification`
- `atlas_registration`
  - `registration`
- `analysis`
  - `summary`

Notes:

- `split`, `validation`, and `core` are helper/internal feature modules.
- Public library calls should use `swctools.api` exports.

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

#### `radii_clean_path(path, *, config_overrides=None) -> dict`

Runs shared radii cleaning on either one SWC file or a folder.

### Validation

Validation radii cleaning uses the same backend implementation as batch radii cleaning.

#### `validation_run_text(swc_text, *, config_overrides=None, feature_overrides=None) -> ValidationReport`

Runs structured validation on SWC text.

#### `validation_run_file(path, *, config_overrides=None, feature_overrides=None) -> ValidationReport`

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

#### `morphology_smart_decimation_text(swc_text, *, config_overrides=None) -> dict`

Runs graph-aware RDP simplification and returns simplified SWC bytes plus stats.

How the backend works:

1. Build directed morphology graph from SWC topology.
2. Protect structural nodes (roots, optional tips, optional bifurcations).
3. Extract anchor-to-anchor linear paths.
4. Apply RDP with `thresholds.epsilon`.
5. Protect radius-sensitive nodes using `thresholds.radius_tolerance`.
6. Rewire parent links to nearest kept ancestors for a valid simplified tree.

Important config parameters:

- `thresholds.epsilon`: higher means stronger simplification.
- `thresholds.radius_tolerance`: lower means stricter radius-preservation.
- `flags.keep_tips`: keep terminal points.
- `flags.keep_bifurcations`: keep branch points.
- `flags.keep_roots`: keep root points.

Common output fields include:

- `dataframe`
- `bytes`
- `original_node_count`
- `new_node_count`
- `reduction_percent`
- `kept_node_ids`
- `removed_node_ids`
- `params_used`
- `protected_counts`

#### `morphology_smart_decimation_file(path, *, out_path=None, write_output=False, config_overrides=None) -> dict`

File wrapper for Smart Decimation.

- if `write_output=True`, writes simplified SWC
- always writes a simplification text log
- returns `log_path`, `input_path`, and optional `output_path`

### Atlas Registration (Placeholder)

#### `register_to_atlas(path, *, atlas_name=None, config_overrides=None) -> FeatureResult`

Returns a structured placeholder response (`ok=False`) until implemented.

### Analysis (Placeholder)

#### `analysis_summary_file(path, *, config_overrides=None) -> dict`

Returns basic morphology summary information.

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

- `precheck`
- `results`
- `summary` (`pass`, `warning`, `fail`, `total`)

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

Validation config:

- `swctools/tools/validation/configs/default.json`

Smart Decimation config:

- `swctools/tools/morphology_editing/configs/simplification.json`

Shared radii-clean config:

- `swctools/tools/batch_processing/configs/radii_cleaning.json`

Use runtime overrides with `config_overrides` (API) or `--config-json` (CLI).

## Minimal Usage Example

```python
import swctools.api as swc

# Validation
report = swc.validation_run_file("data/example.swc")
print(report.summary())

# Smart decimation
result = swc.morphology_smart_decimation_file("data/example.swc", write_output=True)
print(result["original_node_count"], result["new_node_count"], result["log_path"])
```
