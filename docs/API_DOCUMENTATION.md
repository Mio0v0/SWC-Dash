# API Documentation (Library Reference)

This document covers Python-callable APIs in SWC-Studio (Python package: `swctools`).

## How To Read This Doc

- Use `swctools.api` for most integrations.
- Use feature modules directly only when you need tool-specific behavior.
- Use plugin registry APIs to override method implementations.

## API Layers

1. Stable convenience API: `swctools.api`
2. Tool feature modules: `swctools.tools.<tool>.features.<feature>`
3. Plugin registry: `swctools.plugins`

## Recommended Import

```python
import swctools.api as swc
```

## 1) Public Convenience API (`swctools.api`)

Source: `swctools/api.py`

### Data Type

#### `RuleBatchOptions`

Dataclass for auto-typing:

- `soma: bool`
- `axon: bool`
- `apic: bool`
- `basal: bool`
- `rad: bool`
- `zip_output: bool`

### Batch Processing

#### `batch_validate_folder(folder, *, config_overrides=None) -> dict`

Runs validation over all SWC files in a folder.

#### `batch_split_folder(folder, *, config_overrides=None) -> dict`

Splits each SWC by soma-root trees.

#### `batch_auto_typing(folder, *, options=None, config_overrides=None) -> RuleBatchResult`

Runs rule-based auto typing over a folder.

#### `batch_radii_cleaning(folder, *, config_overrides=None) -> dict`

Runs radii cleaning on a folder.

#### `radii_clean_path(path, *, config_overrides=None) -> dict`

Runs radii cleaning on either one file or one folder.

### Validation

#### `validation_run_text(swc_text, *, config_overrides=None, feature_overrides=None) -> ValidationReport`

Runs structured validation from SWC text.

#### `validation_run_file(path, *, config_overrides=None, feature_overrides=None) -> ValidationReport`

Runs structured validation from file path.

#### `auto_fix_text(swc_text, *, config_overrides=None) -> dict`

Runs sanitize + validation from text.

#### `auto_fix_file(path, *, out_path=None, write_output=False, config_overrides=None) -> dict`

Runs sanitize + validation from file, optionally writes output.

#### `validation_auto_typing_file(file_path, *, options=None, config_overrides=None, output_path=None, write_output=True, write_log=True)`

Runs the shared auto-typing logic on a single file.

### Visualization

#### `build_mesh_from_text(swc_text, *, config_overrides=None) -> dict`

Builds mesh summary payload from SWC text.

#### `build_mesh_from_file(path, *, config_overrides=None) -> dict`

Builds mesh summary payload from file.

### Morphology Editing

#### `reassign_subtree_types(swc_text, *, node_id, new_type, config_overrides=None) -> dict`

Reassigns subtree node types in memory.

#### `reassign_subtree_types_in_file(path, *, node_id, new_type, out_path=None, write_output=False, config_overrides=None) -> dict`

File wrapper for subtree type reassignment.

#### `morphology_smart_decimation_text(swc_text, *, config_overrides=None) -> dict`

Runs graph-aware RDP simplification from text.

#### `morphology_smart_decimation_file(path, *, out_path=None, write_output=False, config_overrides=None) -> dict`

File wrapper for smart decimation. Produces simplification log file.

### Atlas Registration (placeholder)

#### `register_to_atlas(path, *, atlas_name=None, config_overrides=None) -> FeatureResult`

Structured placeholder response for atlas registration.

### Analysis

#### `analysis_summary_file(path, *, config_overrides=None) -> dict`

Basic morphology statistics.

### Plugins

#### `load_plugin_module(module_name, *, force_reload=False) -> dict`

Load one plugin module (manifest validation + method registration).
Note: registrations are process-local; use `autoload_plugins_from_environment`
for automatic loading in new CLI processes.

#### `load_plugins(modules) -> list[dict]`

Load multiple plugin modules and return per-module status.

#### `autoload_plugins_from_environment(env_var="SWCTOOLS_PLUGINS") -> list[dict]`

Load comma-separated plugin module list from environment variable.

#### `register_plugin_manifest(manifest) -> PluginManifest`

Register validated plugin manifest metadata.

#### `register_plugin_method(plugin_id, feature_key, method_name, func) -> None`

Register a plugin-owned method for one feature/method key.

#### `unregister_plugin(plugin_id) -> None`

Remove plugin manifest and methods owned by that plugin.

#### `register_method(feature_key, method_name, func) -> None`

Register plugin override method.

#### `unregister_method(feature_key, method_name) -> None`

Remove plugin override method.

#### `list_feature_methods(feature_key) -> dict`

List methods for a single feature key.

#### `list_all_feature_methods() -> dict`

List methods for all feature keys.

#### `list_plugins() -> list[dict]`

List loaded/registered plugin manifests.

#### `get_plugin(plugin_id) -> dict | None`

Get one plugin manifest by plugin id.

## 2) Tool Feature Module Entry Points

These are callable if you want per-feature direct imports.

## Batch Processing

- `swctools.tools.batch_processing.features.batch_validation`
  - `validate_swc_text(...)`
  - `validate_folder(...)`
- `swctools.tools.batch_processing.features.swc_splitter`
  - `split_swc_text(...)`
  - `split_folder(...)`
- `swctools.tools.batch_processing.features.auto_typing`
  - `run_folder(...)`
  - `options_to_dict(...)`
- `swctools.tools.batch_processing.features.radii_cleaning`
  - `clean_swc_text(...)`
  - `clean_file(...)`
  - `clean_folder(...)`
  - `clean_path(...)`

## Validation

- `swctools.tools.validation.features.run_checks`
  - `validate_text(...)`
  - `validate_file(...)`
- `swctools.tools.validation.features.auto_fix`
  - `auto_fix_text(...)`
  - `auto_fix_file(...)`
- `swctools.tools.validation.features.auto_typing`
  - `run_file(...)`
- `swctools.tools.validation.features.radii_cleaning`
  - `clean_path(...)`
  - `clean_file(...)`
  - `clean_folder(...)`

## Visualization

- `swctools.tools.visualization.features.mesh_editing`
  - `build_mesh_from_text(...)`
  - `build_mesh_from_file(...)`

## Morphology Editing

- `swctools.tools.morphology_editing.features.dendrogram_editing`
  - `reassign_subtree_types(...)`
  - `reassign_subtree_types_in_file(...)`
- `swctools.tools.morphology_editing.features.simplification`
  - `simplify_dataframe(...)`
  - `simplify_swc_text(...)`
  - `simplify_file(...)`

## Atlas Registration (placeholder)

- `swctools.tools.atlas_registration.features.registration`
  - `register_to_atlas(...)`

## Analysis

- `swctools.tools.analysis.features.summary`
  - `analyze_text(...)`
  - `analyze_file(...)`

## 3) Validation Engine and Models

Module: `swctools.tools.validation`

Exposed helpers:

- `load_validation_config(...)`
- `build_precheck_summary(...)`
- `run_validation_text(...)`
- `register_check(...)`
- `register_plugin_check(...)`
- `get_check(...)`
- `list_checks(...)`

Result models:

- `PreCheckItem`
- `CheckResult`
- `ValidationReport`

## Smart Decimation (How It Works)

1. Build directed graph from SWC `id`/`parent`.
2. Protect structural nodes (roots, optional tips, optional bifurcations).
3. Extract anchor-to-anchor linear paths.
4. Apply RDP with `thresholds.epsilon`.
5. Protect radius-sensitive nodes where:
   - `abs(node_radius - path_mean_radius) / path_mean_radius > thresholds.radius_tolerance`
6. Rewire kept nodes to nearest kept ancestor.

Key parameters (`simplification.json`):

- `thresholds.epsilon`
- `thresholds.radius_tolerance`
- `flags.keep_tips`
- `flags.keep_bifurcations`
- `flags.keep_roots`

## Plugin Override Pattern

```python
from swctools.plugins import load_plugin_module, register_method

def my_method(*args, **kwargs):
    ...

register_method("batch_processing.auto_typing", "default", my_method)
load_plugin_module("my_lab_plugins.brainglobe_adapter")
```

## Plugin Contract (Versioned)

External plugin modules should expose:

1. `PLUGIN_MANIFEST` dictionary (or `get_plugin_manifest()`)
2. one of:
   - `register_plugin(registrar)` function
   - `PLUGIN_METHODS` mapping/list

Minimal `PLUGIN_MANIFEST`:

```python
PLUGIN_MANIFEST = {
    "plugin_id": "my_lab.brainglobe",
    "name": "BrainGlobe Adapter",
    "version": "0.1.0",
    "api_version": "1",
    "capabilities": ["atlas_registration", "region_annotation"],
}
```

Starter template in this repo:

- `examples/plugins/brainglobe_adapter_template.py`

## Config Files

Per-feature config path pattern:

- `swctools/tools/<tool>/configs/<feature>.json`

Examples:

- `swctools/tools/validation/configs/default.json`
- `swctools/tools/batch_processing/configs/radii_cleaning.json`
- `swctools/tools/morphology_editing/configs/simplification.json`
