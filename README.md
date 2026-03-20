# SWC-Studio

`SWC-Studio` is a modular SWC morphology toolkit (Python package: `swctools`) with:

- a shared Python backend (`swctools/core` + `swctools/tools`)
- a CLI (`swctools`)
- a desktop GUI (`swctools-gui`)

CLI and GUI call the same feature backend functions.

## What This Project Does

Top-level tool areas:

1. Batch Processing
2. Validation
3. Visualization
4. Morphology Editing
5. Atlas Registration (placeholder)
6. Analysis (placeholder)

Core workflows currently include:

- SWC split by soma-root trees
- Rule-based auto typing
- Single-file and batch validation
- Radius outlier cleaning
- Dendrogram subtree type reassignment
- Smart Decimation (RDP-based simplification)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[gui]"
```

CLI-only install:

```bash
pip install -e .
```

## Run

CLI:

```bash
swctools --help
```

GUI:

```bash
swctools-gui
# or
python -m swctools.gui.main
```

## Quick CLI Examples

```bash
swctools batch split ./data
swctools batch validate ./data
swctools batch auto-typing ./data --soma --axon --basal
swctools batch radii-clean ./data

swctools validation rule-guide
swctools validation run ./data/single-soma.swc
swctools validation auto-fix ./data/single-soma.swc --write

swctools morphology smart-decimation ./data/single-soma.swc --write

swctools plugins load my_lab_plugins.brainglobe_adapter
swctools plugins list-loaded
```

## Documentation

Short docs (Markdown):

- [CLI Reference](docs/CLI_REFERENCE.md): command reference and options
- [API / Library Documentation](docs/API_DOCUMENTATION.md): Python API surface
- [Plugin Demonstration](docs/PLUGIN_DEMONSTRATION.md): lab handoff plugin workflow

Comprehensive docs site (Sphinx source):

- Live docs: `https://mio0v0.github.io/SWC-Studio/`
- includes tutorials, architecture, logs/reporting, plugin development, and auto-generated API/module references


## Architecture (High-Level)

- `swctools/core`: shared data models, IO, validation/rules, reporting
- `swctools/tools`: tool/feature backends (actual behavior)
- `swctools/plugins`: registry for builtin + user override methods
- `swctools/cli`: terminal interface layer
- `swctools/gui`: Qt interface layer

## Config

Feature config JSON lives at:

- `swctools/tools/<tool>/configs/<feature>.json`

Examples:

- `swctools/tools/validation/configs/default.json`
- `swctools/tools/batch_processing/configs/radii_cleaning.json`
- `swctools/tools/morphology_editing/configs/simplification.json`

## Notes

- JSON controls parameters and method selection.
- Algorithm/data transformation logic stays in Python.
- No web/server/API service layer is included.

## License

This project is released under the MIT License. See `LICENSE`.

## Plugin Contract (For Many External Plugins)

`swctools` supports plugin modules through a small versioned contract:

1. `PLUGIN_MANIFEST` (or `get_plugin_manifest()`) must provide:
   - `plugin_id`, `name`, `version`, `api_version`
   - optional `description`, `author`, `capabilities`
2. Plugin module must provide either:
   - `register_plugin(registrar)` function, or
   - `PLUGIN_METHODS` dictionary/list
3. Plugin methods register against existing feature keys, e.g.:
   - `batch_processing.auto_typing`
   - `atlas_registration.registration`

This lets you integrate external libraries (like BrainGlobe adapters) without
rewriting their internal algorithms.

For automatic plugin loading in CLI sessions:

```bash
export SWCTOOLS_PLUGINS="my_lab_plugins.brainglobe_adapter,my_lab_plugins.custom_methods"
```

Starter template:

- `examples/plugins/brainglobe_adapter_template.py`
