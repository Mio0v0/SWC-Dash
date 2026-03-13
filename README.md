# swctools

`swctools` is a modular Python package for SWC morphology workflows with a shared backend used by both CLI and GUI interfaces.

## First-Time Setup

If you just cloned this repo and want to run it locally:

```bash
git clone <your-repo-url>
cd SWC-STUDIO
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[gui]"
```

CLI-only install (no GUI dependencies):

```bash
pip install -e .
```

Verify install:

```bash
swctools --help
swctools-gui
```

If `swctools` or `swctools-gui` is not found, make sure the virtual environment is active:

```bash
source .venv/bin/activate
```

## Quick Start

Run CLI examples:

```bash
swctools batch split ./data
swctools batch validate rule-guide
swctools batch validate ./data
swctools batch auto-typing ./data --soma --axon --basal
swctools validation rule-guide
swctools validation run ./data/single-soma.swc
```

Run GUI:

```bash
swctools-gui
```

## Logs and Reports

`swctools` writes text reports from the shared backend for both CLI and GUI.
GUI also auto-opens generated reports in popup windows.

- Single-file validation (`swctools validation run` and GUI Validation):
  - `<input_stem>_validation_report.txt` (next to input SWC)
- Batch split:
  - `<input_folder>/<input_folder>_split/split_report.txt`
- Batch auto-typing:
  - `<input_folder>/<input_folder>_auto_typing/auto_typing_report.txt`
- Batch validation:
  - `<input_folder>/<input_folder>_batch_validation_report.txt`
- Morphology editing session (GUI dendrogram edits):
  - `<input_stem>_morphology_session_log.txt` (one log per session; written when switching file or closing app)

## Architecture

- `swctools/core`: shared algorithms, SWC I/O, and models.
- `swctools/tools`: tool/feature modules organized by domain.
- `swctools/plugins`: method registry for feature-level overrides.
- `swctools/cli`: terminal interface that calls tool/feature APIs.
- `swctools/gui`: Qt desktop interface that calls shared backend logic.

## Documentation

- [CLI Reference](docs/CLI_REFERENCE.md)
- [API / Library Documentation](docs/API_DOCUMENTATION.md)
- [Demo Plan](docs/DEMO_PLAN.md)

## Tool -> Feature map

- Batch Processing
  - `batch_validation`
  - `swc_splitter`
  - `auto_typing`
  - `radii_cleaning`
- Validation
  - `auto_fix`
- Visualization
  - `mesh_editing` (placeholder)
- Morphology Editing
  - `dendrogram_editing`
- Atlas Registration
  - `registration` (placeholder)
- Analysis
  - `summary` (placeholder)

Each feature stores defaults in `swctools/tools/<tool>/configs/<feature>.json`.

## CLI examples

```bash
swctools batch split /path/to/folder
# swctools batch split /Users/tuo/Desktop/SWC-Data
# swctools batch split ./data

swctools batch validate rule-guide
swctools batch validate /path/to/folder
# swctools batch validate ./data
# swctools batch validate ./data --config-json '{"checks":{"no_back_tracking":{"enabled":true}}}'

swctools batch auto-typing /path/to/folder --soma --axon --basal
# swctools batch auto-typing ./data --soma --axon --basal
swctools validation auto-fix /path/to/file.swc --write
swctools validation rule-guide
swctools validation run /path/to/file.swc
# swctools validation rule-guide
swctools validation run ./data/single-soma.swc
swctools morphology dendrogram-edit /path/to/file.swc --node-id 42 --new-type 3 --write
swctools plugins list
```

## Plugin override example

```python
from swctools.plugins import register_method


def my_auto_typing(folder, options, config):
    # custom implementation
    ...

register_method("batch_processing.auto_typing", "default", my_auto_typing)
```

## Notes

- JSON config controls parameters and method selection.
- Algorithms remain in Python modules, not JSON.
- No web/server layer is included.
