# swctools

`swctools` is a modular Python package for SWC morphology workflows with a shared backend used by both CLI and GUI interfaces.

## First-Time Setup

If you just cloned this repo and want to run it locally:

```bash
git clone <your-repo-url>
cd SWC-Studio
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

If `swctools` or `swctools-gui` is not found, make sure your virtual environment is active:

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
swctools batch radii-clean ./data
swctools validation rule-guide
swctools validation run ./data/single-soma.swc
swctools validation radii-clean ./data/single-soma.swc
swctools morphology smart-decimation ./data/single-soma.swc --write
```

Run GUI:

```bash
swctools-gui
# or
python -m swctools.gui.main
```

## Tool -> Feature Map (Current Structure)

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

- `split`, `validation`, and `core` are internal/helper feature modules.
- User-facing CLI commands are listed in [CLI Reference](docs/CLI_REFERENCE.md).

## GUI Notes (Current Behavior)

- Control Center tabs are feature-specific.
- In `Morphology Editing`:
  - `Label Editing` + `Simplification` tabs appear only when an SWC file is open.
  - If no SWC file is open, morphology controls remain empty.
- Smart Decimation workflow:
  - `Process` opens a temporary canvas tab named `Simplified View`.
  - `Apply` saves a new SWC and replaces the active working buffer.
  - `Redo` recomputes simplification.
  - `Cancel` discards temporary simplified data.
- Multiple SWC files can be opened as separate canvas tabs.

## Logs and Reports

`swctools` writes text reports from the shared backend for both CLI and GUI.

- Single-file validation:
  - `<input_stem>_validation_report.txt`
- Batch split:
  - `<input_folder>/<input_folder>_split/split_report.txt`
- Batch auto-typing:
  - `<input_folder>/<input_folder>_auto_typing/auto_typing_report.txt`
- Batch validation:
  - `<input_folder>/<input_folder>_batch_validation_report.txt`
- Radii cleaning (file mode):
  - `<input_stem>_radii_cleaned.swc`
  - `<input_stem>_radii_cleaned_radii_cleaning_report.txt`
- Radii cleaning (folder mode):
  - `<input_folder>/<input_folder>_radii_cleaned/...`
  - `<input_folder>/<input_folder>_radii_cleaned/radii_cleaning_report.txt`
- Morphology editing session log (GUI dendrogram edits):
  - `<input_stem>_morphology_session_log.txt`
- Smart decimation:
  - `<input_stem>_simplification_log.txt`

GUI popup behavior:

- Batch split and batch auto-typing can open a report popup.
- Validation, radii cleaning, and simplification write logs/reports without forcing a popup.

## Architecture

- `swctools/core`: shared algorithms, SWC I/O, models, report formatting.
- `swctools/tools`: tool/feature modules organized by domain.
- `swctools/plugins`: method registry for feature-level overrides.
- `swctools/cli`: terminal interface layer.
- `swctools/gui`: Qt desktop interface layer.

Feature defaults are stored in:

- `swctools/tools/<tool>/configs/<feature>.json`

Shared radii-clean thresholds/config:

- `swctools/tools/batch_processing/configs/radii_cleaning.json`

## CLI Examples

```bash
swctools batch split /path/to/folder
swctools batch validate rule-guide
swctools batch validate /path/to/folder
swctools batch auto-typing /path/to/folder --soma --axon --basal
swctools batch radii-clean /path/to/file_or_folder

swctools validation rule-guide
swctools validation run /path/to/file.swc
swctools validation auto-fix /path/to/file.swc --write
swctools validation radii-clean /path/to/file_or_folder

swctools morphology dendrogram-edit /path/to/file.swc --node-id 42 --new-type 3 --write
swctools morphology smart-decimation /path/to/file.swc --write

swctools plugins list
```

## Documentation

- [CLI Reference](docs/CLI_REFERENCE.md)
- [API / Library Documentation](docs/API_DOCUMENTATION.md)
- [Demo Plan](docs/DEMO_PLAN.md)

## Plugin Override Example

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
