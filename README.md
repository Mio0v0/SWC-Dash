# swctools

`swctools` is a modular Python package for SWC morphology workflows with a shared backend used by both CLI and GUI interfaces.

## Architecture

- `swctools/core`: shared algorithms, SWC I/O, and models.
- `swctools/tools`: tool/feature modules organized by domain.
- `swctools/plugins`: method registry for feature-level overrides.
- `swctools/cli`: terminal interface that calls tool/feature APIs.
- `swctools/gui`: Qt desktop interface that calls shared backend logic.

## Tool -> Feature map

- Batch Processing
  - `batch_validation`
  - `swc_splitter`
  - `auto_typing`
  - `radii_cleaning`
- Validation
  - `auto_fix`
- Visualization
  - `mesh_editing`
- Morphology Editing
  - `dendrogram_editing`
- Atlas Registration
  - `registration` (placeholder)
- Analysis
  - `summary` (starter implementation)

Each feature stores defaults in `swctools/tools/<tool>/configs/<feature>.json`.

## CLI examples

```bash
swctools batch split /path/to/folder
swctools batch auto-typing /path/to/folder --soma --axon --basal
swctools validation auto-fix /path/to/file.swc --write
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
