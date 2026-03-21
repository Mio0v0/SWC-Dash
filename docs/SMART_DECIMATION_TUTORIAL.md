# Smart Decimation Tutorial (RDP)

Smart Decimation simplifies SWC geometry while preserving important morphology structure.

OS note: replace `./data/...` with `.\data\...` on Windows. If `swctools` is not on PATH, use module mode (`python -m swctools.cli.cli ...` on macOS/Linux, `py -m swctools.cli.cli ...` on Windows).

Implementation:

- `swctools.tools.morphology_editing.features.simplification`

Algorithm type:

- graph-aware Ramer-Douglas-Peucker (RDP)

## Core logic

1. Build directed graph from `id` and `parent`.
2. Identify protected nodes:
   - roots/soma roots
   - tips (if `keep_tips`)
   - bifurcations (if `keep_bifurcations`)
3. Split the tree into anchor-to-anchor linear paths.
4. Apply RDP on path interior points using `epsilon`.
5. Protect radius-sensitive points where radius deviates from path mean by more than `radius_tolerance`.
6. Keep protected + selected points, then rewire parent links to nearest kept ancestor.

## Config file

- `swctools/tools/morphology_editing/configs/simplification.json`

Key parameters:

- `thresholds.epsilon`
- `thresholds.radius_tolerance`
- `flags.keep_tips`
- `flags.keep_bifurcations`
- `flags.keep_roots`

## Parameter guidance

- lower `epsilon`: keeps more geometry detail (less reduction)
- higher `epsilon`: stronger simplification
- lower `radius_tolerance`: protects more radius-outlier nodes
- higher `radius_tolerance`: allows stronger point removal around radius variation

## CLI usage

Preview (no file write):

```bash
swctools morphology smart-decimation ./data/single-soma.swc
```

Write simplified file:

```bash
swctools morphology smart-decimation ./data/single-soma.swc --write
```

Custom output path:

```bash
swctools morphology smart-decimation ./data/single-soma.swc --write --out ./data/single-soma_decimated.swc
```

Temporary overrides:

```bash
swctools morphology smart-decimation ./data/single-soma.swc --write --config-json '{"thresholds":{"epsilon":1.2,"radius_tolerance":0.35},"flags":{"keep_tips":true,"keep_bifurcations":true}}'
```

## Outputs and logs

- output SWC (when `--write`)
- simplification log: `<stem>_simplification_log.txt`

Log includes:

- original node count
- new node count
- reduction percent
- parameters used
- protected node statistics
- removed node IDs

## GUI workflow

In Morphology Editing -> Simplification:

1. Open source SWC.
2. Adjust config (JSON editor/rule guide as enabled).
3. Process to preview simplified result.
4. Use Apply/Redo/Cancel controls.
5. Save final file and review simplification log.
