# Tool Tutorials

This page walks through each top-level tool with practical workflows.

## Tool -> Feature map

## 1) Batch Processing

- Batch Validation (`batch validate`)
- SWC Splitter (`batch split`)
- Auto Typing (`batch auto-typing`)
- Radii Cleaning (`batch radii-clean`)

## 2) Validation

- Rule Guide (`validation rule-guide`)
- Run Checks (`validation run`)
- Auto Fix (`validation auto-fix`)
- Radii Cleaning (`validation radii-clean`)

## 3) Visualization

- Mesh Editing (`visualization mesh-editing`)

## 4) Morphology Editing

- Dendrogram Edit (`morphology dendrogram-edit`)
- Smart Decimation (`morphology smart-decimation`)

## 5) Atlas Registration

- Register (`atlas register`) placeholder, plugin-ready

## 6) Analysis

- Summary (`analysis summary`) basic metrics

---

## Batch Processing Tutorials

### A. Batch Validation on a folder

```bash
swctools batch validate ./data
```

What it does:

- runs full validation for each `.swc` in folder
- prints pre-check summary first
- prints per-file grouped results
- writes one batch report file

Show guide only:

```bash
swctools batch validate rule-guide
```

### B. Split all multi-tree SWC files

```bash
swctools batch split ./data
```

Output pattern (default):

- folder: `./data/data_split`
- files: `<original_stem>_tree1.swc`, `<original_stem>_tree2.swc`, ...
- report: `split_report.txt`

### C. Batch Auto Typing

```bash
swctools batch auto-typing ./data --soma --axon --basal
```

Behavior:

- prints auto-typing rule guide before run
- writes outputs into `<folder>/<folder>_auto_typing`
- optional zip if `--zip` enabled
- writes `auto_typing_report.txt`

### D. Batch Radii Cleaning

```bash
swctools batch radii-clean ./data --threshold-mode percentile --percentile-min 1 --percentile-max 99.5
```

or absolute mode:

```bash
swctools batch radii-clean ./data --threshold-mode absolute --abs-min 0.05 --abs-max 30
```

---

## Validation Tutorials

### A. Show Validation Rule Guide

```bash
swctools validation rule-guide
```

### B. Validate one SWC

```bash
swctools validation run ./data/single-soma.swc
```

### C. Auto-fix one SWC

```bash
swctools validation auto-fix ./data/single-soma.swc --write
```

Use custom output path:

```bash
swctools validation auto-fix ./data/single-soma.swc --write --out ./data/single-soma_fixed.swc
```

### D. Validation Radii Cleaning

```bash
swctools validation radii-clean ./data/single-soma.swc --preserve-soma-radii
```

---

## Visualization Tutorial

Build mesh payload summary from one SWC:

```bash
swctools visualization mesh-editing ./data/single-soma.swc --include-edges
```

Used mainly as backend payload for GUI rendering workflows.

---

## Morphology Editing Tutorials

### A. Dendrogram subtree reassignment

```bash
swctools morphology dendrogram-edit ./data/single-soma.swc --node-id 42 --new-type 3 --write
```

### B. Smart Decimation (RDP)

```bash
swctools morphology smart-decimation ./data/single-soma.swc --write
```

Prints rule guide before processing and writes simplification log.

---

## Atlas Registration and Analysis

### Atlas registration (plugin-ready)

```bash
swctools atlas register ./data/single-soma.swc --atlas allen_mouse_25um
```

Default implementation is placeholder. Register plugins to provide real atlas workflows.

### Analysis summary

```bash
swctools analysis summary ./data/single-soma.swc
```

---

## Config override pattern

Most commands support inline temporary overrides:

```bash
swctools validation run ./data/single-soma.swc \
  --config-json '{"checks":{"has_soma":{"enabled":true,"severity":"warning","params":{}}}}'
```

Persistent defaults live in:

- `swctools/tools/<tool>/configs/*.json`
