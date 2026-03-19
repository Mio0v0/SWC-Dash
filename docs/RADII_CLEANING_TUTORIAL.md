# Radii Cleaning Tutorial

Radii Cleaning fixes abnormal radius values while preserving branch continuity.

Shared backend module:

- `swctools.tools.batch_processing.features.radii_cleaning`

Validation tool reuses the same implementation through:

- `swctools.tools.validation.features.radii_cleaning`

So GUI Batch tab, GUI Validation tab, and CLI all use one core behavior.

## What counts as abnormal

Depending on config:

- non-positive radii (`<= 0`)
- non-finite radii (`NaN`, `inf`)
- out-of-bound radii by percentile or absolute thresholds
- local spikes/dips relative to neighboring branch trend

## Replacement strategy (high level)

1. detect abnormal node radii
2. find suitable neighboring normal radii along graph connectivity
3. replace with smoothed/averaged value from parent/children context
4. repeat over configured iterations until stable or iteration limit reached

## Key config file

- `swctools/tools/batch_processing/configs/radii_cleaning.json`

Validation radii cleaning mirrors this config in:

- `swctools/tools/validation/configs/radii_cleaning.json`

## Important config fields

Under `rules`:

- `preserve_soma`: keep soma radius unchanged when `true`
- `small_radius_zero_only`: only treat very small values as abnormal if they are zero
- `threshold_mode`: `percentile` or `absolute`
- `global_percentile_bounds`: min/max percentiles
- `global_absolute_bounds`: min/max absolute radii
- `type_thresholds`: per-type overrides (types 2,3,4 by default)
- `detect_spikes`, `detect_dips`
- `spike_ratio_threshold`, `dip_ratio_threshold`
- `iterations`
- `max_descendant_search_depth`
- `replacement.clamp_min`, `replacement.clamp_max`

## CLI examples

Clean one file with percentile mode:

```bash
swctools batch radii-clean ./data/single-soma.swc \
  --threshold-mode percentile --percentile-min 1 --percentile-max 99.5
```

Clean one file with absolute mode:

```bash
swctools batch radii-clean ./data/single-soma.swc \
  --threshold-mode absolute --abs-min 0.05 --abs-max 30
```

Allow soma radii to be modified:

```bash
swctools batch radii-clean ./data/single-soma.swc --fix-soma-radii
```

Force soma preservation:

```bash
swctools batch radii-clean ./data/single-soma.swc --preserve-soma-radii
```

Validation command path (same backend):

```bash
swctools validation radii-clean ./data/single-soma.swc --preserve-soma-radii
```

## Outputs and logs

- file mode: writes `<stem>_radii_cleaned.swc` and `<stem>_radii_cleaning_report.txt`
- folder mode: writes `<folder>/<folder>_radii_cleaned/` plus folder report

Report includes:

- change counts
- per-file summary (folder mode)
- node-level change lines with old/new radii and reasons
- config used

## GUI notes

- GUI panels expose JSON editor to adjust thresholds
- histogram/statistics view helps choose thresholds
- run-on-loaded-file and run-on-folder are available in appropriate panels
