# GUI Workflow Guide

This page explains how the GUI is structured and how panels interact.

## Layout model

The GUI uses a studio-style layout with shared backend logic:

- top in-app bar: file/edit/view/window/help menus + tool and feature selectors
- center canvas: active SWC visualization/editor tabs
- left panel: Data Explorer (SWC tree, node info, segment info, edit log)
- right panel: Control Center (tool-dependent controls)
- bottom panel: log output/events

The center canvas is the dominant workspace and updates based on current tool/feature.

## Tool and feature switching

Top-level tools:

- Batch Processing
- Validation
- Visualization
- Morphology Editing
- Atlas Registration
- Analysis

When a tool is active, feature buttons are shown for that tool.
The selected feature controls what appears in Control Center.

### Feature mapping in GUI

- Batch Processing: Split, Validation, Auto Label, Radii Cleaning
- Validation: Validation, Auto Label, Radii Cleaning
- Visualization: View Controls
- Morphology Editing: Label Editing, Simplification
- Atlas Registration: Registration (placeholder)
- Analysis: Summary (placeholder)

## Document/canvas behavior

- Multiple SWC files can be open in separate canvas tabs.
- Preview outputs (for some operations) open as temporary comparison tabs.
- Closing a changed tab triggers save/discard flow and session logging.

## Recommended GUI usage sequence

1. **Open file** from menu.
2. Inspect structure in Data Explorer.
3. Switch to **Validation** tool and run checks.
4. Use **Morphology Editing** and **Radii Cleaning** as needed.
5. Review status/log panel and save outputs.

## Validation in GUI

Validation panel supports:

- Rule Guide button (manual display; no forced popup)
- Run Validation
- results table with status/label
- report export controls

Validation uses the same backend as CLI (`swctools.tools.validation` + core engine).

## Auto Typing in GUI

- Batch mode: folder-level auto-labeling
- Validation mode: single-file auto-labeling
- JSON editor for rule parameters
- shared backend logic (same as CLI/API)

## Radii Cleaning in GUI

- shared backend for Batch + Validation + CLI
- supports file/folder usage depending on panel context
- thresholds configurable via JSON
- histogram/statistics visualization for currently loaded file

## Simplification in GUI

- RDP-based Smart Decimation controls in Morphology Editing
- process creates a simplified preview/result path
- action bar supports apply/redo/cancel behavior
- outputs include simplification log with node reduction stats and parameters

## Logs and status

- transient status: status bar and bottom log
- persistent logs: text report files written by backend reporting layer

See [LOGS_AND_REPORTS](LOGS_AND_REPORTS.md) for full report naming conventions.
