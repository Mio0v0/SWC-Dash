# SWC-Dash Session Changes Summary

## New Features

### Per-Tree SWC Split and Download (`validation_core.py`, `callbacks.py`, `layout.py`)
When a multi-soma SWC file is loaded on the Format Validation tab, each tree can now be saved individually. A small **↓** button appears in each `Tree X` column header of the validation table. Clicking it opens a native macOS folder-picker dialog and saves that tree's nodes as `<name>_treeN.swc` to the chosen folder. Result is shown in a status line below the table.

### Save All Trees to Folder (`callbacks.py`, `layout.py`)
A **"Save all trees to folder…"** button (next to "Download validation JSON") opens a folder picker and saves all split trees into a named sub-folder `<name>/` inside the chosen directory. Each tree is saved as `<name>_tree1.swc`, `<name>_tree2.swc`, etc.

### Batch Folder Split (`callbacks.py`, `layout.py`)
A **"📂 Select folder to split…"** button in a new "Batch folder split" panel opens a folder picker. Every multi-tree SWC file found in the selected folder is automatically split: each tree is saved as `<stem>_treeN.swc` inside a sub-folder `<stem>/` created next to the original file. Single-tree files are skipped. A full status report is shown in the UI after completion.

---

## UI Improvements

### Multi-Soma Alert Banner (`callbacks.py`)
A bold **red alert banner** now appears above the validation table whenever a file with multiple somas (trees) is loaded. It lists the number of trees found and their root IDs, and reminds the user to use the split/download buttons.

---

## Implementation Notes

- The tree-splitting logic reuses the existing `_split_swc_by_trees()` function already present in `validation_core.py`.
- All folder-picker dialogs use `osascript` (macOS AppleScript) called server-side, which opens a native Finder dialog on the host machine.
- `dcc.Store(id="store-validate-swc-text")` (already declared in the layout) is now populated on file upload so the split callbacks can access the raw SWC text without re-parsing.
