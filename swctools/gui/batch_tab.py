"""Batch processing controls for split, auto-labeling, and radii cleaning."""

import os
import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.auto_typing import (
    RuleBatchOptions,
    get_auto_rules_config,
    save_auto_rules_config,
)
from swctools.core.config import feature_config_path
from swctools.tools.batch_processing.features.auto_typing import run_folder as run_auto_typing
from swctools.tools.batch_processing.features.swc_splitter import split_folder

_CFG_PATH = feature_config_path("batch_processing", "auto_typing")


class BatchTabWidget(QWidget):
    """Owns three batch control pages used directly by Control Center tabs."""

    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status_boxes: list[QPlainTextEdit] = []
        self._split_page = self._build_split_page()
        self._auto_page = self._build_auto_page()
        self._radii_page = self._build_radii_page()

        # This root widget is not shown directly; pages are used in main window tabs.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(QLabel("Batch controls are shown as tabs in Control Center."))

    # --------------------------------------------------------- Public page access
    def split_tab_widget(self) -> QWidget:
        return self._split_page

    def auto_tab_widget(self) -> QWidget:
        return self._auto_page

    def radii_tab_widget(self) -> QWidget:
        return self._radii_page

    # --------------------------------------------------------- UI builders
    def _build_split_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        desc = QLabel(
            "Select a folder and split each multi-cell SWC into separate trees.\n"
            "Output naming: <original>/<original>_tree1.swc, _tree2.swc, ..."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        layout.addWidget(desc)

        self._btn_split_folder = QPushButton("Select Folder and Process Split…")
        self._btn_split_folder.clicked.connect(self._on_split_folder)
        layout.addWidget(self._btn_split_folder)

        self._split_status = self._new_status_box()
        layout.addWidget(self._split_status, stretch=1)
        return page

    def _build_auto_page(self) -> QWidget:
        # Build a two-column layout: controls on the left, a rules/decision panel on the right
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left column: controls
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        desc = QLabel("Auto labeling with morphology rules for all SWC files in a selected folder.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        layout.addWidget(desc)

        flags_row1 = QHBoxLayout()
        flags_row2 = QHBoxLayout()
        self._flag_soma = QCheckBox("--soma")
        self._flag_axon = QCheckBox("--axon")
        self._flag_dend = QCheckBox("--dendrite")
        self._flag_apic = QCheckBox("--apic")
        self._flag_basal = QCheckBox("--basal")
        self._flag_rad = QCheckBox("--rad")
        self._flag_zip = QCheckBox("--zip")

        self._flag_soma.setChecked(True)
        self._flag_axon.setChecked(True)
        self._flag_basal.setChecked(True)

        for cb in (self._flag_soma, self._flag_axon, self._flag_dend, self._flag_apic):
            flags_row1.addWidget(cb)
        flags_row1.addStretch()
        for cb in (self._flag_basal, self._flag_rad, self._flag_zip):
            flags_row2.addWidget(cb)
        flags_row2.addStretch()
        layout.addLayout(flags_row1)
        layout.addLayout(flags_row2)

        self._btn_run_batch_check = QPushButton("Run Auto Labeling on Folder…")
        self._btn_run_batch_check.clicked.connect(self._on_run_batch_check)
        layout.addWidget(self._btn_run_batch_check)

        self._auto_status = self._new_status_box()
        layout.addWidget(self._auto_status, stretch=1)

        # Right column: Rules / decision panel (collapsible)
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(6)

        title = QLabel("Decision engine — auto-label rules")
        title.setStyleSheet("font-weight: 700; font-size: 13px;")
        r_layout.addWidget(title)

        hint = QLabel(
            "This panel shows the JSON configuration that controls the auto-labeling algorithm.\n"
            "You can edit thresholds and weights and save to change behavior.\n\n"
            "Decision summary:\n"
            "1) Partition branches anchored at soma/roots.\n"
            "2) Compute branch features (path length, radial extent, mean radius, branchiness, z-mean).\n"
            "3) Score each branch for axon/apical/basal using weighted features + prior from existing labels.\n"
            "4) Optionally refine scores via a nearest-centroid (ML) step seeded by confident branches.\n"
            "5) Assign branch-level classes, smooth locally among siblings, then propagate labels to nodes using neighborhood votes.\n"
            "6) Radius rule: copy parent radius into zero/invalid radii when enabled.\n"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 12px; color: #444;")
        r_layout.addWidget(hint)

        self._rules_edit = QPlainTextEdit()
        self._rules_edit.setReadOnly(True)
        self._rules_edit.setMinimumWidth(420)
        self._rules_edit.setMaximumWidth(800)
        r_layout.addWidget(self._rules_edit, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_edit_rules = QPushButton("Edit")
        self._btn_edit_rules.clicked.connect(self._on_toggle_edit_rules)
        btn_row.addWidget(self._btn_edit_rules)

        self._btn_save_rules = QPushButton("Save")
        self._btn_save_rules.clicked.connect(self._on_save_rules)
        self._btn_save_rules.setEnabled(False)
        btn_row.addWidget(self._btn_save_rules)

        self._btn_reload_rules = QPushButton("Reload")
        self._btn_reload_rules.clicked.connect(self._load_rules_text)
        btn_row.addWidget(self._btn_reload_rules)

        self._btn_open_editor = QPushButton("Open in Editor")
        self._btn_open_editor.clicked.connect(self._on_open_in_editor)
        btn_row.addWidget(self._btn_open_editor)

        self._btn_toggle_rules = QPushButton("Hide")
        self._btn_toggle_rules.clicked.connect(lambda: right.setVisible(not right.isVisible()))
        btn_row.addWidget(self._btn_toggle_rules)

        r_layout.addLayout(btn_row)

        # Load initial rules text
        self._load_rules_text()

        # Assemble columns
        root.addWidget(left, stretch=1)
        root.addWidget(right, stretch=0)
        return page

    def _build_radii_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        desc = QLabel("Radii cleaning placeholder. Controls will be added later.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(desc)

        self._radii_status = self._new_status_box()
        self._radii_status.setPlainText("Radii Cleaning tab is ready for future options.")
        layout.addWidget(self._radii_status, stretch=1)
        return page

    def _new_status_box(self) -> QPlainTextEdit:
        w = QPlainTextEdit()
        w.setReadOnly(True)
        w.setMinimumHeight(120)
        w.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._status_boxes.append(w)
        return w

    # --------------------------------------------------------- Public operations
    def run_split_folder(self):
        self._on_split_folder()

    def run_rule_batch(self):
        self._on_run_batch_check()

    def set_active_subtab(self, name: str):
        # Kept for compatibility with older callers.
        _ = name

    # --------------------------------------------------------- Batch logic
    def _set_status(self, text: str):
        for box in self._status_boxes:
            box.setPlainText(text)
        self.log_message.emit(text)

    def _selected_flags(self) -> list[str]:
        flags = []
        for cb in (
            self._flag_soma,
            self._flag_axon,
            self._flag_dend,
            self._flag_apic,
            self._flag_basal,
            self._flag_rad,
            self._flag_zip,
        ):
            if cb.isChecked():
                flags.append(cb.text())
        return flags

    # ---------------- Rules editor helpers ----------------
    def _load_rules_text(self):
        try:
            pretty = json.dumps(get_auto_rules_config(), indent=2, sort_keys=True)
            self._rules_edit.setPlainText(pretty)
            return
        except Exception as e:
            txt = f"// Error loading rules: {e}\n"
        self._rules_edit.setPlainText(txt)

    def _on_toggle_edit_rules(self):
        if self._rules_edit.isReadOnly():
            self._rules_edit.setReadOnly(False)
            self._btn_save_rules.setEnabled(True)
            self._btn_edit_rules.setText("Cancel")
        else:
            self._rules_edit.setReadOnly(True)
            self._btn_save_rules.setEnabled(False)
            self._btn_edit_rules.setText("Edit")
            # reload original file to discard edits
            self._load_rules_text()

    def _on_save_rules(self):
        # Validate JSON then save
        txt = self._rules_edit.toPlainText()
        try:
            j = json.loads(txt)
        except Exception as e:
            self._set_status(f"Failed to parse rules JSON: {e}")
            return
        try:
            save_auto_rules_config(j)
            self._set_status("Rules saved to config.")
            # disable editing
            self._rules_edit.setReadOnly(True)
            self._btn_save_rules.setEnabled(False)
            self._btn_edit_rules.setText("Edit")
        except Exception as e:
            self._set_status(f"Failed to save rules: {e}")

    def _on_open_in_editor(self):
        # Try to launch user's $EDITOR or macOS open
        editor = os.environ.get("EDITOR")
        try:
            if editor:
                os.execvp(editor, [editor, str(_CFG_PATH)])
            else:
                # macOS open
                os.system(f"open '{_CFG_PATH}'")
        except Exception as e:
            self._set_status(f"Failed to open editor: {e}")

    def _on_split_folder(self):
        in_folder = QFileDialog.getExistingDirectory(self, "Choose folder containing SWC files")
        if not in_folder:
            self._set_status("Folder split cancelled.")
            return
        try:
            result = split_folder(in_folder)
        except Exception as e:
            self._set_status(f"Folder split failed:\n{e}")
            return

        summary = [
            "Folder split completed.",
            f"Folder: {result['folder']}",
            f"Processed: {result['files_total']} SWC file(s)",
            f"Split files: {result['files_split']}",
            f"Skipped (<=1 soma-root cell): {result['files_skipped']}",
            f"Saved split files: {result['trees_saved']}",
            f"Failures: {len(result['failures'])}",
        ]
        if result["failures"]:
            summary.extend(["", "First errors:"])
            summary.extend(result["failures"][:5])
        self._set_status("\n".join(summary))

    def _on_run_batch_check(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder with SWC files for rule-based batch processing"
        )
        if not folder_path:
            self._set_status("Rule-based batch processing cancelled.")
            return

        swc_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".swc") and os.path.isfile(os.path.join(folder_path, f))
        ]
        if not swc_files:
            self._set_status(f"No .swc files found in:\n{folder_path}")
            return

        flags = set(self._selected_flags())
        use_dendrite = "--dendrite" in flags
        use_basal = ("--basal" in flags) or use_dendrite
        use_apic = ("--apic" in flags) or use_dendrite
        opts = RuleBatchOptions(
            soma="--soma" in flags,
            axon="--axon" in flags,
            apic=use_apic,
            basal=use_basal,
            rad="--rad" in flags,
            zip_output="--zip" in flags,
        )

        try:
            result = run_auto_typing(folder_path, options=opts)
        except Exception as e:
            self._set_status(f"Rule-based batch processing failed:\n{e}")
            return

        lines = [
            "Rule-based batch processing completed.",
            f"Folder: {result.folder}",
            f"Output folder: {result.out_dir}",
            f"SWC files detected: {result.files_total}",
            f"Processed: {result.files_processed}",
            f"Failed: {result.files_failed}",
            f"Total nodes processed: {result.total_nodes}",
            f"Type changes: {result.total_type_changes}",
            f"Radius changes: {result.total_radius_changes}",
        ]
        if result.zip_path:
            lines.append(f"ZIP output: {result.zip_path}")
        if result.per_file:
            lines.append("")
            lines.append("Per-file summary:")
            lines.extend(result.per_file[:25])
            if len(result.per_file) > 25:
                lines.append(f"... ({len(result.per_file) - 25} more)")
        if result.failures:
            lines.append("")
            lines.append("Errors:")
            lines.extend(result.failures[:10])

        self._set_status("\n".join(lines))
