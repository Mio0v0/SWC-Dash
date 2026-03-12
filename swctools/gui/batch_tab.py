"""Batch processing controls for split, auto-labeling, and radii cleaning."""

import os
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
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
from swctools.tools.batch_processing.features.batch_validation import validate_folder as run_batch_validation
from swctools.tools.batch_processing.features.swc_splitter import split_folder

_CFG_PATH = feature_config_path("batch_processing", "auto_typing")


class _AutoTypingConfigDialog(QDialog):
    """On-demand JSON editor for batch auto-typing config."""

    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Auto-Typing JSON")
        self.resize(820, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        path_label = QLabel(f"Config file: {_CFG_PATH}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(path_label)

        self._editor = QPlainTextEdit()
        self._editor.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        root.addWidget(self._editor, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_reload = QPushButton("Reload")
        self._btn_reload.clicked.connect(self.reload_from_source)
        btn_row.addWidget(self._btn_reload)

        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)

        btn_row.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 12px; color: #555;")
        btn_row.addWidget(self._status)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        btn_row.addWidget(self._btn_close)
        root.addLayout(btn_row)

        self.reload_from_source()

    def reload_from_source(self):
        try:
            txt = json.dumps(get_auto_rules_config(), indent=2, sort_keys=True)
            self._editor.setPlainText(txt)
            self._status.setText("Loaded.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Load failed: {e}")

    def _on_save(self):
        try:
            data = json.loads(self._editor.toPlainText())
            if not isinstance(data, dict):
                raise ValueError("JSON root must be an object")
            save_auto_rules_config(data)
            self._status.setText("Saved.")
            self.saved.emit("Auto-typing JSON saved.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Save failed: {e}")


class BatchTabWidget(QWidget):
    """Owns three batch control pages used directly by Control Center tabs."""

    log_message = Signal(str)
    batch_validation_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status_boxes: list[QPlainTextEdit] = []
        self._config_dialog: _AutoTypingConfigDialog | None = None
        self._split_page = self._build_split_page()
        self._auto_page = self._build_auto_page()
        self._validation_page = self._build_validation_page()
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

    def validation_tab_widget(self) -> QWidget:
        return self._validation_page

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
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignTop)

        desc = QLabel("Auto labeling with morphology rules for all SWC files in a selected folder.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(desc)

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
        root.addLayout(flags_row1)
        root.addLayout(flags_row2)

        self._btn_run_batch_check = QPushButton("Run Auto Labeling on Folder…")
        self._btn_run_batch_check.clicked.connect(self._on_run_batch_check)
        root.addWidget(self._btn_run_batch_check)

        cfg_row = QHBoxLayout()
        self._btn_edit_auto_cfg = QPushButton("Edit Auto-Typing JSON…")
        self._btn_edit_auto_cfg.clicked.connect(self._on_edit_auto_typing_json)
        cfg_row.addWidget(self._btn_edit_auto_cfg)
        cfg_row.addStretch()
        root.addLayout(cfg_row)
        # Keep controls pinned to the top of the tab even when there is extra height.
        root.addStretch(1)
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

    def _build_validation_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignTop)

        desc = QLabel(
            "Run the same validation checks as Validation tool, but for all SWC files in a folder."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(desc)

        self._btn_batch_validate = QPushButton("Run Validation on Folder…")
        self._btn_batch_validate.clicked.connect(self._on_run_batch_validation)
        root.addWidget(self._btn_batch_validate)

        self._batch_validation_status = QLabel("No batch validation run yet.")
        self._batch_validation_status.setWordWrap(True)
        self._batch_validation_status.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(self._batch_validation_status)
        root.addStretch(1)
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
    def _set_status(self, text: str, target: QPlainTextEdit | None = None):
        if target is not None:
            target.setPlainText(text)
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

    def _on_edit_auto_typing_json(self):
        if self._config_dialog is None:
            self._config_dialog = _AutoTypingConfigDialog(self)
            self._config_dialog.saved.connect(self._set_status)
        self._config_dialog.reload_from_source()
        self._config_dialog.show()
        self._config_dialog.raise_()
        self._config_dialog.activateWindow()

    def _on_split_folder(self):
        in_folder = QFileDialog.getExistingDirectory(self, "Choose folder containing SWC files")
        if not in_folder:
            self._set_status("Folder split cancelled.", self._split_status)
            return
        try:
            result = split_folder(in_folder)
        except Exception as e:
            self._set_status(f"Folder split failed:\n{e}", self._split_status)
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
        self._set_status("\n".join(summary), self._split_status)

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

    def _on_run_batch_validation(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder with SWC files for batch validation"
        )
        if not folder_path:
            self._batch_validation_status.setText("Batch validation cancelled.")
            self.log_message.emit("Batch validation cancelled.")
            return
        try:
            out = run_batch_validation(folder_path)
        except Exception as e:  # noqa: BLE001
            msg = f"Batch validation failed: {e}"
            self._batch_validation_status.setText(msg)
            self.log_message.emit(msg)
            return

        totals = dict(out.get("summary_total", {}))
        msg = (
            f"Batch validation completed: files={out.get('files_validated', 0)}/"
            f"{out.get('files_total', 0)}, failed_files={out.get('files_failed', 0)}, "
            f"checks_total={totals.get('total', 0)}, pass={totals.get('pass', 0)}, "
            f"warn={totals.get('warning', 0)}, fail={totals.get('fail', 0)}"
        )
        self._batch_validation_status.setText(msg)
        self.log_message.emit(msg)
        self.batch_validation_ready.emit(out)
