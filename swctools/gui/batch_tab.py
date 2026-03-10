"""Batch processing controls for split, auto-labeling, and radii cleaning."""

import os
from pathlib import Path

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

from .rule_batch_processor import RuleBatchOptions, run_rule_batch
from .validation_core import _split_swc_by_soma_roots


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
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
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

    def _on_split_folder(self):
        in_folder = QFileDialog.getExistingDirectory(self, "Choose folder containing SWC files")
        if not in_folder:
            self._set_status("Folder split cancelled.")
            return

        swc_files = sorted(
            p for p in Path(in_folder).iterdir()
            if p.is_file() and p.suffix.lower() == ".swc"
        )
        if not swc_files:
            self._set_status(f"No .swc files found in:\n{in_folder}")
            return

        files_split = 0
        files_skipped = 0
        cells_saved = 0
        failures = []

        for swc_path in swc_files:
            try:
                with open(swc_path, "r", encoding="utf-8", errors="ignore") as f:
                    swc_text = f.read()
                trees = _split_swc_by_soma_roots(swc_text)

                if len(trees) <= 1:
                    files_skipped += 1
                    continue

                files_split += 1
                stem = swc_path.stem
                out_dir = Path(in_folder) / stem
                out_dir.mkdir(parents=True, exist_ok=True)

                for idx, (_root_id, sub_text, _node_count) in enumerate(trees, start=1):
                    out_path = out_dir / f"{stem}_tree{idx}.swc"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(sub_text)
                    cells_saved += 1
            except Exception as e:
                failures.append(f"{swc_path.name}: {e}")

        summary = (
            f"Folder split completed.\n"
            f"Folder: {in_folder}\n"
            f"Processed: {len(swc_files)} SWC file(s)\n"
            f"Split files: {files_split}\n"
            f"Skipped (<=1 soma-root cell): {files_skipped}\n"
            f"Saved split files: {cells_saved}\n"
            f"Failures: {len(failures)}"
        )
        if failures:
            summary += "\n\nFirst errors:\n" + "\n".join(failures[:5])
        self._set_status(summary)

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
            result = run_rule_batch(folder_path, opts)
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
