"""Batch processing tab for folder-level SWC operations."""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFileDialog, QMessageBox, QPlainTextEdit,
)

from validation_core import _split_swc_by_soma_roots
from rule_batch_processor import RuleBatchOptions, run_rule_batch


class BatchTabWidget(QWidget):
    """Batch operations: split folders and apply rule-based SWC type processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Batch Processing")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #333;")
        layout.addWidget(title)

        intro = QLabel(
            "Run folder-level tools on all SWC files in a selected directory."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: #555;")
        layout.addWidget(intro)

        # --- Batch split section ---
        split_title = QLabel("Batch folder split")
        split_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #333;")
        layout.addWidget(split_title)

        split_desc = QLabel(
            "For each multi-cell SWC, create a folder named after the file and save "
            "split files as <original>_tree1.swc, <original>_tree2.swc, ..."
        )
        split_desc.setWordWrap(True)
        split_desc.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(split_desc)

        self._btn_split_folder = QPushButton("Split Folder SWCs…")
        self._btn_split_folder.clicked.connect(self._on_split_folder)
        layout.addWidget(self._btn_split_folder)

        # --- Native rule-based section ---
        batch_title = QLabel("Rule-Based Batch Processing")
        batch_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #333;")
        layout.addWidget(batch_title)

        batch_desc = QLabel(
            "Apply morphology rules to assign/fix SWC types for all files in a folder. "
            "Outputs are saved to a sibling folder named <original>_batch process."
        )
        batch_desc.setWordWrap(True)
        batch_desc.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(batch_desc)

        flags_row = QHBoxLayout()
        self._flag_soma = QCheckBox("--soma")
        self._flag_axon = QCheckBox("--axon")
        self._flag_dend = QCheckBox("--dendrite")
        self._flag_apic = QCheckBox("--apic")
        self._flag_basal = QCheckBox("--basal")
        self._flag_rad = QCheckBox("--rad")
        self._flag_zip = QCheckBox("--zip")

        self._flag_axon.setChecked(True)
        self._flag_dend.setChecked(True)

        for cb in (
            self._flag_soma, self._flag_axon, self._flag_dend, self._flag_apic,
            self._flag_basal, self._flag_rad, self._flag_zip,
        ):
            flags_row.addWidget(cb)
        flags_row.addStretch()
        layout.addLayout(flags_row)

        self._btn_run_batch_check = QPushButton("Run Rule-Based Batch on Folder…")
        self._btn_run_batch_check.clicked.connect(self._on_run_batch_check)
        layout.addWidget(self._btn_run_batch_check)

        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setMinimumHeight(180)
        self._status.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        layout.addWidget(self._status, stretch=1)

    def _set_status(self, text: str):
        self._status.setPlainText(text)

    def _selected_flags(self) -> list[str]:
        flags = []
        for cb in (
            self._flag_soma, self._flag_axon, self._flag_dend, self._flag_apic,
            self._flag_basal, self._flag_rad, self._flag_zip,
        ):
            if cb.isChecked():
                flags.append(cb.text())
        return flags

    def _on_split_folder(self):
        in_folder = QFileDialog.getExistingDirectory(
            self, "Choose folder containing SWC files"
        )
        if not in_folder:
            return

        swc_files = sorted(
            p for p in Path(in_folder).iterdir()
            if p.is_file() and p.suffix.lower() == ".swc"
        )
        if not swc_files:
            QMessageBox.information(self, "No SWC files", "No .swc files found in the selected folder.")
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
            QMessageBox.warning(self, "Folder split completed with errors", summary)
        else:
            QMessageBox.information(self, "Folder split completed", summary)
        self._set_status(summary)

    def _on_run_batch_check(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder with SWC files for rule-based batch processing"
        )
        if not folder_path:
            return

        swc_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".swc") and os.path.isfile(os.path.join(folder_path, f))
        ]
        if not swc_files:
            QMessageBox.information(self, "No SWC files", f"No .swc files found in:\n{folder_path}")
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
            msg = f"Rule-based batch processing failed:\n{e}"
            self._set_status(msg)
            QMessageBox.critical(self, "Batch processing failed", msg)
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

        msg = "\n".join(lines)
        self._set_status(msg)
        if result.files_failed:
            QMessageBox.warning(self, "Batch processing completed with errors", msg)
        else:
            QMessageBox.information(self, "Batch processing completed", msg)
