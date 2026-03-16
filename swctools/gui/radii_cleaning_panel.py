"""Reusable GUI panel for radii cleaning (file/folder)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.config import feature_config_path, load_feature_config, save_feature_config
from swctools.tools.batch_processing.features.radii_cleaning import clean_path

_CFG_TOOL = "batch_processing"
_CFG_FEATURE = "radii_cleaning"
_CFG_PATH = feature_config_path(_CFG_TOOL, _CFG_FEATURE)


class _RadiiConfigDialog(QDialog):
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Radii Cleaning JSON")
        self.resize(820, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        path_label = QLabel(f"Config file (shared by Batch + Validation + CLI): {_CFG_PATH}")
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

        row = QHBoxLayout()
        b_reload = QPushButton("Reload")
        b_reload.clicked.connect(self.reload_from_source)
        row.addWidget(b_reload)

        b_save = QPushButton("Save")
        b_save.clicked.connect(self._on_save)
        row.addWidget(b_save)

        row.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 12px; color: #555;")
        row.addWidget(self._status)

        b_close = QPushButton("Close")
        b_close.clicked.connect(self.close)
        row.addWidget(b_close)
        root.addLayout(row)

        self.reload_from_source()

    def reload_from_source(self):
        try:
            cfg = load_feature_config(_CFG_TOOL, _CFG_FEATURE, default={})
            self._editor.setPlainText(json.dumps(cfg, indent=2, sort_keys=True))
            self._status.setText("Loaded.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Load failed: {e}")

    def _on_save(self):
        try:
            payload = json.loads(self._editor.toPlainText())
            if not isinstance(payload, dict):
                raise ValueError("JSON root must be an object")
            save_feature_config(_CFG_TOOL, _CFG_FEATURE, payload)
            self._status.setText("Saved.")
            self.saved.emit("Radii-clean JSON saved.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Save failed: {e}")


class RadiiCleaningPanel(QWidget):
    """Run shared radii cleaning backend for either file or folder."""

    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg_dialog: _RadiiConfigDialog | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        desc = QLabel(
            "Detects abnormal radii (zero/non-finite/spikes) and replaces each abnormal value with "
            "the local mean of parent + children radii. Thresholds are JSON-configurable."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(desc)

        row = QHBoxLayout()
        b_file = QPushButton("Run on File…")
        b_file.clicked.connect(self._on_run_file)
        row.addWidget(b_file)

        b_folder = QPushButton("Run on Folder…")
        b_folder.clicked.connect(self._on_run_folder)
        row.addWidget(b_folder)

        b_cfg = QPushButton("Edit Radii JSON…")
        b_cfg.clicked.connect(self._on_edit_cfg)
        row.addWidget(b_cfg)
        row.addStretch()
        root.addLayout(row)

        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setMinimumHeight(140)
        self._status.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._status.setPlainText("Radii cleaning ready.")
        root.addWidget(self._status, stretch=1)

    def _set_status(self, text: str):
        self._status.setPlainText(text)
        self.log_message.emit(text)

    def _on_edit_cfg(self):
        if self._cfg_dialog is None:
            self._cfg_dialog = _RadiiConfigDialog(self)
            self._cfg_dialog.saved.connect(self.log_message.emit)
        self._cfg_dialog.reload_from_source()
        self._cfg_dialog.show()
        self._cfg_dialog.raise_()
        self._cfg_dialog.activateWindow()

    def _on_run_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SWC file for radii cleaning", "", "SWC Files (*.swc)")
        if not path:
            self._set_status("Radii cleaning cancelled.")
            return
        self._run_path(path)

    def _on_run_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with SWC files for radii cleaning")
        if not folder:
            self._set_status("Radii cleaning cancelled.")
            return
        self._run_path(folder)

    def _run_path(self, path: str):
        try:
            out = clean_path(path)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"Radii cleaning failed:\n{e}")
            return

        mode = str(out.get("mode", ""))
        if mode == "file":
            lines = [
                "Radii cleaning completed (file).",
                f"Input: {out.get('input_path', '')}",
                f"Output: {out.get('output_path', '')}",
                f"Radius changes: {out.get('radius_changes', 0)}",
                f"Log: {out.get('log_path', '')}",
            ]
            detail = list(out.get("change_lines", []))
            if detail:
                lines.append("")
                lines.append("Node changes:")
                lines.extend(detail[:80])
                if len(detail) > 80:
                    lines.append(f"... ({len(detail) - 80} more)")
            self._set_status("\n".join(lines))
            return

        lines = [
            "Radii cleaning completed (folder).",
            f"Folder: {out.get('folder', '')}",
            f"Output folder: {out.get('out_dir', '')}",
            f"SWC files detected: {out.get('files_total', 0)}",
            f"Processed: {out.get('files_processed', 0)}",
            f"Failed: {out.get('files_failed', 0)}",
            f"Total radius changes: {out.get('total_radius_changes', 0)}",
            f"Log: {out.get('log_path', '')}",
        ]
        per_file = list(out.get("per_file", []))
        if per_file:
            lines.append("")
            lines.append("Per-file summary:")
            for row in per_file[:40]:
                lines.append(f"{row.get('file', '')}: radius_changes={row.get('radius_changes', 0)}")
            if len(per_file) > 40:
                lines.append(f"... ({len(per_file) - 40} more)")
        self._set_status("\n".join(lines))
