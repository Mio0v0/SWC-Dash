"""Validation control panel: single-file auto labeling with preview/apply/cancel."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.auto_typing import RuleBatchOptions, get_auto_rules_config, save_auto_rules_config
from swctools.core.config import feature_config_path

_CFG_PATH = feature_config_path("batch_processing", "auto_typing")


class _AutoTypingConfigDialog(QDialog):
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


class ValidationAutoLabelPanel(QWidget):
    process_requested = Signal(object)
    apply_requested = Signal()
    cancel_requested = Signal()
    guide_requested = Signal()
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_dialog: _AutoTypingConfigDialog | None = None
        self._build_ui()
        self.set_preview_state(False, None)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignTop)

        desc = QLabel("Auto labeling on current SWC file using the same rule engine as Batch Auto Label.")
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

        self._flag_soma.setChecked(True)
        self._flag_axon.setChecked(True)
        self._flag_basal.setChecked(True)

        for cb in (self._flag_soma, self._flag_axon, self._flag_dend, self._flag_apic):
            flags_row1.addWidget(cb)
        flags_row1.addStretch()
        for cb in (self._flag_basal, self._flag_rad):
            flags_row2.addWidget(cb)
        flags_row2.addStretch()
        root.addLayout(flags_row1)
        root.addLayout(flags_row2)

        top_btns = QHBoxLayout()
        self._btn_rule_guide = QPushButton("Show Auto Typing Guide")
        self._btn_rule_guide.clicked.connect(self.guide_requested.emit)
        top_btns.addWidget(self._btn_rule_guide)
        self._btn_run = QPushButton("Run Auto Labeling on Current File")
        self._btn_run.clicked.connect(lambda: self.process_requested.emit(self.current_options()))
        top_btns.addWidget(self._btn_run)
        top_btns.addStretch()
        root.addLayout(top_btns)

        cfg_row = QHBoxLayout()
        self._btn_edit_cfg = QPushButton("Edit Auto-Typing JSON…")
        self._btn_edit_cfg.clicked.connect(self._on_edit_auto_typing_json)
        cfg_row.addWidget(self._btn_edit_cfg)
        cfg_row.addStretch()
        root.addLayout(cfg_row)

        action_row = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self.apply_requested.emit)
        action_row.addWidget(self._btn_apply)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.cancel_requested.emit)
        action_row.addWidget(self._btn_cancel)
        action_row.addStretch()
        root.addLayout(action_row)

        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMinimumHeight(160)
        self._summary.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        root.addWidget(self._summary, stretch=1)

    def current_options(self) -> RuleBatchOptions:
        use_dendrite = bool(self._flag_dend.isChecked())
        use_basal = bool(self._flag_basal.isChecked() or use_dendrite)
        use_apic = bool(self._flag_apic.isChecked() or use_dendrite)
        return RuleBatchOptions(
            soma=bool(self._flag_soma.isChecked()),
            axon=bool(self._flag_axon.isChecked()),
            apic=use_apic,
            basal=use_basal,
            rad=bool(self._flag_rad.isChecked()),
            zip_output=False,
        )

    def _on_edit_auto_typing_json(self):
        if self._config_dialog is None:
            self._config_dialog = _AutoTypingConfigDialog(self)
            self._config_dialog.saved.connect(self.log_message.emit)
        self._config_dialog.reload_from_source()
        self._config_dialog.show()
        self._config_dialog.raise_()
        self._config_dialog.activateWindow()

    def set_preview_state(self, has_preview: bool, summary: dict | None):
        self._btn_apply.setEnabled(bool(has_preview))
        self._btn_cancel.setEnabled(bool(has_preview))

        if not summary:
            self._summary.setPlainText(
                "No auto-label preview yet.\n"
                "Use Run Auto Labeling on Current File to generate a preview tab."
            )
            return

        out_counts = dict(summary.get("out_type_counts", {}))
        lines = [
            "Auto Label Preview",
            "------------------",
            f"Nodes processed: {summary.get('nodes_total', 0)}",
            f"Type changes: {summary.get('type_changes', 0)}",
            f"Radius changes: {summary.get('radius_changes', 0)}",
            (
                "Output types (soma/axon/basal/apic): "
                f"{out_counts.get(1, 0)}/{out_counts.get(2, 0)}/{out_counts.get(3, 0)}/{out_counts.get(4, 0)}"
            ),
        ]
        log_path = str(summary.get("log_path", "") or "").strip()
        if log_path:
            lines.extend(["", f"Log file: {log_path}"])
        self._summary.setPlainText("\n".join(lines))

    def set_status_text(self, text: str):
        self._summary.setPlainText(str(text or ""))
