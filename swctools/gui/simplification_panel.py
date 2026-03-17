"""Morphology Smart Decimation (RDP) control panel."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.config import feature_config_path, load_feature_config, save_feature_config

_DEFAULT_CFG: dict[str, Any] = {
    "thresholds": {
        "epsilon": 2.0,
        "radius_tolerance": 0.5,
    },
    "flags": {
        "keep_tips": True,
        "keep_bifurcations": True,
        "keep_roots": True,
    },
}


class SimplificationPanel(QWidget):
    """UI-only control panel for Smart Decimation workflow."""

    process_requested = Signal(dict)
    apply_requested = Signal()
    redo_requested = Signal()
    cancel_requested = Signal()
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg_path = feature_config_path("morphology_editing", "simplification")
        self._cfg_dialog: _SimplificationConfigDialog | None = None
        self._build_ui()
        self._load_from_json()
        self.set_preview_state(False, None, None)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignTop)

        title = QLabel("Smart Decimation (RDP)")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #333;")
        root.addWidget(title)

        desc = QLabel(
            "Process creates a temporary Simplified View tab.\n"
            "Apply saves a new SWC and replaces the current working buffer.\n"
            "Redo recalculates simplification. Cancel discards temporary preview."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(desc)

        cfg_row = QHBoxLayout()
        self._btn_reload_json = QPushButton("Reload JSON")
        self._btn_reload_json.clicked.connect(self._load_from_json)
        cfg_row.addWidget(self._btn_reload_json)

        self._btn_open_json = QPushButton("Open JSON")
        self._btn_open_json.clicked.connect(self._open_json)
        cfg_row.addWidget(self._btn_open_json)
        cfg_row.addStretch()
        root.addLayout(cfg_row)

        eps_row = QHBoxLayout()
        eps_row.addWidget(QLabel("Epsilon (RDP):"))
        self._epsilon = QDoubleSpinBox()
        self._epsilon.setDecimals(4)
        self._epsilon.setRange(0.0, 100000.0)
        self._epsilon.setSingleStep(0.1)
        eps_row.addWidget(self._epsilon)
        root.addLayout(eps_row)

        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Radius Tolerance:"))
        self._radius_tol = QDoubleSpinBox()
        self._radius_tol.setDecimals(4)
        self._radius_tol.setRange(0.0, 1000.0)
        self._radius_tol.setSingleStep(0.05)
        rad_row.addWidget(self._radius_tol)
        root.addLayout(rad_row)

        flag_row = QHBoxLayout()
        self._keep_tips = QCheckBox("Keep Tips")
        self._keep_bifs = QCheckBox("Keep Bifurcations")
        flag_row.addWidget(self._keep_tips)
        flag_row.addWidget(self._keep_bifs)
        flag_row.addStretch()
        root.addLayout(flag_row)

        self._btn_process = QPushButton("Process")
        self._btn_process.clicked.connect(self._on_process)
        root.addWidget(self._btn_process)

        action_bar = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self.apply_requested.emit)
        action_bar.addWidget(self._btn_apply)

        self._btn_redo = QPushButton("Redo")
        self._btn_redo.clicked.connect(self.redo_requested.emit)
        action_bar.addWidget(self._btn_redo)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.cancel_requested.emit)
        action_bar.addWidget(self._btn_cancel)
        action_bar.addStretch()
        root.addLayout(action_bar)

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

    def _open_json(self):
        if self._cfg_dialog is None:
            self._cfg_dialog = _SimplificationConfigDialog(self)
            self._cfg_dialog.saved.connect(self.log_message.emit)
            self._cfg_dialog.saved.connect(lambda _msg: self._load_from_json())
        self._cfg_dialog.reload_from_source()
        self._cfg_dialog.show()
        self._cfg_dialog.raise_()
        self._cfg_dialog.activateWindow()

    def _load_from_json(self):
        cfg = load_feature_config("morphology_editing", "simplification", default=_DEFAULT_CFG)
        thr = dict(cfg.get("thresholds", {}))
        flags = dict(cfg.get("flags", {}))
        self._epsilon.setValue(float(thr.get("epsilon", _DEFAULT_CFG["thresholds"]["epsilon"])))
        self._radius_tol.setValue(
            float(thr.get("radius_tolerance", _DEFAULT_CFG["thresholds"]["radius_tolerance"]))
        )
        self._keep_tips.setChecked(bool(flags.get("keep_tips", True)))
        self._keep_bifs.setChecked(bool(flags.get("keep_bifurcations", True)))
        self.log_message.emit(f"Loaded simplification config: {self._cfg_path}")

    def _on_process(self):
        self.process_requested.emit(self.current_overrides())

    def current_overrides(self) -> dict[str, Any]:
        return {
            "thresholds": {
                "epsilon": float(self._epsilon.value()),
                "radius_tolerance": float(self._radius_tol.value()),
            },
            "flags": {
                "keep_tips": bool(self._keep_tips.isChecked()),
                "keep_bifurcations": bool(self._keep_bifs.isChecked()),
            },
        }

    def set_preview_state(
        self,
        has_preview: bool,
        summary: dict[str, Any] | None,
        log_path: str | None,
    ):
        self._btn_apply.setEnabled(bool(has_preview))
        self._btn_redo.setEnabled(bool(has_preview))
        self._btn_cancel.setEnabled(bool(has_preview))

        if not summary:
            self._summary.setPlainText(
                "No simplification preview yet.\n"
                "Use Process to open a temporary Simplified View tab."
            )
            return

        lines = [
            "Smart Decimation Preview",
            "------------------------",
            f"Original Node Count: {summary.get('original_node_count', 0)}",
            f"New Node Count: {summary.get('new_node_count', 0)}",
            f"Reduction (%): {float(summary.get('reduction_percent', 0.0)):.2f}",
            "",
            "Parameters Used:",
        ]
        for k, v in sorted(dict(summary.get("params_used", {})).items()):
            lines.append(f"- {k}: {v}")
        if log_path:
            lines.extend(["", f"Log file: {log_path}"])
        self._summary.setPlainText("\n".join(lines))

    def set_log_text(self, text: str):
        self._summary.setPlainText(str(text or ""))

    def export_current_json_text(self) -> str:
        data = {
            "thresholds": {
                "epsilon": float(self._epsilon.value()),
                "radius_tolerance": float(self._radius_tol.value()),
            },
            "flags": {
                "keep_tips": bool(self._keep_tips.isChecked()),
                "keep_bifurcations": bool(self._keep_bifs.isChecked()),
            },
        }
        return json.dumps(data, indent=2, sort_keys=True)


class _SimplificationConfigDialog(QDialog):
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Simplification JSON")
        self.resize(820, 620)
        self._tool = "morphology_editing"
        self._feature = "simplification"
        self._cfg_path = feature_config_path(self._tool, self._feature)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        path_label = QLabel(f"Config file: {self._cfg_path}")
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
        btn_reload = QPushButton("Reload")
        btn_reload.clicked.connect(self.reload_from_source)
        btn_row.addWidget(btn_reload)

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_save)

        btn_row.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 12px; color: #555;")
        btn_row.addWidget(self._status)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        self.reload_from_source()

    def reload_from_source(self):
        try:
            cfg = load_feature_config(self._tool, self._feature, default=_DEFAULT_CFG)
            self._editor.setPlainText(json.dumps(cfg, indent=2, sort_keys=True))
            self._status.setText("Loaded.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Load failed: {e}")

    def _on_save(self):
        try:
            payload = json.loads(self._editor.toPlainText())
            if not isinstance(payload, dict):
                raise ValueError("JSON root must be an object")
            save_feature_config(self._tool, self._feature, payload)
            self._status.setText("Saved.")
            self.saved.emit("Simplification JSON saved.")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"Save failed: {e}")
