"""Morphology Smart Decimation (RDP) control panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swctools.core.config import feature_config_path, load_feature_config

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
        p = Path(self._cfg_path)
        if not p.exists():
            self.log_message.emit(f"Simplification config not found: {p}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

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
