"""Validation widgets for SWC-QT.

Contains:
1) ValidationPrecheckWidget: grouped rule summary + explanations
2) ValidationTabWidget: validation results/details/actions
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from swctools.core.validation import _split_swc_by_soma_roots
from swctools.tools.validation.features.core import run_checks_text
from swctools.validation.catalog import CHECK_CATALOG, CHECK_ORDER


class _ValidationWorker(QObject):
    finished = Signal(int, dict)
    failed = Signal(int, str)

    def __init__(self, run_id: int, swc_text: str):
        super().__init__()
        self._run_id = int(run_id)
        self._swc_text = swc_text

    @Slot()
    def run(self):
        try:
            report = run_checks_text(self._swc_text, profile="default")
            self.finished.emit(self._run_id, report.to_dict())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self._run_id, str(e))


class ValidationPrecheckWidget(QWidget):
    """Floating pre-check summary grouped by validation category."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.populate_catalog()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Pre-check Summary (Rule Guide)")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #222;")
        layout.addWidget(title)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Label", "Rule"])
        self._tree.setUniformRowHeights(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setWordWrap(True)
        self._tree.setStyleSheet(
            "QTreeWidget { font-size: 12px; gridline-color: #ddd; }"
            "QHeaderView::section { font-weight: 600; padding: 4px; }"
        )
        self._tree.setColumnWidth(0, 250)
        self._tree.setColumnWidth(1, 640)
        layout.addWidget(self._tree, stretch=1)

    def populate_catalog(self):
        self._tree.clear()
        for category, checks in CHECK_CATALOG:
            group = QTreeWidgetItem([category, ""])
            group.setFirstColumnSpanned(True)
            group.setExpanded(True)
            group.setFont(0, QFont("", 11, QFont.Bold))
            self._tree.addTopLevelItem(group)
            for _key, label, rule in checks:
                item = QTreeWidgetItem([label, rule])
                item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
                item.setTextAlignment(1, Qt.AlignLeft | Qt.AlignVCenter)
                group.addChild(item)
        self._tree.expandAll()


class ValidationTabWidget(QWidget):
    """Validation results panel with lazy execution for fast SWC loading."""

    def __init__(self, parent=None, as_panel: bool = True):
        super().__init__(parent)
        self._as_panel = as_panel
        self._source_stem = "file"
        self._df: pd.DataFrame | None = None
        self._swc_text: str = ""
        self._swc_dirty = True
        self._trees: list = []
        self._report: dict | None = None
        self._results_rows: list[dict] = []
        self._show_save_all = False
        self._run_id = 0
        self._worker_thread: QThread | None = None
        self._worker: _ValidationWorker | None = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Validation Results")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #333;")
        header.addWidget(title)
        header.addStretch()
        self._btn_run = QPushButton("Run Validation")
        self._btn_run.clicked.connect(self.run_validation)
        header.addWidget(self._btn_run)
        layout.addLayout(header)

        self._alert_banner = QLabel("")
        self._alert_banner.setVisible(False)
        self._alert_banner.setWordWrap(True)
        self._alert_banner.setStyleSheet(
            "QLabel {"
            "  padding: 8px 10px; border-radius: 6px; border: 1px solid #c0392b;"
            "  background-color: #fdecea; color: #7b1a12; font-size: 12px;"
            "}"
        )
        layout.addWidget(self._alert_banner)

        split = QSplitter(Qt.Vertical)
        layout.addWidget(split, stretch=1)

        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(4)
        results_title = QLabel("Post-check Results")
        results_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #333;")
        results_layout.addWidget(results_title)
        self._results_table = QTableWidget()
        self._results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._results_table.setSelectionMode(QTableWidget.SingleSelection)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setStyleSheet(
            "QTableWidget { font-size: 12px; gridline-color: #ddd; }"
            "QHeaderView::section { font-weight: 600; padding: 4px; }"
        )
        self._results_table.cellClicked.connect(self._on_result_row_clicked)
        results_layout.addWidget(self._results_table, stretch=1)
        split.addWidget(results_panel)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(4)
        detail_title = QLabel("Check Detail")
        detail_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #333;")
        detail_layout.addWidget(detail_title)
        self._detail_text = QPlainTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        detail_layout.addWidget(self._detail_text, stretch=1)
        split.addWidget(detail_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        btn_layout = QHBoxLayout()
        self._btn_save_all = QPushButton("Save All Trees")
        self._btn_save_all.setVisible(False)
        self._btn_save_all.clicked.connect(self._on_save_all)
        btn_layout.addWidget(self._btn_save_all)
        self._btn_export_json = QPushButton("Export Validation JSON")
        self._btn_export_json.setEnabled(False)
        self._btn_export_json.clicked.connect(self._on_export_json)
        btn_layout.addWidget(self._btn_export_json)
        btn_layout.addStretch()
        self._save_status = QLabel("")
        self._save_status.setStyleSheet("color: #555; font-size: 12px;")
        btn_layout.addWidget(self._save_status)
        layout.addLayout(btn_layout)

        self._clear_results_ui("Load an SWC and click Run Validation.")

    # --------------------------------------------------------- Public API
    def has_results(self) -> bool:
        return bool(self._report)

    def load_swc(self, df: pd.DataFrame, filename: str, auto_run: bool = True):
        self._source_stem = Path(filename or "file").stem or "file"
        self._df = df.copy()
        self._swc_text = ""
        self._swc_dirty = True
        self._trees = []
        self._report = None
        self._results_rows = []
        self._show_save_all = int((self._df["type"] == 1).sum()) > 1
        self._btn_save_all.setVisible(self._show_save_all)
        self._btn_save_all.setEnabled(self._show_save_all)

        if self._show_save_all:
            self._alert_banner.setText(
                "Multiple soma nodes detected. Use Save All Trees to split output files."
            )
            self._alert_banner.setVisible(True)
        else:
            self._alert_banner.clear()
            self._alert_banner.setVisible(False)

        self._btn_export_json.setEnabled(False)
        self._clear_results_ui("Load complete. Click Run Validation to compute results.")
        if auto_run:
            self.run_validation()

    def run_validation(self):
        if self._df is None or self._df.empty:
            self._save_status.setText("No SWC loaded.")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
            return
        try:
            self._ensure_swc_text()
            self._start_validation_worker(self._swc_text)
        except Exception as e:  # noqa: BLE001
            self._save_status.setText(f"Validation error: {e}")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")

    def _start_validation_worker(self, swc_text: str):
        self._run_id += 1
        run_id = int(self._run_id)
        self._btn_run.setEnabled(False)
        self._save_status.setText("Validation running...")
        self._save_status.setStyleSheet("color: #555; font-size: 12px;")

        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(100)

        self._worker_thread = QThread(self)
        self._worker = _ValidationWorker(run_id, swc_text)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_validation_finished)
        self._worker.failed.connect(self._on_validation_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker_refs)
        self._worker_thread.start()

    @Slot()
    def _cleanup_worker_refs(self):
        self._worker = None
        self._worker_thread = None

    @Slot(int, dict)
    def _on_validation_finished(self, run_id: int, report_dict: dict):
        if int(run_id) != int(self._run_id):
            return
        self._report = dict(report_dict)
        rows = list(self._report.get("results", []))
        rows.sort(key=self._result_sort_key)
        self._results_rows = rows
        self._populate_results_table(rows)
        self._btn_export_json.setEnabled(True)
        self._btn_run.setEnabled(True)
        self._detail_text.setPlainText("Select a row to inspect details.")
        self._save_status.setText("Validation completed.")
        self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")

    @Slot(int, str)
    def _on_validation_failed(self, run_id: int, error_text: str):
        if int(run_id) != int(self._run_id):
            return
        self._btn_run.setEnabled(True)
        self._save_status.setText(f"Validation error: {error_text}")
        self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")

    # --------------------------------------------------------- Internal helpers
    def _ensure_swc_text(self):
        if not self._swc_dirty and self._swc_text:
            return
        if self._df is None or self._df.empty:
            self._swc_text = ""
            self._swc_dirty = False
            return

        arr = self._df[["id", "type", "x", "y", "z", "radius", "parent"]].to_numpy(copy=False)
        buf = io.StringIO()
        np.savetxt(
            buf,
            arr,
            fmt=["%d", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%d"],
            delimiter=" ",
        )
        self._swc_text = "# id type x y z radius parent\n" + buf.getvalue()
        self._swc_dirty = False

    def _ensure_trees(self):
        if self._trees:
            return
        self._ensure_swc_text()
        if not self._swc_text:
            return
        self._trees = _split_swc_by_soma_roots(self._swc_text)

    def _clear_results_ui(self, message: str):
        self._results_table.clear()
        self._results_table.setColumnCount(2)
        self._results_table.setHorizontalHeaderLabels(["Status", "Label"])
        self._results_table.setRowCount(0)
        self._results_table.setColumnWidth(0, 80)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._detail_text.setPlainText(message)

    def _status_cell(self, status: str) -> tuple[str, str]:
        s = (status or "").lower()
        if s == "pass":
            return "PASS", "#2ca02c"
        if s == "warning":
            return "WARN", "#ff9900"
        if s == "error":
            return "ERR", "#9467bd"
        return "FAIL", "#d62728"

    def _result_sort_key(self, row: dict) -> tuple[int, str]:
        key = str(row.get("key", ""))
        label = str(row.get("label", ""))
        return (CHECK_ORDER.get(key, 1000), label.lower())

    def _populate_results_table(self, rows: list[dict]):
        self._results_table.clear()
        self._results_table.setColumnCount(2)
        self._results_table.setHorizontalHeaderLabels(["Status", "Label"])
        self._results_table.setRowCount(len(rows))
        self._results_table.setColumnWidth(0, 80)
        self._results_table.setColumnWidth(1, 500)
        for i, row in enumerate(rows):
            status_txt, color = self._status_cell(str(row.get("status", "")))
            vals = [status_txt, str(row.get("label", ""))]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if j == 0:
                    it.setFont(QFont("", 11, QFont.Bold))
                    it.setForeground(QBrush(QColor(color)))
                self._results_table.setItem(i, j, it)
        self._results_table.horizontalHeader().setStretchLastSection(True)

    # --------------------------------------------------------- Detail / actions
    def _on_result_row_clicked(self, row: int, _col: int):
        if row < 0 or row >= len(self._results_rows):
            return
        r = self._results_rows[row]
        detail = [
            f"label: {r.get('label')}",
            f"status: {r.get('status')}",
            f"message: {r.get('message')}",
            f"params_used: {json.dumps(r.get('params_used', {}), sort_keys=True)}",
            f"thresholds_used: {json.dumps(r.get('thresholds_used', {}), sort_keys=True)}",
            f"failing_node_ids: {r.get('failing_node_ids', [])}",
            f"failing_section_ids: {r.get('failing_section_ids', [])}",
            f"metrics: {json.dumps(r.get('metrics', {}), sort_keys=True)}",
        ]
        self._detail_text.setPlainText("\n".join(detail))

    def _on_save_all(self):
        if self._df is None or self._df.empty:
            self._save_status.setText("No SWC loaded.")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
            return

        self._ensure_trees()
        if not self._trees:
            self._save_status.setText("No split trees available to save.")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
            return

        folder = QFileDialog.getExistingDirectory(self, "Choose folder to save all trees")
        if not folder:
            self._save_status.setText("Save all trees cancelled.")
            self._save_status.setStyleSheet("color: #777; font-size: 12px;")
            return

        try:
            saved = 0
            for tidx, (_root_id, sub_text, _node_count) in enumerate(self._trees, start=1):
                out_name = f"{self._source_stem}_tree{tidx}.swc"
                out_path = os.path.join(folder, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(sub_text)
                saved += 1
            self._save_status.setText(f"Saved {saved} tree(s) to {folder}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:  # noqa: BLE001
            self._save_status.setText(f"Save error: {e}")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")

    def _on_export_json(self):
        if not self._report:
            self._save_status.setText("No validation results to export.")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Validation JSON",
            "validation.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            self._save_status.setText("Export validation JSON cancelled.")
            self._save_status.setStyleSheet("color: #777; font-size: 12px;")
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._report, f, indent=2, default=str)
            self._save_status.setText(f"Exported JSON to {os.path.basename(path)}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:  # noqa: BLE001
            self._save_status.setText(f"Export error: {e}")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
