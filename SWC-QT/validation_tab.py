"""Format Validation tab widget for the SWC-QT application."""

import json
import os

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QSizePolicy,
)

from validation_core import run_per_tree_validation, _split_swc_by_trees
from constants import SWC_COLS


# Non-fatal checks that show orange instead of red
WARNING_CHECKS = {
    "Has apical dendrite",
    "Has any unifurcation",
    "No single-child chains",
    "No flat neurites",
    "No ultra-narrow sections",
    "No ultra-narrow starts",
    "No \u201cfat\u201d terminal ends",
}


class ValidationTabWidget(QWidget):
    """Validation tab: runs per-tree checks and displays a colored grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._swc_text: str | None = None
        self._trees: list = []
        self._json_rows: list = []

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Alert banner (hidden by default)
        self._alert_banner = QLabel()
        self._alert_banner.setVisible(False)
        self._alert_banner.setWordWrap(True)
        layout.addWidget(self._alert_banner)

        # Validation grid table
        self._grid = QTableWidget()
        self._grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self._grid.setSelectionMode(QTableWidget.NoSelection)
        self._grid.verticalHeader().setVisible(False)
        self._grid.setAlternatingRowColors(True)
        self._grid.setStyleSheet(
            "QTableWidget { font-size: 13px; gridline-color: #ddd; }"
            "QHeaderView::section { font-weight: 600; padding: 4px; }"
        )
        layout.addWidget(self._grid, stretch=1)

        # Button bar
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)

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

    # --------------------------------------------------------- Public API
    def load_swc(self, df: pd.DataFrame, filename: str):
        """Run validation on the loaded SWC data."""
        # Build SWC text from the DataFrame
        lines = ["# id type x y z radius parent"]
        for _, row in df.iterrows():
            lines.append(
                f"{int(row['id'])} {int(row['type'])} "
                f"{row['x']:.4f} {row['y']:.4f} {row['z']:.4f} "
                f"{row['radius']:.4f} {int(row['parent'])}"
            )
        self._swc_text = "\n".join(lines) + "\n"

        # Run validation
        check_names, tree_results = run_per_tree_validation(self._swc_text)

        if not check_names or not tree_results:
            self._grid.clear()
            self._grid.setRowCount(0)
            self._alert_banner.setText("No validation checks returned.")
            self._alert_banner.setStyleSheet(
                "color: #888; padding: 10px; font-size: 13px;"
            )
            self._alert_banner.setVisible(True)
            return

        # Split trees for saving
        self._trees = _split_swc_by_trees(self._swc_text)

        n_trees = len(tree_results)
        is_multi = n_trees > 1

        # Multi-soma alert banner
        if is_multi:
            soma_ids = [root_id for root_id, _, _ in tree_results]
            self._alert_banner.setText(
                f"⚠  Multiple somas detected — {n_trees} trees found "
                f"(root IDs: {', '.join(str(s) for s in soma_ids)}). "
                "Use 'Save All Trees' to split into individual files."
            )
            self._alert_banner.setStyleSheet(
                "QLabel {"
                "  padding: 12px 16px; margin-bottom: 8px;"
                "  border-radius: 6px; border: 2px solid #c0392b;"
                "  background-color: #fdecea; color: #7b1a12;"
                "  font-weight: 600; font-size: 14px;"
                "}"
            )
            self._alert_banner.setVisible(True)
            self._btn_save_all.setVisible(True)
        else:
            self._alert_banner.setVisible(False)
            self._btn_save_all.setVisible(False)

        # Build grid
        self._populate_grid(check_names, tree_results, is_multi)
        self._btn_export_json.setEnabled(True)

    # ------------------------------------------------- Grid builder
    def _populate_grid(self, check_names, tree_results, is_multi):
        """Fill the QTableWidget with validation results."""
        n_checks = len(check_names)
        n_trees = len(tree_results)

        # Columns: Check Name + one per tree
        col_headers = ["Check"]
        for tidx, (root_id, node_count, _) in enumerate(tree_results):
            if is_multi:
                col_headers.append(f"Tree {tidx + 1}")
            else:
                col_headers.append("Status")

        self._grid.setColumnCount(len(col_headers))
        self._grid.setRowCount(n_checks)
        self._grid.setHorizontalHeaderLabels(col_headers)

        # First column wider
        self._grid.setColumnWidth(0, 260)
        for c in range(1, len(col_headers)):
            self._grid.setColumnWidth(c, 100)

        self._json_rows = []

        for row_idx, (code, friendly) in enumerate(check_names):
            # Check name cell
            name_item = QTableWidgetItem(friendly)
            name_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            name_item.setFont(QFont("", 12))
            self._grid.setItem(row_idx, 0, name_item)

            json_entry = {"check": friendly}

            for col_idx, (root_id, node_count, results) in enumerate(tree_results):
                val = results.get(code, "N/A")

                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)

                if isinstance(val, bool):
                    if val:
                        item.setText("■")
                        item.setForeground(QBrush(QColor("#2ca02c")))
                    else:
                        if friendly in WARNING_CHECKS:
                            item.setText("■")
                            item.setForeground(QBrush(QColor("#ff9900")))
                        else:
                            item.setText("■")
                            item.setForeground(QBrush(QColor("#d62728")))
                else:
                    item.setText("⚠")
                    item.setForeground(QBrush(QColor("#999")))

                item.setFont(QFont("", 16, QFont.Bold))
                self._grid.setItem(row_idx, col_idx + 1, item)

                key = f"tree_{root_id}" if is_multi else "status"
                json_entry[key] = val

            self._json_rows.append(json_entry)

        self._grid.horizontalHeader().setStretchLastSection(True)

    # ------------------------------------------------- Save actions
    def _on_save_all(self):
        """Save all trees as individual SWC files to a folder."""
        if not self._trees:
            return

        folder = QFileDialog.getExistingDirectory(self, "Choose folder to save all trees")
        if not folder:
            return

        try:
            saved = 0
            for tidx, (root_id, sub_text, node_count) in enumerate(self._trees):
                out_path = os.path.join(folder, f"tree_{tidx + 1}_root{root_id}.swc")
                with open(out_path, "w") as f:
                    f.write(sub_text)
                saved += 1

            self._save_status.setText(f"✓ Saved {saved} tree(s) to {folder}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _on_export_json(self):
        """Export validation results as JSON."""
        if not self._json_rows:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation JSON", "validation.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                json.dump(self._json_rows, f, indent=2, default=str)
            self._save_status.setText(f"✓ Exported JSON to {os.path.basename(path)}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
