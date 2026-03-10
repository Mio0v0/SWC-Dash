"""Validation panel widget for the SWC-QT application."""

import json
import os
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QSizePolicy,
)

from .validation_core import run_per_tree_validation, _split_swc_by_soma_roots


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
    """Validation panel: runs per-tree checks and displays colored results."""

    EXPANDED_WIDTH = 360
    COLLAPSED_WIDTH = 72

    def __init__(self, parent=None, as_panel: bool = True):
        super().__init__(parent)
        self._as_panel = as_panel
        self._swc_text: str | None = None
        self._source_stem = "file"
        self._trees: list = []
        self._json_rows: list = []
        self._has_results = False
        self._has_alert = False
        self._show_save_all = False
        self._is_collapsed = False

        self._build_ui()

    def _build_ui(self):
        if self._as_panel:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel("Validation Checks")
        self._title.setStyleSheet("font-size: 14px; font-weight: 600; color: #333;")
        header.addWidget(self._title)

        header.addStretch()

        self._toggle_btn = QPushButton(">")
        self._toggle_btn.setFixedWidth(28)
        self._toggle_btn.setToolTip("Minimize validation panel")
        self._toggle_btn.clicked.connect(self._toggle_collapsed)
        header.addWidget(self._toggle_btn)
        self._toggle_btn.setVisible(self._as_panel)

        layout.addLayout(header)

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

        # Compact (collapsed) color-only grid
        self._compact_grid = QTableWidget()
        self._compact_grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self._compact_grid.setSelectionMode(QTableWidget.NoSelection)
        self._compact_grid.verticalHeader().setVisible(False)
        self._compact_grid.horizontalHeader().setVisible(False)
        self._compact_grid.setShowGrid(False)
        self._compact_grid.setVisible(False)
        self._compact_grid.setStyleSheet(
            "QTableWidget { font-size: 13px; background: #fff; }"
        )
        layout.addWidget(self._compact_grid, stretch=1)

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
        self._apply_panel_mode()

    # --------------------------------------------------------- Public API
    def load_swc(self, df: pd.DataFrame, filename: str):
        """Run validation on the loaded SWC data."""
        self._source_stem = Path(filename or "file").stem or "file"

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
            self._has_results = False
            self._has_alert = True
            self._show_save_all = False
            self._trees = []
            self._json_rows = []

            self._grid.clear()
            self._grid.setRowCount(0)
            self._compact_grid.clear()
            self._compact_grid.setColumnCount(1)
            self._compact_grid.setRowCount(0)
            self._alert_banner.setText("No validation checks returned.")
            self._alert_banner.setStyleSheet(
                "color: #888; padding: 10px; font-size: 13px;"
            )
            self._btn_save_all.setEnabled(False)
            self._btn_export_json.setEnabled(False)
            self._apply_panel_mode()
            return

        # Split trees for saving
        self._trees = _split_swc_by_soma_roots(self._swc_text)
        self._has_results = True

        n_trees = len(tree_results)
        is_multi = n_trees > 1

        # Multi-soma alert banner (strict soma-root criterion)
        if len(self._trees) > 1:
            self._has_alert = True
            self._show_save_all = True
            soma_ids = [root_id for root_id, _, _ in self._trees]
            self._alert_banner.setText(
                f"⚠  Multiple somas detected — {len(self._trees)} cells found "
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
        else:
            self._has_alert = False
            self._show_save_all = False
            self._alert_banner.setVisible(False)
            self._alert_banner.clear()

        # Build grid
        self._populate_grid(check_names, tree_results, is_multi)
        self._populate_compact_grid(check_names, tree_results)
        self._btn_save_all.setEnabled(self._show_save_all)
        self._btn_export_json.setEnabled(True)
        self._apply_panel_mode()

    def _toggle_collapsed(self):
        self._is_collapsed = not self._is_collapsed
        self._apply_panel_mode()

    def _apply_panel_mode(self):
        if not self._as_panel:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self._title.setVisible(True)
            self._toggle_btn.setVisible(False)
            self._alert_banner.setVisible(self._has_alert)
            self._grid.setVisible(self._has_results)
            self._compact_grid.setVisible(False)
            self._btn_save_all.setVisible(self._show_save_all)
            self._btn_export_json.setVisible(True)
            self._btn_export_json.setEnabled(self._has_results)
            self._save_status.setVisible(True)
            return

        if self._is_collapsed:
            self.setMinimumWidth(self.COLLAPSED_WIDTH)
            self.setMaximumWidth(self.COLLAPSED_WIDTH)
            self._title.setVisible(False)
            self._toggle_btn.setText("<")
            self._toggle_btn.setToolTip("Expand validation panel")
            self._alert_banner.setVisible(False)
            self._grid.setVisible(False)
            self._compact_grid.setVisible(self._has_results)
            self._btn_save_all.setVisible(False)
            self._btn_export_json.setVisible(False)
            self._save_status.setVisible(False)
            return

        self.setMinimumWidth(self.EXPANDED_WIDTH)
        self.setMaximumWidth(self.EXPANDED_WIDTH)
        self._title.setVisible(True)
        self._toggle_btn.setText(">")
        self._toggle_btn.setToolTip("Minimize validation panel")
        self._alert_banner.setVisible(self._has_alert)
        self._grid.setVisible(self._has_results)
        self._compact_grid.setVisible(False)
        self._btn_save_all.setVisible(self._show_save_all)
        self._btn_export_json.setVisible(True)
        self._btn_export_json.setEnabled(self._has_results)
        self._save_status.setVisible(True)

    # ------------------------------------------------- Grid builder
    def _populate_grid(self, check_names, tree_results, is_multi):
        """Fill the QTableWidget with validation results."""
        n_checks = len(check_names)

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
                marker, color = self._marker_and_color(val, friendly)
                item.setText(marker)
                item.setForeground(QBrush(QColor(color)))

                item.setFont(QFont("", 16, QFont.Bold))
                self._grid.setItem(row_idx, col_idx + 1, item)

                key = f"tree_{root_id}" if is_multi else "status"
                json_entry[key] = val

            self._json_rows.append(json_entry)

        self._grid.horizontalHeader().setStretchLastSection(True)

    def _populate_compact_grid(self, check_names, tree_results):
        """Build compact single-column color view used when panel is minimized."""
        self._compact_grid.clear()
        self._compact_grid.setColumnCount(1)
        self._compact_grid.setRowCount(len(check_names))
        self._compact_grid.setColumnWidth(0, self.COLLAPSED_WIDTH - 24)

        for row_idx, (code, friendly) in enumerate(check_names):
            status_values = [results.get(code, "N/A") for _, _, results in tree_results]
            marker, color = self._compact_marker_and_color(status_values, friendly)

            item = QTableWidgetItem(marker)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont("", 14, QFont.Bold))
            item.setForeground(QBrush(QColor(color)))
            item.setToolTip(friendly)
            self._compact_grid.setItem(row_idx, 0, item)

        self._compact_grid.horizontalHeader().setStretchLastSection(True)

    def _marker_and_color(self, value, friendly):
        if isinstance(value, bool):
            if value:
                return "■", "#2ca02c"
            if friendly in WARNING_CHECKS:
                return "■", "#ff9900"
            return "■", "#d62728"
        return "⚠", "#999"

    def _compact_marker_and_color(self, values, friendly):
        bool_values = [v for v in values if isinstance(v, bool)]
        if len(bool_values) != len(values):
            return "⚠", "#999"
        if all(bool_values):
            return "■", "#2ca02c"
        if friendly in WARNING_CHECKS:
            return "■", "#ff9900"
        return "■", "#d62728"

    # ------------------------------------------------- Save actions
    def _on_save_all(self):
        """Save all trees as individual SWC files to a folder."""
        if not self._trees:
            self._save_status.setText("No trees available to save.")
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

            self._save_status.setText(f"✓ Saved {saved} tree(s) to {folder}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:
            self._save_status.setText(f"Save error: {e}")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")

    def _on_export_json(self):
        """Export validation results as JSON."""
        if not self._json_rows:
            self._save_status.setText("No validation results to export.")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation JSON", "validation.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            self._save_status.setText("Export validation JSON cancelled.")
            self._save_status.setStyleSheet("color: #777; font-size: 12px;")
            return

        try:
            with open(path, "w") as f:
                json.dump(self._json_rows, f, indent=2, default=str)
            self._save_status.setText(f"✓ Exported JSON to {os.path.basename(path)}")
            self._save_status.setStyleSheet("color: #2ca02c; font-size: 12px;")
        except Exception as e:
            self._save_status.setText(f"Export error: {e}")
            self._save_status.setStyleSheet("color: #d62728; font-size: 12px;")
