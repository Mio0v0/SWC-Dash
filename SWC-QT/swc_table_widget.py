"""Reusable widget showing an SWC DataFrame in a table view."""

import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QHeaderView, QTableView, QAbstractItemView, QSizePolicy,
)

from constants import SWC_COLS, color_for_type, label_for_type


# ------------------------------------------------------------------ Model
class _SWCTableModel(QAbstractTableModel):
    """Read-only table model backed by a pandas DataFrame."""

    _HEADERS = SWC_COLS  # id, type, x, y, z, radius, parent

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame(columns=SWC_COLS)

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    # --- required overrides ---
    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._HEADERS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        value = self._df.iloc[row, col]

        if role == Qt.DisplayRole:
            col_name = self._HEADERS[col]
            if col_name in ("id", "type", "parent"):
                return str(int(value))
            if col_name == "radius":
                return f"{value:.2f}"
            if col_name in ("x", "y", "z"):
                return f"{value:.2f}"
            return str(value)

        if role == Qt.BackgroundRole and self._HEADERS[col] == "type":
            hex_color = color_for_type(int(value))
            c = QColor(hex_color)
            c.setAlpha(50)
            return c

        if role == Qt.ToolTipRole and self._HEADERS[col] == "type":
            return label_for_type(int(value))

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._HEADERS[section]
            return str(section + 1)
        return None


# ------------------------------------------------------------------ Widget
class SWCTableWidget(QWidget):
    """Encapsulates a collapsible SWC table panel."""

    EXPANDED_WIDTH = 360
    COLLAPSED_WIDTH = 72

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_collapsed = False
        self._has_data = False

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel("SWC Data")
        self._title.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #444; padding: 4px 0;"
        )
        header_layout.addWidget(self._title)
        header_layout.addStretch()

        self._toggle_btn = QPushButton("<")
        self._toggle_btn.setFixedWidth(28)
        self._toggle_btn.setToolTip("Minimize SWC panel")
        self._toggle_btn.clicked.connect(self._toggle_collapsed)
        header_layout.addWidget(self._toggle_btn)
        layout.addLayout(header_layout)

        self._model = _SWCTableModel()
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setAlternatingRowColors(True)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._view.verticalHeader().setDefaultSectionSize(22)
        self._view.horizontalHeader().setStretchLastSection(True)
        self._view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._view.setStyleSheet(
            "QTableView { font-size: 13px; gridline-color: #ddd; }"
            "QTableView::item:selected { background: #cde8ff; color: #000; }"
        )
        layout.addWidget(self._view)

        self._empty = QLabel("Load an SWC file to view node table.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color: #777; font-size: 12px;")
        layout.addWidget(self._empty)

        self._apply_panel_mode()

    def load_dataframe(self, df: pd.DataFrame):
        self._model.set_dataframe(df)
        self._title.setText(f"SWC Data — {len(df)} rows")
        self._has_data = True

        # Auto-resize columns to content
        for i in range(self._model.columnCount()):
            self._view.resizeColumnToContents(i)

        self._apply_panel_mode()

    def _toggle_collapsed(self):
        self._is_collapsed = not self._is_collapsed
        self._apply_panel_mode()

    def _apply_panel_mode(self):
        if self._is_collapsed:
            self.setMinimumWidth(self.COLLAPSED_WIDTH)
            self.setMaximumWidth(self.COLLAPSED_WIDTH)
            self._title.setVisible(False)
            self._toggle_btn.setText(">")
            self._toggle_btn.setToolTip("Expand SWC panel")
            self._view.setVisible(False)
            self._empty.setVisible(False)
            return

        self.setMinimumWidth(self.EXPANDED_WIDTH)
        self.setMaximumWidth(self.EXPANDED_WIDTH)
        self._title.setVisible(True)
        self._toggle_btn.setText("<")
        self._toggle_btn.setToolTip("Minimize SWC panel")
        self._view.setVisible(self._has_data)
        self._empty.setVisible(not self._has_data)
