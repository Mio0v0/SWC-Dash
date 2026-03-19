"""Reusable widget showing an SWC DataFrame in a table view."""

import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QHeaderView, QTableView, QAbstractItemView, QSizePolicy,
)

from .constants import SWC_COLS, color_for_type, label_for_type


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

    EXPANDED_MIN_WIDTH = 80
    COLLAPSED_WIDTH = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_collapsed = False
        self._has_data = False
        self._filename = "No SWC loaded"

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(4)

        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self._toggle_btn = QPushButton("▾")
        self._toggle_btn.setFixedWidth(20)
        self._toggle_btn.setToolTip("Collapse/expand SWC rows")
        self._toggle_btn.setStyleSheet(
            "QPushButton { border: none; padding: 0px; color: #444; font-weight: 700; }"
            "QPushButton:hover { color: #111; }"
        )
        self._toggle_btn.clicked.connect(self._toggle_collapsed)
        header_layout.addWidget(self._toggle_btn)

        self._title = QLabel("SWC Data")
        self._title.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #444; padding: 4px 0;"
        )
        self._title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._title.setMinimumWidth(0)
        self._title.mousePressEvent = lambda _e: self._toggle_collapsed()
        header_layout.addWidget(self._title)
        header_layout.addStretch()
        layout.addWidget(header_container, stretch=0)

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

        self._empty = QLabel("Load an SWC file to view node table.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color: #777; font-size: 12px;")

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)
        body_layout.addWidget(self._view, stretch=1)
        body_layout.addWidget(self._empty, stretch=1)
        layout.addWidget(self._body, stretch=1)

        # Used only in collapsed mode to keep filename row pinned to top.
        self._collapsed_spacer = QWidget()
        self._collapsed_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._collapsed_spacer.setVisible(False)
        layout.addWidget(self._collapsed_spacer, stretch=1)

        self._apply_panel_mode()

    def load_dataframe(self, df: pd.DataFrame, filename: str = ""):
        self._model.set_dataframe(df)
        self._filename = filename or self._filename
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
            self.setMaximumWidth(16777215)
            self._title.setVisible(True)
            self._title.setText(f"{self._filename}")
            self._toggle_btn.setText("▸")
            self._toggle_btn.setToolTip("Expand to show SWC rows")
            self._body.setVisible(False)
            self._collapsed_spacer.setVisible(True)
            return

        self.setMinimumWidth(self.EXPANDED_MIN_WIDTH)
        self.setMaximumWidth(16777215)
        self._title.setVisible(True)
        row_info = f"{self._model.rowCount()} rows" if self._has_data else "no rows"
        full_title = f"{self._filename}  ({row_info})"
        self._title.setToolTip(full_title)
        self._title.setText(full_title)
        self._toggle_btn.setText("▾")
        self._toggle_btn.setToolTip("Collapse to filename only")
        self._body.setVisible(True)
        self._collapsed_spacer.setVisible(False)
        self._view.setVisible(self._has_data)
        self._empty.setVisible(not self._has_data)
