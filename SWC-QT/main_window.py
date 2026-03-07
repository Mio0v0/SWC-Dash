"""Main window with tabbed interface and SWC file loading."""

import os

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QMessageBox, QStatusBar,
)

from swc_table_widget import SWCTableWidget
from validation_tab import ValidationTabWidget
from editor_tab import EditorTab
from constants import SWC_COLS


class SWCMainWindow(QMainWindow):
    """Top-level window: menu bar, tabs, status bar, drag-and-drop."""

    swc_loaded = Signal(pd.DataFrame, str)  # df, filename

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWC-QT — Neuron Editor")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self._df: pd.DataFrame | None = None
        self._filename: str = ""

        self._build_menu()
        self._build_ui()
        self._build_status_bar()

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        open_action = QAction("&Open SWC…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_ui(self):
        self._tabs = QTabWidget()

        # --- Tab 1: Format Validation ---
        self._validation_tab = ValidationTabWidget(as_panel=False)

        # --- Tab 2: Dendrogram Editor + 3D View ---
        self._editor_tab = EditorTab()
        self._editor_tab.df_changed.connect(self._on_editor_df_changed)

        # --- Tab 3: Radii Cleaner ---
        self._radii_tab = QWidget()
        radii_layout = QVBoxLayout(self._radii_tab)
        radii_layout.addWidget(QLabel("Radii Cleaner — Coming in Milestone 5."))
        radii_layout.addStretch()

        self._tabs.addTab(self._validation_tab, "Format Validation")
        self._tabs.addTab(self._editor_tab, "Dendrogram Editor")
        self._tabs.addTab(self._radii_tab, "Radii Cleaner")

        # Shared validation panel on the right side for non-validation tabs
        self._validation_panel = ValidationTabWidget(as_panel=True)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.setCurrentIndex(0)

        # --- Central layout with drop zone + tabs ---
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 0)

        # Drop zone / file info banner (clickable)
        self._file_banner = QLabel(
            "📂  Drag & drop an SWC file here, or click to browse"
        )
        self._file_banner.setAlignment(Qt.AlignCenter)
        self._file_banner.setCursor(Qt.PointingHandCursor)
        self._file_banner.setStyleSheet(
            "QLabel {"
            "  border: 2px dashed #999; border-radius: 8px;"
            "  padding: 18px; font-size: 15px; color: #555;"
            "  background: #fafafa;"
            "}"
            "QLabel:hover {"
            "  border-color: #4a9; background: #f0faf5;"
            "}"
        )
        self._file_banner.setMinimumHeight(60)
        self._file_banner.mousePressEvent = lambda e: self._on_open()
        main_layout.addWidget(self._file_banner)

        # Shared SWC table panel on the left side for all tabs
        self._table_widget = SWCTableWidget()

        tab_row = QWidget()
        tab_row_layout = QHBoxLayout(tab_row)
        tab_row_layout.setContentsMargins(0, 0, 0, 0)
        tab_row_layout.setSpacing(8)
        tab_row_layout.addWidget(self._table_widget)
        tab_row_layout.addWidget(self._tabs, stretch=1)
        tab_row_layout.addWidget(self._validation_panel)
        main_layout.addWidget(tab_row, stretch=3)

        self.setCentralWidget(central)
        self._on_tab_changed(self._tabs.currentIndex())

    def _build_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open an SWC file to start.")

    # --------------------------------------------------------- File loading
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SWC file", "",
            "SWC Files (*.swc);;All Files (*)"
        )
        if path:
            self._load_swc(path)

    def _load_swc(self, path: str):
        """Parse SWC file and populate the UI."""
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                comment="#",
                names=SWC_COLS,
            )
            # Basic validation
            if df.empty:
                QMessageBox.warning(self, "Empty file", f"{path} contains no data rows.")
                return

            # Ensure integer columns
            for col in ("id", "type", "parent"):
                df[col] = df[col].astype(int)

            self._df = df
            self._filename = os.path.basename(path)

            # Update UI
            n_roots = int((df["parent"] == -1).sum())
            n_soma = int((df["type"] == 1).sum())
            self._file_banner.setText(
                f"📄 {self._filename}  —  {len(df)} nodes, {n_roots} root(s), {n_soma} soma node(s)"
            )
            self._file_banner.setStyleSheet(
                "QLabel {"
                "  border: 2px solid #4a9; border-radius: 8px;"
                "  padding: 12px; font-size: 14px; color: #333;"
                "  background: #e8f8f0;"
                "}"
            )

            self._table_widget.load_dataframe(df)

            # Run validation
            self._validation_tab.load_swc(df, self._filename)
            self._validation_panel.load_swc(df, self._filename)

            # Load dendrogram + 3D view
            self._editor_tab.load_swc(df, self._filename)

            self._status.showMessage(
                f"Loaded {self._filename}: {len(df)} nodes, {n_roots} root(s), {n_soma} soma(s)",
                5000,
            )

            # Emit signal for other components
            self.swc_loaded.emit(df, self._filename)

        except Exception as e:
            QMessageBox.critical(self, "Error loading SWC", str(e))

    # --------------------------------------------------- Drag & drop
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(".swc") for u in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".swc"):
                self._load_swc(path)
                break

    def _on_tab_changed(self, tab_index: int):
        """Show side validation panel for non-validation tabs only."""
        self._validation_panel.setVisible(tab_index != 0)

    def _on_editor_df_changed(self, df: pd.DataFrame):
        """Refresh table + validation when dendrogram edits modify SWC data."""
        self._df = df.copy()
        self._table_widget.load_dataframe(self._df)
        self._validation_tab.load_swc(self._df, self._filename)
        self._validation_panel.load_swc(self._df, self._filename)
