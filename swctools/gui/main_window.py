"""Main window layout for SWC-QT with tabbed Home/Tools top bar."""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .batch_tab import BatchTabWidget
from .auto_typing_guide import AutoTypingGuideWidget
from .constants import SWC_COLS
from .editor_tab import EditorTab
from .neuron_3d_widget import Neuron3DWidget
from .swc_table_widget import SWCTableWidget
from .validation_tab import ValidationPrecheckWidget, ValidationTabWidget
from swctools.core.reporting import (
    format_morphology_session_log_text,
    morphology_session_log_path,
    write_text_report,
)
from .report_popup import ReportPopupDialog


class SWCMainWindow(QMainWindow):
    """Top-level app window with tabbed top bar, workspace, side panels, and log."""

    swc_loaded = Signal(pd.DataFrame, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWC-QT — Neuron Editor")
        self.resize(1480, 920)
        self.setAcceptDrops(False)

        self._df: pd.DataFrame | None = None
        self._filename: str = ""
        self._file_path: str = ""
        self._recent_paths: list[str] = []
        self._active_tool: str = ""
        self._morph_session_changes: list[dict] = []
        self._morph_session_started_at: str = ""
        self._morph_session_source_path: str = ""
        self._morph_seq: int = 0

        self._build_ui()
        self._build_status_bar()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Use an in-window top strip instead of the OS menu bar.
        self.menuBar().setVisible(False)

        # ---------------- Top combined bar: Home / Tools ----------------
        self._top_tabs = self._build_top_tabs()

        # ---------------- Center workspace ----------------
        self._editor_tab = EditorTab()
        self._editor_tab.df_changed.connect(self._on_editor_df_changed)
        self._dendro_controls = self._editor_tab.take_dendrogram_controls_panel()

        # ---------------- Right side: dockable Data Explorer + Control Center ----------------
        self._data_tabs = QTabWidget()
        self._table_widget = SWCTableWidget()

        self._info_label = QLabel("No SWC file loaded.")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size: 13px; color: #444; padding: 8px;")
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(6, 6, 6, 6)
        info_layout.addWidget(self._info_label)
        info_layout.addStretch()

        self._segment_label = QLabel(
            "Segment Info\n\nLoad an SWC file and select nodes in dendrogram mode."
        )
        self._segment_label.setWordWrap(True)
        self._segment_label.setStyleSheet("font-size: 13px; color: #555; padding: 8px;")
        seg_panel = QWidget()
        seg_layout = QVBoxLayout(seg_panel)
        seg_layout.setContentsMargins(6, 6, 6, 6)
        seg_layout.addWidget(self._segment_label)
        seg_layout.addStretch()

        self._edit_log_text = QPlainTextEdit()
        self._edit_log_text.setReadOnly(True)
        self._edit_log_text.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #333;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._edit_log_text.setPlainText("No morphology edits recorded for this session yet.")

        self._data_tabs.addTab(self._table_widget, "SWC File")
        self._data_tabs.addTab(info_panel, "Node Info")
        self._data_tabs.addTab(seg_panel, "Segment Info")
        self._data_tabs.addTab(self._edit_log_text, "Edit Log")

        self._control_tabs = QTabWidget()
        self._batch_tab = BatchTabWidget()
        self._batch_tab.batch_validation_ready.connect(self._on_batch_validation_ready)
        self._batch_tab.precheck_requested.connect(self._on_precheck_requested)
        self._validation_tab = ValidationTabWidget(as_panel=False)
        self._validation_tab.precheck_requested.connect(self._on_precheck_requested)
        self._validation_precheck = ValidationPrecheckWidget()
        self._auto_typing_guide = AutoTypingGuideWidget()
        self._viz_control = self._build_visualization_control_panel()
        self._control_tabs.currentChanged.connect(self._on_control_tab_changed)
        self._set_control_tabs_for_feature("")

        self._data_dock = QDockWidget("Data Explorer", self)
        self._data_dock.setObjectName("DataExplorerDock")
        self._data_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._data_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._data_dock.setMinimumWidth(120)
        self._data_dock.setWidget(self._data_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self._data_dock)

        self._control_dock = QDockWidget("Control Center", self)
        self._control_dock.setObjectName("ControlCenterDock")
        self._control_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._control_dock.setMinimumWidth(140)
        self._control_dock.setWidget(self._control_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self._control_dock)
        self.splitDockWidget(self._data_dock, self._control_dock, Qt.Vertical)

        self._precheck_dock = QDockWidget("Rule Guide", self)
        self._precheck_dock.setObjectName("ValidationPrecheckDock")
        self._precheck_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._precheck_dock.setAllowedAreas(
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self._precheck_dock.setWidget(self._validation_precheck)
        self.addDockWidget(Qt.TopDockWidgetArea, self._precheck_dock)
        self._precheck_dock.hide()

        self._auto_guide_dock = QDockWidget("Auto Typing Guide", self)
        self._auto_guide_dock.setObjectName("AutoTypingGuideDock")
        self._auto_guide_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._auto_guide_dock.setAllowedAreas(
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self._auto_guide_dock.setWidget(self._auto_typing_guide)
        self.addDockWidget(Qt.TopDockWidgetArea, self._auto_guide_dock)
        self._auto_guide_dock.hide()

        # ---------------- Bottom log ----------------
        self._log_console = QPlainTextEdit()
        self._log_console.setReadOnly(True)
        self._log_console.setMinimumHeight(110)
        self._log_console.setMaximumHeight(240)
        self._log_console.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #111; border: 1px solid #333; color: #f1f1f1;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        self._batch_tab.log_message.connect(lambda msg: self._append_log(msg, "BATCH"))

        # ---------------- Root layout ----------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 0)
        root.setSpacing(6)
        root.addWidget(self._top_tabs, stretch=0)
        root.addWidget(self._editor_tab, stretch=1)
        root.addWidget(self._log_console, stretch=0)
        self.setCentralWidget(central)
        # Make the dock/canvas boundary easier to grab and resize.
        self.setStyleSheet(
            "QMainWindow::separator {"
            "  background: #bdbdbd;"
            "  width: 8px;"
            "  height: 8px;"
            "}"
        )

        self._reset_layout()
        self._append_log("UI initialized. Open an SWC file from File menu.", "INFO")

    def _build_top_tabs(self) -> QTabWidget:
        top_bg = "#f2f2f2"
        top_fg = "#222222"
        top_border = "#c7c7c7"
        top_hover = "#e9e9e9"
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setElideMode(Qt.ElideRight)
        tabs.setStyleSheet(
            "QTabWidget::pane {"
            f"  border: 1px solid {top_border}; background: {top_bg};"
            "}"
            "QTabBar::tab {"
            f"  padding: 6px 16px; font-weight: 600; color: {top_fg};"
            f"  background: {top_bg}; border: 1px solid {top_border};"
            "  border-bottom: none; margin-right: 1px;"
            "}"
            "QTabBar::tab:selected {"
            f"  background: {top_bg};"
            "}"
            "QTabBar::tab:hover {"
            f"  background: {top_hover};"
            "}"
        )

        # Home tab: classic dropdown menus.
        home_page = QWidget()
        home_page.setStyleSheet(f"background: {top_bg};")
        home_layout = QVBoxLayout(home_page)
        home_layout.setContentsMargins(0, 0, 0, 0)
        home_layout.setSpacing(0)

        self._home_menu_bar = QMenuBar(home_page)
        self._home_menu_bar.setNativeMenuBar(False)
        self._home_menu_bar.setStyleSheet(
            "QMenuBar {"
            f"  background: {top_bg}; border-bottom: 1px solid {top_border}; color: {top_fg};"
            "}"
            "QMenuBar::item {"
            "  padding: 6px 12px; background: transparent;"
            "}"
            "QMenuBar::item:selected {"
            f"  background: {top_hover};"
            "}"
            "QMenu {"
            f"  background: {top_bg}; color: {top_fg}; border: 1px solid {top_border};"
            "}"
            "QMenu::item {"
            "  padding: 6px 20px 6px 24px; background: transparent;"
            "}"
            "QMenu::item:selected {"
            f"  background: {top_hover};"
            "}"
        )
        self._populate_home_menus(self._home_menu_bar)
        home_layout.addWidget(self._home_menu_bar, stretch=0)

        # Tools tab: feature buttons.
        tools_page = QWidget()
        tools_page.setStyleSheet(
            "QWidget {"
            f"  background: {top_bg}; color: {top_fg};"
            "}"
            "QPushButton {"
            f"  background: {top_bg}; color: {top_fg}; border: 1px solid {top_border};"
            "  border-radius: 4px; padding: 6px 10px;"
            "}"
            "QPushButton:hover {"
            f"  background: {top_hover};"
            "}"
            "QPushButton:pressed {"
            f"  background: {top_hover};"
            "}"
        )
        tools_layout = QHBoxLayout(tools_page)
        tools_layout.setContentsMargins(8, 6, 8, 6)
        tools_layout.setSpacing(8)

        b_batch = QPushButton("Batch Processing")
        b_batch.clicked.connect(lambda: self._activate_feature("batch"))
        tools_layout.addWidget(b_batch)

        b_validation = QPushButton("Validation")
        b_validation.clicked.connect(lambda: self._activate_feature("validation"))
        tools_layout.addWidget(b_validation)

        b_visual = QPushButton("Visualization")
        b_visual.clicked.connect(lambda: self._activate_feature("visualization"))
        tools_layout.addWidget(b_visual)

        b_morph = QPushButton("Morphology Editing")
        b_morph.clicked.connect(lambda: self._activate_feature("morphology_editing"))
        tools_layout.addWidget(b_morph)

        b_atlas = QPushButton("Atlas Registration")
        b_atlas.clicked.connect(lambda: self._activate_feature("atlas_registration"))
        tools_layout.addWidget(b_atlas)

        b_analysis = QPushButton("Analysis")
        b_analysis.clicked.connect(lambda: self._activate_feature("analysis"))
        tools_layout.addWidget(b_analysis)

        tools_layout.addStretch()
        self._feature_label = QLabel("Active feature: None")
        self._feature_label.setStyleSheet("font-size: 12px; color: #555;")
        tools_layout.addWidget(self._feature_label)
        self._current_file_label = QLabel("Current file: (none)")
        self._current_file_label.setStyleSheet("font-size: 12px; color: #555;")
        self._current_file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._current_file_label.setMinimumWidth(0)
        self._current_file_label.setMaximumWidth(320)
        tools_layout.addWidget(self._current_file_label)
        self._set_current_file_label_text("")

        tabs.addTab(home_page, "Home")
        tabs.addTab(tools_page, "Tools")
        return tabs

    def _populate_home_menus(self, menu: QMenuBar):
        menu.clear()

        # File
        file_menu = menu.addMenu("File")
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save As", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        self._recent_menu = file_menu.addMenu("Recent Files")
        self._recent_menu.addAction("(empty)").setEnabled(False)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit
        edit_menu = menu.addMenu("Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo_edit)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self._redo_edit)
        edit_menu.addAction(redo_action)

        pref_action = QAction("Preferences", self)
        pref_action.triggered.connect(
            lambda: self._append_log("Preferences dialog is not implemented yet.", "INFO")
        )
        edit_menu.addAction(pref_action)

        # View
        view_menu = menu.addMenu("View")
        show_log_action = QAction("Show/Hide Log", self)
        show_log_action.triggered.connect(
            lambda: self._toggle_log_panel(not self._log_console.isVisible())
        )
        view_menu.addAction(show_log_action)

        view_menu.addSeparator()
        cam_iso_action = QAction("Camera Isometric", self)
        cam_iso_action.triggered.connect(lambda: self._set_camera("iso"))
        view_menu.addAction(cam_iso_action)
        cam_top_action = QAction("Camera Top", self)
        cam_top_action.triggered.connect(lambda: self._set_camera("top"))
        view_menu.addAction(cam_top_action)
        cam_front_action = QAction("Camera Front", self)
        cam_front_action.triggered.connect(lambda: self._set_camera("front"))
        view_menu.addAction(cam_front_action)
        cam_side_action = QAction("Camera Side", self)
        cam_side_action.triggered.connect(lambda: self._set_camera("side"))
        view_menu.addAction(cam_side_action)

        # Window
        window_menu = menu.addMenu("Window")
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self._reset_layout)
        window_menu.addAction(reset_layout_action)

        show_data_action = QAction("Show/Hide Data Explorer", self)
        show_data_action.triggered.connect(
            lambda: self._toggle_data_panel(not self._data_dock.isVisible())
        )
        window_menu.addAction(show_data_action)

        show_control_action = QAction("Show/Hide Control Center", self)
        show_control_action.triggered.connect(
            lambda: self._toggle_control_panel(not self._control_dock.isVisible())
        )
        window_menu.addAction(show_control_action)

        show_precheck_action = QAction("Show/Hide Rule Guide", self)
        show_precheck_action.triggered.connect(
            lambda: self._toggle_precheck_panel(not self._precheck_dock.isVisible())
        )
        window_menu.addAction(show_precheck_action)

        show_auto_guide_action = QAction("Show/Hide Auto Typing Guide", self)
        show_auto_guide_action.triggered.connect(
            lambda: self._toggle_auto_typing_guide_panel(not self._auto_guide_dock.isVisible())
        )
        window_menu.addAction(show_auto_guide_action)

        # Help
        help_menu = menu.addMenu("Help")
        quick_action = QAction("Quick Manual", self)
        quick_action.triggered.connect(self._show_quick_manual)
        help_menu.addAction(quick_action)
        short_action = QAction("Shortcuts", self)
        short_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(short_action)
        about_action = QAction("About", self)
        about_action.triggered.connect(
            lambda: self._append_log("SWC-QT — neuron visualization and editing workspace.", "INFO")
        )
        help_menu.addAction(about_action)

    def _build_visualization_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        row1 = QHBoxLayout()
        b_iso = QPushButton("Isometric")
        b_iso.clicked.connect(lambda: self._set_camera("iso"))
        row1.addWidget(b_iso)
        b_top = QPushButton("Top")
        b_top.clicked.connect(lambda: self._set_camera("top"))
        row1.addWidget(b_top)
        b_front = QPushButton("Front")
        b_front.clicked.connect(lambda: self._set_camera("front"))
        row1.addWidget(b_front)
        b_side = QPushButton("Side")
        b_side.clicked.connect(lambda: self._set_camera("side"))
        row1.addWidget(b_side)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        b_reset = QPushButton("Reset Camera")
        b_reset.clicked.connect(self._reset_camera)
        row2.addWidget(b_reset)
        row2.addWidget(QLabel("Render mode:"))
        self._render_combo = QComboBox()
        self._render_combo.addItem("Lines", Neuron3DWidget.MODE_LINES)
        self._render_combo.addItem("Spheres", Neuron3DWidget.MODE_SPHERES)
        self._render_combo.addItem("Frustum", Neuron3DWidget.MODE_FRUSTUM)
        self._render_combo.currentIndexChanged.connect(self._on_render_mode_changed)
        row2.addWidget(self._render_combo)
        layout.addLayout(row2)

        hint = QLabel(
            "Visualization mode shows:\n"
            "- one 3D view on top\n"
            "- three 2D projections (top/front/side) below"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 12px; color: #555;")
        layout.addWidget(hint)
        layout.addStretch()
        return panel

    def _build_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open an SWC file to start.")

    def _set_control_tabs_for_feature(self, feature: str):
        """Show only control tabs relevant to the active feature."""
        key = (feature or "").strip().lower()
        while self._control_tabs.count() > 0:
            self._control_tabs.removeTab(0)

        if key in ("", "none"):
            return

        if key == "batch":
            self._control_tabs.addTab(self._batch_tab.split_tab_widget(), "Split")
            self._control_tabs.addTab(self._batch_tab.validation_tab_widget(), "Validation")
            self._control_tabs.addTab(self._batch_tab.auto_tab_widget(), "Auto Label")
            self._control_tabs.addTab(self._batch_tab.radii_tab_widget(), "Radii Cleaning")
            self._control_tabs.setCurrentIndex(0)
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            return

        if key == "validation":
            self._control_tabs.addTab(self._validation_tab, "Validation")
            self._control_tabs.setCurrentWidget(self._validation_tab)
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            return

        if key in ("morphology_editing", "dendrogram"):
            self._control_tabs.addTab(self._dendro_controls, "Label Editing")
            self._control_tabs.setCurrentWidget(self._dendro_controls)
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            return

        if key in ("atlas_registration", "analysis"):
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            return

        # default: visualization
        self._control_tabs.addTab(self._viz_control, "View Controls")
        self._control_tabs.setCurrentWidget(self._viz_control)
        self._on_control_tab_changed(self._control_tabs.currentIndex())

    # --------------------------------------------------------- Feature routing
    def _activate_feature(self, name: str):
        key = (name or "").strip().lower()
        if key == "batch":
            self._active_tool = "batch"
            self._editor_tab.set_mode(EditorTab.MODE_EMPTY)
            self._set_control_tabs_for_feature("batch")
            self._control_dock.show()
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            self._feature_label.setText("Active feature: Batch Processing")
            self._append_log("Feature switched: Batch Processing", "INFO")
            return
        if key == "validation":
            self._active_tool = "validation"
            self._editor_tab.set_mode(EditorTab.MODE_CANVAS)
            self._set_control_tabs_for_feature("validation")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._feature_label.setText("Active feature: Validation")
            self._append_log("Feature switched: Validation", "INFO")
            return
        if key == "visualization":
            self._active_tool = "visualization"
            self._editor_tab.set_mode(EditorTab.MODE_VIS)
            self._set_control_tabs_for_feature("visualization")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._feature_label.setText("Active feature: Visualization")
            self._append_log("Feature switched: Visualization", "INFO")
            return
        if key in ("morphology_editing", "dendrogram"):
            self._active_tool = "morphology_editing"
            self._editor_tab.set_mode(EditorTab.MODE_DENDRO)
            self._set_control_tabs_for_feature("morphology_editing")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._feature_label.setText("Active feature: Morphology Editing")
            self._append_log("Feature switched: Morphology Editing", "INFO")
            return
        if key == "atlas_registration":
            self._active_tool = "atlas_registration"
            self._editor_tab.set_mode(EditorTab.MODE_CANVAS)
            self._set_control_tabs_for_feature("atlas_registration")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._feature_label.setText("Active feature: Atlas Registration")
            self._append_log("Feature switched: Atlas Registration (placeholder)", "INFO")
            return
        if key == "analysis":
            self._active_tool = "analysis"
            self._editor_tab.set_mode(EditorTab.MODE_CANVAS)
            self._set_control_tabs_for_feature("analysis")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._feature_label.setText("Active feature: Analysis")
            self._append_log("Feature switched: Analysis (placeholder)", "INFO")
            return
        self._active_tool = ""
        self._editor_tab.set_mode(EditorTab.MODE_CANVAS)
        self._set_control_tabs_for_feature("")
        self._precheck_dock.hide()
        self._auto_guide_dock.hide()
        self._feature_label.setText("Active feature: None")
        self._append_log("No active tool selected.", "INFO")

    def _set_camera(self, preset: str):
        self._editor_tab.set_camera_view(preset)
        self._append_log(f"Camera preset: {preset}", "INFO")

    def _reset_camera(self):
        self._editor_tab.reset_camera()
        self._append_log("Camera reset.", "INFO")

    def _on_render_mode_changed(self, index: int):
        mode = self._render_combo.currentData()
        if mode is None:
            return
        self._editor_tab.set_render_mode(int(mode))
        self._append_log(f"Render mode set to {self._render_combo.currentText()}.", "INFO")

    # --------------------------------------------------------- File loading
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SWC file", "", "SWC Files (*.swc);;All Files (*)"
        )
        if path:
            self._load_swc(path)

    def _load_swc(self, path: str):
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", names=SWC_COLS)
            if df.empty:
                self._append_log(f"Empty file: {path} contains no data rows.", "WARN")
                return

            for col in ("id", "type", "parent"):
                df[col] = df[col].astype(int)

            if self._morph_session_changes:
                self._finalize_morphology_session(show_popup=True)

            self._df = df
            self._filename = os.path.basename(path)
            self._file_path = path
            self._set_current_file_label_text(self._filename)

            n_roots = int((df["parent"] == -1).sum())
            n_soma = int((df["type"] == 1).sum())

            self._table_widget.load_dataframe(df, self._filename)
            self._update_info_label(df, n_roots, n_soma)
            self._update_recent_files(path)

            self._validation_tab.load_swc(df, self._filename, file_path=path, auto_run=True)
            self._editor_tab.load_swc(df, self._filename)
            if self._active_tool:
                self._activate_feature(self._active_tool)
            else:
                self._editor_tab.set_mode(EditorTab.MODE_CANVAS)
                self._set_control_tabs_for_feature("")
                self._feature_label.setText("Active feature: None")

            self._status.showMessage(
                f"Loaded {self._filename}: {len(df)} nodes, {n_roots} root(s), {n_soma} soma(s)",
                5000,
            )
            self._append_log(
                f"Loaded {self._filename}: nodes={len(df)}, roots={n_roots}, soma={n_soma}",
                "INFO",
            )
            self._start_morphology_session(path)
            self.swc_loaded.emit(df, self._filename)
        except Exception as e:
            self._append_log(f"Error loading SWC: {e}", "ERROR")

    def _on_save(self):
        if self._df is None or self._df.empty:
            self._append_log("No SWC loaded. Nothing to save.", "WARN")
            return
        if not self._file_path:
            self._on_save_as()
            return
        self._write_swc_file(self._file_path, self._df)
        self._append_log(f"Saved {self._file_path}", "INFO")

    def _on_save_as(self):
        if self._df is None or self._df.empty:
            self._append_log("No SWC loaded. Nothing to save.", "WARN")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SWC As", self._filename or "edited.swc", "SWC Files (*.swc);;All Files (*)"
        )
        if not path:
            self._append_log("Save As cancelled.", "INFO")
            return
        self._write_swc_file(path, self._df)
        self._file_path = path
        self._filename = os.path.basename(path)
        self._morph_session_source_path = path
        self._set_current_file_label_text(self._filename)
        self._append_log(f"Saved As {path}", "INFO")
        self._update_recent_files(path)

    def _on_export(self):
        if self._df is None or self._df.empty:
            self._append_log("No SWC loaded. Nothing to export.", "WARN")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SWC", f"export_{self._filename or 'swc'}.swc", "SWC Files (*.swc)"
        )
        if not path:
            self._append_log("Export cancelled.", "INFO")
            return
        self._write_swc_file(path, self._df)
        self._append_log(f"Exported {path}", "INFO")

    def _write_swc_file(self, path: str, df: pd.DataFrame):
        lines = ["# id type x y z radius parent"]
        for _, row in df.iterrows():
            lines.append(
                f"{int(row['id'])} {int(row['type'])} "
                f"{row['x']:.4f} {row['y']:.4f} {row['z']:.4f} "
                f"{row['radius']:.4f} {int(row['parent'])}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # --------------------------------------------------- Drag & drop
    def dragEnterEvent(self, event: QDragEnterEvent):
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        event.ignore()

    # --------------------------------------------------- Sync
    def _on_editor_df_changed(self, df: pd.DataFrame):
        old_df = self._df.copy() if self._df is not None else None
        self._df = df.copy()
        self._table_widget.load_dataframe(self._df, self._filename)
        n_roots = int((self._df["parent"] == -1).sum())
        n_soma = int((self._df["type"] == 1).sum())
        self._update_info_label(self._df, n_roots, n_soma)
        self._record_morph_type_changes(old_df, self._df)
        # Avoid running full validation on every edit; user can re-run from Validation panel.
        self._validation_tab.load_swc(self._df, self._filename, file_path=self._file_path, auto_run=False)
        self._append_log("Dendrogram edits applied to current SWC.", "INFO")

    def _start_morphology_session(self, source_path: str):
        self._morph_session_source_path = str(source_path or "")
        self._morph_session_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._morph_session_changes = []
        self._morph_seq = 0
        self._refresh_morph_edit_tab()

    def _record_morph_type_changes(self, old_df: pd.DataFrame | None, new_df: pd.DataFrame | None):
        if old_df is None or new_df is None or old_df.empty or new_df.empty:
            return
        old_types = {int(r["id"]): int(r["type"]) for _, r in old_df[["id", "type"]].iterrows()}
        changed_rows = []
        for _, row in new_df[["id", "type"]].iterrows():
            nid = int(row["id"])
            new_t = int(row["type"])
            old_t = old_types.get(nid)
            if old_t is None or int(old_t) == int(new_t):
                continue
            self._morph_seq += 1
            changed_rows.append(
                {
                    "seq": self._morph_seq,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "node_id": nid,
                    "old_type": int(old_t),
                    "new_type": int(new_t),
                }
            )
        if changed_rows:
            self._morph_session_changes.extend(changed_rows)
            self._refresh_morph_edit_tab()

    def _refresh_morph_edit_tab(self):
        if not self._morph_session_changes:
            self._edit_log_text.setPlainText("No morphology edits recorded for this session yet.")
            return
        lines = ["Session changes (current SWC):", "Seq\tTime\tNodeID\tOldType\tNewType"]
        for c in self._morph_session_changes:
            lines.append(
                f"{c.get('seq')}\t{c.get('time')}\t{c.get('node_id')}\t{c.get('old_type')}\t{c.get('new_type')}"
            )
        self._edit_log_text.setPlainText("\n".join(lines))

    def _finalize_morphology_session(self, *, show_popup: bool):
        if not self._morph_session_changes:
            return
        source = self._morph_session_source_path or self._file_path
        if source:
            log_path = morphology_session_log_path(source)
            source_name = os.path.basename(source)
        else:
            source_name = self._filename or "swc"
            log_path = Path.cwd() / f"{Path(source_name).stem}_morphology_session_log.txt"
        txt = format_morphology_session_log_text(
            source_file=source_name,
            session_started=self._morph_session_started_at or "",
            session_ended=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            changes=list(self._morph_session_changes),
        )
        out_path = write_text_report(log_path, txt)
        self._append_log(f"Morphology session log written: {out_path}", "INFO")
        if show_popup:
            try:
                ReportPopupDialog.open_report(self, title="Morphology Session Log", report_path=out_path)
            except Exception as e:  # noqa: BLE001
                self._append_log(f"Could not open morphology log popup: {e}", "WARN")
        self._morph_session_changes = []
        self._morph_seq = 0
        self._refresh_morph_edit_tab()

    # --------------------------------------------------- Helpers
    def _append_log(self, text: str, level: str = "INFO"):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] [{level}] {text}".rstrip()
        self._log_console.appendPlainText(line)
        sb = self._log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_current_file_label_text(self, filename: str):
        full = f"Current file: {filename or '(none)'}"
        self._current_file_label.setToolTip(full)
        # Keep top ribbon shrinkable when filenames are long.
        fm = QFontMetrics(self._current_file_label.font())
        max_px = max(120, self._current_file_label.maximumWidth())
        self._current_file_label.setText(fm.elidedText(full, Qt.ElideMiddle, max_px))

    def _update_info_label(self, df: pd.DataFrame, n_roots: int, n_soma: int):
        self._info_label.setText(
            f"File: {self._filename}\n"
            f"Nodes: {len(df)}\n"
            f"Roots: {n_roots}\n"
            f"Soma nodes: {n_soma}\n"
            f"Type counts:\n"
            f"  Soma (1): {(df['type'] == 1).sum()}\n"
            f"  Axon (2): {(df['type'] == 2).sum()}\n"
            f"  Basal (3): {(df['type'] == 3).sum()}\n"
            f"  Apical (4): {(df['type'] == 4).sum()}"
        )

    def _update_recent_files(self, path: str):
        path = os.path.abspath(path)
        if path in self._recent_paths:
            self._recent_paths.remove(path)
        self._recent_paths.insert(0, path)
        self._recent_paths = self._recent_paths[:10]

        self._recent_menu.clear()
        for p in self._recent_paths:
            act = QAction(p, self)
            act.triggered.connect(lambda _=False, sp=p: self._load_swc(sp))
            self._recent_menu.addAction(act)

    def _reset_layout(self):
        self._data_dock.show()
        self._control_dock.show()
        self._precheck_dock.hide()
        if self._is_auto_label_control_active():
            self._show_auto_typing_guide_floating()
        else:
            self._auto_guide_dock.hide()
        # Don't force exact dock sizes here; allow users to drag boundaries.
        try:
            # Enable animated/interactive docks so the sash is draggable.
            self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
            # Set a practical default width so canvas keeps most space while right panel remains usable.
            self.resizeDocks([self._data_dock], [340], Qt.Horizontal)
            self.resizeDocks([self._data_dock, self._control_dock], [1, 1], Qt.Vertical)
        except Exception:
            # Fallback: nothing to do if setDockOptions unavailable
            pass
        self._append_log("Layout reset (docks movable).", "INFO")

    def _toggle_data_panel(self, checked: bool):
        self._data_dock.setVisible(bool(checked))

    def _toggle_control_panel(self, checked: bool):
        self._control_dock.setVisible(bool(checked))

    def _toggle_precheck_panel(self, checked: bool):
        if checked:
            self._show_precheck_floating()
        else:
            self._precheck_dock.hide()

    def _toggle_auto_typing_guide_panel(self, checked: bool):
        if checked:
            if self._is_auto_label_control_active():
                self._show_auto_typing_guide_floating()
            else:
                self._auto_guide_dock.hide()
                self._append_log(
                    "Auto Typing Guide opens when Batch Processing -> Auto Label tab is active.",
                    "INFO",
                )
        else:
            self._auto_guide_dock.hide()

    def _toggle_log_panel(self, checked: bool):
        self._log_console.setVisible(bool(checked))

    def _show_precheck_floating(self):
        self._precheck_dock.show()
        self._precheck_dock.setFloating(True)
        g = self.geometry()
        w = max(760, int(g.width() * 0.62))
        h = max(360, int(g.height() * 0.36))
        x = g.x() + max(40, int((g.width() - w) * 0.5))
        y = g.y() + 120
        self._precheck_dock.setGeometry(x, y, w, h)
        self._precheck_dock.raise_()

    def _show_auto_typing_guide_floating(self):
        self._auto_typing_guide.refresh()
        self._auto_guide_dock.show()
        self._auto_guide_dock.setFloating(True)
        g = self.geometry()
        w = max(760, int(g.width() * 0.62))
        h = max(360, int(g.height() * 0.36))
        x = g.x() + max(40, int((g.width() - w) * 0.5))
        y = g.y() + 160
        self._auto_guide_dock.setGeometry(x, y, w, h)
        self._auto_guide_dock.raise_()

    def _is_auto_label_control_active(self) -> bool:
        if self._active_tool != "batch":
            return False
        idx = self._control_tabs.currentIndex()
        if idx < 0:
            return False
        label = self._control_tabs.tabText(idx).strip().lower()
        return label == "auto label"

    def _is_batch_validation_control_active(self) -> bool:
        if self._active_tool != "batch":
            return False
        idx = self._control_tabs.currentIndex()
        if idx < 0:
            return False
        label = self._control_tabs.tabText(idx).strip().lower()
        return label == "validation"

    def _on_control_tab_changed(self, _index: int):
        if self._is_auto_label_control_active():
            self._show_auto_typing_guide_floating()
        else:
            self._auto_guide_dock.hide()
        self._precheck_dock.hide()
        if self._active_tool == "batch":
            if self._is_batch_validation_control_active():
                self._editor_tab.set_mode(EditorTab.MODE_BATCH)
            else:
                self._editor_tab.set_mode(EditorTab.MODE_EMPTY)

    def _on_precheck_requested(self):
        self._show_precheck_floating()

    def _on_batch_validation_ready(self, report: dict):
        self._editor_tab.show_batch_validation_results(report)
        if self._active_tool == "batch":
            if self._is_batch_validation_control_active():
                self._editor_tab.set_mode(EditorTab.MODE_BATCH)
            else:
                self._editor_tab.set_mode(EditorTab.MODE_EMPTY)
            self._feature_label.setText("Active feature: Batch Processing")
        totals = dict(report.get("summary_total", {}))
        self._append_log(
            "Batch validation results loaded to canvas: "
            f"files={report.get('files_validated', 0)}/{report.get('files_total', 0)}, "
            f"pass={totals.get('pass', 0)}, warn={totals.get('warning', 0)}, "
            f"fail={totals.get('fail', 0)}",
            "INFO",
        )

    def _undo_edit(self):
        if hasattr(self._editor_tab, "_dendro"):
            self._editor_tab._dendro._undo_stack.undo()
            self._append_log("Undo.", "INFO")

    def _redo_edit(self):
        if hasattr(self._editor_tab, "_dendro"):
            self._editor_tab._dendro._undo_stack.redo()
            self._append_log("Redo.", "INFO")

    def closeEvent(self, event):
        try:
            self._finalize_morphology_session(show_popup=False)
        finally:
            super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._set_current_file_label_text(self._filename)

    # ---------------- Help ----------------
    def _show_quick_manual(self):
        text = (
            "Quick Manual:\n"
            "1) Top tab 'Home': File/Edit/View/Window/Help dropdown menus.\n"
            "2) Top tab 'Tools': Batch Processing, Validation, Visualization, Morphology Editing, "
            "Atlas Registration, Analysis buttons.\n"
            "3) Data Explorer and Control Center are dock windows (close, float, resize, move).\n"
            "4) Validation uses a floating Rule Guide window above the canvas.\n"
            "5) Auto Label uses a floating Auto Typing Guide window with decision boundaries.\n"
            "6) After selection, Control Center tabs switch to that feature only.\n"
            "   Batch Processing includes: Split, Validation, Auto Labeling, Radii Cleaning.\n"
            "7) Canvas keeps 3D for standard visualization/editing; Batch Processing uses table views.\n"
            "8) Bottom panel shows all logs and warnings."
        )
        self._append_log(text, "HELP")

    def _show_shortcuts(self):
        text = (
            "Shortcuts:\n"
            "Ctrl+O: Open\n"
            "Ctrl+S: Save\n"
            "Ctrl+Shift+S: Save As\n"
            "Ctrl+Z: Undo\n"
            "Ctrl+Shift+Z: Redo"
        )
        self._append_log(text, "HELP")
