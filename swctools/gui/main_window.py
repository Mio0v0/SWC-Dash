"""Main window layout for SWC-QT with tabbed Home/Tools top bar."""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
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
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .auto_typing_guide import AutoTypingGuideWidget
from .batch_tab import BatchTabWidget
from .constants import SWC_COLS
from .editor_tab import EditorTab
from .neuron_3d_widget import Neuron3DWidget
from .report_popup import ReportPopupDialog
from .radii_cleaning_panel import RadiiCleaningPanel
from .simplification_panel import SimplificationPanel
from .swc_table_widget import SWCTableWidget
from .validation_auto_label_panel import ValidationAutoLabelPanel
from .validation_tab import ValidationPrecheckWidget, ValidationTabWidget
from swctools.core.reporting import (
    format_auto_typing_report_text,
    format_morphology_session_log_text,
    format_simplification_report_text,
    morphology_session_log_path,
    write_text_report,
)
from swctools.tools.morphology_editing.features.simplification import simplify_dataframe
from swctools.tools.validation.features.auto_typing import run_file as run_validation_auto_typing_file


@dataclass
class _DocumentState:
    """Open SWC document state bound to one editor tab/window."""

    editor: EditorTab
    controls: QWidget
    df: pd.DataFrame
    filename: str
    file_path: str
    session_started_at: str = ""
    session_changes: list[dict] = field(default_factory=list)
    session_seq: int = 0
    is_preview: bool = False
    source_editor: EditorTab | None = None
    preview_kind: str = ""


class _CanvasTabs(QTabWidget):
    """Tab widget for open SWC documents with drag-out support."""

    detach_requested = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self._drag_tab_index = -1
        self._drag_start_global = None
        self.tabBar().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.tabBar():
            et = event.type()
            if et == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                self._drag_tab_index = self.tabBar().tabAt(pos)
                self._drag_start_global = (
                    event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
                )
            elif et == QEvent.MouseMove and self._drag_tab_index >= 0 and self._drag_start_global is not None:
                now_global = (
                    event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
                )
                if (now_global - self._drag_start_global).manhattanLength() >= QApplication.startDragDistance():
                    local = self.tabBar().mapFromGlobal(now_global)
                    if not self.tabBar().rect().adjusted(-32, -24, 32, 24).contains(local):
                        idx = int(self._drag_tab_index)
                        self._drag_tab_index = -1
                        self._drag_start_global = None
                        self.detach_requested.emit(idx, int(now_global.x()), int(now_global.y()))
                        return True
            elif et in (QEvent.MouseButtonRelease, QEvent.Leave):
                self._drag_tab_index = -1
                self._drag_start_global = None
        return super().eventFilter(watched, event)


class _DetachedEditorWindow(QMainWindow):
    """Floating window hosting one detached SWC editor."""

    editor_closing = Signal(QWidget)

    def __init__(self, editor: EditorTab, title: str, parent=None):
        super().__init__(parent)
        self._editor = editor
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.resize(980, 760)
        self.setCentralWidget(editor)

    def closeEvent(self, event):
        self.editor_closing.emit(self._editor)
        super().closeEvent(event)


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
        self._documents: dict[EditorTab, _DocumentState] = {}
        self._detached_windows: dict[EditorTab, _DetachedEditorWindow] = {}
        self._simplify_preview_by_source: dict[EditorTab, EditorTab] = {}
        self._simplify_source_by_preview: dict[EditorTab, EditorTab] = {}
        self._simplify_result_by_preview: dict[EditorTab, dict] = {}
        self._auto_label_preview_by_source: dict[EditorTab, EditorTab] = {}
        self._auto_label_source_by_preview: dict[EditorTab, EditorTab] = {}
        self._auto_label_result_by_preview: dict[EditorTab, dict] = {}
        self._batch_has_results: bool = False
        self._closing_app: bool = False

        self._build_ui()
        self._build_status_bar()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Use an in-window top strip instead of the OS menu bar.
        self.menuBar().setVisible(False)

        # ---------------- Top combined bar: Home / Tools ----------------
        self._top_tabs = self._build_top_tabs()

        # ---------------- Center workspace ----------------
        self._canvas_empty = QWidget()
        self._canvas_empty.setStyleSheet("background: #000;")

        self._canvas_tabs = _CanvasTabs()
        self._canvas_tabs.currentChanged.connect(self._on_document_tab_changed)
        self._canvas_tabs.tabCloseRequested.connect(self._on_document_tab_close_requested)
        self._canvas_tabs.detach_requested.connect(self._on_document_detach_requested)

        self._batch_canvas = EditorTab()
        self._batch_canvas.set_mode(EditorTab.MODE_BATCH)

        self._canvas_stack = QStackedWidget()
        self._canvas_stack.addWidget(self._canvas_empty)
        self._canvas_stack.addWidget(self._canvas_tabs)
        self._canvas_stack.addWidget(self._batch_canvas)

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
        self._validation_auto_label_panel = ValidationAutoLabelPanel(self)
        self._validation_auto_label_panel.guide_requested.connect(self._show_auto_typing_guide_floating)
        self._validation_auto_label_panel.log_message.connect(lambda msg: self._append_log(msg, "AUTO"))
        self._validation_auto_label_panel.process_requested.connect(
            self._on_validation_auto_label_process_requested
        )
        self._validation_auto_label_panel.apply_requested.connect(
            self._on_validation_auto_label_apply_requested
        )
        self._validation_auto_label_panel.cancel_requested.connect(
            self._on_validation_auto_label_cancel_requested
        )
        self._validation_radii_panel = RadiiCleaningPanel(self)
        self._validation_radii_panel.log_message.connect(lambda msg: self._append_log(msg, "RADII"))
        self._validation_precheck = ValidationPrecheckWidget()
        self._auto_typing_guide = AutoTypingGuideWidget()
        self._simplification_panel = SimplificationPanel(self)
        self._simplification_panel.log_message.connect(lambda msg: self._append_log(msg, "SIMPLIFY"))
        self._simplification_panel.process_requested.connect(self._on_simplification_process_requested)
        self._simplification_panel.apply_requested.connect(self._on_simplification_apply_requested)
        self._simplification_panel.redo_requested.connect(self._on_simplification_redo_requested)
        self._simplification_panel.cancel_requested.connect(self._on_simplification_cancel_requested)
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
        root.addWidget(self._canvas_stack, stretch=1)
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

        self._refresh_canvas_surface()
        self._reset_layout()
        self._append_log("UI initialized. Open SWC files from File menu.", "INFO")

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

    # --------------------------------------------------------- Document helpers
    def _active_editor(self) -> EditorTab | None:
        idx = self._canvas_tabs.currentIndex()
        if idx < 0:
            return None
        w = self._canvas_tabs.widget(idx)
        return w if isinstance(w, EditorTab) else None

    def _active_document(self) -> _DocumentState | None:
        ed = self._active_editor()
        if ed is None:
            return None
        return self._documents.get(ed)

    def _editor_mode_for_feature(self) -> str:
        key = (self._active_tool or "").strip().lower()
        if key == "visualization":
            return EditorTab.MODE_VIS
        if key in ("morphology_editing", "dendrogram"):
            return EditorTab.MODE_DENDRO
        return EditorTab.MODE_CANVAS

    def _apply_editor_modes(self):
        mode = self._editor_mode_for_feature()
        for doc in self._documents.values():
            doc.editor.set_mode(mode)

    def _refresh_canvas_surface(self):
        if self._active_tool == "batch":
            if self._is_batch_validation_control_active():
                self._batch_canvas.set_mode(EditorTab.MODE_BATCH)
                self._canvas_stack.setCurrentWidget(self._batch_canvas)
            else:
                self._canvas_stack.setCurrentWidget(self._canvas_empty)
            return

        if self._canvas_tabs.count() > 0:
            self._canvas_stack.setCurrentWidget(self._canvas_tabs)
        else:
            self._canvas_stack.setCurrentWidget(self._canvas_empty)

    def _sync_from_active_document(self, *, auto_run_validation: bool):
        doc = self._active_document()
        if doc is None:
            self._df = None
            self._filename = ""
            self._file_path = ""
            self._set_current_file_label_text("")
            self._table_widget.load_dataframe(pd.DataFrame(columns=SWC_COLS), "No SWC loaded")
            self._info_label.setText("No SWC file loaded.")
            self._edit_log_text.setPlainText("No morphology edits recorded for this session yet.")
            self._refresh_canvas_surface()
            self._refresh_simplification_panel_state()
            self._refresh_validation_auto_label_panel_state()
            return

        self._df = doc.df.copy()
        self._filename = doc.filename
        self._file_path = doc.file_path
        self._set_current_file_label_text(doc.filename)

        n_roots = int((doc.df["parent"] == -1).sum())
        n_soma = int((doc.df["type"] == 1).sum())
        self._table_widget.load_dataframe(doc.df, doc.filename)
        self._update_info_label(doc.df, n_roots, n_soma, filename=doc.filename)
        self._refresh_morph_edit_tab(doc)
        self._validation_tab.load_swc(
            doc.df,
            doc.filename,
            file_path=doc.file_path,
            auto_run=bool(auto_run_validation and not doc.is_preview),
        )
        self._refresh_canvas_surface()
        self._refresh_simplification_panel_state()
        self._refresh_validation_auto_label_panel_state()

    def _start_morphology_session(self, doc: _DocumentState):
        if doc.is_preview:
            return
        doc.session_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc.session_changes = []
        doc.session_seq = 0
        if doc is self._active_document():
            self._refresh_morph_edit_tab(doc)

    def _record_morph_type_changes(
        self,
        doc: _DocumentState,
        old_df: pd.DataFrame | None,
        new_df: pd.DataFrame | None,
    ):
        if doc.is_preview:
            return
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
            doc.session_seq += 1
            changed_rows.append(
                {
                    "seq": doc.session_seq,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "node_id": nid,
                    "old_type": int(old_t),
                    "new_type": int(new_t),
                }
            )
        if changed_rows:
            doc.session_changes.extend(changed_rows)
            if doc is self._active_document():
                self._refresh_morph_edit_tab(doc)

    def _refresh_morph_edit_tab(self, doc: _DocumentState | None = None):
        row = doc or self._active_document()
        if row is None or not row.session_changes:
            self._edit_log_text.setPlainText("No morphology edits recorded for this session yet.")
            return

        lines = [
            f"Session changes ({row.filename}):",
            "Seq\tTime\tNodeID\tOldType\tNewType",
        ]
        for c in row.session_changes:
            lines.append(
                f"{c.get('seq')}\t{c.get('time')}\t{c.get('node_id')}\t{c.get('old_type')}\t{c.get('new_type')}"
            )
        self._edit_log_text.setPlainText("\n".join(lines))

    def _finalize_morphology_session(
        self,
        doc: _DocumentState,
        *,
        show_popup: bool,
        source_override: str | None = None,
    ) -> str | None:
        if not doc.session_changes:
            return None

        source = str(source_override or doc.file_path or "")
        if source:
            log_path = morphology_session_log_path(source)
            source_name = os.path.basename(source)
        else:
            source_name = doc.filename or "swc"
            log_path = Path.cwd() / f"{Path(source_name).stem}_morphology_session_log.txt"

        txt = format_morphology_session_log_text(
            source_file=source_name,
            session_started=doc.session_started_at or "",
            session_ended=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            changes=list(doc.session_changes),
        )
        out_path = write_text_report(log_path, txt)
        self._append_log(f"Morphology session log written: {out_path}", "INFO")
        if show_popup:
            try:
                ReportPopupDialog.open_report(self, title="Morphology Session Log", report_path=out_path)
            except Exception as e:  # noqa: BLE001
                self._append_log(f"Could not open morphology log popup: {e}", "WARN")

        doc.session_changes = []
        doc.session_seq = 0
        if doc is self._active_document():
            self._refresh_morph_edit_tab(doc)
        return str(out_path)

    def _next_closed_output_path(self, doc: _DocumentState) -> Path:
        base = Path(doc.file_path) if str(doc.file_path or "").strip() else Path.cwd() / (doc.filename or "swc")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cand = base.with_name(f"{base.stem}_closed_{ts}.swc")
        i = 1
        while cand.exists():
            cand = base.with_name(f"{base.stem}_closed_{ts}_{i}.swc")
            i += 1
        return cand

    def _write_closed_copy_and_log(self, doc: _DocumentState):
        if not doc.session_changes:
            return
        out_swc = self._next_closed_output_path(doc)
        self._write_swc_file(str(out_swc), doc.df)
        self._append_log(f"Closed tab saved to: {out_swc}", "INFO")
        self._finalize_morphology_session(doc, show_popup=False, source_override=str(out_swc))

    # --------------------------------------------------------- Controls per feature
    def _is_simplification_preview(self, doc: _DocumentState | None) -> bool:
        return bool(doc and doc.is_preview and doc.preview_kind == "simplification")

    def _source_document_for(self, doc: _DocumentState | None) -> _DocumentState | None:
        if doc is None:
            return None
        if self._is_simplification_preview(doc):
            src_editor = doc.source_editor or self._simplify_source_by_preview.get(doc.editor)
            if src_editor is None:
                return None
            return self._documents.get(src_editor)
        if self._is_validation_auto_label_preview(doc):
            src_editor = doc.source_editor or self._auto_label_source_by_preview.get(doc.editor)
            if src_editor is None:
                return None
            return self._documents.get(src_editor)
        return doc

    def _active_source_document(self) -> _DocumentState | None:
        return self._source_document_for(self._active_document())

    def _simplification_log_payload(
        self,
        source_doc: _DocumentState,
        result: dict,
        *,
        output_path: str | None,
    ) -> dict:
        return {
            "mode": "gui",
            "input_path": str(source_doc.file_path or source_doc.filename),
            "output_path": str(output_path or ""),
            "original_node_count": int(result.get("original_node_count", 0)),
            "new_node_count": int(result.get("new_node_count", 0)),
            "reduction_percent": float(result.get("reduction_percent", 0.0)),
            "params_used": dict(result.get("params_used", {})),
            "protected_counts": dict(result.get("protected_counts", {})),
            "removed_node_ids": list(result.get("removed_node_ids", [])),
        }

    def _write_simplification_log(self, payload: dict) -> str:
        output_path = str(payload.get("output_path", "") or "").strip()
        input_path = str(payload.get("input_path", "") or "").strip()

        if output_path:
            base = Path(output_path)
        elif input_path:
            base = Path(input_path)
        else:
            base = Path.cwd() / "simplified_preview.swc"

        if not base.suffix:
            base = base.with_suffix(".swc")

        log_path = base.with_name(f"{base.stem}_simplification_log.txt")
        return write_text_report(log_path, format_simplification_report_text(payload))

    def _resolve_simplification_context(self) -> tuple[_DocumentState | None, _DocumentState | None, dict | None]:
        active = self._active_document()
        if active is None:
            return None, None, None

        if self._is_simplification_preview(active):
            source_doc = self._source_document_for(active)
            result = self._simplify_result_by_preview.get(active.editor)
            return source_doc, active, result

        preview_editor = self._simplify_preview_by_source.get(active.editor)
        if preview_editor is None:
            return active, None, None

        preview_doc = self._documents.get(preview_editor)
        result = self._simplify_result_by_preview.get(preview_editor)
        return active, preview_doc, result

    def _remove_simplification_preview(self, preview_editor: EditorTab, *, switch_to_source: bool):
        preview_doc = self._documents.get(preview_editor)
        source_editor = self._simplify_source_by_preview.pop(preview_editor, None)
        if source_editor is not None:
            if self._simplify_preview_by_source.get(source_editor) is preview_editor:
                self._simplify_preview_by_source.pop(source_editor, None)
        self._simplify_result_by_preview.pop(preview_editor, None)

        if preview_doc is not None:
            self._documents.pop(preview_editor, None)

        idx = self._canvas_tabs.indexOf(preview_editor)
        if idx >= 0:
            self._canvas_tabs.removeTab(idx)

        float_win = self._detached_windows.pop(preview_editor, None)
        if float_win is not None:
            try:
                float_win.editor_closing.disconnect(self._on_detached_editor_closing)
            except Exception:
                pass
            float_win.close()

        if switch_to_source and source_editor is not None:
            src_idx = self._canvas_tabs.indexOf(source_editor)
            if src_idx >= 0:
                self._canvas_tabs.setCurrentIndex(src_idx)

    def _refresh_simplification_panel_state(self):
        source_doc, preview_doc, result = self._resolve_simplification_context()
        if source_doc is None or preview_doc is None or not isinstance(result, dict):
            self._simplification_panel.set_preview_state(False, None, None)
            return
        summary = {
            "original_node_count": int(result.get("original_node_count", 0)),
            "new_node_count": int(result.get("new_node_count", 0)),
            "reduction_percent": float(result.get("reduction_percent", 0.0)),
            "params_used": dict(result.get("params_used", {})),
        }
        self._simplification_panel.set_preview_state(
            True,
            summary,
            str(result.get("log_path", "") or ""),
        )

    def _on_simplification_process_requested(self, config_overrides: dict):
        source_doc = self._active_source_document()
        if source_doc is None or source_doc.df is None or source_doc.df.empty:
            self._append_log("Smart Decimation: no active SWC document.", "WARN")
            return

        try:
            result = simplify_dataframe(source_doc.df, config_overrides=dict(config_overrides or {}))
        except Exception as e:  # noqa: BLE001
            self._append_log(f"Smart Decimation failed: {e}", "ERROR")
            return

        simplified_df = result.get("dataframe")
        if not isinstance(simplified_df, pd.DataFrame) or simplified_df.empty:
            self._append_log("Smart Decimation produced empty output.", "WARN")
            return

        preview_editor = self._simplify_preview_by_source.get(source_doc.editor)
        preview_doc = self._documents.get(preview_editor) if preview_editor is not None else None

        preview_name = "Simplified View"
        if preview_doc is None:
            preview_editor = EditorTab()
            preview_editor.df_changed.connect(
                lambda new_df, ed=preview_editor: self._on_editor_df_changed(ed, new_df)
            )
            preview_controls = preview_editor.take_dendrogram_controls_panel()
            preview_doc = _DocumentState(
                editor=preview_editor,
                controls=preview_controls,
                df=simplified_df.copy(),
                filename=preview_name,
                file_path=str(source_doc.file_path or ""),
                is_preview=True,
                source_editor=source_doc.editor,
                preview_kind="simplification",
            )
            self._documents[preview_editor] = preview_doc
            self._simplify_preview_by_source[source_doc.editor] = preview_editor
            self._simplify_source_by_preview[preview_editor] = source_doc.editor
            tab_idx = self._canvas_tabs.addTab(preview_editor, preview_name)
            self._canvas_tabs.setCurrentIndex(tab_idx)
        else:
            preview_doc.df = simplified_df.copy()
            preview_doc.file_path = str(source_doc.file_path or "")
            tab_idx = self._canvas_tabs.indexOf(preview_doc.editor)
            if tab_idx >= 0:
                self._canvas_tabs.setCurrentIndex(tab_idx)

        preview_doc.df = simplified_df.copy()
        preview_doc.editor.load_swc(simplified_df, preview_name)
        preview_doc.editor.set_mode(self._editor_mode_for_feature())

        payload = self._simplification_log_payload(source_doc, result, output_path=None)
        log_path = self._write_simplification_log(payload)
        result["summary"] = payload
        result["log_path"] = log_path
        self._simplify_result_by_preview[preview_doc.editor] = result

        self._append_log(
            "Smart Decimation preview created: "
            f"{payload.get('original_node_count', 0)} -> {payload.get('new_node_count', 0)} "
            f"({payload.get('reduction_percent', 0.0):.2f}%).",
            "INFO",
        )
        self._append_log(f"Smart Decimation log written: {log_path}", "INFO")

        self._refresh_simplification_panel_state()
        self._sync_from_active_document(auto_run_validation=False)

    def _on_simplification_apply_requested(self):
        source_doc, preview_doc, result = self._resolve_simplification_context()
        if source_doc is None or preview_doc is None or preview_doc.df is None or preview_doc.df.empty:
            self._append_log("Smart Decimation Apply: no preview available.", "WARN")
            return

        src_path = str(source_doc.file_path or "")
        input_ref = str(source_doc.file_path or source_doc.filename)
        if src_path:
            src_p = Path(src_path)
            default_out = str(src_p.with_name(f"{src_p.stem}_simplified{src_p.suffix}"))
        else:
            default_out = f"{Path(source_doc.filename or 'swc').stem}_simplified.swc"

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Simplified SWC",
            default_out,
            "SWC Files (*.swc);;All Files (*)",
        )
        if not out_path:
            self._append_log("Smart Decimation Apply cancelled.", "INFO")
            return

        self._write_swc_file(out_path, preview_doc.df)

        source_doc.df = preview_doc.df.copy()
        source_doc.file_path = str(out_path)
        source_doc.filename = os.path.basename(out_path)
        source_doc.editor.load_swc(source_doc.df, source_doc.filename)
        source_doc.editor.set_mode(self._editor_mode_for_feature())

        src_idx = self._canvas_tabs.indexOf(source_doc.editor)
        if src_idx >= 0:
            self._canvas_tabs.setTabText(src_idx, source_doc.filename)

        if not isinstance(result, dict):
            result = {}
        payload = self._simplification_log_payload(source_doc, result, output_path=str(out_path))
        payload["input_path"] = input_ref
        log_path = self._write_simplification_log(payload)
        result["summary"] = payload
        result["log_path"] = log_path

        self._remove_simplification_preview(preview_doc.editor, switch_to_source=True)
        self._update_recent_files(out_path)
        self._start_morphology_session(source_doc)
        self._sync_from_active_document(auto_run_validation=False)
        self._append_log(f"Smart Decimation applied: {out_path}", "INFO")
        self._append_log(f"Smart Decimation log written: {log_path}", "INFO")
        self._refresh_simplification_panel_state()

    def _on_simplification_redo_requested(self):
        source_doc, preview_doc, _result = self._resolve_simplification_context()
        if source_doc is None:
            self._append_log("Smart Decimation Redo: no active SWC document.", "WARN")
            return
        if preview_doc is not None:
            self._remove_simplification_preview(preview_doc.editor, switch_to_source=True)
        self._on_simplification_process_requested(self._simplification_panel.current_overrides())

    def _on_simplification_cancel_requested(self):
        source_doc, preview_doc, _result = self._resolve_simplification_context()
        if preview_doc is None:
            self._append_log("Smart Decimation Cancel: no preview to discard.", "INFO")
            return
        self._remove_simplification_preview(preview_doc.editor, switch_to_source=True)
        if source_doc is not None:
            src_idx = self._canvas_tabs.indexOf(source_doc.editor)
            if src_idx >= 0:
                self._canvas_tabs.setCurrentIndex(src_idx)
        self._sync_from_active_document(auto_run_validation=False)
        self._append_log("Smart Decimation preview discarded.", "INFO")
        self._refresh_simplification_panel_state()

    def _is_validation_auto_label_preview(self, doc: _DocumentState | None) -> bool:
        return bool(doc and doc.is_preview and doc.preview_kind == "validation_auto_label")

    def _resolve_validation_auto_label_context(
        self,
    ) -> tuple[_DocumentState | None, _DocumentState | None, dict | None]:
        active = self._active_document()
        if active is None:
            return None, None, None

        if self._is_validation_auto_label_preview(active):
            source_doc = self._source_document_for(active)
            result = self._auto_label_result_by_preview.get(active.editor)
            return source_doc, active, result

        preview_editor = self._auto_label_preview_by_source.get(active.editor)
        if preview_editor is None:
            return active, None, None

        preview_doc = self._documents.get(preview_editor)
        result = self._auto_label_result_by_preview.get(preview_editor)
        return active, preview_doc, result

    def _remove_validation_auto_label_preview(self, preview_editor: EditorTab, *, switch_to_source: bool):
        preview_doc = self._documents.get(preview_editor)
        source_editor = self._auto_label_source_by_preview.pop(preview_editor, None)
        if source_editor is not None:
            if self._auto_label_preview_by_source.get(source_editor) is preview_editor:
                self._auto_label_preview_by_source.pop(source_editor, None)
        self._auto_label_result_by_preview.pop(preview_editor, None)

        if preview_doc is not None:
            self._documents.pop(preview_editor, None)

        idx = self._canvas_tabs.indexOf(preview_editor)
        if idx >= 0:
            self._canvas_tabs.removeTab(idx)

        float_win = self._detached_windows.pop(preview_editor, None)
        if float_win is not None:
            try:
                float_win.editor_closing.disconnect(self._on_detached_editor_closing)
            except Exception:
                pass
            float_win.close()

        if switch_to_source and source_editor is not None:
            src_idx = self._canvas_tabs.indexOf(source_editor)
            if src_idx >= 0:
                self._canvas_tabs.setCurrentIndex(src_idx)

    def _auto_label_result_to_dataframe(self, result: object) -> pd.DataFrame:
        rows = list(getattr(result, "rows", []))
        types = list(getattr(result, "types", []))
        radii = list(getattr(result, "radii", []))
        if not rows:
            return pd.DataFrame(columns=SWC_COLS)
        data = []
        for i, row in enumerate(rows):
            data.append(
                {
                    "id": int(row.get("id", 0)),
                    "type": int(types[i] if i < len(types) else row.get("type", 0)),
                    "x": float(row.get("x", 0.0)),
                    "y": float(row.get("y", 0.0)),
                    "z": float(row.get("z", 0.0)),
                    "radius": float(radii[i] if i < len(radii) else row.get("radius", 0.0)),
                    "parent": int(row.get("parent", -1)),
                }
            )
        return pd.DataFrame(data, columns=SWC_COLS)

    def _refresh_validation_auto_label_panel_state(self):
        source_doc, preview_doc, result = self._resolve_validation_auto_label_context()
        if (
            source_doc is None
            or preview_doc is None
            or result is None
            or not isinstance(result, dict)
        ):
            self._validation_auto_label_panel.set_preview_state(False, None)
            return
        self._validation_auto_label_panel.set_preview_state(True, result)

    def _on_validation_auto_label_process_requested(self, options: object):
        source_doc = self._active_source_document()
        if source_doc is None or source_doc.df is None or source_doc.df.empty:
            self._append_log("Validation Auto Label: no active SWC document.", "WARN")
            self._validation_auto_label_panel.set_status_text("No active SWC loaded.")
            return

        tmp_fd, tmp_in = tempfile.mkstemp(prefix="swctools_auto_label_", suffix=".swc")
        os.close(tmp_fd)
        tmp_path = Path(tmp_in)
        try:
            self._write_swc_file(str(tmp_path), source_doc.df)
            result_obj = run_validation_auto_typing_file(
                str(tmp_path),
                options=options,
                write_output=False,
                write_log=False,
            )
        except Exception as e:  # noqa: BLE001
            self._append_log(f"Validation Auto Label failed: {e}", "ERROR")
            self._validation_auto_label_panel.set_status_text(f"Auto Label failed:\n{e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        preview_df = self._auto_label_result_to_dataframe(result_obj)
        if preview_df.empty:
            self._append_log("Validation Auto Label produced empty output.", "WARN")
            self._validation_auto_label_panel.set_status_text("Auto Label output is empty.")
            return

        preview_editor = self._auto_label_preview_by_source.get(source_doc.editor)
        preview_doc = self._documents.get(preview_editor) if preview_editor is not None else None

        preview_name = "Auto-Labeled View"
        if preview_doc is None:
            preview_editor = EditorTab()
            preview_editor.df_changed.connect(
                lambda new_df, ed=preview_editor: self._on_editor_df_changed(ed, new_df)
            )
            preview_controls = preview_editor.take_dendrogram_controls_panel()
            preview_doc = _DocumentState(
                editor=preview_editor,
                controls=preview_controls,
                df=preview_df.copy(),
                filename=preview_name,
                file_path=str(source_doc.file_path or ""),
                is_preview=True,
                source_editor=source_doc.editor,
                preview_kind="validation_auto_label",
            )
            self._documents[preview_editor] = preview_doc
            self._auto_label_preview_by_source[source_doc.editor] = preview_editor
            self._auto_label_source_by_preview[preview_editor] = source_doc.editor
            tab_idx = self._canvas_tabs.addTab(preview_editor, preview_name)
            self._canvas_tabs.setCurrentIndex(tab_idx)
        else:
            preview_doc.df = preview_df.copy()
            preview_doc.file_path = str(source_doc.file_path or "")
            tab_idx = self._canvas_tabs.indexOf(preview_doc.editor)
            if tab_idx >= 0:
                self._canvas_tabs.setCurrentIndex(tab_idx)

        preview_doc.df = preview_df.copy()
        preview_doc.editor.load_swc(preview_df, preview_name)
        preview_doc.editor.set_mode(self._editor_mode_for_feature())

        opts = options
        opts_dict = {
            "soma": bool(getattr(opts, "soma", False)),
            "axon": bool(getattr(opts, "axon", False)),
            "apic": bool(getattr(opts, "apic", False)),
            "basal": bool(getattr(opts, "basal", False)),
            "rad": bool(getattr(opts, "rad", False)),
        }
        result_payload = {
            "input_file": str(source_doc.file_path or source_doc.filename),
            "nodes_total": int(getattr(result_obj, "nodes_total", 0)),
            "type_changes": int(getattr(result_obj, "type_changes", 0)),
            "radius_changes": int(getattr(result_obj, "radius_changes", 0)),
            "out_type_counts": dict(getattr(result_obj, "out_type_counts", {}) or {}),
            "change_details": list(getattr(result_obj, "change_details", []) or []),
            "options": opts_dict,
            "log_path": "",
        }
        self._auto_label_result_by_preview[preview_doc.editor] = result_payload
        self._validation_auto_label_panel.set_preview_state(True, result_payload)

        self._append_log(
            "Validation Auto Label preview created: "
            f"nodes={result_payload['nodes_total']}, "
            f"type_changes={result_payload['type_changes']}, "
            f"radius_changes={result_payload['radius_changes']}",
            "INFO",
        )
        self._sync_from_active_document(auto_run_validation=False)

    def _write_validation_auto_label_log(
        self,
        source_doc: _DocumentState,
        result: dict,
        *,
        output_path: str,
    ) -> str:
        out_p = Path(output_path)
        out_counts = dict(result.get("out_type_counts", {}))
        per_file = [
            f"{Path(source_doc.filename or 'swc').name}: "
            f"nodes={int(result.get('nodes_total', 0))}, "
            f"type_changes={int(result.get('type_changes', 0))}, "
            f"radius_changes={int(result.get('radius_changes', 0))}, "
            f"out_types(soma/axon/basal/apic)="
            f"{out_counts.get(1, 0)}/{out_counts.get(2, 0)}/{out_counts.get(3, 0)}/{out_counts.get(4, 0)}"
        ]
        payload = {
            "folder": str(out_p.parent),
            "out_dir": str(out_p.parent),
            "zip_path": None,
            "files_total": 1,
            "files_processed": 1,
            "files_failed": 0,
            "total_nodes": int(result.get("nodes_total", 0)),
            "total_type_changes": int(result.get("type_changes", 0)),
            "total_radius_changes": int(result.get("radius_changes", 0)),
            "failures": [],
            "per_file": per_file,
            "change_details": list(result.get("change_details", [])),
        }
        log_path = out_p.with_name(f"{out_p.stem}_auto_typing_report.txt")
        return write_text_report(log_path, format_auto_typing_report_text(payload))

    def _on_validation_auto_label_apply_requested(self):
        source_doc, preview_doc, result = self._resolve_validation_auto_label_context()
        if source_doc is None or preview_doc is None or preview_doc.df is None or preview_doc.df.empty:
            self._append_log("Validation Auto Label Apply: no preview available.", "WARN")
            return
        if not isinstance(result, dict):
            result = {}

        src_path = str(source_doc.file_path or "")
        if src_path:
            src_p = Path(src_path)
            default_out = str(src_p.with_name(f"{src_p.stem}_auto_labeled{src_p.suffix}"))
        else:
            default_out = f"{Path(source_doc.filename or 'swc').stem}_auto_labeled.swc"

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Auto-Labeled SWC",
            default_out,
            "SWC Files (*.swc);;All Files (*)",
        )
        if not out_path:
            self._append_log("Validation Auto Label Apply cancelled.", "INFO")
            return

        self._write_swc_file(out_path, preview_doc.df)
        log_path = self._write_validation_auto_label_log(source_doc, result, output_path=out_path)
        result["log_path"] = log_path

        source_doc.df = preview_doc.df.copy()
        source_doc.file_path = str(out_path)
        source_doc.filename = os.path.basename(out_path)
        source_doc.editor.load_swc(source_doc.df, source_doc.filename)
        source_doc.editor.set_mode(self._editor_mode_for_feature())

        src_idx = self._canvas_tabs.indexOf(source_doc.editor)
        if src_idx >= 0:
            self._canvas_tabs.setTabText(src_idx, source_doc.filename)

        self._remove_validation_auto_label_preview(preview_doc.editor, switch_to_source=True)
        self._update_recent_files(out_path)
        self._start_morphology_session(source_doc)
        self._sync_from_active_document(auto_run_validation=False)
        self._append_log(f"Validation Auto Label applied: {out_path}", "INFO")
        self._append_log(f"Validation Auto Label log written: {log_path}", "INFO")

    def _on_validation_auto_label_cancel_requested(self):
        source_doc, preview_doc, _result = self._resolve_validation_auto_label_context()
        if preview_doc is None:
            self._append_log("Validation Auto Label Cancel: no preview to discard.", "INFO")
            return
        self._remove_validation_auto_label_preview(preview_doc.editor, switch_to_source=True)
        if source_doc is not None:
            src_idx = self._canvas_tabs.indexOf(source_doc.editor)
            if src_idx >= 0:
                self._canvas_tabs.setCurrentIndex(src_idx)
        self._sync_from_active_document(auto_run_validation=False)
        self._append_log("Validation Auto Label preview discarded.", "INFO")

    def _set_control_tabs_for_feature(self, feature: str):
        """Show only control tabs relevant to the active feature."""
        key = (feature or "").strip().lower()
        previous_label = ""
        if self._control_tabs.count() > 0 and self._control_tabs.currentIndex() >= 0:
            previous_label = self._control_tabs.tabText(self._control_tabs.currentIndex()).strip().lower()
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
            self._control_tabs.addTab(self._validation_auto_label_panel, "Auto Label")
            self._control_tabs.addTab(self._validation_radii_panel, "Radii Cleaning")
            self._control_tabs.setCurrentWidget(self._validation_tab)
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            return

        if key in ("morphology_editing", "dendrogram"):
            doc = self._active_document()
            if doc is None:
                self._on_control_tab_changed(self._control_tabs.currentIndex())
                return
            self._control_tabs.addTab(doc.controls, "Label Editing")
            self._control_tabs.addTab(self._simplification_panel, "Simplification")
            if previous_label == "simplification":
                self._control_tabs.setCurrentWidget(self._simplification_panel)
            else:
                self._control_tabs.setCurrentIndex(0)
            self._refresh_simplification_panel_state()
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
            self._set_control_tabs_for_feature("batch")
            self._control_dock.show()
            self._on_control_tab_changed(self._control_tabs.currentIndex())
            self._feature_label.setText("Active feature: Batch Processing")
            self._append_log("Feature switched: Batch Processing", "INFO")
            return
        if key == "validation":
            self._active_tool = "validation"
            self._set_control_tabs_for_feature("validation")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._apply_editor_modes()
            self._refresh_canvas_surface()
            self._feature_label.setText("Active feature: Validation")
            self._append_log("Feature switched: Validation", "INFO")
            return
        if key == "visualization":
            self._active_tool = "visualization"
            self._set_control_tabs_for_feature("visualization")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._apply_editor_modes()
            self._refresh_canvas_surface()
            self._feature_label.setText("Active feature: Visualization")
            self._append_log("Feature switched: Visualization", "INFO")
            return
        if key in ("morphology_editing", "dendrogram"):
            self._active_tool = "morphology_editing"
            self._set_control_tabs_for_feature("morphology_editing")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._apply_editor_modes()
            self._refresh_canvas_surface()
            self._feature_label.setText("Active feature: Morphology Editing")
            self._append_log("Feature switched: Morphology Editing", "INFO")
            return
        if key == "atlas_registration":
            self._active_tool = "atlas_registration"
            self._set_control_tabs_for_feature("atlas_registration")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._apply_editor_modes()
            self._refresh_canvas_surface()
            self._feature_label.setText("Active feature: Atlas Registration")
            self._append_log("Feature switched: Atlas Registration (placeholder)", "INFO")
            return
        if key == "analysis":
            self._active_tool = "analysis"
            self._set_control_tabs_for_feature("analysis")
            self._control_dock.show()
            self._precheck_dock.hide()
            self._auto_guide_dock.hide()
            self._apply_editor_modes()
            self._refresh_canvas_surface()
            self._feature_label.setText("Active feature: Analysis")
            self._append_log("Feature switched: Analysis (placeholder)", "INFO")
            return

        self._active_tool = ""
        self._set_control_tabs_for_feature("")
        self._precheck_dock.hide()
        self._auto_guide_dock.hide()
        self._apply_editor_modes()
        self._refresh_canvas_surface()
        self._feature_label.setText("Active feature: None")
        self._append_log("No active tool selected.", "INFO")

    def _set_camera(self, preset: str):
        ed = self._active_editor()
        if ed is None:
            return
        ed.set_camera_view(preset)
        self._append_log(f"Camera preset: {preset}", "INFO")

    def _reset_camera(self):
        ed = self._active_editor()
        if ed is None:
            return
        ed.reset_camera()
        self._append_log("Camera reset.", "INFO")

    def _on_render_mode_changed(self, index: int):
        mode = self._render_combo.currentData()
        if mode is None:
            return
        ed = self._active_editor()
        if ed is None:
            return
        ed.set_render_mode(int(mode))
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

            filename = os.path.basename(path)
            editor = EditorTab()
            editor.df_changed.connect(lambda new_df, ed=editor: self._on_editor_df_changed(ed, new_df))
            controls = editor.take_dendrogram_controls_panel()
            editor.load_swc(df, filename)

            doc = _DocumentState(
                editor=editor,
                controls=controls,
                df=df.copy(),
                filename=filename,
                file_path=str(path),
            )
            self._documents[editor] = doc
            self._start_morphology_session(doc)

            idx = self._canvas_tabs.addTab(editor, filename)
            self._canvas_tabs.setCurrentIndex(idx)

            self._update_recent_files(path)
            self._apply_editor_modes()
            self._sync_from_active_document(auto_run_validation=True)
            if self._active_tool in ("morphology_editing", "dendrogram"):
                self._set_control_tabs_for_feature("morphology_editing")

            n_roots = int((df["parent"] == -1).sum())
            n_soma = int((df["type"] == 1).sum())
            self._status.showMessage(
                f"Loaded {filename}: {len(df)} nodes, {n_roots} root(s), {n_soma} soma(s)",
                5000,
            )
            self._append_log(
                f"Loaded {filename}: nodes={len(df)}, roots={n_roots}, soma={n_soma}",
                "INFO",
            )
            self._refresh_canvas_surface()
            self.swc_loaded.emit(df, filename)
        except Exception as e:
            self._append_log(f"Error loading SWC: {e}", "ERROR")

    def _on_save(self):
        doc = self._active_document()
        if doc is None or doc.df.empty:
            self._append_log("No SWC loaded. Nothing to save.", "WARN")
            return
        if not doc.file_path:
            self._on_save_as()
            return
        self._write_swc_file(doc.file_path, doc.df)
        self._append_log(f"Saved {doc.file_path}", "INFO")

    def _on_save_as(self):
        doc = self._active_document()
        if doc is None or doc.df.empty:
            self._append_log("No SWC loaded. Nothing to save.", "WARN")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SWC As",
            doc.filename or "edited.swc",
            "SWC Files (*.swc);;All Files (*)",
        )
        if not path:
            self._append_log("Save As cancelled.", "INFO")
            return

        self._write_swc_file(path, doc.df)
        doc.file_path = str(path)
        doc.filename = os.path.basename(path)
        tab_idx = self._canvas_tabs.indexOf(doc.editor)
        if tab_idx >= 0:
            self._canvas_tabs.setTabText(tab_idx, doc.filename)
        self._sync_from_active_document(auto_run_validation=False)
        self._append_log(f"Saved As {path}", "INFO")
        self._update_recent_files(path)

    def _on_export(self):
        doc = self._active_document()
        if doc is None or doc.df.empty:
            self._append_log("No SWC loaded. Nothing to export.", "WARN")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export SWC",
            f"export_{doc.filename or 'swc'}.swc",
            "SWC Files (*.swc)",
        )
        if not path:
            self._append_log("Export cancelled.", "INFO")
            return
        self._write_swc_file(path, doc.df)
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

    # --------------------------------------------------- Sync from editor
    def _on_editor_df_changed(self, editor: EditorTab, df: pd.DataFrame):
        doc = self._documents.get(editor)
        if doc is None:
            return

        old_df = doc.df.copy() if doc.df is not None else None
        doc.df = df.copy()
        self._record_morph_type_changes(doc, old_df, doc.df)

        if editor is self._active_editor():
            self._sync_from_active_document(auto_run_validation=False)
            if not doc.is_preview:
                self._append_log("Dendrogram edits applied to current SWC.", "INFO")
            else:
                self._append_log("Dendrogram edits applied to simplification preview.", "INFO")

    # --------------------------------------------------- Document tabs/windows
    def _on_document_tab_changed(self, _index: int):
        self._sync_from_active_document(auto_run_validation=False)
        if self._active_tool in ("morphology_editing", "dendrogram"):
            self._set_control_tabs_for_feature("morphology_editing")
            self._refresh_simplification_panel_state()
        self._apply_editor_modes()

    def _on_document_tab_close_requested(self, index: int):
        editor = self._canvas_tabs.widget(index)
        if isinstance(editor, EditorTab):
            self._close_document_editor(editor, from_detached_window=False)

    def _on_document_detach_requested(self, index: int, x: int, y: int):
        editor = self._canvas_tabs.widget(index)
        if not isinstance(editor, EditorTab):
            return
        doc = self._documents.get(editor)
        if doc is None:
            return

        self._canvas_tabs.removeTab(index)
        float_win = _DetachedEditorWindow(editor, doc.filename, self)
        float_win.editor_closing.connect(self._on_detached_editor_closing)
        float_win.move(max(0, int(x - 120)), max(0, int(y - 20)))
        float_win.show()
        self._detached_windows[editor] = float_win

        self._append_log(f"Detached tab: {doc.filename}", "INFO")
        self._sync_from_active_document(auto_run_validation=False)
        self._refresh_canvas_surface()
        if self._active_tool in ("morphology_editing", "dendrogram"):
            self._set_control_tabs_for_feature("morphology_editing")

    def _on_detached_editor_closing(self, editor_widget: QWidget):
        if self._closing_app:
            return
        if isinstance(editor_widget, EditorTab):
            self._close_document_editor(editor_widget, from_detached_window=True)

    def _close_document_editor(self, editor: EditorTab, *, from_detached_window: bool):
        doc = self._documents.get(editor)
        if doc is None:
            return

        # If this source has a simplification preview tab, close preview first.
        preview_editor = self._simplify_preview_by_source.get(editor)
        if preview_editor is not None:
            self._remove_simplification_preview(preview_editor, switch_to_source=False)
        # If this source has an auto-label preview tab, close preview first.
        auto_preview_editor = self._auto_label_preview_by_source.get(editor)
        if auto_preview_editor is not None:
            self._remove_validation_auto_label_preview(auto_preview_editor, switch_to_source=False)

        # If closing a preview tab, just drop preview state and UI tab.
        if self._is_simplification_preview(doc):
            self._remove_simplification_preview(editor, switch_to_source=False)
            self._append_log(f"Closed tab: {doc.filename}", "INFO")
            self._sync_from_active_document(auto_run_validation=False)
            self._refresh_canvas_surface()
            if self._active_tool in ("morphology_editing", "dendrogram"):
                self._set_control_tabs_for_feature("morphology_editing")
                self._refresh_simplification_panel_state()
            return
        if self._is_validation_auto_label_preview(doc):
            self._remove_validation_auto_label_preview(editor, switch_to_source=False)
            self._append_log(f"Closed tab: {doc.filename}", "INFO")
            self._sync_from_active_document(auto_run_validation=False)
            self._refresh_canvas_surface()
            return

        self._documents.pop(editor, None)
        self._write_closed_copy_and_log(doc)

        if not from_detached_window:
            idx = self._canvas_tabs.indexOf(editor)
            if idx >= 0:
                self._canvas_tabs.removeTab(idx)

        float_win = self._detached_windows.pop(editor, None)
        if float_win is not None and not from_detached_window:
            try:
                float_win.editor_closing.disconnect(self._on_detached_editor_closing)
            except Exception:
                pass
            float_win.close()

        self._append_log(f"Closed tab: {doc.filename}", "INFO")
        self._sync_from_active_document(auto_run_validation=False)
        self._refresh_canvas_surface()
        if self._active_tool in ("morphology_editing", "dendrogram"):
            self._set_control_tabs_for_feature("morphology_editing")
            self._refresh_simplification_panel_state()
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

    def _update_info_label(self, df: pd.DataFrame, n_roots: int, n_soma: int, *, filename: str):
        self._info_label.setText(
            f"File: {filename}\n"
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
                    "Auto Typing Guide opens when an Auto Label control tab is active.",
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
        if self._active_tool not in ("batch", "validation"):
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
        if self._active_tool in ("morphology_editing", "dendrogram"):
            self._refresh_simplification_panel_state()
        self._refresh_canvas_surface()

    def _on_precheck_requested(self):
        self._show_precheck_floating()

    def _on_batch_validation_ready(self, report: dict):
        self._batch_canvas.show_batch_validation_results(report)
        self._batch_canvas.set_mode(EditorTab.MODE_BATCH)
        self._batch_has_results = True
        self._refresh_canvas_surface()
        if self._active_tool == "batch":
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
        ed = self._active_editor()
        if ed is not None and hasattr(ed, "_dendro"):
            ed._dendro._undo_stack.undo()
            self._append_log("Undo.", "INFO")

    def _redo_edit(self):
        ed = self._active_editor()
        if ed is not None and hasattr(ed, "_dendro"):
            ed._dendro._undo_stack.redo()
            self._append_log("Redo.", "INFO")

    def closeEvent(self, event):
        self._closing_app = True
        try:
            for doc in list(self._documents.values()):
                self._write_closed_copy_and_log(doc)
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
            "4) You can open multiple SWC files as canvas tabs.\n"
            "5) Drag a canvas tab outside the tab bar to detach it into a floating window.\n"
            "6) Closing a SWC tab/window auto-saves edited SWC copy and morphology log.\n"
            "7) Validation uses a floating Rule Guide window above the canvas.\n"
            "8) Auto Label tabs use a floating Auto Typing Guide window with decision boundaries.\n"
            "9) Bottom panel shows all logs and warnings."
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
