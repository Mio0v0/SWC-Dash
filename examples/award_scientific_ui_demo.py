"""Studio-split scientific UI demo (structure only, no backend logic).

Design goals reflected:
- No Session group in ribbon
- Data Explorer on left
- Full-height vertical Control Center on right
- Tool subfeatures shown in Context (horizontal scroll strip)
- Clicking a subfeature updates Control Center content
"""

from __future__ import annotations

import datetime as _dt
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AwardScientificUIDemo(QMainWindow):
    """High-level UI template for a clean scientific desktop workflow."""

    TOOL_LABELS = {
        "": "None",
        "inspect": "Inspect",
        "validation": "Validation",
        "morph": "Morphology Editing",
        "batch": "Batch Processing",
        "visualization": "Visualization",
        "atlas": "Atlas Registration",
        "analysis": "Analysis",
    }

    SUBFEATURES = {
        "inspect": [("overview", "Overview")],
        "batch": [
            ("split", "Split"),
            ("validation", "Validation"),
            ("auto_label", "Auto Label"),
            ("radii", "Radii Cleaning"),
        ],
        "validation": [
            ("validation", "Validation"),
            ("auto_label", "Auto Label"),
            ("radii", "Radii Cleaning"),
        ],
        "morph": [
            ("label_edit", "Label Editing"),
            ("simplification", "Simplification"),
        ],
        "visualization": [("view_controls", "View Controls")],
        "atlas": [
            ("registration", "Registration"),
            ("plugins", "Plugins"),
        ],
        "analysis": [
            ("summary", "Summary"),
            ("plugins", "Plugins"),
        ],
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWC Studio - Scientific Workstation Template")
        self.resize(1680, 980)

        self._active_tool = ""
        self._active_subfeature = ""
        self._active_subfeature_label = ""

        self._build_ui()
        self._set_tool("inspect")

    # ------------------------------------------------------------------ shell
    def _build_ui(self):
        self.setDockOptions(
            QMainWindow.AnimatedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
        )
        self.setStyleSheet(
            "QWidget { font-family: 'Helvetica Neue', 'Segoe UI', Arial; font-size: 12px; }"
            "QMainWindow::separator { background: #b7bfc8; width: 8px; height: 8px; }"
            "QTabWidget::pane { border: 1px solid #c9d1d9; background: #f7f9fb; }"
            "QTabBar::tab { background: #e9eef3; border: 1px solid #c9d1d9; padding: 6px 12px; margin-right: 1px; }"
            "QTabBar::tab:selected { background: #ffffff; font-weight: 600; }"
            "QTabBar::tab:hover { background: #dfe7ef; }"
            "QPushButton { background: #f3f6f9; border: 1px solid #b8c2cd; border-radius: 5px; padding: 6px 10px; }"
            "QPushButton:hover { background: #e8eef5; }"
            "QPushButton:pressed { background: #dce6f0; }"
            "QGroupBox { border: 1px solid #d1d8df; border-radius: 6px; margin-top: 8px; background: #ffffff; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #35516b; font-weight: 600; }"
        )

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._build_top_system_bar(), stretch=0)
        root.addWidget(self._build_workspace_area(), stretch=1)
        self.setCentralWidget(central)

        self._build_data_explorer_dock()
        self._build_control_center_dock()
        self._build_log_dock()
        self._build_guides_docks()

        self.resizeDocks([self._data_dock, self._control_dock], [280, 380], Qt.Horizontal)
        self.resizeDocks([self._log_dock], [170], Qt.Vertical)

    # ------------------------------------------------------------------ top bar
    def _build_top_system_bar(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #f2f5f8; border: 1px solid #c8d0d8; }")

        root = QVBoxLayout(frame)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        menu_bar = QMenuBar(frame)
        menu_bar.setNativeMenuBar(False)
        menu_bar.setStyleSheet(
            "QMenuBar { background: #f2f5f8; border-bottom: 1px solid #d0d7de; }"
            "QMenuBar::item { padding: 6px 12px; }"
            "QMenuBar::item:selected { background: #e4ebf2; }"
        )
        self._populate_menu_bar(menu_bar)
        root.addWidget(menu_bar)

        root.addWidget(self._build_ribbon())
        return frame

    def _populate_menu_bar(self, menu_bar: QMenuBar):
        file_menu = menu_bar.addMenu("File")
        for txt in ("Open", "Save", "Save As", "Export", "Recent Files", "Exit"):
            file_menu.addAction(QAction(txt, self))

        edit_menu = menu_bar.addMenu("Edit")
        for txt in ("Undo", "Redo", "Preferences"):
            edit_menu.addAction(QAction(txt, self))

        view_menu = menu_bar.addMenu("View")
        for txt in ("Reset Layout", "Show/Hide Panels", "Theme", "Font Size"):
            view_menu.addAction(QAction(txt, self))

        window_menu = menu_bar.addMenu("Window")
        for txt in ("Data Explorer", "Control Center", "Log", "Rule Guide", "Auto Typing Guide"):
            window_menu.addAction(QAction(txt, self))

        help_menu = menu_bar.addMenu("Help")
        for txt in ("Quick Manual", "Shortcuts", "About"):
            help_menu.addAction(QAction(txt, self))

    def _build_ribbon(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        group = QGroupBox("Tools + Features")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(6)

        tools_row = QHBoxLayout()
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(6)

        self._tool_button_map = {}
        tool_map = [
            ("Inspect", "inspect"),
            ("Validate", "validation"),
            ("Edit Morph", "morph"),
            ("Batch", "batch"),
            ("Visualize", "visualization"),
            ("Atlas", "atlas"),
            ("Analysis", "analysis"),
        ]
        for label, key in tool_map:
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self._set_tool(k))
            tools_row.addWidget(b)
            self._tool_button_map[key] = b

        clear_tool = QPushButton("Clear Tool")
        clear_tool.clicked.connect(lambda: self._set_tool(""))
        tools_row.addWidget(clear_tool)
        tools_row.addStretch()
        group_layout.addLayout(tools_row)

        feature_caption = QLabel("Features")
        feature_caption.setStyleSheet("color: #4d6073; font-weight: 600;")
        group_layout.addWidget(feature_caption)

        self._subfeature_scroll = QScrollArea()
        self._subfeature_scroll.setWidgetResizable(True)
        self._subfeature_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._subfeature_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._subfeature_scroll.setFrameShape(QFrame.NoFrame)

        self._subfeature_strip = QWidget()
        self._subfeature_row = QHBoxLayout(self._subfeature_strip)
        self._subfeature_row.setContentsMargins(0, 0, 0, 0)
        self._subfeature_row.setSpacing(6)
        self._subfeature_row.addStretch()
        self._subfeature_scroll.setWidget(self._subfeature_strip)
        group_layout.addWidget(self._subfeature_scroll)

        root.addWidget(group)
        return panel
    # ------------------------------------------------------------------ workspace
    def _build_workspace_area(self) -> QWidget:
        wrapper = QWidget()
        root = QVBoxLayout(wrapper)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self._workspace_stack = QStackedWidget()
        self._workspace_stack.addWidget(self._build_default_workspace())
        self._workspace_stack.addWidget(self._build_visualization_workspace())
        self._workspace_stack.addWidget(self._build_batch_workspace())
        root.addWidget(self._workspace_stack, stretch=1)
        return wrapper

    def _build_default_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self._doc_tabs_default = QTabWidget()
        self._doc_tabs_default.setTabsClosable(True)
        self._doc_tabs_default.setMovable(True)
        self._doc_tabs_default.addTab(self._build_black_canvas("3D Workspace - cell_A.swc"), "cell_A.swc")
        self._doc_tabs_default.addTab(self._build_black_canvas("3D Workspace - cell_B.swc"), "cell_B.swc")
        self._doc_tabs_default.addTab(self._build_black_canvas("Preview Workspace - auto label"), "cell_A [Preview]")
        layout.addWidget(self._doc_tabs_default)
        return page

    def _build_visualization_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.setTabsClosable(True)
        tabs.setMovable(True)
        tabs.addTab(self._build_visual_page("cell_A.swc"), "cell_A.swc")
        tabs.addTab(self._build_visual_page("cell_B.swc"), "cell_B.swc")
        layout.addWidget(tabs)
        return page

    def _build_batch_workspace(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        grid.addWidget(self._metric_card("Input Folder", "/demo/project/data_batch"), 0, 0)
        grid.addWidget(self._metric_card("Files", "152"), 0, 1)
        grid.addWidget(self._metric_card("Processed", "148"), 0, 2)
        grid.addWidget(self._metric_card("Warnings", "21"), 0, 3)

        table = QTableWidget(8, 4)
        table.setHorizontalHeaderLabels(["File", "Task", "Status", "Notes"])
        for r in range(8):
            table.setItem(r, 0, QTableWidgetItem(f"cell_{r+1:03d}.swc"))
            table.setItem(r, 1, QTableWidgetItem("Validation + Auto Label"))
            table.setItem(r, 2, QTableWidgetItem("Completed" if r % 3 else "Warning"))
            table.setItem(r, 3, QTableWidgetItem("Report downloadable"))
        table.horizontalHeader().setStretchLastSection(True)
        grid.addWidget(table, 1, 0, 1, 4)

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(87)
        grid.addWidget(progress, 2, 0, 1, 4)
        return page

    def _metric_card(self, title: str, value: str) -> QWidget:
        w = QFrame()
        w.setFrameShape(QFrame.StyledPanel)
        w.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #d1d8df; border-radius: 6px; }")
        l = QVBoxLayout(w)
        t = QLabel(title)
        t.setStyleSheet("color: #5a6f83; font-size: 11px;")
        v = QLabel(value)
        v.setStyleSheet("color: #1f3550; font-size: 18px; font-weight: 700;")
        l.addWidget(t)
        l.addWidget(v)
        return w

    def _build_black_canvas(self, text: str) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #000000; border: 1px solid #303030; }")
        l = QVBoxLayout(frame)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("QLabel { color: #a9b7c6; font-size: 13px; }")
        l.addWidget(lbl)
        return frame

    def _build_visual_page(self, title: str) -> QWidget:
        page = QWidget()
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)

        top = self._build_black_canvas(f"3D Visualization - {title}")
        split.addWidget(top)

        bottom = QWidget()
        row = QHBoxLayout(bottom)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._soft_placeholder("Top 2D"))
        row.addWidget(self._soft_placeholder("Front 2D"))
        row.addWidget(self._soft_placeholder("Side 2D"))
        split.addWidget(bottom)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        return page

    def _soft_placeholder(self, text: str) -> QWidget:
        w = QFrame()
        w.setFrameShape(QFrame.StyledPanel)
        w.setStyleSheet("QFrame { background: #f8fafc; border: 1px solid #d3dbe3; }")
        l = QVBoxLayout(w)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("QLabel { color: #68829a; }")
        l.addWidget(lbl)
        return w

    # ------------------------------------------------------------------ docks
    def _build_data_explorer_dock(self):
        tabs = QTabWidget()
        tabs.addTab(self._build_swc_file_tab(), "SWC File")
        tabs.addTab(self._soft_placeholder("Node Info"), "Node Info")
        tabs.addTab(self._soft_placeholder("Segment Info"), "Segment Info")
        tabs.addTab(self._soft_placeholder("Edit Log"), "Edit Log")

        self._data_dock = QDockWidget("Data Explorer", self)
        self._data_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._data_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._data_dock.setWidget(tabs)
        self._data_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._data_dock)

    def _build_control_center_dock(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Control Center")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f4d68;")
        layout.addWidget(title)

        self._control_scroll = QScrollArea()
        self._control_scroll.setWidgetResizable(True)
        self._control_scroll.setFrameShape(QFrame.NoFrame)

        self._control_content = QWidget()
        self._control_content_layout = QVBoxLayout(self._control_content)
        self._control_content_layout.setContentsMargins(0, 0, 0, 0)
        self._control_content_layout.setSpacing(8)
        self._control_content_layout.addStretch()

        self._control_scroll.setWidget(self._control_content)
        layout.addWidget(self._control_scroll, stretch=1)

        self._control_dock = QDockWidget("Control Center", self)
        self._control_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._control_dock.setWidget(container)
        self._control_dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, self._control_dock)

    def _build_log_dock(self):
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "QPlainTextEdit { background: #0f141a; color: #d6dde5; border: 1px solid #2b3947;"
            " font-family: Menlo, Consolas, monospace; font-size: 12px; }"
        )
        self._append_log("Template loaded")
        self._append_log("UI-only demonstration. No processing backend attached.")

        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self._log_dock.setWidget(self._log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)

    def _build_guides_docks(self):
        rule_text = QPlainTextEdit()
        rule_text.setReadOnly(True)
        rule_text.setPlainText("Rule Guide\n\nClick from Context or Control Center when needed.")
        self._rule_guide_dock = QDockWidget("Rule Guide", self)
        self._rule_guide_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._rule_guide_dock.setWidget(rule_text)
        self.addDockWidget(Qt.TopDockWidgetArea, self._rule_guide_dock)
        self._rule_guide_dock.hide()

        auto_text = QPlainTextEdit()
        auto_text.setReadOnly(True)
        auto_text.setPlainText("Auto Typing Guide\n\nDecision boundaries, thresholds, and method notes.")
        self._auto_guide_dock = QDockWidget("Auto Typing Guide", self)
        self._auto_guide_dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self._auto_guide_dock.setWidget(auto_text)
        self.addDockWidget(Qt.TopDockWidgetArea, self._auto_guide_dock)
        self._auto_guide_dock.hide()

    # ------------------------------------------------------------------ behavior
    def _set_tool(self, key: str):
        self._active_tool = str(key or "").strip().lower()
        label = self.TOOL_LABELS.get(self._active_tool, "None")

        for tool_key, tool_btn in self._tool_button_map.items():
            selected = tool_key == self._active_tool and bool(self._active_tool)
            tool_btn.setChecked(selected)
            if selected:
                tool_btn.setStyleSheet("QPushButton { background: #d7e8f8; border: 1px solid #7ea8cf; font-weight: 700; }")
            else:
                tool_btn.setStyleSheet("")

        # workspace mode
        if self._active_tool == "visualization":
            self._workspace_stack.setCurrentIndex(1)
        elif self._active_tool == "batch":
            self._workspace_stack.setCurrentIndex(2)
        else:
            self._workspace_stack.setCurrentIndex(0)

        self._refresh_subfeature_strip()
        self._append_log(f"Tool switched: {label}")

    def _refresh_subfeature_strip(self):
        # clear old buttons
        while self._subfeature_row.count() > 0:
            item = self._subfeature_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        items = self.SUBFEATURES.get(self._active_tool, [])
        if not items:
            self._active_subfeature = ""
            self._active_subfeature_label = ""
            self._render_control_center()
            msg = QLabel("No subfeatures for this mode")
            msg.setStyleSheet("color: #687887;")
            self._subfeature_row.addWidget(msg)
            self._subfeature_row.addStretch()
            return

        for sub_key, sub_label in items:
            b = QPushButton(sub_label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=sub_key, lb=sub_label: self._set_subfeature(k, lb))
            self._subfeature_row.addWidget(b)

        self._subfeature_row.addStretch()
        first_key, first_label = items[0]
        self._set_subfeature(first_key, first_label)

    def _set_subfeature(self, key: str, label: str):
        self._active_subfeature = str(key)
        self._active_subfeature_label = str(label)

        # highlight selected button
        for i in range(self._subfeature_row.count()):
            it = self._subfeature_row.itemAt(i)
            w = it.widget()
            if isinstance(w, QPushButton) and w.isCheckable():
                w.setChecked(w.text() == label)
                if w.isChecked():
                    w.setStyleSheet("QPushButton { background: #d7e8f8; border: 1px solid #7ea8cf; font-weight: 700; }")
                else:
                    w.setStyleSheet("")

        self._render_control_center()
        self._append_log(f"Subfeature selected: {label}")

    def _render_control_center(self):
        lay = self._control_content_layout
        while lay.count() > 0:
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self._active_tool:
            lbl = QLabel("No active tool selected.")
            lbl.setStyleSheet("color: #687887;")
            lay.addWidget(lbl)
            lay.addStretch()
            return

        head = QLabel(f"{self.TOOL_LABELS.get(self._active_tool, 'None')} / {self._active_subfeature_label}")
        head.setStyleSheet("font-size: 13px; font-weight: 700; color: #2f4d68;")
        lay.addWidget(head)

        status = QLabel("Status: Ready")
        status.setStyleSheet("color: #1f7a35; font-weight: 600;")
        lay.addWidget(status)

        params = QGroupBox("Parameters")
        pl = QVBoxLayout(params)
        for txt in (
            "Threshold A",
            "Threshold B",
            "Method selector",
            "Rule/Config editor",
        ):
            pl.addWidget(QPushButton(txt))
        lay.addWidget(params)

        actions = QGroupBox("Actions")
        al = QHBoxLayout(actions)
        for txt in ("Run", "Apply", "Cancel", "Download Report"):
            al.addWidget(QPushButton(txt))
        lay.addWidget(actions)

        guides = QGroupBox("Guides")
        gl = QHBoxLayout(guides)
        b_rule = QPushButton("Open Rule Guide")
        b_rule.clicked.connect(lambda: self._rule_guide_dock.show())
        gl.addWidget(b_rule)
        b_auto = QPushButton("Open Auto Typing Guide")
        b_auto.clicked.connect(lambda: self._auto_guide_dock.show())
        gl.addWidget(b_auto)
        gl.addStretch()
        lay.addWidget(guides)

        lay.addStretch()

    # ------------------------------------------------------------------ misc
    def _build_swc_file_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Loaded SWC Files"])

        root_a = QTreeWidgetItem(["cell_A.swc"])
        root_a.addChild(QTreeWidgetItem(["nodes: 28,075 | status: validated"]))
        root_a.setExpanded(False)

        root_b = QTreeWidgetItem(["cell_B.swc"])
        root_b.addChild(QTreeWidgetItem(["nodes: 6,155 | status: edited"]))
        root_b.setExpanded(False)

        tree.addTopLevelItem(root_a)
        tree.addTopLevelItem(root_b)
        layout.addWidget(tree)
        return panel

    def _append_log(self, msg: str):
        t = _dt.datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{t}] {msg}")



def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SWC Scientific UI Template")
    app.setStyle("Fusion")

    win = AwardScientificUIDemo()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
