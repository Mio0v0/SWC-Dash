"""Studio Split scientific UI demo (structure only, no backend logic).

Layout goals:
- Left narrow tool sidebar (icon-like buttons)
- Top thin contextual toolbar that changes per active tool
- Center tabbed canvas (dominant)
- Right fixed panes: Inspector (top) and Actions (bottom)
- Bottom compact event log
"""

from __future__ import annotations

import datetime as _dt
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StudioSplitUIDemo(QMainWindow):
    TOOL_LABELS = {
        "inspect": "Inspect",
        "validation": "Validation",
        "morph": "Morphology Editing",
        "batch": "Batch Processing",
        "visualization": "Visualization",
        "atlas": "Atlas Registration",
        "analysis": "Analysis",
    }

    TOOL_FEATURES = {
        "inspect": ["Overview", "Node Browse", "Segment Browse"],
        "validation": ["Run Validation", "Auto Label", "Radii Cleaning", "Rule Guide"],
        "morph": ["Dendrogram", "Type Reassign", "Simplification"],
        "batch": ["Split Folder", "Batch Validation", "Auto Typing", "Radii Cleaning"],
        "visualization": ["3D View", "2D Orthogonal", "Camera", "Color Mapping"],
        "atlas": ["Registration", "Template Select", "Plugin Method"],
        "analysis": ["Metrics", "Reports", "Plugin Method"],
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWC Studio - Studio Split UI Demo")
        self.resize(1680, 980)

        self._active_tool = "inspect"
        self._tool_buttons: dict[str, QToolButton] = {}

        self._build_ui()
        self._activate_tool("inspect")

    def _build_ui(self):
        self.setStyleSheet(
            "QWidget { font-family: 'Helvetica Neue', 'Segoe UI', Arial; font-size: 12px; }"
            "QPushButton { background: #f4f7fa; border: 1px solid #b8c2cd; border-radius: 4px; padding: 4px 8px; }"
            "QPushButton:hover { background: #e9eff6; }"
            "QToolButton { background: #eef3f8; border: 1px solid #c5ced8; border-radius: 6px; padding: 8px 4px; min-height: 52px; }"
            "QToolButton:checked { background: #d7e8f8; border: 1px solid #7ea8cf; font-weight: 700; }"
            "QTabWidget::pane { border: 1px solid #c9d1d9; background: #f7f9fb; }"
            "QTabBar::tab { background: #e9eef3; border: 1px solid #c9d1d9; padding: 6px 12px; margin-right: 1px; }"
            "QTabBar::tab:selected { background: #ffffff; font-weight: 600; }"
            "QMainWindow::separator { background: #b7bfc8; width: 8px; height: 8px; }"
        )

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_menu_and_context_bar(), stretch=0)
        root.addWidget(self._build_body(), stretch=1)
        root.addWidget(self._build_compact_log(), stretch=0)

        self.setCentralWidget(central)

    def _build_menu_and_context_bar(self) -> QWidget:
        wrapper = QFrame()
        wrapper.setFrameShape(QFrame.StyledPanel)
        wrapper.setStyleSheet("QFrame { background: #f2f5f8; border: 1px solid #c8d0d8; }")

        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        menu_bar = QMenuBar(wrapper)
        menu_bar.setNativeMenuBar(False)
        self._populate_menu_bar(menu_bar)
        layout.addWidget(menu_bar)

        context_bar = QWidget()
        row = QHBoxLayout(context_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._tool_title = QLabel("Tool: Inspect")
        self._tool_title.setStyleSheet("font-weight: 700; color: #29425b;")
        row.addWidget(self._tool_title)

        sep = QLabel("|")
        sep.setStyleSheet("color: #728497;")
        row.addWidget(sep)

        self._feature_scroll_host = QWidget()
        self._feature_row = QHBoxLayout(self._feature_scroll_host)
        self._feature_row.setContentsMargins(0, 0, 0, 0)
        self._feature_row.setSpacing(6)
        row.addWidget(self._feature_scroll_host, stretch=1)

        self._command_search = QLineEdit()
        self._command_search.setPlaceholderText("Command / search...")
        self._command_search.setMaximumWidth(280)
        row.addWidget(self._command_search)

        layout.addWidget(context_bar)
        return wrapper

    def _populate_menu_bar(self, menu_bar: QMenuBar):
        file_menu = menu_bar.addMenu("File")
        for txt in ("Open", "Save", "Save As", "Export", "Exit"):
            file_menu.addAction(QAction(txt, self))

        edit_menu = menu_bar.addMenu("Edit")
        for txt in ("Undo", "Redo", "Preferences"):
            edit_menu.addAction(QAction(txt, self))

        view_menu = menu_bar.addMenu("View")
        for txt in ("Reset Layout", "Theme", "Font Size"):
            view_menu.addAction(QAction(txt, self))

        window_menu = menu_bar.addMenu("Window")
        for txt in ("Show Inspector", "Show Actions", "Show Log"):
            window_menu.addAction(QAction(txt, self))

        help_menu = menu_bar.addMenu("Help")
        for txt in ("Quick Manual", "Shortcuts", "About"):
            help_menu.addAction(QAction(txt, self))

    def _build_body(self) -> QWidget:
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        split.addWidget(self._build_left_tool_sidebar())
        split.addWidget(self._build_center_canvas())
        split.addWidget(self._build_right_stack())

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([100, 1050, 420])
        return split

    def _build_left_tool_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet("QFrame { background: #f6f8fb; border: 1px solid #ccd5df; }")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        items = [
            ("inspect", "I", "Inspect"),
            ("validation", "V", "Validation"),
            ("morph", "M", "Morph"),
            ("batch", "B", "Batch"),
            ("visualization", "3D", "Visualize"),
            ("atlas", "A", "Atlas"),
            ("analysis", "N", "Analysis"),
        ]
        for key, short, caption in items:
            b = QToolButton()
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setText(f"{short}\n{caption}")
            b.clicked.connect(lambda _=False, k=key: self._activate_tool(k))
            layout.addWidget(b)
            self._tool_buttons[key] = b

        layout.addStretch()
        panel.setMinimumWidth(92)
        panel.setMaximumWidth(116)
        return panel

    def _build_center_canvas(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setTabsClosable(True)
        tabs.setMovable(True)

        tabs.addTab(self._build_black_canvas("3D Canvas - cell_A.swc"), "cell_A.swc")
        tabs.addTab(self._build_black_canvas("3D Canvas - cell_B.swc"), "cell_B.swc")
        tabs.addTab(self._build_black_canvas("Preview Canvas - auto label"), "cell_A [Preview]")
        return tabs

    def _build_black_canvas(self, text: str) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #000000; border: 1px solid #303030; }")
        lay = QVBoxLayout(frame)

        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("QLabel { color: #a9b7c6; font-size: 13px; }")
        lay.addWidget(lbl)
        return frame

    def _build_right_stack(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)

        inspector = self._build_inspector_pane()
        actions = self._build_actions_pane()

        vertical.addWidget(inspector)
        vertical.addWidget(actions)
        vertical.setStretchFactor(0, 2)
        vertical.setStretchFactor(1, 3)
        vertical.setSizes([310, 520])

        root.addWidget(vertical)
        panel.setMinimumWidth(360)
        return panel

    def _build_inspector_pane(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #ccd5df; }")

        root = QVBoxLayout(frame)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("Inspector")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #2f4d68;")
        root.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_swc_tree(), "SWC")
        tabs.addTab(self._build_info_table("Node Info"), "Node")
        tabs.addTab(self._build_info_table("Segment Info"), "Segment")
        root.addWidget(tabs)
        return frame

    def _build_swc_tree(self) -> QWidget:
        tree = QTreeWidget()
        tree.setHeaderLabels(["Loaded SWC Files"])

        root_a = QTreeWidgetItem(["cell_A.swc"])
        root_a.addChild(QTreeWidgetItem(["nodes: 28,075 | status: validated"]))

        root_b = QTreeWidgetItem(["cell_B.swc"])
        root_b.addChild(QTreeWidgetItem(["nodes: 6,155 | status: edited"]))

        tree.addTopLevelItem(root_a)
        tree.addTopLevelItem(root_b)
        return tree

    def _build_info_table(self, kind: str) -> QWidget:
        table = QTableWidget(5, 2)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        rows = [
            ("View", kind),
            ("Current ID", "24207"),
            ("Type", "2 (axon)"),
            ("Radius", "0.52"),
            ("Parent", "24206"),
        ]
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(k))
            table.setItem(i, 1, QTableWidgetItem(v))
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _build_actions_pane(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #ccd5df; }")

        root = QVBoxLayout(frame)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._actions_title = QLabel("Actions")
        self._actions_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #2f4d68;")
        root.addWidget(self._actions_title)

        self._actions_host = QWidget()
        self._actions_grid = QGridLayout(self._actions_host)
        self._actions_grid.setContentsMargins(0, 0, 0, 0)
        self._actions_grid.setHorizontalSpacing(6)
        self._actions_grid.setVerticalSpacing(6)
        root.addWidget(self._actions_host)

        root.addStretch()
        return frame

    def _build_compact_log(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #11161d; border: 1px solid #2a3744; }")

        row = QVBoxLayout(frame)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(4)

        head = QLabel("Event Log")
        head.setStyleSheet("QLabel { color: #b7c4d1; font-size: 11px; font-weight: 600; }")
        row.addWidget(head)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        self._log.setStyleSheet(
            "QPlainTextEdit { background: #0f141a; color: #d6dde5; border: 1px solid #2b3947;"
            " font-family: Menlo, Consolas, monospace; font-size: 11px; }"
        )
        row.addWidget(self._log)
        return frame

    def _activate_tool(self, key: str):
        if key not in self.TOOL_LABELS:
            return

        self._active_tool = key
        self._tool_title.setText(f"Tool: {self.TOOL_LABELS[key]}")

        for tool_key, button in self._tool_buttons.items():
            selected = tool_key == key
            button.setChecked(selected)

        self._rebuild_feature_buttons()
        self._rebuild_actions_panel()
        self._append_log(f"Tool selected: {self.TOOL_LABELS[key]}")

    def _rebuild_feature_buttons(self):
        while self._feature_row.count() > 0:
            item = self._feature_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        features = self.TOOL_FEATURES.get(self._active_tool, [])
        for feature in features:
            b = QPushButton(feature)
            b.clicked.connect(lambda _=False, f=feature: self._select_feature(f))
            self._feature_row.addWidget(b)

        self._feature_row.addStretch()

    def _select_feature(self, feature: str):
        self._actions_title.setText(f"Actions - {feature}")
        self._append_log(f"Feature selected: {feature}")

    def _rebuild_actions_panel(self):
        while self._actions_grid.count() > 0:
            item = self._actions_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        items = {
            "inspect": ["Focus Node", "Center View", "Show Segment"],
            "validation": ["Run Validation", "Download Report", "Rule Guide", "Edit Validation JSON"],
            "morph": ["Assign Type", "Apply", "Undo", "Cancel"],
            "batch": ["Select Folder", "Run Batch", "Edit JSON", "Download Summary"],
            "visualization": ["Reset Camera", "Top", "Front", "Side", "Color by Type"],
            "atlas": ["Choose Atlas", "Run Registration", "Plugin Method"],
            "analysis": ["Run Metrics", "Export CSV", "Plugin Method"],
        }.get(self._active_tool, [])

        for i, text in enumerate(items):
            r = i // 2
            c = i % 2
            self._actions_grid.addWidget(QPushButton(text), r, c)

    def _append_log(self, message: str):
        t = _dt.datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{t}] {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = StudioSplitUIDemo()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
