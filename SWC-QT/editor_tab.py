"""Central visualization workspace for 3D, dendrogram, and multiview layouts."""

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from constants import DEFAULT_COLORS, label_for_type
from dendrogram_widget import DendrogramWidget
from neuron_3d_widget import Neuron3DWidget


class _Projection2DWidget(QWidget):
    """Simple 2D projected view of SWC segments."""

    def __init__(self, title: str, x_col: str, y_col: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._x_col = x_col
        self._y_col = y_col
        self._df: pd.DataFrame | None = None
        self._highlight_id: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 12px; color: #333;")
        layout.addWidget(label)

        self._plot = pg.PlotWidget(background="white")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.showGrid(x=False, y=False, alpha=0.1)
        self._plot.getAxis("bottom").setLabel(x_col)
        self._plot.getAxis("left").setLabel(y_col)
        layout.addWidget(self._plot, stretch=1)

    def load_swc(self, df: pd.DataFrame):
        self._df = df.copy()
        self._draw()

    def refresh(self, df: pd.DataFrame):
        self._df = df.copy()
        self._draw()

    def highlight_node(self, swc_id: int):
        self._highlight_id = swc_id
        self._draw()

    def _draw(self):
        self._plot.clear()
        if self._df is None or self._df.empty:
            return

        df = self._df
        ids = df["id"].to_numpy(dtype=int)
        types = df["type"].to_numpy(dtype=int)
        parents = df["parent"].to_numpy(dtype=int)
        x = df[self._x_col].to_numpy(dtype=float)
        y = df[self._y_col].to_numpy(dtype=float)
        id2idx = {int(ids[i]): i for i in range(len(ids))}

        lines_by_label: dict[str, list[float]] = {}
        for i in range(len(ids)):
            pid = int(parents[i])
            if pid < 0 or pid not in id2idx:
                continue
            p_idx = id2idx[pid]
            label = label_for_type(int(types[i]))
            lines = lines_by_label.setdefault(label, [])
            lines.extend([x[p_idx], x[i], np.nan])

        y_by_label: dict[str, list[float]] = {}
        for i in range(len(ids)):
            pid = int(parents[i])
            if pid < 0 or pid not in id2idx:
                continue
            p_idx = id2idx[pid]
            label = label_for_type(int(types[i]))
            rows = y_by_label.setdefault(label, [])
            rows.extend([y[p_idx], y[i], np.nan])

        for label, color in DEFAULT_COLORS.items():
            xs = lines_by_label.get(label)
            ys = y_by_label.get(label)
            if not xs or not ys:
                continue
            self._plot.plot(
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                pen=pg.mkPen(color=color, width=1.2),
                connect="finite",
            )

        if self._highlight_id is not None:
            row = df[df["id"] == int(self._highlight_id)]
            if not row.empty:
                hx = float(row.iloc[0][self._x_col])
                hy = float(row.iloc[0][self._y_col])
                self._plot.addItem(
                    pg.ScatterPlotItem(
                        [hx], [hy], size=10, symbol="o",
                        pen=pg.mkPen("#e11", width=1.5),
                        brush=pg.mkBrush("#ff444466"),
                    )
                )


class EditorTab(QWidget):
    """Workspace with mode switching: canvas, dendrogram, and visualization."""

    df_changed = Signal(pd.DataFrame)

    MODE_CANVAS = "canvas"
    MODE_DENDRO = "dendrogram"
    MODE_VIS = "visualization"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._mode = self.MODE_CANVAS
        self._has_data = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        root_layout.addWidget(self._stack)

        self._page_empty = QWidget()
        self._page_empty.setStyleSheet("background: #000;")
        self._stack.addWidget(self._page_empty)

        self._view3d_canvas = Neuron3DWidget()
        self._page_canvas = QWidget()
        canvas_layout = QVBoxLayout(self._page_canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._view3d_canvas)
        self._stack.addWidget(self._page_canvas)

        self._view3d_dendro = Neuron3DWidget()
        self._dendro = DendrogramWidget()
        self._page_dendro = QWidget()
        dendro_layout = QVBoxLayout(self._page_dendro)
        dendro_layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._view3d_dendro)
        split.addWidget(self._dendro)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        dendro_layout.addWidget(split)
        self._stack.addWidget(self._page_dendro)

        self._view3d_visual = Neuron3DWidget()
        self._proj_xy = _Projection2DWidget("Top View (X-Y)", "x", "y")
        self._proj_xz = _Projection2DWidget("Front View (X-Z)", "x", "z")
        self._proj_yz = _Projection2DWidget("Side View (Y-Z)", "y", "z")

        proj_row = QWidget()
        proj_layout = QHBoxLayout(proj_row)
        proj_layout.setContentsMargins(0, 0, 0, 0)
        proj_layout.setSpacing(6)
        proj_layout.addWidget(self._proj_xy)
        proj_layout.addWidget(self._proj_xz)
        proj_layout.addWidget(self._proj_yz)

        self._page_visual = QWidget()
        visual_layout = QVBoxLayout(self._page_visual)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_split = QSplitter(Qt.Vertical)
        visual_split.addWidget(self._view3d_visual)
        visual_split.addWidget(proj_row)
        visual_split.setStretchFactor(0, 3)
        visual_split.setStretchFactor(1, 2)
        visual_layout.addWidget(visual_split)
        self._stack.addWidget(self._page_visual)

        self._dendro.df_changed.connect(self._on_df_changed)
        self._dendro.node_selected.connect(self._on_node_selected)

        self._show_current_mode()

    # --------------------------------------------------------- Public API
    def load_swc(self, df: pd.DataFrame, filename: str = ""):
        self._df = df.copy()
        self._has_data = True
        self._dendro.load_swc(df, filename)
        self._view3d_canvas.load_swc(df, filename)
        self._view3d_dendro.load_swc(df, filename)
        self._view3d_visual.load_swc(df, filename)
        self._proj_xy.load_swc(df)
        self._proj_xz.load_swc(df)
        self._proj_yz.load_swc(df)
        self._show_current_mode()

    def set_mode(self, mode: str):
        if mode in (self.MODE_CANVAS, self.MODE_DENDRO, self.MODE_VIS):
            self._mode = mode
            self._show_current_mode()

    def take_dendrogram_controls_panel(self) -> QWidget:
        return self._dendro.take_controls_panel()

    def set_render_mode(self, mode_id: int):
        self._view3d_canvas.set_render_mode(mode_id)
        self._view3d_dendro.set_render_mode(mode_id)
        self._view3d_visual.set_render_mode(mode_id)

    def set_camera_view(self, preset: str):
        self._active_view().set_camera_view(preset)

    def reset_camera(self):
        self._active_view().reset_camera()

    def _active_view(self) -> Neuron3DWidget:
        if self._mode == self.MODE_DENDRO:
            return self._view3d_dendro
        if self._mode == self.MODE_VIS:
            return self._view3d_visual
        return self._view3d_canvas

    # ------------------------------------------------- Sync
    def _on_df_changed(self, df: pd.DataFrame):
        self._df = df.copy()
        self._view3d_canvas.refresh(df)
        self._view3d_dendro.refresh(df)
        self._view3d_visual.refresh(df)
        self._proj_xy.refresh(df)
        self._proj_xz.refresh(df)
        self._proj_yz.refresh(df)
        self.df_changed.emit(df)

    def _on_node_selected(self, swc_id: int, node_type: int, level: int):
        self._view3d_canvas.highlight_node(swc_id)
        self._view3d_dendro.highlight_node(swc_id)
        self._view3d_visual.highlight_node(swc_id)
        self._proj_xy.highlight_node(swc_id)
        self._proj_xz.highlight_node(swc_id)
        self._proj_yz.highlight_node(swc_id)

    def _show_current_mode(self):
        if not self._has_data:
            self._stack.setCurrentWidget(self._page_empty)
            return
        if self._mode == self.MODE_DENDRO:
            self._stack.setCurrentWidget(self._page_dendro)
            return
        if self._mode == self.MODE_VIS:
            self._stack.setCurrentWidget(self._page_visual)
            return
        self._stack.setCurrentWidget(self._page_canvas)
