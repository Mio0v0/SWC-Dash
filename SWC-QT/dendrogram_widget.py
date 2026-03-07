"""2D Dendrogram editor widget using pyqtgraph for instant rendering."""

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QColor, QFont, QUndoStack, QUndoCommand,
    QPainter, QPen, QBrush, QFontMetrics, QPainterPath, QPolygonF,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QSpinBox, QCheckBox,
    QGroupBox, QSizePolicy, QScrollArea, QSplitter,
    QGraphicsItem,
)

from graph_utils import (
    build_tree_cache, find_all_roots, cumlens_from_root_cache,
    layout_y_positions_cache, compute_levels, merge_dangling_trees,
)
from constants import (
    color_for_type, label_for_type, DEFAULT_COLORS, SWC_COLS,
)


# ------------------------------------------------------------------ Undo
class TypeChangeCommand(QUndoCommand):
    """Undoable command for changing node type(s)."""

    def __init__(self, widget, indices: list, old_types: list, new_type: int, description=""):
        super().__init__(description or f"Change {len(indices)} node(s) to type {new_type}")
        self._widget = widget
        self._indices = indices
        self._old_types = old_types
        self._new_type = new_type

    def redo(self):
        df = self._widget._df
        for idx in self._indices:
            df.at[idx, "type"] = self._new_type
        self._widget._rebuild_and_draw()

    def undo(self):
        df = self._widget._df
        for idx, old_t in zip(self._indices, self._old_types):
            df.at[idx, "type"] = old_t
        self._widget._rebuild_and_draw()


# --------------------------------------------------------- Speech-bubble tooltip
class BubbleTooltip(QGraphicsItem):
    """A speech-bubble shaped tooltip: rounded rect + triangle pointer at bottom."""

    PADDING = 6
    ARROW_W = 8
    ARROW_H = 6
    RADIUS = 4

    def __init__(self):
        super().__init__()
        self._text = ""
        self._font = QFont("sans-serif", 10)
        self._bg = QColor(230, 230, 230, 230)
        self._border = QColor(180, 180, 180)
        self._text_color = QColor(50, 50, 50)
        self._rect = QRectF()
        self.setVisible(False)
        self.setZValue(1000)
        # Paint in screen pixels, not data coordinates — prevents flip
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def set_text(self, text: str):
        self.prepareGeometryChange()
        self._text = text
        fm = QFontMetrics(self._font)
        tw = fm.horizontalAdvance(text) + 2 * self.PADDING
        th = fm.height() + 2 * self.PADDING
        # Box centered above the anchor point; arrow at bottom-center
        self._rect = QRectF(-tw / 2, -(th + self.ARROW_H), tw, th)

    def set_color(self, hex_color: str):
        """Tint the bubble to match a type color."""
        c = QColor(hex_color)
        self._border = c
        # Light tinted background
        self._bg = QColor(c.red(), c.green(), c.blue(), 45)
        self._text_color = c.darker(150)
        self.update()

    def boundingRect(self):
        r = self._rect
        return QRectF(r.x() - 2, r.y() - 2,
                      r.width() + 4, r.height() + self.ARROW_H + 4)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """Return None so pyqtgraph ignores this item in auto-range."""
        return None

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        r = self._rect

        # Build path: rounded rect + triangle pointer
        path = QPainterPath()
        path.addRoundedRect(r, self.RADIUS, self.RADIUS)

        # Triangle at bottom-center of the rect
        tri = QPolygonF([
            QPointF(-self.ARROW_W / 2, r.bottom()),
            QPointF(self.ARROW_W / 2, r.bottom()),
            QPointF(0, r.bottom() + self.ARROW_H),
        ])
        path.addPolygon(tri)

        painter.setPen(QPen(self._border, 1))
        painter.setBrush(QBrush(self._bg))
        painter.drawPath(path)

        # Draw text
        painter.setPen(QPen(self._text_color))
        painter.setFont(self._font)
        painter.drawText(r, Qt.AlignCenter, self._text)


# ------------------------------------------------------------------ Widget
class DendrogramWidget(QWidget):
    """Interactive 2D dendrogram using pyqtgraph."""

    node_selected = Signal(int, int, int)   # swc_id, type, level
    df_changed = Signal(pd.DataFrame)       # emitted after edits

    TYPE_LABELS = {
        0: "undefined", 1: "soma", 2: "axon",
        3: "basal dendrite", 4: "apical dendrite", 5: "custom",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._tree = None
        self._roots = []
        self._compress = True
        self._selected_idx: int | None = None  # index into tree arrays
        self._level_val: int | None = None

        self._undo_stack = QUndoStack(self)

        # Internal arrays (recomputed on rebuild)
        self._cum = None
        self._y = None
        self._levels = None

        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # Left: plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(background="w")
        self._plot.setLabel("bottom", "Path length from soma (µm)")
        self._plot.getAxis("left").setStyle(showValues=False)
        self._plot.getAxis("left").setTicks([])
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.scene().sigMouseClicked.connect(self._on_click)

        # Hover tooltip
        self._hover_label = pg.TextItem(anchor=(0, 1), color="#333")
        self._hover_label.setVisible(False)
        self._plot.addItem(self._hover_label)
        self._hover_proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_hover
        )

        plot_layout.addWidget(self._plot)
        splitter.addWidget(plot_container)

        # Right: controls panel
        controls = QWidget()
        controls.setMaximumWidth(280)
        controls.setMinimumWidth(220)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)

        # --- Node Info ---
        info_group = QGroupBox("Selected Node")
        info_layout = QVBoxLayout(info_group)
        self._info_label = QLabel("Click an edge to select a node.")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size: 13px; color: #444;")
        info_layout.addWidget(self._info_label)
        ctrl_layout.addWidget(info_group)

        # --- Type Change ---
        type_group = QGroupBox("Change Type")
        type_layout = QVBoxLayout(type_group)

        self._type_buttons = QButtonGroup(self)
        for t, name in self.TYPE_LABELS.items():
            rb = QRadioButton(f"{t} — {name}")
            rb.setStyleSheet(f"color: {color_for_type(t)}; font-weight: 600;")
            self._type_buttons.addButton(rb, t)
            type_layout.addWidget(rb)

        # Scope
        scope_layout = QHBoxLayout()
        self._scope_single = QRadioButton("Single node")
        self._scope_subtree = QRadioButton("Whole subtree")
        self._scope_single.setChecked(True)
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self._scope_single, 0)
        self._scope_group.addButton(self._scope_subtree, 1)
        scope_layout.addWidget(self._scope_single)
        scope_layout.addWidget(self._scope_subtree)
        type_layout.addLayout(scope_layout)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self._btn_apply)

        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._undo_stack.undo)
        self._undo_stack.canUndoChanged.connect(self._btn_undo.setEnabled)
        btn_row.addWidget(self._btn_undo)
        type_layout.addLayout(btn_row)

        self._apply_msg = QLabel("")
        self._apply_msg.setStyleSheet("font-size: 11px; color: #555;")
        type_layout.addWidget(self._apply_msg)

        ctrl_layout.addWidget(type_group)

        # --- View Controls ---
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        self._compress_cb = QCheckBox("Compress x-axis (√)")
        self._compress_cb.setChecked(True)
        self._compress_cb.stateChanged.connect(self._on_compress_toggle)
        view_layout.addWidget(self._compress_cb)

        self._boxzoom_cb = QCheckBox("Box zoom (drag to zoom)")
        self._boxzoom_cb.setChecked(False)
        self._boxzoom_cb.stateChanged.connect(self._on_boxzoom_toggle)
        view_layout.addWidget(self._boxzoom_cb)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level:"))
        self._level_spin = QSpinBox()
        self._level_spin.setRange(0, 9999)
        self._level_spin.setValue(0)
        self._level_spin.setSpecialValueText("—")
        level_row.addWidget(self._level_spin)
        self._btn_show_level = QPushButton("Show")
        self._btn_show_level.clicked.connect(self._on_show_level)
        level_row.addWidget(self._btn_show_level)
        self._btn_clear_level = QPushButton("Clear")
        self._btn_clear_level.clicked.connect(self._on_clear_level)
        level_row.addWidget(self._btn_clear_level)
        view_layout.addLayout(level_row)

        self._level_info = QLabel("")
        self._level_info.setStyleSheet("font-size: 11px; color: #555;")
        view_layout.addWidget(self._level_info)

        ctrl_layout.addWidget(view_group)
        ctrl_layout.addStretch()

        # --- Download ---
        dl_group = QGroupBox("Export")
        dl_layout = QVBoxLayout(dl_group)
        self._btn_dl_swc = QPushButton("Download edited SWC")
        self._btn_dl_swc.setEnabled(False)
        self._btn_dl_swc.clicked.connect(self._on_download_swc)
        dl_layout.addWidget(self._btn_dl_swc)
        ctrl_layout.addWidget(dl_group)

        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    # --------------------------------------------------------- Public API
    def load_swc(self, df: pd.DataFrame, filename: str = ""):
        """Load an SWC DataFrame and render the dendrogram."""
        self._df = df.copy()
        self._undo_stack.clear()
        self._selected_idx = None
        self._level_val = None
        self._btn_dl_swc.setEnabled(True)
        self._rebuild_and_draw()

    # ------------------------------------------------- Core rendering
    def _rebuild_and_draw(self):
        """Recompute tree layout and redraw all edges."""
        if self._df is None or self._df.empty:
            return

        df = self._df
        tree = build_tree_cache(df)
        self._tree = tree

        roots = find_all_roots(tree)

        # Classify soma vs dangling roots
        soma_roots = []
        dangling_roots = []
        for root in roots:
            lvls = compute_levels(tree, root)
            members = np.flatnonzero(lvls >= 0)
            if np.any(tree.types[members] == 1):
                soma_roots.append(root)
            else:
                dangling_roots.append(root)

        if not soma_roots:
            soma_roots = roots
            dangling_roots = []

        # Assign dangling to nearest soma tree
        dangling_assign = {}
        if dangling_roots and soma_roots:
            soma_mask = tree.types == 1
            soma_idxs = np.flatnonzero(soma_mask)
            if soma_idxs.size > 0:
                soma_xyz = tree.xyz[soma_idxs].astype(np.float64)
                soma_to_tree = {}
                for si, sroot in enumerate(soma_roots):
                    sl = compute_levels(tree, sroot)
                    for idx in np.flatnonzero(sl >= 0):
                        if tree.types[idx] == 1:
                            soma_to_tree[idx] = si
                for dr in dangling_roots:
                    dxyz = tree.xyz[dr].astype(np.float64)
                    dists = np.linalg.norm(soma_xyz - dxyz, axis=1)
                    nearest = int(np.argmin(dists))
                    target = soma_to_tree.get(int(soma_idxs[nearest]), 0)
                    dangling_assign[dr] = target

        # Build root groups
        root_groups = [[] for _ in soma_roots]
        for i, sr in enumerate(soma_roots):
            root_groups[i].append(sr)
        for dr, ti in dangling_assign.items():
            root_groups[ti].append(dr)

        # We'll render only the first group for now (single-tree view)
        # Multi-tree could use tabs later
        primary_root = root_groups[0][0]
        cum = cumlens_from_root_cache(tree, primary_root)
        y_arr = layout_y_positions_cache(tree, primary_root)
        levels = compute_levels(tree, primary_root)
        tree_mask = levels >= 0

        # Include dangling roots in same figure
        for extra in root_groups[0][1:]:
            ec = cumlens_from_root_cache(tree, extra)
            ey = layout_y_positions_cache(tree, extra)
            el = compute_levels(tree, extra)
            em = el >= 0
            if np.any(tree_mask):
                y_min = float(y_arr[tree_mask].min())
            else:
                y_min = 0.0
            ey[em] += y_min - 2.0
            cum[em] = ec[em]
            y_arr[em] = ey[em]
            levels[em] = el[em]
            tree_mask = tree_mask | em

        self._cum_raw = cum.copy()
        self._y = y_arr
        self._levels = levels
        self._tree_mask = tree_mask
        self._roots = soma_roots

        self._draw()

    def _draw(self):
        """Draw edges on the pyqtgraph plot."""
        self._plot.clear()

        # Re-add hover tooltip (speech bubble)
        self._hover_bubble = BubbleTooltip()
        self._plot.addItem(self._hover_bubble)

        tree = self._tree
        if tree is None:
            return

        cum_raw = self._cum_raw
        cum = np.sqrt(cum_raw) if self._compress else cum_raw
        self._cum = cum
        y = self._y
        levels = self._levels
        tree_mask = self._tree_mask

        offsets = tree.child_offsets
        child_indices = tree.child_indices
        child_counts = np.diff(offsets)
        labels_arr = np.array([label_for_type(int(t)) for t in tree.types.tolist()], dtype=object)

        parent_indices_all = np.repeat(np.arange(tree.size, dtype=np.int32), child_counts)
        edge_children_all = child_indices

        edge_in_tree = tree_mask[edge_children_all]
        parent_indices = parent_indices_all[edge_in_tree]
        edge_children = edge_children_all[edge_in_tree]

        # Store edge data for click detection
        self._edge_children = edge_children
        self._edge_parents = parent_indices

        # Draw vertical connectors (per type)
        if edge_children.size > 0:
            edge_labels = labels_arr[edge_children]
            for label, hex_color in DEFAULT_COLORS.items():
                mask = edge_labels == label
                if not np.any(mask):
                    continue
                p_idx = parent_indices[mask]
                c_idx = edge_children[mask]
                m = int(mask.sum())

                vx = np.empty(m * 3, dtype=np.float32)
                vy = np.empty(m * 3, dtype=np.float32)
                vx[0::3] = cum[p_idx]; vx[1::3] = cum[p_idx]; vx[2::3] = np.nan
                vy[0::3] = y[c_idx];   vy[1::3] = y[p_idx];   vy[2::3] = np.nan

                pen = pg.mkPen(color=hex_color, width=1.5)
                self._plot.plot(vx, vy, pen=pen, connect="finite")

        # Draw horizontal edges (per type)
        if edge_children.size > 0:
            edge_labels = labels_arr[edge_children]
            for label, hex_color in DEFAULT_COLORS.items():
                mask = edge_labels == label
                if not np.any(mask):
                    continue
                p_idx = parent_indices[mask]
                c_idx = edge_children[mask]
                m = int(mask.sum())

                hx = np.empty(m * 3, dtype=np.float32)
                hy = np.empty(m * 3, dtype=np.float32)
                hx[0::3] = cum[p_idx]; hx[1::3] = cum[c_idx]; hx[2::3] = np.nan
                hy[0::3] = y[c_idx];   hy[1::3] = y[c_idx];   hy[2::3] = np.nan

                pen = pg.mkPen(color=hex_color, width=1.5)
                self._plot.plot(hx, hy, pen=pen, connect="finite")

        # Root dot
        if self._roots:
            root = self._roots[0]
            root_color = DEFAULT_COLORS.get(label_for_type(int(tree.types[root])), "#666")
            scatter = pg.ScatterPlotItem(
                [float(cum[root])], [float(y[root])],
                size=8, pen=pg.mkPen("#333", width=1),
                brush=pg.mkBrush(root_color),
            )
            self._plot.addItem(scatter)

        # Selection highlight — solid red X, visible on white
        if self._selected_idx is not None and self._selected_idx < tree.size:
            sx = float(cum[self._selected_idx])
            sy = float(y[self._selected_idx])
            sel_scatter = pg.ScatterPlotItem(
                [sx], [sy], size=10, symbol="x",
                pen=pg.mkPen("#e00", width=1.5),
                brush=pg.mkBrush("#e00"),
            )
            self._plot.addItem(sel_scatter)

        # Level overlay
        if self._level_val is not None:
            target = self._level_val - 1  # match Dash behavior
            level_mask = (levels == target) & tree_mask
            if np.any(level_mask):
                lx = cum[level_mask]
                ly = y[level_mask]
                lvl_scatter = pg.ScatterPlotItem(
                    lx, ly, size=6, symbol="d",
                    pen=pg.mkPen("#333", width=0.5),
                    brush=pg.mkBrush("#ffd600"),
                )
                self._plot.addItem(lvl_scatter)

        # Update axis label
        if self._compress:
            self._plot.setLabel("bottom", "Path length from soma (√µm, compressed)")
        else:
            self._plot.setLabel("bottom", "Path length from soma (µm)")

    # ------------------------------------------------- Click handling
    def _on_click(self, event):
        """Handle click on the plot to select a node."""
        if self._tree is None or self._cum is None:
            return

        pos = event.scenePos()
        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        # Find nearest edge child node
        cum = self._cum
        y = self._y
        tree_mask = self._tree_mask

        candidates = np.flatnonzero(tree_mask)
        if candidates.size == 0:
            return

        cx = cum[candidates]
        cy = y[candidates]

        # Normalize distances by axis ranges
        x_range = max(1e-9, float(np.nanmax(cx) - np.nanmin(cx)))
        y_range = max(1e-9, float(np.nanmax(cy) - np.nanmin(cy)))

        dx = (cx - mx) / x_range
        dy = (cy - my) / y_range
        dists = dx**2 + dy**2

        nearest_local = int(np.argmin(dists))
        nearest_idx = int(candidates[nearest_local])

        # Check if close enough (threshold in normalized coords)
        if dists[nearest_local] > 0.001:
            return

        self._selected_idx = nearest_idx
        tree = self._tree
        swc_id = int(tree.ids[nearest_idx])
        node_type = int(tree.types[nearest_idx])
        level = int(self._levels[nearest_idx]) if self._levels is not None else -1

        self._info_label.setText(
            f"<b>ID:</b> {swc_id}<br>"
            f"<b>Type:</b> {label_for_type(node_type)} ({node_type})<br>"
            f"<b>Level:</b> {level}"
        )

        # Pre-select the current type
        btn = self._type_buttons.button(node_type)
        if btn:
            btn.setChecked(True)
        self._btn_apply.setEnabled(True)

        self.node_selected.emit(swc_id, node_type, level)
        self._draw()

    # ------------------------------------------------- Type changes
    def _on_apply(self):
        """Apply type change to selected node(s)."""
        if self._selected_idx is None or self._tree is None:
            return

        new_type = self._type_buttons.checkedId()
        if new_type < 0:
            return

        tree = self._tree
        subtree_mode = self._scope_subtree.isChecked()

        if subtree_mode:
            # BFS from selected node to get all descendants
            indices = self._get_subtree_indices(self._selected_idx)
        else:
            indices = [self._selected_idx]

        # Map tree indices back to DataFrame row indices
        df_indices = []
        old_types = []
        for idx in indices:
            swc_id = int(tree.ids[idx])
            df_row = self._df.index[self._df["id"] == swc_id]
            if len(df_row) > 0:
                df_idx = df_row[0]
                df_indices.append(df_idx)
                old_types.append(int(self._df.at[df_idx, "type"]))

        if not df_indices:
            return

        cmd = TypeChangeCommand(self, df_indices, old_types, new_type,
                                f"Set {len(df_indices)} node(s) to {label_for_type(new_type)}")
        self._undo_stack.push(cmd)

        self._apply_msg.setText(
            f"✓ Changed {len(df_indices)} node(s) to {label_for_type(new_type)}"
        )
        self._apply_msg.setStyleSheet("font-size: 11px; color: #2ca02c;")
        self.df_changed.emit(self._df)

    def _get_subtree_indices(self, root_idx: int) -> list:
        """BFS from root_idx to collect all descendants (tree-array indices)."""
        tree = self._tree
        visited = []
        stack = [root_idx]
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            visited.append(u)
            start = int(tree.child_offsets[u])
            end = int(tree.child_offsets[u + 1])
            for c in tree.child_indices[start:end]:
                stack.append(int(c))
        return visited

    # ------------------------------------------------- View controls
    def _on_compress_toggle(self, state):
        self._compress = bool(state)
        self._draw()
        # Reset view to fit data after redraw
        self._plot.autoRange()

    def _on_boxzoom_toggle(self, state):
        """Toggle between pan mode and box-zoom mode."""
        vb = self._plot.plotItem.vb
        if state:
            # Box-zoom: left-drag draws a rectangle, releasing zooms to it
            vb.setMouseMode(vb.RectMode)
            self._plot.setMouseEnabled(x=False, y=False)
        else:
            # Normal pan mode
            vb.setMouseMode(vb.PanMode)
            self._plot.setMouseEnabled(x=True, y=True)


    def _on_show_level(self):
        self._level_val = self._level_spin.value()
        target = self._level_val - 1
        if self._levels is not None and self._tree_mask is not None:
            count = int(((self._levels == target) & self._tree_mask).sum())
            self._level_info.setText(f"{count} nodes at level {self._level_val}")
        self._draw()

    def _on_clear_level(self):
        self._level_val = None
        self._level_info.setText("")
        self._draw()

    # ------------------------------------------------- Export
    def _on_download_swc(self):
        if self._df is None:
            return

        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save edited SWC", "edited.swc",
            "SWC Files (*.swc);;All Files (*)"
        )
        if not path:
            return

        df = self._df
        lines = ["# id type x y z radius parent"]
        for _, row in df.iterrows():
            lines.append(
                f"{int(row['id'])} {int(row['type'])} "
                f"{row['x']:.4f} {row['y']:.4f} {row['z']:.4f} "
                f"{row['radius']:.4f} {int(row['parent'])}"
            )
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self._apply_msg.setText(f"✓ Saved to {path}")

    # ------------------------------------------------- Hover tooltip
    def _on_hover(self, evt):
        """Show tooltip with id/type/level near the hovered node."""
        if self._tree is None or self._cum is None:
            return

        pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(pos):
            self._hover_bubble.setVisible(False)
            return

        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        cum = self._cum
        y = self._y
        tree_mask = self._tree_mask
        candidates = np.flatnonzero(tree_mask)
        if candidates.size == 0:
            self._hover_bubble.setVisible(False)
            return

        cx = cum[candidates]
        cy = y[candidates]

        x_range = max(1e-9, float(np.nanmax(cx) - np.nanmin(cx)))
        y_range = max(1e-9, float(np.nanmax(cy) - np.nanmin(cy)))

        dx = (cx - mx) / x_range
        dy = (cy - my) / y_range
        dists = dx**2 + dy**2

        nearest_local = int(np.argmin(dists))
        if dists[nearest_local] > 0.0005:
            self._hover_bubble.setVisible(False)
            return

        nearest_idx = int(candidates[nearest_local])
        tree = self._tree
        swc_id = int(tree.ids[nearest_idx])
        node_type = int(tree.types[nearest_idx])
        level = int(self._levels[nearest_idx]) if self._levels is not None else -1
        label = label_for_type(node_type)

        node_x = float(cum[nearest_idx])
        node_y = float(y[nearest_idx])

        # Show speech-bubble tooltip above the node, colored by type
        type_color = DEFAULT_COLORS.get(label, "#666")
        self._hover_bubble.set_text(f"id={swc_id}, type={label} ({node_type}), level={level}")
        self._hover_bubble.set_color(type_color)
        self._hover_bubble.setPos(node_x, node_y)
        self._hover_bubble.setVisible(True)
