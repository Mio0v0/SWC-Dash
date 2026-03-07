"""Combined editor tab: Dendrogram (left) + 3D Neuron View (right)."""

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter

from dendrogram_widget import DendrogramWidget
from neuron_3d_widget import Neuron3DWidget


class EditorTab(QWidget):
    """Tab combining the 2D dendrogram editor and 3D neuron view side by side."""

    df_changed = Signal(pd.DataFrame)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left: 3D neuron view
        self._view3d = Neuron3DWidget()
        splitter.addWidget(self._view3d)

        # Right: dendrogram editor
        self._dendro = DendrogramWidget()
        splitter.addWidget(self._dendro)

        # 50/50 split
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Sync: when type changes in dendrogram, refresh 3D view
        self._dendro.df_changed.connect(self._on_df_changed)

        # Sync: when a node is selected in dendrogram, highlight in 3D
        self._dendro.node_selected.connect(self._on_node_selected)

    # --------------------------------------------------------- Public API
    def load_swc(self, df: pd.DataFrame, filename: str = ""):
        """Load SWC data into both views."""
        self._dendro.load_swc(df, filename)
        self._view3d.load_swc(df, filename)

    # ------------------------------------------------- Sync
    def _on_df_changed(self, df: pd.DataFrame):
        """Refresh 3D view when dendrogram edits are applied."""
        self._view3d.refresh(df)
        self.df_changed.emit(df)

    def _on_node_selected(self, swc_id: int, node_type: int, level: int):
        """Highlight the selected node in the 3D view."""
        self._view3d.highlight_node(swc_id)
