"""Floating guide widget for auto-typing rules and decision boundaries."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from swctools.core.auto_typing_catalog import get_auto_typing_guide


class AutoTypingGuideWidget(QWidget):
    """Guide panel shown as a floating dock for Auto Labeling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._title = QLabel("Auto Typing Rule Guide")
        self._title.setStyleSheet("font-size: 14px; font-weight: 700; color: #222;")
        layout.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._body.setStyleSheet(
            "QLabel {"
            "  font-size: 12px;"
            "  color: #444;"
            "  background: #f6f6f6;"
            "  border: 1px solid #ddd;"
            "  padding: 8px;"
            "}"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self._body)
        layout.addWidget(scroll, stretch=1)

    def refresh(self):
        guide = get_auto_typing_guide()
        self._title.setText(str(guide.get("title", "Auto Typing Rule Guide")))
        self._body.setText(str(guide.get("body", "")))
