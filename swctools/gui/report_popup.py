"""Reusable popup dialog for generated text reports."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ReportPopupDialog(QDialog):
    def __init__(self, *, title: str, report_path: str, report_text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        path_label = QLabel(f"Report file: {report_path}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(path_label)

        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(report_text)
        editor.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #fafafa; border: 1px solid #ddd; color: #222;"
            "  font-family: Menlo, Consolas, monospace; font-size: 12px;"
            "}"
        )
        root.addWidget(editor, stretch=1)

        row = QHBoxLayout()
        row.addStretch()
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        root.addLayout(row)

    @staticmethod
    def open_report(parent: QWidget | None, *, title: str, report_path: str) -> None:
        p = Path(report_path)
        txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        dlg = ReportPopupDialog(title=title, report_path=str(p), report_text=txt, parent=parent)
        dlg.exec()
