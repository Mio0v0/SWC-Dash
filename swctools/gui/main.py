"""SWC-QT — PySide6-based SWC neuron editor."""

import sys

from .main_window import SWCMainWindow


def _pick_app_font():
    """Select a concrete installed UI font to avoid expensive alias fallback."""
    from PySide6.QtGui import QFont, QFontDatabase

    families = set(QFontDatabase.families())
    for name in ("Helvetica Neue", "SF Pro Text", "Arial", "DejaVu Sans"):
        if name in families:
            return QFont(name, 11)
    # Fallback to Qt's concrete general system font rather than abstract aliases.
    f = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    if f.pointSize() <= 0:
        f.setPointSize(11)
    return f


def main():
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("SWC-QT")
    app.setStyle("Fusion")
    app.setFont(_pick_app_font())

    window = SWCMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
