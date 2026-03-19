"""SWC-QT — PySide6-based SWC neuron editor."""

import sys

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from .main_window import SWCMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SWC-QT")
    app.setStyle("Fusion")

    families = set(QFontDatabase.families())
    for name in ("Helvetica Neue", "SF Pro Text", "Arial", "DejaVu Sans"):
        if name in families:
            app.setFont(QFont(name, 11))
            break

    window = SWCMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
