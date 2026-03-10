"""SWC-QT — PySide6-based SWC neuron editor."""

import sys
import os

from PySide6.QtWidgets import QApplication
from .main_window import SWCMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SWC-QT")
    app.setStyle("Fusion")

    window = SWCMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
