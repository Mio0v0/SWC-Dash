"""Convenience launcher for the SWC Qt GUI.
Run from repo root with the project's venv active:

  source .venv/bin/activate
  python run_gui.py

This simply forwards to the package entry at swctools.gui.main
"""

import os
import sys

# Make the GUI package's directory importable by bare module names (the GUI
# package uses a mix of package-relative and plain imports). This ensures
# running `python run_gui.py` from the repo root works the same as running
# from inside the GUI folder.
HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.join(HERE, "swctools", "gui")
if GUI_DIR not in sys.path:
  sys.path.insert(0, GUI_DIR)

from swctools.gui import main as _m

if __name__ == "__main__":
  _m.main()
