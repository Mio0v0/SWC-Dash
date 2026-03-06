#!/bin/bash
./.venv/bin/pip freeze > requirements.txt
./.venv/bin/pip install pyinstaller -r requirements.txt
rm -rf build dist SWC-Dash.spec
./.venv/bin/pyinstaller --noconfirm --clean --windowed --onefile --name "SWC-Dash" --copy-metadata neurom --collect-submodules neurom --collect-data neurom --copy-metadata morphio --collect-submodules morphio --collect-data morphio --collect-binaries morphio --add-data "assets:assets" main.py
