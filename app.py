from dash import Dash
from flask import request
from layout import build_layout
from callbacks import register_callbacks
from constants import APP_TITLE
import os
import sys

def get_assets_path():
    """Return correct path for assets whether running normally or as frozen app."""
    if getattr(sys, "frozen", False):  # running as PyInstaller executable
        return os.path.join(sys._MEIPASS, "assets")
    return os.path.join(os.path.dirname(__file__), "assets")


def create_app() -> Dash:
    # Tell Dash where to find assets (important for PyInstaller builds)
    app = Dash(__name__, assets_folder=get_assets_path(), suppress_callback_exceptions=True)
    app.title = APP_TITLE
    app.layout = build_layout()
    register_callbacks(app)

    server = app.server  # Flask app underneath

    # --- Shutdown route ---
    def _shutdown_server():
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        else:
            os._exit(0)

    @server.route("/_shutdown", methods=["POST"])
    def _shutdown():
        _shutdown_server()
        return "OK"

    return app

