from dash import Dash
from layout import build_layout
from callbacks import register_callbacks
from constants import APP_TITLE

def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = APP_TITLE
    app.layout = build_layout()
    register_callbacks(app)
    return app


