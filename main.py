# main.py
import os
import sys
import threading
import webbrowser
import logging
import traceback
from app import create_app

HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8050"))
URL = f"http://{HOST}:{PORT}"


def setup_logging() -> str:
    """
    Set up logging to a file in the user's home directory.
    This works for both normal and PyInstaller-frozen builds.
    """
    # e.g. /Users/xxx/.swc_dash_logs/swc_dash_error.log
    log_dir = os.path.join(os.path.expanduser("~"), ".swc_dash_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "swc_dash_error.log")

    # Basic config: file only; console handler added below for dev mode
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # In non-frozen (dev) mode, also log to stderr so you see errors directly.
    if not getattr(sys, "frozen", False):
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(console)

    return log_path


def open_browser():
    """
    Try to open the default browser to the app URL.
    Fail silently into logs if something goes wrong.
    """
    try:
        webbrowser.open(URL)
    except Exception as e:
        logging.warning(f"Failed to open browser automatically: {e}")


def main():
    is_frozen = getattr(sys, "frozen", False)
    log_path = setup_logging()

    logging.info("Starting SWC-Dash (frozen=%s) on %s", is_frozen, URL)

    # --- Create Dash app ---
    try:
        app = create_app()
    except Exception:
        logging.exception("Error while creating Dash app")
        # Also force-write full traceback to the log file
        with open(log_path, "a") as f:
            f.write("\n" + traceback.format_exc() + "\n")
        # Re-raise so if they run from Terminal, they see the error
        raise

    # --- Open browser after server starts (no autoreloader for frozen app) ---
    threading.Timer(1.0, open_browser).start()

    # --- Run server ---
    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=not is_frozen,   # debug=False in exe -> real 500s, no Werkzeug debugger
            use_reloader=False,    # important for PyInstaller
        )
    except Exception:
        logging.exception("Error while running SWC-Dash server")
        with open(log_path, "a") as f:
            f.write("\n" + traceback.format_exc() + "\n")
        raise


if __name__ == "__main__":
    main()

