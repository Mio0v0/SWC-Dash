# main.py
from app import create_app
import threading, webbrowser, os, sys

PORT = int(os.getenv("PORT", "8050"))
URL  = f"http://127.0.0.1:{PORT}"

def open_browser():
    webbrowser.open(URL)

if __name__ == "__main__":
    app = create_app()

    # Open browser once server starts
    threading.Timer(1.0, open_browser).start()

    # Turn off auto-reloader for packaged apps
    is_frozen = getattr(sys, "frozen", False)
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=not is_frozen,
        use_reloader=False
    )
