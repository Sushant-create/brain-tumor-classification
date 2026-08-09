#!/usr/bin/env python3
"""
Cross-platform launcher for the Brain Tumor MRI Classifier.

- Installs dependencies if missing
- Checks whether a trained model is present (warns but still launches if not)
- Starts the Flask server
- Opens the dashboard in the default browser

Usage:
    python launch.py
"""

import os
import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "tumor_classifier.keras")
HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def ensure_dependencies():
    try:
        import flask  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError:
        print("Installing dependencies from requirements.txt (first run only)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            os.path.join(ROOT, "requirements.txt"),
        ])


def check_model():
    if not os.path.exists(MODEL_PATH):
        print(
            "\nWARNING: no trained model found at "
            f"{MODEL_PATH}\n"
            "The dashboard will still start, but predictions will return an "
            "error until you run:\n"
            "    python train.py --data-dir <path-to-brain-tumor-mri-dataset>\n"
        )


def wait_for_server(timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(f"{URL}/health", timeout=1)
            return True
        except URLError:
            time.sleep(0.5)
    return False


def main():
    ensure_dependencies()
    check_model()

    env = os.environ.copy()
    env["PORT"] = str(PORT)

    print(f"Starting server at {URL} ...")
    server = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "main.py")],
        env=env,
        cwd=ROOT,
    )

    try:
        if wait_for_server():
            webbrowser.open(URL)
        else:
            print("Server did not respond in time; check the logs above.")
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
