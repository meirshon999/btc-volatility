"""Start the Streamlit dashboard on a free port and open it in the browser."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from src import config


def free_port(start: int = 8501, attempts: int = 50) -> int:
    """First port from `start` on which nothing is listening."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found for the dashboard")


def wait_until_up(url: str, timeout: float = 90.0) -> bool:
    """Poll Streamlit's health endpoint until the server answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def launch_dashboard(open_browser: bool = True) -> None:
    """Run the dashboard until the user presses Ctrl+C (or closes the window)."""
    port = free_port()
    url = f"http://localhost:{port}"
    # headless: Streamlit must not ask for an e-mail on the first run; we open the browser ourselves
    command = [sys.executable, "-m", "streamlit", "run", str(config.ROOT / "app.py"),
               "--server.port", str(port), "--server.headless", "true", "--server.address", "localhost"]
    process = subprocess.Popen(command, cwd=config.ROOT)
    try:
        if wait_until_up(url):
            print(f"\n  Dashboard: {url}")
            print("  Keep this window open while you use it; press Ctrl+C here to stop.\n")
            if open_browser:
                webbrowser.open(url)
        else:
            print("  The dashboard did not start within 90 seconds; see the messages above.")
        process.wait()
    except KeyboardInterrupt:
        print("\n  Stopping the dashboard...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
