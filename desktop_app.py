import subprocess
import threading
import time
import requests
import webview

STREAMLIT_URL = "http://localhost:8501"

def start_streamlit():
    """Launch Streamlit app in a separate process."""
    subprocess.run(["streamlit", "run", "app.py"], check=True)

def wait_for_streamlit(url, timeout=30):
    """Wait until Streamlit server is up and running."""
    start_time = time.time()
    while True:
        try:
            requests.get(url)
            break  # Server is up
        except requests.exceptions.ConnectionError:
            if time.time() - start_time > timeout:
                raise RuntimeError("Streamlit did not start in time")
            time.sleep(0.5)

if __name__ == "__main__":
    # Start Streamlit in a separate thread
    threading.Thread(target=start_streamlit, daemon=True).start()

    # Wait until Streamlit is ready
    wait_for_streamlit(STREAMLIT_URL)

    # Open the Streamlit app in a native window
    webview.create_window("E-Learning Eyetracker", STREAMLIT_URL)
    webview.start()
