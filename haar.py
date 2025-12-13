# bundle_resources.py
# This file is just for PyInstaller to bundle resources

import os

# List all files you want PyInstaller to include
haarcascade_path = os.path.join("fer", "data", "haarcascade_frontalface_default.xml")
dummy = haarcascade_path  # just reference it so PyInstaller sees it
