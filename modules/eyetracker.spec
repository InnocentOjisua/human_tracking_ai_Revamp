# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_all

# --- Collect all data for external packages ---
datas = []
binaries = []
hiddenimports = []

# Packages to collect
packages = [
    "streamlit",
    "pywebview",
    "mediapipe",
    "torch",
    "facenet_pytorch",
    "fer",
    "moviepy",
    "imageio",
    "imageio.plugins",
    "plotly",
    "pandas",
    "numpy",
    "yaml",
]

for pkg in packages:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Add your local folders (core, modules, ui)
datas += [
    ("core", "core"),
    ("modules", "modules"),
    ("ui", "ui")
]

# --- PyInstaller Spec ---
block_cipher = None

a = Analysis(
    ['app.py'],           # Main app
    pathex=[],            # Add project path if needed
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EyeTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # Set True to debug
    icon=None
)
