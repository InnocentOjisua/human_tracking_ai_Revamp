# Human-Signal AI: Real-Time Cognitive State Tracking

## 📌 Overview
A real-time computer vision system that tracks human **attention**, **fatigue**, and **stress** from live video using face/pose landmarks, engineered signals, and on-screen overlays.

---

## 🛠️ Tools & Libraries
- `opencv-python`, `mediapipe`
- `numpy`, `scipy`
- `streamlit`, `plotly`
- `pydantic`, `pyyaml`

---

## 🚀 Features
- 🎥 **Live webcam input** — face & posture detection (MediaPipe)
- 💡 **Real-time graphs** — attention, fatigue, stress trendlines (Plotly)
- ⚡ **Responsive overlays** — status badges + confidence gauges
- 🧠 **Pop-up alerts** — *Distracted*, *Tired*, *Stressed* (with cooldowns)
- ⚙️ **Config-driven** — thresholds & UI via `configs/*.yaml`
- 🌐 **Streamlit UI** — one-click local dashboard

---

## 🎯 Applications
- Learning platforms — detect disengagement, prompt breaks
- Driver monitoring — PERCLOS/blink patterns for drowsiness
- Workplace wellness — early stress signals via micro-behaviour change

---

## 🎥 Demo
![Demo](configs/demo.gif)

---

## 🔍 Methodology (Signals → States)
**Landmarks → engineered signals → smoothing → thresholds → state flags**
- **Attention:** head yaw/pitch + gaze proxy (eye-corner/iris ratios); hysteresis to avoid flicker
- **Fatigue:** eye aspect ratio (EAR), **PERCLOS**, blink frequency/duration (rolling windows)
- **Stress (proxy):** brow-motion variance, landmark micro-jitter RMS vs baseline  
Signals are smoothed (EMA / Savitzky–Golay), compared to thresholds with hysteresis & cooldowns, and surfaced as overlays with confidence (0–1).

---

## 📊 Results & Observations
*(Qualitative placeholders — replace with your measurements when available.)*
- **Latency:** ~20–35 ms/frame at 720p CPU on common laptops (≈28–50 FPS)
- **Attention:** robust to brief glance-aways (<1 s) due to hysteresis
- **Fatigue:** PERCLOS rises under prolonged closures; low false positives with good lighting
- **Stress:** best interpreted as *relative change* rather than absolute diagnosis

---

## 🚀 How to Run (ONE single block: clone + install + imports + run)
```bash
# Clone the repository
git clone https://github.com/<YOU>/<REPO>.git
cd <REPO>

# (Optional) Create and activate a virtual environment
# python -m venv .venv && source .venv/bin/activate    # macOS/Linux
# py -m venv .venv && .venv\Scripts\activate           # Windows

# Install dependencies
python -m pip install --upgrade pip
pip install opencv-python mediapipe numpy scipy streamlit plotly pydantic pyyaml

# --- Reference Imports (paste into a notebook's first cell if running interactively) ---
# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.signal import savgol_filter
# import streamlit as st
# import plotly.graph_objects as go
# from pydantic import BaseModel, validator
# import yaml, time, math
# from collections import deque

# Run Streamlit app (adjust the path if needed)
streamlit run app/main.py -- --config configs/default.yaml
