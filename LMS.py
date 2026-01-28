# Human Signal AI (Revamped MVP) — Streamlit Cloud version
import time
from collections import deque
from pathlib import Path
import csv
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml
from datetime import datetime
from core.tracking import Tracker
from core.metrics import EyeTracker
from core.features import compute_features
from core.filters import ema
from core.quality import estimate_quality
from modules.fatigue import fatigue_score
from modules.attention import attention_score
from modules.emotion_detection import detect_emotion
from modules.stress import stress_score
from modules.posture import posture_status
from modules.ergonomics_distance import distance_status
from ui.overlays import draw_labels

# --- Streamlit page ---
st.set_page_config(page_title="E-Learning Eyetracker — Live Monitor (BrainEyeCore)", layout="wide")
st.title("🧠 E-Learning Eyetracker — Live Education Monitor")
st.caption("On-device. Education features only — not a medical device.")

# --- Load config ---
DEFAULT_CFG = {
    "video": {"fps": 30},
    "windows": {"fatigue_seconds": 60, "attention_seconds": 30, "stress_seconds": 30, "update_hz": 5},
    "thresholds": {
        "perclos_drowsy": 0.25, "blink_rate_high": 25, "gaze_offscreen_secs": 3.0,
        "stress_tension_high": 0.7, "neck_angle_slouch": 22,
        "yawn_mar": 0.6, "yawn_min_secs": 0.5,
        "distance_face_ratio_close": 0.32, "distance_face_ratio_far": 0.12,
    },
    "smoothing": {"ema_alpha": 0.25},
    "quality": {"min_brightness": 60, "max_motion_px": 5, "min_confidence": 0.5},
}
cfg_path = Path("configs/default.yaml")
user_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
cfg = DEFAULT_CFG | {k: (DEFAULT_CFG.get(k, {}) | user_cfg.get(k, {})) for k in DEFAULT_CFG.keys()}

# --- Alerts ---
ALERTS = {
    "DISTRACTED": {"cond": lambda att, fat, strx: att < 50.0},
    "TIRED": {"cond": lambda att, fat, strx: fat > 50.0},
    "STRESSED": {"cond": lambda att, fat, strx: strx > 50.0},
}
ALERT_DEBOUNCE_ON = 0.40
ALERT_HOLD_OFF = 1.50
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {name: {"is_on": False, "t_on": 0.0, "t_off": 0.0} for name in ALERTS.keys()}

def _update_alert_states(att_val, fat_val, str_val, now):
    states = st.session_state.alert_state
    for name, spec in ALERTS.items():
        active = bool(spec["cond"](att_val, fat_val, str_val))
        s = states[name]
        if active:
            if not s["is_on"]:
                if s["t_on"] == 0.0: s["t_on"] = now
                if now - s["t_on"] >= ALERT_DEBOUNCE_ON:
                    s["is_on"] = True
                    s["t_off"] = 0.0
            else:
                s["t_off"] = 0.0
        else:
            s["t_on"] = 0.0
            if s["is_on"]:
                if s["t_off"] == 0.0: s["t_off"] = now
                if now - s["t_off"] >= ALERT_HOLD_OFF:
                    s["is_on"] = False

def _draw_alert_banner(img_bgr, labels):
    if not labels: return img_bgr
    h, w = img_bgr.shape[:2]
    text = "  •  ".join(labels)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 28, 16
    box_w, box_h = tw + pad_x*2, th + pad_y*2
    x0, y0 = (w - box_w)//2, 40
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+box_w, y0+box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)
    cv2.putText(img_bgr, text, (x0+pad_x, y0+pad_y+th), font, scale, (255,255,255), thick, cv2.LINE_AA)
    return img_bgr

def _draw_futuristic_overlay(img_bgr, face_landmarks):
    if face_landmarks is None or len(face_landmarks) < 478: return img_bgr
    pts = face_landmarks[:, :2].astype(np.int32)
    idxs = {"left_eye":33,"right_eye":263,"nose_tip":1,"chin":152,"forehead":10,
            "mouth_left":61,"mouth_right":291,"left_iris":468,"right_iris":473}
    c1, c2 = (255,255,255),(90,220,255)
    P = lambda i: tuple(pts[i])
    overlay = img_bgr.copy()
    cv2.line(overlay, P(idxs["left_eye"]), P(idxs["right_eye"]), c1,1,cv2.LINE_AA)
    cv2.line(overlay, P(idxs["nose_tip"]), P(idxs["chin"]), c1,1,cv2.LINE_AA)
    cv2.line(overlay, P(idxs["forehead"]), P(idxs["nose_tip"]), c1,1,cv2.LINE_AA)
    cv2.line(overlay, P(idxs["mouth_left"]), P(idxs["mouth_right"]), c1,1,cv2.LINE_AA)
    for i in [idxs["left_iris"], idxs["right_iris"]]:
        cv2.circle(overlay, P(i), 3, c2,1,cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0, img_bgr)
    return img_bgr

# --- Buffers & placeholders ---
sec_window = 60
target_fps = int(cfg["video"].get("fps",30))
buf_len = sec_window * max(1, target_fps)
if "fatigue_buf" not in st.session_state:
    st.session_state.fatigue_buf = deque(maxlen=buf_len)
    st.session_state.attention_buf = deque(maxlen=buf_len)
    st.session_state.stress_buf = deque(maxlen=buf_len)
video_placeholder = st.empty()
c1,c2,c3 = st.columns(3)
chart1, chart2, chart3 = c1.empty(), c2.empty(), c3.empty()
status_placeholder = st.empty()
if "fatigue_sm" not in st.session_state: st.session_state.fatigue_sm = 0.0
if "attention_sm" not in st.session_state: st.session_state.attention_sm = 0.0
if "stress_sm" not in st.session_state: st.session_state.stress_sm = 0.0
if "last_csv_write" not in st.session_state: st.session_state.last_csv_write = 0.0

# --- Sidebar & CSV setup ---
with st.sidebar:
    st.header("Controls")
    show_futuristic = st.toggle("Futuristic overlay", value=True, key="show_futuristic")
    st.markdown("---")
    if "csv_path" not in st.session_state or not st.session_state.csv_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        st.session_state.csv_path = f"{ts}_metrics.csv"
        if not Path(st.session_state.csv_path).exists():
            with open(st.session_state.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "t_epoch","session_time","face_id","fatigue","attention","stress",
                    "posture","distance","emotion","blink_rate","perclos","yawn_count",
                    "yawn_event","pupil_diameter","fixation_duration","saccades","fixation",
                    "recommendation"
                ])
    csv_file = Path(st.session_state.csv_path)
    st.download_button(
        label="📥 Download CSV",
        data=open(csv_file, 'r').read(),
        file_name=csv_file.name,
        mime="text/csv",
        key="download_csv"
    )

# --- Initialize Tracker & EyeTracker ---
tracker = Tracker()
eye_tracker = EyeTracker(max_len=60*cfg["video"]["fps"])
current_content = {"title":"Quadratic Equations - Exercise 3","difficulty":"hard"}

# --- Camera input ---
uploaded_file = st.camera_input("📷 Position your face for tracking", key="browser_cam")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is not None:
        # --- Process frame ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = tracker.process(frame_rgb)
        feats = compute_features(det)
        quality = estimate_quality(frame_rgb, det, feats, cfg)

        # --- Metrics ---
        post = posture_status(feats, quality, cfg)
        fat  = fatigue_score(feats, quality, cfg, posture_state=post["state"])
        att  = attention_score(feats, quality, cfg)
        strx = stress_score(feats, quality, cfg)
        dist = distance_status(det, cfg)
        emotion_val = detect_emotion(frame_rgb)

        # --- Smooth metrics ---
        alpha = cfg["smoothing"]["ema_alpha"]
        st.session_state.fatigue_sm   = ema(fat.get("score",0.0), st.session_state.fatigue_sm, alpha)
        st.session_state.attention_sm = ema(att.get("score",0.0), st.session_state.attention_sm, alpha)
        st.session_state.stress_sm    = ema(strx.get("score",0.0), st.session_state.stress_sm, alpha)

        # --- EyeTracker update ---
        left_iris = feats.get("left_iris")
        right_iris = feats.get("right_iris")
        left_eye_lms = feats.get("left_eye_landmarks")
        right_eye_lms = feats.get("right_eye_landmarks")
        if left_iris is not None and right_iris is not None:
            eye_tracker.update(left_iris, right_iris, left_eye_lms, right_eye_lms)
        else:
            eye_tracker.update(np.array([[0,0]]), np.array([[0,0]]), None, None)
        fixation_duration_val, saccades_val, fixation_val, *_ = eye_tracker.compute_metrics()

        # --- Alerts ---
        _update_alert_states(st.session_state.attention_sm, st.session_state.fatigue_sm, st.session_state.stress_sm, time.time())
        active_labels = [name for name,s in st.session_state.alert_state.items() if s["is_on"]]

        # --- Append buffers ---
        st.session_state.fatigue_buf.append(st.session_state.fatigue_sm)
        st.session_state.attention_buf.append(st.session_state.attention_sm)
        st.session_state.stress_buf.append(st.session_state.stress_sm)

        # --- Draw overlays ---
        labeled = draw_labels(frame, post, dist)
        if show_futuristic:
            labeled = _draw_futuristic_overlay(labeled, feats.get("face", None))
        labeled = _draw_alert_banner(labeled, active_labels)
        video_placeholder.image(labeled[:, :, ::-1], channels="RGB", use_container_width=True)

        # --- Recommendation ---
        if st.session_state.fatigue_sm > 55:
            recommendation_val = "Take a long break"
        elif st.session_state.attention_sm < 40:
            recommendation_val = "Refocus"
        else:
            recommendation_val = "Keep learning"

        # --- CSV logging ---
        t_now = time.time()
        if t_now - st.session_state.last_csv_write >= 1.0:
            t_epoch = t_now
            t_human = datetime.fromtimestamp(t_now).strftime("%H:%M:%S")
            face_id = feats.get("face_id", 0)
            with open(st.session_state.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    t_epoch, t_human, face_id, f"{st.session_state.fatigue_sm:.2f}",
                    f"{st.session_state.attention_sm:.2f}", f"{st.session_state.stress_sm:.2f}",
                    str(post["state"]), str(dist["state"]), str(emotion_val),
                    f"{fat.get('blink_rate',0.0):.1f}", f"{fat.get('perclos',0.0):.2f}",
                    fat.get("yawn_count",0), 1 if fat.get("yawn_counted_this_yawn",False) else 0,
                    f"{feats.get('pupil_diameter',0.0):.2f}", f"{fixation_duration_val:.2f}",
                    f"{saccades_val:.2f}", f"{fixation_val:.2f}", recommendation_val
                ])
            st.session_state.last_csv_write = t_now

        # --- Status ---
        status_placeholder.info(
            f"Fatigue: {st.session_state.fatigue_sm:0.0f} | Attention: {st.session_state.attention_sm:0.0f} | "
            f"Stress: {st.session_state.stress_sm:0.0f} | Blink Rate: {fat.get('blink_rate',0.0):.1f} | "
            f"Perclos: {fat.get('perclos',0.0):.2f} | Yawn: {fat.get('yawn_count',0)} | "
            f"Emotion: {emotion_val} | Recommendation: {recommendation_val}"
        )

        # --- Plot charts ---
        def plot_series(container, ys, title, color):
            xs = np.arange(len(ys))/max(1,cfg["video"]["fps"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3)))
            fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10), title=title, yaxis=dict(range=[0,100]))
            container.plotly_chart(fig, use_container_width=True)

        plot_series(chart1, list(st.session_state.fatigue_buf), "Fatigue", "#1f77b4")
        plot_series(chart2, list(st.session_state.attention_buf), "Attention", "#2ca02c")
        plot_series(chart3, list(st.session_state.stress_buf), "Stress", "#d62728")
