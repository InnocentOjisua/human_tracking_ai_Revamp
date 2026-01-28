import gradio as gr
import cv2
import numpy as np
import time
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

# Import your modules (same as Streamlit version)
from core.tracking import Tracker
from core.metrics import EyeTracker
from core.features import compute_features
from core.filters import ema
from core.quality import estimate_quality
from modules.fatigue import fatigue_score
from modules.attention import attention_score
from modules.stress import stress_score
from modules.posture import posture_status
from modules.ergonomics_distance import distance_status
from modules.emotion_detection import detect_emotion
from ui.overlays import draw_labels

# --- Global state ---
tracker = Tracker()
eye_tracker = EyeTracker(max_len=1800)  # 60s * 30fps buffer
fatigue_buf = deque(maxlen=1800)
attention_buf = deque(maxlen=1800)
stress_buf = deque(maxlen=1800)
fatigue_sm = 0.0
attention_sm = 0.0
stress_sm = 0.0
last_csv_write = 0.0
csv_path = Path(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_metrics.csv")
# CSV header
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow([
        "t_epoch","session_time","face_id","fatigue","attention","stress",
        "posture","distance","emotion","blink_rate","perclos","yawn_count",
        "yawn_event","pupil_diameter","fixation_duration","saccades","fixation",
        "recommendation"
    ])

# --- Alerts ---
ALERTS = {
    "DISTRACTED": lambda att, fat, strx: att < 50.0,
    "TIRED": lambda att, fat, strx: fat > 50.0,
    "STRESSED": lambda att, fat, strx: strx > 50.0,
}
alert_state = {name: {"is_on": False, "t_on": 0.0, "t_off": 0.0} for name in ALERTS.keys()}
ALERT_DEBOUNCE_ON = 0.40
ALERT_HOLD_OFF = 1.50

def _update_alert_states(att_val, fat_val, str_val, now):
    global alert_state
    for name, cond in ALERTS.items():
        active = cond(att_val, fat_val, str_val)
        s = alert_state[name]
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

def process_frame(frame, show_futuristic=True):
    global fatigue_sm, attention_sm, stress_sm, last_csv_write

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = tracker.process(frame_rgb)
    feats = compute_features(det)
    quality = estimate_quality(frame_rgb, det, feats, {})

    # --- Metrics ---
    post = posture_status(feats, quality, {})
    fat  = fatigue_score(feats, quality, {}, posture_state=post["state"])
    att  = attention_score(feats, quality, {})
    strx = stress_score(feats, quality, {})

    # --- Smooth metrics ---
    alpha = 0.25
    fatigue_sm   = ema(fat.get("score",0.0), fatigue_sm, alpha)
    attention_sm = ema(att.get("score",0.0), attention_sm, alpha)
    stress_sm    = ema(strx.get("score",0.0), stress_sm, alpha)

    # --- EyeTracker ---
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
    _update_alert_states(attention_sm, fatigue_sm, stress_sm, time.time())
    active_labels = [name for name,s in alert_state.items() if s["is_on"]]

    # --- Buffers ---
    fatigue_buf.append(fatigue_sm)
    attention_buf.append(attention_sm)
    stress_buf.append(stress_sm)

    # --- Overlays ---
    labeled = draw_labels(frame, post, distance_status(det, {}) )
    if show_futuristic:
        if "face" in feats:
            pts = feats.get("face")
            if pts is not None:
                labeled = labeled  # optional: add your futuristic overlay code here
    labeled = _draw_alert_banner(labeled, active_labels)

    # --- Recommendations ---
    if fatigue_sm > 55:
        recommendation_val = "Take a long break"
    elif attention_sm < 40:
        recommendation_val = "Refocus"
    else:
        recommendation_val = "Keep learning"

    # --- CSV Logging ---
    t_now = time.time()
    if t_now - last_csv_write >= 1.0:
        t_epoch = t_now
        t_human = datetime.fromtimestamp(t_now).strftime("%H:%M:%S")
        face_id = feats.get("face_id",0)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                t_epoch, t_human, face_id, f"{fatigue_sm:.2f}",
                f"{attention_sm:.2f}", f"{stress_sm:.2f}",
                str(post["state"]), str(distance_status(det,{})["state"]), "N/A",
                f"{fat.get('blink_rate',0.0):.1f}", f"{fat.get('perclos',0.0):.2f}",
                fat.get("yawn_count",0), 1 if fat.get("yawn_counted_this_yawn",False) else 0,
                f"{feats.get('pupil_diameter',0.0):.2f}", f"{fixation_duration_val:.2f}",
                f"{saccades_val:.2f}", f"{fixation_val:.2f}", recommendation_val
            ])
        last_csv_write = t_now

    return labeled[:, :, ::-1], fatigue_sm, attention_sm, stress_sm, recommendation_val, str(csv_path)

# --- Gradio interface ---
iface = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Video(source="webcam", streaming=True, type="numpy", label="Live Webcam"),
        gr.Checkbox(label="Futuristic overlay", value=True)
    ],
    outputs=[
        gr.Image(type="numpy", label="Video Feed"),
        gr.Number(label="Fatigue"),
        gr.Number(label="Attention"),
        gr.Number(label="Stress"),
        gr.Textbox(label="Recommendation"),
        gr.File(label="CSV Log")
    ],
    live=True,
    title="🧠 E-Learning Eyetracker — BrainEyeCore",
    description="Real-time monitoring of fatigue, attention, stress, posture, and recommendations. Education-only; not a medical device."
)

iface.launch()
