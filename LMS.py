import time
import numpy as np
import cv2
import streamlit as st
import csv
from datetime import datetime
import plotly.graph_objects as go
import joblib
from collections import deque


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

# --- ML Model ---
model = joblib.load("fatigue_model.pkl")
FATIGUE_LABELS = ["low", "medium", "high"]

# --- Emotion mapping ---
EMOTION_MAP = {"neutral":0,"happy":1,"sad":2,"angry":3,"surprise":4,"fear":5,"disgust":6,"confused":7}

# --- Config ---
cfg = {
    "video": {"source": 0, "fps": 30, "width": 1280, "height": 720},  # <-- add this
    "quality": {"min_brightness": 60, "max_motion_px": 5, "min_confidence": 0.5},
    "smoothing": {"ema_alpha": 0.25},
    "thresholds": {
        "perclos_drowsy": 0.25,
        "blink_rate_high": 25,
        "gaze_offscreen_secs": 3.0,   # required by attention_score
        "neck_angle_slouch": 22,      # required by posture_status
        "stress_tension_high": 0.7,   # used in stress_score
        "yawn_mar": 0.6,              # used in fatigue_score
        "yawn_min_secs": 0.5,
        "distance_face_ratio_close": 0.32,
        "distance_face_ratio_far": 0.12
    },
    "windows": {"update_hz": 5},
}


# --- Alerts ---
ALERTS = {
    "DISTRACTED": {"cond": lambda att, fat, strx: att < 50.0},
    "TIRED": {"cond": lambda att, fat, strx: fat > 50.0},
    "STRESSED": {"cond": lambda att, fat, strx: strx > 50.0}
}
ALERT_DEBOUNCE_ON = 0.40
ALERT_HOLD_OFF = 1.50
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {name: {"is_on": False, "t_on":0.0,"t_off":0.0} for name in ALERTS.keys()}

def _update_alert_states(att_val,fat_val,str_val,now):
    for name,spec in ALERTS.items():
        active = bool(spec["cond"](att_val,fat_val,str_val))
        s = st.session_state.alert_state[name]
        if active:
            if not s["is_on"]:
                if s["t_on"]==0.0: s["t_on"]=now
                if now-s["t_on"]>=ALERT_DEBOUNCE_ON: s["is_on"]=True; s["t_off"]=0.0
            else: s["t_off"]=0.0
        else:
            s["t_on"]=0.0
            if s["is_on"]:
                if s["t_off"]==0.0: s["t_off"]=now
                if now-s["t_off"]>=ALERT_HOLD_OFF: s["is_on"]=False

def _draw_alert_banner(img_bgr,labels):
    if not labels: return img_bgr
    h,w = img_bgr.shape[:2]
    text = "  •  ".join(labels)
    font,scale,thick = cv2.FONT_HERSHEY_SIMPLEX,1.0,2
    (tw,th),_ = cv2.getTextSize(text,font,scale,thick)
    pad_x,pad_y = 28,16
    box_w,box_h = tw+pad_x*2, th+pad_y*2
    x0,y0 = (w-box_w)//2,40
    overlay = img_bgr.copy()
    cv2.rectangle(overlay,(x0,y0),(x0+box_w,y0+box_h),(0,0,0),-1)
    cv2.addWeighted(overlay,0.55,img_bgr,0.45,0,img_bgr)
    cv2.putText(img_bgr,text,(x0+pad_x,y0+pad_y+th),font,scale,(255,255,255),thick,cv2.LINE_AA)
    return img_bgr

def _draw_futuristic_overlay(img_bgr,face_landmarks):
    if face_landmarks is None or len(face_landmarks)<478: return img_bgr
    pts = face_landmarks[:,:2].astype(np.int32)
    idxs = {"left_eye":33,"right_eye":263,"nose_tip":1,"chin":152,"forehead":10,
            "mouth_left":61,"mouth_right":291,"left_iris":468,"right_iris":473}
    c1,c2 = (255,255,255),(90,220,255)
    P = lambda i: tuple(pts[i])
    overlay = img_bgr.copy()
    cv2.line(overlay,P(idxs["left_eye"]),P(idxs["right_eye"]),c1,1,cv2.LINE_AA)
    cv2.line(overlay,P(idxs["nose_tip"]),P(idxs["chin"]),c1,1,cv2.LINE_AA)
    cv2.line(overlay,P(idxs["forehead"]),P(idxs["nose_tip"]),c1,1,cv2.LINE_AA)
    cv2.line(overlay,P(idxs["mouth_left"]),P(idxs["mouth_right"]),c1,1,cv2.LINE_AA)
    for i in [idxs["left_iris"],idxs["right_iris"]]: cv2.circle(overlay,P(i),3,c2,1,cv2.LINE_AA)
    cv2.addWeighted(overlay,0.45,img_bgr,0.55,0,img_bgr)
    return img_bgr

# --- Streamlit Page ---
st.set_page_config(page_title="E-Learning Eyetracker — Live Monitor",layout="wide")
st.title("🧠 E-Learning Eyetracker — Live Education Monitor (MVP)")
st.caption("On-device. Education features only — not a medical device.")

# --- Initialize ---
tracker = Tracker()
eye_tracker = EyeTracker(max_len=60*cfg["video"]["fps"])
target_fps = cfg["video"]["fps"]
buf_len = 60 * target_fps
fatigue_buf, attention_buf, stress_buf = (deque(maxlen=buf_len) for _ in range(3))
video_placeholder = st.empty()
c1,c2,c3 = st.columns(3)
chart1, chart2, chart3 = c1.empty(),c2.empty(),c3.empty()
status_placeholder = st.empty()
UPDATE_HZ = cfg["windows"]["update_hz"]
_chart_every = max(1,int(target_fps/UPDATE_HZ))
frame_count = 0
fatigue_sm = attention_sm = stress_sm = 0.0
csv_write_interval = 1.0
t_last = time.time()

# --- Video Capture ---
cap = cv2.VideoCapture(int(cfg["video"]["source"]), cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS,target_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg["video"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg["video"]["height"])

# --- CSV Setup ---
if "csv_path" not in st.session_state:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.csv_path = f"{ts}_metrics.csv"
    with open(st.session_state.csv_path,"w",newline="") as f:
        csv.writer(f).writerow([
            "t_epoch","session_time","face_id","fatigue","attention","stress",
            "posture","distance","emotion","blink_rate","perclos","yawn_count",
            "yawn_event","pupil_diameter","fixation_duration","saccades","fixation","recommendation"
        ])
st.session_state.last_csv_write = 0.0

# --- Download Button ---
with open(st.session_state.csv_path, "rb") as f:
    st.download_button(
        label="📥 Download Metrics CSV",
        data=f,
        file_name=st.session_state.csv_path,
        mime="text/csv"
    )

# --- Buffers & placeholders ---
buf_len = 60 * target_fps
fatigue_buf, attention_buf, stress_buf = (deque(maxlen=buf_len) for _ in range(3))
blink_buf = deque(maxlen=buf_len)
yawn_buf = deque(maxlen=buf_len)
fatigue_level_buf = deque(maxlen=buf_len)
video_placeholder = st.empty()
c1,c2,c3 = st.columns(3)
chart1, chart2, chart3 = c1.empty(), c2.empty(), c3.empty()
c4,c5,c6 = st.columns(3)
chart4, chart5, chart6 = c4.empty(), c5.empty(), c6.empty()
status_placeholder = st.empty()
_chart_every = max(1,int(target_fps/UPDATE_HZ))
frame_count = 0
fatigue_sm = attention_sm = stress_sm = 0.0
csv_write_interval = 1.0
t_last = time.time()

# --- Main Loop ---
try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        det = tracker.process(frame_rgb)
        feats = compute_features(det)
        quality = estimate_quality(frame_rgb,det,feats,cfg)

        # --- Metrics ---
        post = posture_status(feats,quality,cfg)
        fat = fatigue_score(feats,quality,cfg,posture_state=post["state"])
        att = attention_score(feats,quality,cfg)
        strx = stress_score(feats,quality,cfg)
        dist = distance_status(det,cfg)
        emotion_val = detect_emotion(frame_rgb)
        emotion_num = EMOTION_MAP.get(emotion_val,0)

        # --- EMA smoothing ---
        alpha = cfg["smoothing"]["ema_alpha"]
        fatigue_sm = ema(fat.get("score",0.0),fatigue_sm,alpha)
        attention_sm = ema(att.get("score",0.0),attention_sm,alpha)
        stress_sm = ema(strx.get("score",0.0),stress_sm,alpha)

        # --- EyeTracker ---
        left_iris = feats.get("left_iris")
        right_iris = feats.get("right_iris")
        left_lms = feats.get("left_eye_landmarks")
        right_lms = feats.get("right_eye_landmarks")
        if left_iris is not None and right_iris is not None:
            eye_tracker.update(left_iris,right_iris,left_lms,right_lms)
        else:
            eye_tracker.update(np.array([[0,0]]),np.array([[0,0]]),None,None)
        fixation_dur, saccades_val, fixation_val, _ = eye_tracker.compute_metrics()

        # --- ML Prediction ---
        X_input = np.array([[ 
            datetime.now().timestamp(),
            float(datetime.now().strftime("%H%M%S")),
            feats.get("face_id",0),
            attention_sm,
            stress_sm,
            {"upright":0,"slouch":1,"leaning":2}.get(post["state"],-1),
            {"too_close":0,"normal":1,"too_far":2}.get(dist["state"],-1),
            emotion_num,
            fat.get("blink_rate",0.0),
            fat.get("perclos",0.0),
            fat.get("yawn_count",0),
            1 if fat.get("yawn_counted_this_yawn",False) else 0,
            feats.get("pupil_diameter",0.0),
            fixation_dur,
            saccades_val,
            fixation_val
        ]])
        try:
            pred_encoded = model.predict(X_input)[0]
            fatigue_pred_ml = FATIGUE_LABELS[pred_encoded]
        except: fatigue_pred_ml = "medium"
        fatigue_score_ml = {"low":20,"medium":50,"high":80}[fatigue_pred_ml]

        # --- Buffers ---
        fatigue_buf.append(fatigue_score_ml)
        attention_buf.append(attention_sm)
        stress_buf.append(stress_sm)
        blink_buf.append(fat.get("blink_rate",0.0))
        yawn_buf.append(fat.get("yawn_count",0))
        fatigue_level_buf.append(fatigue_score_ml)

        # --- Recommendation ---
        if fatigue_score_ml>55 or (50<=fatigue_score_ml<=55 and attention_sm<30):
            recommendation_val="Take a long break"
        elif attention_sm<40:
            recommendation_val="Refocus"
        else:
            recommendation_val="Keep learning"

        # --- Alerts ---
        _update_alert_states(attention_sm,fatigue_score_ml,stress_sm,time.time())
        active_labels = [name for name,s in st.session_state.alert_state.items() if s["is_on"]]

        # --- Draw overlays ---
        labeled = draw_labels(frame,post,dist)
        labeled = _draw_futuristic_overlay(labeled,feats.get("face",None))
        labeled = _draw_alert_banner(labeled,active_labels)
        video_placeholder.image(labeled[:,:,::-1],channels="RGB",use_container_width=True)

        # --- CSV Logging ---
        t_now = time.time()
        if t_now-st.session_state.last_csv_write>=csv_write_interval:
            with open(st.session_state.csv_path,"a",newline="") as f:
                csv.writer(f).writerow([
                    t_now,
                    datetime.fromtimestamp(t_now).strftime("%H:%M:%S"),
                    feats.get("face_id",0),
                    f"{fatigue_score_ml:.2f}",
                    f"{attention_sm:.2f}",
                    f"{stress_sm:.2f}",
                    str(post["state"]),
                    str(dist["state"]),
                    str(emotion_val),
                    f"{fat.get('blink_rate',0.0):.1f}",
                    f"{fat.get('perclos',0.0):.2f}",
                    fat.get("yawn_count",0),
                    1 if fat.get("yawn_counted_this_yawn",False) else 0,
                    f"{feats.get('pupil_diameter',0.0):.2f}",
                    f"{fixation_dur:.2f}",
                    f"{saccades_val:.2f}",
                    f"{fixation_val:.2f}",
                    recommendation_val
                ])
            st.session_state.last_csv_write = t_now

        # --- Status Display ---
        status_placeholder.info(
    f"Fatigue: {fatigue_score_ml:.0f} ({fatigue_pred_ml}) | "
    f"Attention: {attention_sm:.0f} | Stress: {stress_sm:.0f} | "
    f"Blink: {fat.get('blink_rate',0.0):.1f} | "
    f"Yawn: {fat.get('yawn_count',0)} | "
    f"Recommendation: {recommendation_val}"
)
        


        # --- Charts ---
        frame_count += 1
        if frame_count%_chart_every==0:
            def plot_series(container,ys,title,color):
                xs = np.arange(len(ys))/max(1,target_fps)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color=color,width=3)))
                fig.update_layout(height=220,margin=dict(l=10,r=10,t=30,b=10),title=title,yaxis=dict(range=[0,100]))
                container.plotly_chart(fig,use_container_width=True)

            # Main charts
            plot_series(chart1,list(fatigue_buf),"Fatigue","#1f77b4")
            plot_series(chart2,list(attention_buf),"Attention","#2ca02c")
            plot_series(chart3,list(stress_buf),"Stress","#d62728")
            
           

        # --- Frame timing ---
        dt = time.time()-t_last
        target_dt = 1.0/UPDATE_HZ
        if dt<target_dt: time.sleep(target_dt-dt)
        t_last = time.time()

except KeyboardInterrupt:
    pass
finally:
    cap.release()
