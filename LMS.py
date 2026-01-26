import time
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from collections import deque
from pathlib import Path
import csv
import threading

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
from ui.overlays import draw_labels


class EyetrackerApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.UPDATE_HZ = 5

        # Video capture
        self.cap = cv2.VideoCapture(cfg["video"]["source"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["video"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["video"]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cfg["video"]["fps"])

        self.tracker = Tracker()
        self.eye_tracker = EyeTracker(max_len=60 * cfg["video"]["fps"])

        # Buffers
        buf_len = cfg["video"]["fps"] * 60
        self.fatigue_buf = deque(maxlen=buf_len)
        self.attention_buf = deque(maxlen=buf_len)
        self.stress_buf = deque(maxlen=buf_len)

        # Smoothed values
        self.fatigue_sm = 0.0
        self.attention_sm = 0.0
        self.stress_sm = 0.0

        # CSV setup
        self.csv_path = Path("metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["t_epoch", "fatigue", "attention", "stress"])

        # Alerts
        self.ALERTS = {
            "DISTRACTED": lambda att, fat, strx: att < 50,
            "TIRED": lambda att, fat, strx: fat > 50,
            "STRESSED": lambda att, fat, strx: strx > 50,
        }
        self.alert_state = {name: {"is_on": False, "t_on": 0, "t_off": 0}
                            for name in self.ALERTS.keys()}
        self.ALERT_DEBOUNCE_ON = 0.4
        self.ALERT_HOLD_OFF = 1.5

        # Latest frame
        self.latest_frame = np.zeros((cfg["video"]["height"], cfg["video"]["width"], 3), dtype=np.uint8)
        self.latest_labels = []

        # Thread control
        self.running = True
        self.lock = threading.Lock()

        # GUI setup
        self._setup_gui()

        # Start background processing
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()

    # ---------------- GUI Setup ----------------
    def _setup_gui(self):
        dpg.create_context()
        self.video_tag = "video_texture"

        with dpg.window(label="E-Learning Eyetracker", width=800, height=700):
            dpg.add_text("Status: initializing...", tag="status_text")
            dpg.add_dynamic_texture(
                self.cfg["video"]["width"], self.cfg["video"]["height"],
                np.zeros((self.cfg["video"]["height"], self.cfg["video"]["width"], 4), dtype=np.float32),
                tag=self.video_tag
            )

            with dpg.plot(label="Metrics", height=200, width=780):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag="y_axis")
                dpg.add_line_series([], [], label="Fatigue", parent="y_axis", tag="fatigue_series")
                dpg.add_line_series([], [], label="Attention", parent="y_axis", tag="attention_series")
                dpg.add_line_series([], [], label="Stress", parent="y_axis", tag="stress_series")

            dpg.add_button(label="Download CSV", callback=self._download_csv)

        dpg.create_viewport(title='E-Learning Eyetracker', width=820, height=720)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # GUI update timer
        dpg.add_timer(1 / self.UPDATE_HZ, callback=self.update_gui, tag="update_timer", repeat=True)

    def _download_csv(self):
        import shutil
        shutil.copy(self.csv_path, "metrics_download.csv")
        print("CSV copied to metrics_download.csv")

    # ---------------- Alerts ----------------
    def _update_alerts(self, att, fat, strx):
        now = time.time()
        labels = []
        for name, cond in self.ALERTS.items():
            active = cond(att, fat, strx)
            s = self.alert_state[name]
            if active:
                if not s["is_on"]:
                    if s["t_on"] == 0: s["t_on"] = now
                    if now - s["t_on"] >= self.ALERT_DEBOUNCE_ON:
                        s["is_on"] = True
                        s["t_off"] = 0
                else:
                    s["t_off"] = 0
            else:
                s["t_on"] = 0
                if s["is_on"]:
                    if s["t_off"] == 0: s["t_off"] = now
                    if now - s["t_off"] >= self.ALERT_HOLD_OFF:
                        s["is_on"] = False
            if s["is_on"]:
                labels.append(name)
        return labels

    # ---------------- Background processing ----------------
    def _background_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = self.tracker.process(frame_rgb)
            feats = compute_features(det)
            quality = estimate_quality(frame_rgb, det, feats, self.cfg)

            # Metrics
            fat = fatigue_score(feats, quality, self.cfg)
            att = attention_score(feats, quality, self.cfg)
            strx = stress_score(feats, quality, self.cfg)
            alpha = self.cfg["smoothing"]["ema_alpha"]
            self.fatigue_sm = ema(fat.get("score", 0), self.fatigue_sm, alpha)
            self.attention_sm = ema(att.get("score", 0), self.attention_sm, alpha)
            self.stress_sm = ema(strx.get("score", 0), self.stress_sm, alpha)

            # Buffers
            self.fatigue_buf.append(self.fatigue_sm)
            self.attention_buf.append(self.attention_sm)
            self.stress_buf.append(self.stress_sm)

            # Video overlay
            labeled = draw_labels(frame, posture_status(feats, quality, self.cfg),
                                  distance_status(det, self.cfg))
            labeled = self._draw_futuristic_overlay(labeled, feats.get("face", None))
            labels = self._update_alerts(self.attention_sm, self.fatigue_sm, self.stress_sm)
            labeled = self._draw_alert_banner(labeled, labels)

            # Store latest frame and labels thread-safely
            with self.lock:
                self.latest_frame = labeled.copy()
                self.latest_labels = labels

            # CSV logging
            t_now = time.time()
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([t_now, self.fatigue_sm, self.attention_sm, self.stress_sm])

            time.sleep(1 / self.UPDATE_HZ)

    # ---------------- GUI Update ----------------
    def update_gui(self):
        # Get the latest frame safely
        with self.lock:
            frame = self.latest_frame.copy()
            labels = self.latest_labels.copy()

        # Convert to RGBA for DearPyGui
        frame_tex = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) / 255.0
        dpg.set_value(self.video_tag, frame_tex)

        # Update status
        dpg.set_value("status_text",
                      f"Fatigue: {self.fatigue_sm:.1f} | Attention: {self.attention_sm:.1f} | Stress: {self.stress_sm:.1f}")

        # Update charts
        x_vals = list(range(len(self.fatigue_buf)))
        dpg.set_value("fatigue_series", [x_vals, list(self.fatigue_buf)])
        dpg.set_value("attention_series", [x_vals, list(self.attention_buf)])
        dpg.set_value("stress_series", [x_vals, list(self.stress_buf)])

    # ---------------- Overlays ----------------
    def _draw_alert_banner(self, img_bgr, labels):
        if not labels: return img_bgr
        h, w = img_bgr.shape[:2]
        text = "  •  ".join(labels)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad_x, pad_y = 28, 16
        box_w, box_h = tw + pad_x * 2, th + pad_y * 2
        x0, y0 = (w - box_w) // 2, 40
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)
        cv2.putText(img_bgr, text, (x0 + pad_x, y0 + pad_y + th), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
        return img_bgr

    def _draw_futuristic_overlay(self, img_bgr, face_landmarks):
        if face_landmarks is None or len(face_landmarks) < 478:
            return img_bgr
        pts = face_landmarks[:, :2].astype(np.int32)
        idxs = {
            "left_eye": 33, "right_eye": 263, "nose_tip": 1, "chin": 152,
            "forehead": 10, "mouth_left": 61, "mouth_right": 291,
            "left_iris": 468, "right_iris": 473
        }
        c1, c2 = (255, 255, 255), (90, 220, 255)
        P = lambda i: tuple(pts[i])
        overlay = img_bgr.copy()
        cv2.line(overlay, P(idxs["left_eye"]), P(idxs["right_eye"]), c1, 1, cv2.LINE_AA)
        cv2.line(overlay, P(idxs["nose_tip"]), P(idxs["chin"]), c1, 1, cv2.LINE_AA)
        cv2.line(overlay, P(idxs["forehead"]), P(idxs["nose_tip"]), c1, 1, cv2.LINE_AA)
        cv2.line(overlay, P(idxs["mouth_left"]), P(idxs["mouth_right"]), c1, 1, cv2.LINE_AA)
        for i in [idxs["left_iris"], idxs["right_iris"]]:
            cv2.circle(overlay, P(i), 3, c2, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0, img_bgr)
        return img_bgr


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    cfg = {
        "video": {"source": 0, "fps": 30, "width": 640, "height": 480},
        "smoothing": {"ema_alpha": 0.25},
    }
    app = EyetrackerApp(cfg)

    try:
        dpg.start_dearpygui()
    finally:
        app.running = False
        app.thread.join()
        app.cap.release()
        dpg.destroy_context()
