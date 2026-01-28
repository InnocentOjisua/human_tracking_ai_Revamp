# --- Determine if running locally ---
import os
is_local = "CLOUD_ENV" not in os.environ  # Streamlit Cloud sets environment variables

# --- Initialize camera (local or Streamlit Cloud) ---
if is_local:
    cam_source = cfg["video"].get("source", 0)
    target_fps = int(cfg["video"].get("fps", 30))
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["video"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["video"]["height"])
    if not cap.isOpened():
        st.error(f"Failed to open local camera: {cam_source}")
        st.stop()
else:
    st.info("Running on Streamlit Cloud — using browser camera input")
    cap = None  # we will use st.camera_input

# --- Main loop ---
try:
    while True:
        # --- Get frame ---
        if is_local:
            ok, frame = cap.read()
            if not ok or frame is None:
                st.error("Failed to read from local camera.")
                break
        else:
            uploaded_file = st.camera_input("📷 Position your face for tracking")
            if uploaded_file is None:
                st.warning("Waiting for camera input...")
                time.sleep(0.1)
                continue
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Failed to read frame from browser camera.")
                continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Tracking & Features ---
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
        fatigue_sm   = ema(fat.get("score",0.0), fatigue_sm, alpha)
        attention_sm = ema(att.get("score",0.0), attention_sm, alpha)
        stress_sm    = ema(strx.get("score",0.0), stress_sm, alpha)

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

        # --- Alerts & Buffers ---
        _update_alert_states(attention_sm, fatigue_sm, stress_sm, time.time())
        active_labels = [name for name,s in st.session_state.alert_state.items() if s["is_on"]]
        fatigue_buf.append(fatigue_sm)
        attention_buf.append(attention_sm)
        stress_buf.append(stress_sm)

        # --- Draw overlays ---
        labeled = draw_labels(frame, post, dist)
        if show_futuristic:
            labeled = _draw_futuristic_overlay(labeled, feats.get("face", None))
        labeled = _draw_alert_banner(labeled, active_labels)
        video_placeholder.image(labeled[:, :, ::-1], channels="RGB", use_container_width=True)

        # --- Recommendation ---
        if fatigue_sm > 55:
            recommendation_val = "Take a long break"
        elif attention_sm < 40:
            recommendation_val = "Refocus"
        else:
            recommendation_val = "Keep learning"

        # --- CSV logging ---
        t_now = time.time()
        if t_now - st.session_state.last_csv_write >= csv_write_interval:
            t_epoch = t_now
            t_human = datetime.fromtimestamp(t_now).strftime("%H:%M:%S")
            face_id = feats.get("face_id", 0)
            with open(st.session_state.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    t_epoch, t_human, face_id, f"{fatigue_sm:.2f}", f"{attention_sm:.2f}", f"{stress_sm:.2f}",
                    str(post["state"]), str(dist["state"]), str(emotion_val), f"{fatigue_sm:.1f}", f"{attention_sm:.2f}",
                    fat.get("yawn_count",0), 1 if fat.get("yawn_counted_this_yawn", False) else 0,
                    f"{feats.get('pupil_diameter',0.0):.2f}", f"{fixation_duration_val:.2f}", f"{saccades_val:.2f}",
                    f"{fixation_val:.2f}", recommendation_val
                ])
            st.session_state.last_csv_write = t_now

        # --- Status display ---
        status_placeholder.info(
            f"Fatigue: {fatigue_sm:0.0f} | Attention: {attention_sm:0.0f} | Stress: {stress_sm:0.0f} | Recommendation: {recommendation_val}"
        )

        # --- Plot charts ---
        frame_count += 1
        if frame_count % _chart_every == 0:
            def plot_series(container, ys, title, color):
                xs = np.arange(len(ys))/max(1,target_fps)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3)))
                fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10), title=title, yaxis=dict(range=[0,100]))
                container.plotly_chart(fig, use_container_width=True)
            plot_series(chart1, list(fatigue_buf), "Fatigue", "#1f77b4")
            plot_series(chart2, list(attention_buf), "Attention", "#2ca02c")
            plot_series(chart3, list(stress_buf), "Stress", "#d62728")

        # --- Frame timing ---
        dt = time.time() - t_last
        target_dt = 1.0 / UPDATE_HZ
        if dt < target_dt:
            time.sleep(target_dt - dt)
        t_last = time.time()

finally:
    if cap is not None:
        cap.release()
