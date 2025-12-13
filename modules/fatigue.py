# modules/fatigue.py
import time
import numpy as np

# ============================
# CONFIG / CONSTANTS
# ============================

EAR_CLOSED = 0.20          # threshold for eye-closure
BLINK_MIN_FRAMES = 2       # frames to count as blink

DECAY_PER_SEC       = 20.0   # global decay toward 0
SLOUCH_BUMP_PER_SEC = 6.0
MOTION_THRESHOLD    = 0.20
MOTION_GAIN         = 180.0
OCCLUSION_DROP_FRAC = 0.35
OCCLUSION_BUMP      = 18.0

PERCLOS_WINDOW      = 60.0   # seconds for closed-time integration
PERCLOS_BASE_GAIN   = 25.0   # gain from PERCLOS → baseline fatigue

# ============================
# INTERNAL STATE
# ============================

_state = {
    "closed_frames": 0,
    "last_closed": False,
    "closed_intervals": [],   # stores (start, end) for closed eyes
    "closed_start": None,

    "yawn_start": None,
    "yawn_count": 0,          # counts yawns
    "score": 0.0,

    "last_face_area": None,
    "last_t": None,
}

# ============================
# RESET
# ============================

def reset():
    _state.update({
        "closed_frames": 0,
        "last_closed": False,
        "closed_intervals": [],
        "closed_start": None,

        "yawn_start": None,
        "yawn_count": 0,
        "score": 0.0,

        "last_face_area": None,
        "last_t": None,
    })

# ============================
# HELPERS
# ============================

def _update_closed_intervals(now, closed, fps):
    """Track closed-eye durations for PERCLOS and microsleep."""
    if closed and _state["closed_start"] is None:
        _state["closed_start"] = now

    if not closed and _state["closed_start"] is not None:
        start = _state["closed_start"]
        end = now
        _state["closed_intervals"].append((start, end))
        _state["closed_start"] = None

    # purge old intervals past PERCLOS_WINDOW
    cutoff = now - PERCLOS_WINDOW
    _state["closed_intervals"] = [(s, e) for (s, e) in _state["closed_intervals"] if e >= cutoff]

    total_closed = sum(max(0, min(e, now) - max(s, cutoff))
                       for (s, e) in _state["closed_intervals"])
    perclos = np.clip(total_closed / PERCLOS_WINDOW, 0.0, 1.0)
    return perclos

# ============================
# MAIN FUNCTION
# ============================

def fatigue_score(feats, quality, cfg=None, calibrating=False,
                  posture_state=None, motion_energy=0.0):
    """
    High-quality fatigue estimation model with PERCLOS, microsleep,
    yawn detection (count + peak), slouch, motion/occlusion bumps,
    attention, and global decay.
    """

    # Safe defaults for cfg
    cfg = cfg or {}
    fps = cfg.get("video", {}).get("fps", 30)
    yawn_thr = cfg.get("thresholds", {}).get("yawn_mar", 0.55)
    yawn_min = cfg.get("thresholds", {}).get("yawn_min_secs", 0.3)

    face = feats.get("face")
    mar = feats.get("mar")
    le, re = feats.get("ear_left"), feats.get("ear_right")
    attention = feats.get("attention_score")

    # Compute average EAR safely
    if le is None and re is None:
        ear = None
    elif le is None:
        ear = re
    elif re is None:
        ear = le
    else:
        ear = (le + re) / 2.0

    now = time.time()

    # Calibration resets state
    if calibrating:
        reset()
        _state["last_t"] = now
        return {
            "score": 0.0,
            "blink_rate": 0.0,
            "perclos": 0.0,
            "yawn_count": 0,
            "conf": (quality.get("conf_base", 1.0)) * 0.8
        }

    # dt
    if _state["last_t"] is None:
        _state["last_t"] = now
    dt = max(0.0, min(0.25, now - _state["last_t"]))
    _state["last_t"] = now

    # ============================
    # EYE CLOSURE
    # ============================
    closed = (ear is not None) and (ear < EAR_CLOSED)
    _state["closed_frames"] = _state["closed_frames"] + 1 if closed else 0
    perclos = _update_closed_intervals(now, closed, fps)
    perclos_fatigue = PERCLOS_BASE_GAIN * perclos

    # ============================
    # LONG CLOSURE / MICROSLEEP
    # ============================
    if closed:
        closed_seconds = _state["closed_frames"] / max(1, fps)
        if closed_seconds >= 0.5:
            _state["score"] = min(100.0, _state["score"] + 35.0 * dt)
        if closed_seconds >= 1.0:
            _state["score"] = max(_state["score"], 90.0)

    # ============================
    # YAWN → instant top + count
    # ============================
    if mar is not None and mar >= yawn_thr:
        if _state["yawn_start"] is None:
            _state["yawn_start"] = now
            _state["yawn_counted_this_yawn"] = False
        elif now - _state["yawn_start"] >= yawn_min:
            over = np.clip((mar - yawn_thr) / 0.35, 0.0, 1.0)
            peak = 95.0 + 5.0 * over
            _state["score"] = max(_state["score"], peak)

            # increment yawn count only once per yawn
            if not _state.get("yawn_counted_this_yawn", False):
                _state["yawn_count"] += 1
                _state["yawn_counted_this_yawn"] = True
    else:
        _state["yawn_start"] = None
        _state["yawn_counted_this_yawn"] = False

    # ============================
    # MOTION BUMP
    # ============================
    if motion_energy is not None and motion_energy > MOTION_THRESHOLD:
        bump = (motion_energy - MOTION_THRESHOLD) * MOTION_GAIN
        _state["score"] = min(100.0, _state["score"] + bump)

    # ============================
    # OCCLUSION BUMP
    # ============================
    if face is not None:
        xs, ys = face[:, 0], face[:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1e-6
        if _state["last_face_area"] is not None:
            if area < (1.0 - OCCLUSION_DROP_FRAC) * _state["last_face_area"]:
                _state["score"] = min(100.0, _state["score"] + OCCLUSION_BUMP)
        _state["last_face_area"] = area

    # ============================
    # SLOUCHING DRIFT
    # ============================
    if posture_state == "slouching":
        _state["score"] = min(100.0, _state["score"] + SLOUCH_BUMP_PER_SEC * dt)

    # ============================
    # ATTENTION-BASED FATIGUE
    # ============================
    if attention is not None:
        if attention < 30:
            _state["score"] = min(100.0, _state["score"] + 25.0 * dt)
        if attention < 20:
            _state["score"] = min(100.0, _state["score"] + 55.0 * dt)

    # ============================
    # GLOBAL DECAY
    # ============================
    if _state["score"] > 0:
        _state["score"] = max(0.0, _state["score"] - DECAY_PER_SEC * dt)

    # ============================
    # FINAL SCORE
    # ============================
    final_score = max(perclos_fatigue, _state["score"])
    final_score = float(np.clip(final_score, 0.0, 100.0))
    conf = (quality.get("conf_base", 1.0)) * (1.0 if ear is not None else 0.6)
    blink_rate = len(_state["closed_intervals"])

    return {
        "score": final_score,
        "blink_rate": blink_rate,
        "perclos": float(perclos),
        "yawn_count": _state["yawn_count"],
        "conf": conf
    }
