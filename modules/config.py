# config.py

# Emotion categories (integer-based)
EMOTION_MAP = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Surprised",
    5: "Disgust",
    6: "Fear"
}

# Posture categories
POSTURE_MAP = {
    0: "Upright",
    1: "Slouching",
    2: "Leaning Forward",
    3: "Leaning Back",
    4: "Unknown"
}

# Distance categories
DISTANCE_MAP = {
    0: "Optimal",
    1: "Too Close",
    2: "Too Far",
    3: "Unknown"
}

# General configuration
cfg = {
    "video": {"fps": 30},
    "smoothing": {"ema_alpha": 0.25},
    "show_futuristic": True
}