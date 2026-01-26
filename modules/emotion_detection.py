# modules/emotion_detection.py
from fer import FER
import cv2

# Initialize FER detector once
detector = FER()

# Keep track of fatigue multiplier per session
emotion_fatigue_state = {"factor": 0.0, "last_update": 0.0}

def detect_emotion(frame, detections=None):
    """
    Detect dominant emotion from a frame.
    Returns a string, e.g., "happy", "neutral", "tired", etc.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb)
    if not results:
        return "neutral"
    emotions = results[0]["emotions"]
    dominant = max(emotions, key=emotions.get)
    # Map FER emotions to simplified categories
    if dominant in ["angry", "disgust", "fear", "sad", "tired", "bored"]:
        return "tired"
    return dominant

def emotion_fatigue_factor(emotion, max_boost=15.0, decay=0.9):
    """
    Returns a dynamic fatigue boost based on detected emotion.
    Consecutive 'tired' frames increase the factor gradually.
    """
    now = cv2.getTickCount() / cv2.getTickFrequency()
    if emotion == "tired":
        # Increase gradually, up to max_boost
        emotion_fatigue_state["factor"] = min(max_boost, emotion_fatigue_state["factor"] + 0.5)
    else:
        # Decay gradually when not tired
        emotion_fatigue_state["factor"] *= decay
    emotion_fatigue_state["last_update"] = now
    return emotion_fatigue_state["factor"]
