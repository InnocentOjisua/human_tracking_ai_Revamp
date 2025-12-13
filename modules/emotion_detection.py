# modules/emotion_detection.py

from fer import FER
import cv2

# Initialize FER detector once
detector = FER()

def detect_emotion(frame):
    """
    Detect dominant emotion from a frame without bounding boxes.

    Args:
        frame (ndarray): BGR frame from OpenCV.

    Returns:
        str: dominant emotion (e.g., "happy", "sad", "neutral")
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb)

    if not results:
        return "neutral"  # fallback if no faces detected

    # Pick the first detected face
    emotions = results[0]["emotions"]
    dominant = max(emotions, key=emotions.get)
    return dominant
