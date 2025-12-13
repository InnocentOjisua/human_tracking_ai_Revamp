import cv2
import numpy as np

def draw_futuristic_overlay(frame_bgr, det=None, attention=None):
    """
    Draws a futuristic bounding box around detected face.
    
    Args:
        frame_bgr: BGR frame from OpenCV
        det: detector output containing face box: {"bbox": [x, y, w, h]}
        attention: attention score or state (0-1 or low/medium/high)
    
    Returns:
        Frame with futuristic bounding box
    """
    img = frame_bgr.copy()

    if det is None or "bbox" not in det:
        return img

    x, y, w, h = det["bbox"]

    # Choose color based on attention level
    # You can modify this rule
    if attention is not None and (attention == "low" or attention < 0.4):
        color = (0, 255, 0)   # Green = distracted or low attention
    else:
        color = (255, 0, 0)   # Blue = focused

    # Make coordinates safe
    x, y = max(x, 0), max(y, 0)
    x2, y2 = x + w, y + h

    thickness = 2

    # Draw a futuristic corner box (not full rectangle)
    corner_len = int(w * 0.25)

    # Top-left corner
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y), (x2 - corner_len, y), color, thickness)
    cv2.line(img, (x2, y), (x2, y + corner_len), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x, y2), (x + corner_len, y2), color, thickness)
    cv2.line(img, (x, y2), (x, y2 - corner_len), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

    return img
