# modules/utils.py
import numpy as np
import cv2

def compute_features(detections):
    features = []
    for det in detections:
        feats = {
            "blink_rate": np.random.rand(),
            "pupil_diameter": np.random.rand() * 5,
            "fixation_duration": np.random.rand() * 2
        }
        features.append(feats)
    return features

def estimate_quality(frame, detections, feats_list, cfg):
    quality_list = []
    for det in detections:
        sharpness = np.random.rand()
        lighting = np.random.rand()

        # Confidence baseline: weighted average of sharpness & lighting
        conf_base = 0.6 * sharpness + 0.4 * lighting

        quality = {
            "sharpness": sharpness,
            "lighting": lighting,
            "conf_base": conf_base   # ✅ realistic confidence baseline
        }
        quality_list.append(quality)
    return quality_list