import numpy as np
import time
from collections import deque

class EyeTracker:
    """Tracks eye positions and computes fixation, saccade, stability, blink rate, and pupil diameter."""

    def __init__(self, max_len=1800, blink_threshold=0.25, blink_window_sec=2.0, ema_alpha=0.3):
        self.left_eye_buf = deque(maxlen=max_len)
        self.right_eye_buf = deque(maxlen=max_len)
        self.time_buf = deque(maxlen=max_len)
        self.blink_times = deque(maxlen=300)
        self.blink_threshold = blink_threshold
        self.blink_window_sec = blink_window_sec
        self.ema_alpha = ema_alpha

        # EMA placeholders
        self.fixation_ema = 0.0
        self.saccades_ema = 0.0
        self.stability_ema = 0.0
        self.blink_rate_ema = 0.0
        self.pupil_left_ema = 0.0
        self.pupil_right_ema = 0.0

    def update(self, left_iris, right_iris, left_eye_landmarks=None, right_eye_landmarks=None, pupil_left=None, pupil_right=None):
        """Update iris positions, blink detection, and pupil diameter."""
        if left_iris is None or right_iris is None:
            return

        left_iris = np.array(left_iris).reshape(-1)[:2]
        right_iris = np.array(right_iris).reshape(-1)[:2]

        self.left_eye_buf.append(left_iris)
        self.right_eye_buf.append(right_iris)
        self.time_buf.append(time.time())

        # Blink detection
        if self._is_blink(left_eye_landmarks) or self._is_blink(right_eye_landmarks):
            self.blink_times.append(time.time())

        # Pupil diameter EMA
        if pupil_left is not None:
            self.pupil_left_ema = self.ema_alpha * pupil_left + (1 - self.ema_alpha) * self.pupil_left_ema
        if pupil_right is not None:
            self.pupil_right_ema = self.ema_alpha * pupil_right + (1 - self.ema_alpha) * self.pupil_right_ema

    def _is_blink(self, eye_landmarks):
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return False
        p1, p2, p3, p4, p5, p6 = np.array(eye_landmarks[:6])
        ear = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * (np.linalg.norm(p1-p4)+1e-6))
        return ear < self.blink_threshold

    def _compute_blink_rate(self):
        now = time.time()
        while self.blink_times and now - self.blink_times[0] > self.blink_window_sec:
            self.blink_times.popleft()
        rate = len(self.blink_times) * (60.0 / max(1e-3, self.blink_window_sec))
        self.blink_rate_ema = self.ema_alpha * rate + (1 - self.ema_alpha) * self.blink_rate_ema
        return self.blink_rate_ema

    def compute_metrics(self):
        """Compute normalized metrics (0-100) for fixation, saccades, stability, blink rate, and pupil diameters."""
        if len(self.left_eye_buf) < 2 or len(self.right_eye_buf) < 2:
            return (self.fixation_ema, self.saccades_ema, self.stability_ema,
                    self._compute_blink_rate(), self.pupil_left_ema, self.pupil_right_ema)

        left = np.array(self.left_eye_buf)
        right = np.array(self.right_eye_buf)
        centers = (left + right) / 2.0

        deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        times = np.diff(np.array(self.time_buf))
        min_len = min(len(deltas), len(times))
        deltas = deltas[:min_len]
        times = times[:min_len]

        velocities = deltas / (times + 1e-6)

        SACCADE_VEL = 50.0
        FIXATION_VEL = 10.0

        saccades_count = np.sum(velocities > SACCADE_VEL)
        fixation_frames = np.sum(velocities < FIXATION_VEL)
        total_time = self.time_buf[-1] - self.time_buf[0] + 1e-6
        fixation_duration_sec = (fixation_frames / len(velocities)) * total_time

        fixation_norm = min(fixation_duration_sec / 2.0 * 100, 100)
        saccades_norm = min(saccades_count / 10.0 * 100, 100)
        stability_norm = (fixation_frames / len(velocities)) * 100

        self.fixation_ema = self.ema_alpha * fixation_norm + (1 - self.ema_alpha) * self.fixation_ema
        self.saccades_ema = self.ema_alpha * saccades_norm + (1 - self.ema_alpha) * self.saccades_ema
        self.stability_ema = self.ema_alpha * stability_norm + (1 - self.ema_alpha) * self.stability_ema

        return (self.fixation_ema, self.saccades_ema, self.stability_ema,
                self._compute_blink_rate(), self.pupil_left_ema, self.pupil_right_ema)
