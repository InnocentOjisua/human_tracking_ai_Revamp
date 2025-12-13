import numpy as np

# FaceMesh landmark indices for eyes/mouth (6 points per eye for EAR, 2 vertical points for MAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye landmarks
MOUTH = [61, 291, 13, 14, 78, 308]         # lips corners & vertical

def _aspect_ratio(pts):
    """EAR/MAR style: vertical / horizontal ratio"""
    p = pts
    vert = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
    horiz = np.linalg.norm(p[0]-p[3]) + 1e-6
    return vert / (2.0 * horiz)

def eye_aspect_ratio(face):
    if face is None: return None, None
    le = face[LEFT_EYE, :2]
    re = face[RIGHT_EYE, :2]
    return _aspect_ratio(le), _aspect_ratio(re)

def mouth_aspect_ratio(face):
    if face is None: return None
    m = face[MOUTH, :2]
    vert = np.linalg.norm(m[2]-m[3])
    horiz = np.linalg.norm(m[0]-m[1]) + 1e-6
    return vert / horiz

def gaze_proxy(face):
    if face is None: return 0.0
    xs, ys = face[:,0], face[:,1]
    cx, cy = xs.mean(), ys.mean()
    left_center = face[np.array(LEFT_EYE), :2].mean(axis=0)
    right_center = face[np.array(RIGHT_EYE), :2].mean(axis=0)
    eye_center = (left_center + right_center) / 2.0
    dist = np.linalg.norm(eye_center - np.array([cx, cy]))
    face_diag = np.hypot(xs.max()-xs.min(), ys.max()-ys.min()) + 1e-6
    return float(np.clip(1.0 - (dist / (0.35 * face_diag)), 0.0, 1.0))

def neck_angle_deg(pose):
    if pose is None: return None
    try:
        l_sh, r_sh = pose[11,:2], pose[12,:2]
        l_ear, r_ear = pose[7,:2], pose[8,:2]
        shoulder_mid = (l_sh + r_sh)/2.0
        ear_mid = (l_ear + r_ear)/2.0
        v = ear_mid - shoulder_mid
        v_norm = v / (np.linalg.norm(v)+1e-6)
        dot = np.dot(v_norm, np.array([0.0,-1.0]))
        ang = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return float(ang)
    except Exception:
        return None

def pupil_diameter(face):
    """Estimate pupil diameter (pixels) from iris landmarks"""
    if face is None or face.shape[0] < 478:
        return None, None

    left_iris_pts = face[468:473, :2]  # left iris 5 points
    right_iris_pts = face[473:478, :2] # right iris 5 points

    def diameter(pts):
        c = pts.mean(axis=0)
        return float(np.mean(np.linalg.norm(pts - c, axis=1)) * 2.0)  # approximate diameter

    return diameter(left_iris_pts), diameter(right_iris_pts)

def compute_features(det):
    face = det["face"]
    pose = det["pose"]

    # --- Eye & mouth metrics ---
    le, re = eye_aspect_ratio(face)
    mar = mouth_aspect_ratio(face)
    gaze = gaze_proxy(face)
    neck = neck_angle_deg(pose)

    # --- Pupil diameter ---
    pd_left, pd_right = pupil_diameter(face)
    pupil_diameter_val = np.mean([d for d in [pd_left, pd_right] if d is not None]) if pd_left or pd_right else 0.0

    # --- EyeTracker specific ---
    left_iris = face[468, :2] if face is not None else None
    right_iris = face[473, :2] if face is not None else None
    left_eye_landmarks = face[LEFT_EYE, :2] if face is not None else None
    right_eye_landmarks = face[RIGHT_EYE, :2] if face is not None else None

    return {
        "face": face,
        "pose": pose,
        "ear_left": le,
        "ear_right": re,
        "mar": mar,
        "gaze": gaze,
        "neck_angle": neck,
        "pupil_diameter": pupil_diameter_val,
        "left_iris": left_iris,
        "right_iris": right_iris,
        "left_eye_landmarks": left_eye_landmarks,
        "right_eye_landmarks": right_eye_landmarks
    }
