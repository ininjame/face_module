"""Single source of truth for landmark extraction."""
import numpy as np
from config import ALL_LANDMARK_INDICES


def extract_landmarks_from_result(result, indices=ALL_LANDMARK_INDICES):
    """MediaPipe FaceMesh result -> (N, 2) float32 array in [0, 1], or None."""
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([[lm[i].x, lm[i].y] for i in indices], dtype=np.float32)
    return np.clip(pts, 0.0, 1.0)
