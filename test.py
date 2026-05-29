# check what the scaler expects
import pickle
from face_tracker import extract_au_features

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print(f"Scaler expects: {scaler.n_features_in_} features")

# check what extract_au_features produces live
import mediapipe as mp
import numpy as np

# dummy landmark flat array (468 landmarks * 3)
dummy = np.zeros(468 * 3, dtype=np.float32)
features = extract_au_features(dummy)
print(f"extract_au_features produces: {len(features)} features")