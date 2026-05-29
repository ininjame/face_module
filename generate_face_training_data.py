import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import numpy as np
import pandas as pd
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh

# Expression label mapping
# FRE 2013
# LABEL_MAP = {
#     'neutral':  0,
#     'happy':      1,
#     'angry':    2,
#     'sad':  3,
#     'surprise': 4,
#     'disgust':  5,
#     'fear':     6,
# }

# AffectNet
LABEL_MAP = {
    'neutral':  0,
    'happy':      1,
    'anger':    2,
    'sad':  3,
    'surprise': 4,
    'disgust':  5,
    'fear':     6,
    'contempt': 7,
}

def extract_landmarks(img):
    """Returns flat array of 468*3=1404 values (x,y,z per landmark), or None."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        # flatten to [x0,y0,z0, x1,y1,z1, ...]
        return np.array([[p.x, p.y, p.z] for p in lm]).flatten()

def build_dataset(data_root, output_csv):
    """
    Expects folder structure:
      data_root/
        neutral/  img1.jpg img2.jpg ...
        joy/      img1.jpg ...
        anger/    ...
        ...
    """
    records = []
    failed  = 0

    for label_name, label_idx in LABEL_MAP.items():
        folder = Path(data_root) / label_name
        if not folder.exists():
            print(f"Skipping {label_name} — folder not found")
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        print(f"Processing {label_name}: {len(images)} images")

        failed_single_cat = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            landmarks = extract_landmarks(img)
            if landmarks is None:
                failed += 1
                failed_single_cat += 1
                continue

            records.append([label_idx, label_name] + landmarks.tolist())
        
        print(f"Failed (no face detected) in category '{label_name}': {failed_single_cat} / {len(images)}")

    # column names
    coord_cols = []
    for i in range(468):
        coord_cols += [f"x{i}", f"y{i}", f"z{i}"]

    df = pd.DataFrame(records, columns=["label", "label_name"] + coord_cols)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} samples to {output_csv}")
    print(f"Failed (no face detected): {failed}")
    return df

df = build_dataset("affectnet/Train", "keypoints_dataset_affectnet.csv")
print(df["label_name"].value_counts())