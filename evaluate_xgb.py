import cv2
import numpy as np
import pickle
import mediapipe as mp
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

LABELS   = ['neutral', 'happy', 'anger', 'sad', 'surprise', 'disgust', 'fear', 'contempt']
TEST_DIR = "affectnet/Test"


def extract_au_features(lm_flat):
    pts   = lm_flat.reshape(-1, 3)
    nose  = pts[1].copy()
    pts  -= nose
    scale = np.max(np.abs(pts)) + 1e-6
    pts  /= scale
    p     = pts

    face_width = np.linalg.norm(p[33] - p[263]) + 1e-6

    feats = {}
    feats['ear_right']          = np.linalg.norm(p[159] - p[145]) / face_width
    feats['ear_left']           = np.linalg.norm(p[386] - p[374]) / face_width
    feats['brow_raise_right']   = (p[159][1] - p[52][1])          / face_width
    feats['brow_raise_left']    = (p[386][1] - p[282][1])         / face_width
    feats['brow_inner_raise']   = ((p[159][1] - p[65][1] +
                                    p[386][1] - p[295][1]) / 2)   / face_width
    feats['brow_furrow']        = np.linalg.norm(p[55]  - p[285]) / face_width
    feats['mar']                = np.linalg.norm(p[13]  - p[14])  / face_width
    feats['mouth_width']        = np.linalg.norm(p[61]  - p[291]) / face_width
    feats['lip_corner_right_x'] = p[61][0]   / face_width
    feats['lip_corner_left_x']  = p[291][0]  / face_width
    feats['lip_corner_right_y'] = p[61][1]   / face_width
    feats['lip_corner_left_y']  = p[291][1]  / face_width
    feats['upper_lip_raise']    = p[0][1]    / face_width
    feats['nose_width']         = np.linalg.norm(p[98]  - p[327]) / face_width
    feats['jaw_drop']           = p[152][1]  / face_width
    feats['eye_squint_right']   = np.linalg.norm(p[33]  - p[133]) / face_width
    feats['eye_squint_left']    = np.linalg.norm(p[362] - p[263]) / face_width
    feats['cheek_raise']        = (p[50][1] + p[280][1]) / 2      / face_width

    au_array = np.array(list(feats.values()))
    return np.concatenate([pts.flatten(), au_array])


if __name__ == "__main__":

    # --- mediapipe ---
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    # --- load model and scaler ---
    print("Loading XGBoost model and scaler...")
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Model loaded")

    # --- scan test folder, extract landmarks and features ---
    features, y_true = [], []
    failed = 0

    for label_idx, label_name in enumerate(LABELS):
        folder = os.path.join(TEST_DIR, label_name)
        if not os.path.isdir(folder):
            print(f"Skipping {label_name} — folder not found")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for fname in tqdm(files, desc=label_name):
            img_path = os.path.join(folder, fname)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                failed += 1
                continue

            # extract landmarks
            rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = mp_face.process(rgb)

            if result.multi_face_landmarks:
                lm     = result.multi_face_landmarks[0].landmark
                # get x,y,z for first 468 landmarks
                lm_flat = np.array([[p.x, p.y, p.z] for p in lm],
                                    dtype=np.float32).flatten()
            else:
                lm_flat = np.zeros(468 * 3, dtype=np.float32)
                failed += 1

            feat = extract_au_features(lm_flat)
            features.append(feat)
            y_true.append(label_idx)

    print(f"\nTotal: {len(y_true)} samples  (failed/no face: {failed})")

    # --- predict ---
    print("Running XGBoost predictions...")
    X      = scaler.transform(np.array(features, dtype=np.float32))
    y_pred = model.predict(X).astype(int)
    y_true = np.array(y_true)

    # --- results ---
    acc = np.mean(y_pred == y_true)
    print(f"\nTest accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=LABELS, yticklabels=LABELS,
                cmap='Blues')
    plt.title(f'XGBoost Test Set  (acc={acc:.3f})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("xgb_test_confusion.png")
    print("Saved xgb_test_confusion.png")