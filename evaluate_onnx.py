import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import pickle
import os

LABELS   = ['neutral', 'happy', 'anger', 'sad', 'surprise', 'disgust', 'fear', 'contempt']
TEST_DIR = "affectnet/Test"


def get_landmarks(img_bgr, mp_face, img_path=None, landmark_cache=None):
    """
    Get landmarks for an image.
    Uses cache if provided and image path is in cache,
    otherwise runs MediaPipe live.
    """
    if landmark_cache is not None and img_path in landmark_cache:
        return landmark_cache[img_path], True  # (68,2), cache_hit

    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    if result.multi_face_landmarks:
        lm     = result.multi_face_landmarks[0].landmark
        coords = np.array([[p.x, p.y] for p in lm[:68]], dtype=np.float32)
    else:
        coords = np.zeros((68, 2), dtype=np.float32)

    return coords, False  # (68,2), cache_hit=False


def evaluate_nkf(test_dir, model_path="nkf_model.onnx",
                 cache_path=None):
    """
    Evaluate NKF ONNX model on test set.

    Args:
        test_dir   : path to test folder with class subfolders
        model_path : path to .onnx model file
        cache_path : optional path to landmarks .pkl cache file.
                     If provided, uses cached landmarks where available
                     and falls back to live MediaPipe for misses.
    """

    # --- load landmark cache if provided ---
    landmark_cache = None
    if cache_path is not None:
        print(f"Loading landmark cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            landmark_cache = pickle.load(f)
        print(f"Cache contains {len(landmark_cache)} entries")

    # --- mediapipe (used for cache misses or when no cache) ---
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    # --- onnx session ---
    print(f"Loading NKF ONNX model from {model_path}...")
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )

    img_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- scan test folder and run inference ---
    y_pred, y_true = [], []
    cache_hits, cache_misses, failed = 0, 0, 0

    for label_idx, label_name in enumerate(LABELS):
        folder = os.path.join(test_dir, label_name)
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

            # get landmarks — from cache or live
            coords, hit = get_landmarks(
                img_bgr, mp_face, img_path, landmark_cache
            )
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1

            # preprocess image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_t   = img_transform(img_pil).unsqueeze(0).numpy()  # (1, 3, 96, 96)
            print(img_t.dtype)
            return
            lm_t    = coords[np.newaxis].astype(np.float32)         # (1, 68, 2)

            # run inference
            logits = session.run(
                ['logits'],
                {'image': img_t, 'landmarks': lm_t}
            )[0]

            y_pred.append(int(np.argmax(logits)))
            y_true.append(label_idx)

    # --- summary ---
    if landmark_cache is not None:
        print(f"\nCache hits: {cache_hits}  misses: {cache_misses}  "
              f"failed: {failed}")
        if cache_misses > 0:
            print(f"WARNING: {cache_misses} images ran live MediaPipe "
                  f"— landmarks may differ from training")
    else:
        print(f"\nTotal: {len(y_true)} samples  (failed/no face: {failed})")

    # --- results ---
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    acc = np.mean(y_pred == y_true)
    print(f"\nTest accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=LABELS, yticklabels=LABELS,
                cmap='Greens')
    plt.title(f'NKF ONNX Test Set  (acc={acc:.3f})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("nkf_test_confusion.png")
    print("Saved nkf_test_confusion.png")

    return acc, y_pred, y_true


if __name__ == "__main__":

    # Option A — use cache (matches training evaluation exactly)
    evaluate_nkf(
        test_dir   = TEST_DIR,
        model_path = "nkf_model.onnx",
        cache_path = "landmarks_test.pkl"
    )

    # Option B — live MediaPipe, no cache
    # evaluate_nkf(
    #     test_dir   = TEST_DIR,
    #     model_path = "nkf_model.onnx",
    #     cache_path = None
    # )