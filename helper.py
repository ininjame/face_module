import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from face_state import FaceState

LABELS = ['neutral', 'happy', 'anger', 'sad', 'surprise', 'disgust', 'fear', 'contempt']

def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    X  = df.drop(columns=["label", "label_name"]).values
    y  = df["label"].values

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(pd.Series(y).value_counts())

    print(f"Total samples: {len(df)}")
    print(f"Total classes: {df['label'].nunique()}")
    print()
    for i, label in enumerate(LABELS):
        count = len(df[df['label'] == i])
        print(f"  {i} {label:<12} {count:>6} samples")

    # --- normalize ---
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # save scaler for inference later
    with open(f"scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # --- split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_on_csv(model, scaler, csv_file, label, save_prefix):
    """Evaluate a trained model on any CSV dataset."""
    print(f"\n--- Evaluating on {label} ---")
    df = pd.read_csv(csv_file)

    meta_cols = ["label", "label_name"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = scaler.transform(df[feat_cols].values.astype(np.float32))
    y = df["label"].values

    y_pred = model.predict(X)
    acc    = np.mean(y_pred == y)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=LABELS))

    cm = confusion_matrix(y, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=LABELS, yticklabels=LABELS,
                cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'XGBoost {label}  (acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_xgb_{save_prefix}.png")
    print(f"Saved confusion_matrix_xgb_{save_prefix}.png")

    return acc, cm


def print_face_state(face_state: FaceState, min_confidence: float = None) -> dict:
    """
    Pretty print FaceState output matching the output specification.

    Args:
        face_state      : FaceState object to print
        min_confidence  : if set, suppress low confidence frames and
                          reprint last reliable result instead

    Returns:
        dict of the printed face state (after confidence filtering)
    """
    # --- confidence filtering ---
    if not hasattr(print_face_state, "last_reliable"):
        print_face_state.last_reliable = None

    current_conf = face_state.to_dict()["confidence"]
    suppressed   = False

    if min_confidence is not None:
        if current_conf >= min_confidence:
            print_face_state.last_reliable = face_state
        else:
            suppressed = True
            if print_face_state.last_reliable is not None:
                face_state = print_face_state.last_reliable

    d = face_state.to_dict()

    # --- render ---
    conf_bar = "█" * int(d["confidence"] * 20)

    print("\033[2J\033[H", end="")
    print("=" * 50)
    print("  FACE TRACKER OUTPUT")
    print("=" * 50)

    if suppressed:
        print(f"\n  ⚠  low confidence ({current_conf:.0%}) — showing last reliable frame")

    print(f"\n  EXPRESSION   : {d['expression'].upper():<12}  "
          f"conf: {conf_bar:<20} {d['confidence']:.0%}")

    left_bar  = "█" * int(d["eye_open_left"]  * 20)
    right_bar = "█" * int(d["eye_open_right"] * 20)
    print(f"\n  EYES")
    print(f"    left        : {left_bar:<20}  {d['eye_open_left']:.2f}"
          f"  {'BLINK' if d['blinking_left'] else '     '}")
    print(f"    right       : {right_bar:<20}  {d['eye_open_right']:.2f}"
          f"  {'BLINK' if d['blinking_right'] else '     '}")

    bl = "█" * int(d["brow_raise_left"]  * 20)
    br = "█" * int(d["brow_raise_right"] * 20)
    print(f"\n  EYEBROWS")
    print(f"    left raise  : {bl:<20}  {d['brow_raise_left']:.2f}")
    print(f"    right raise : {br:<20}  {d['brow_raise_right']:.2f}")

    mo = "█" * int(d["mouth_open_norm"]  * 20)
    mw = "█" * int(d["mouth_width_norm"] * 20)
    print(f"\n  MOUTH")
    print(f"    open        : {mo:<20}  {d['mouth_open_norm']:.2f}"
          f"  {'OPEN' if d['mouth_open'] else 'closed'}")
    print(f"    width       : {mw:<20}  {d['mouth_width_norm']:.2f}")

    print(f"\n  timestamp     : {d['timestamp']:.3f}")
    print("=" * 50)

    return


def get_face_crop(frame, face_landmarks, w, h, padding=0.2):
    """Crop face region from frame using landmark bounding box."""
    lm = face_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    x_min = max(0, int((min(xs) - padding) * w))
    x_max = min(w, int((max(xs) + padding) * w))
    y_min = max(0, int((min(ys) - padding) * h))
    y_max = min(h, int((max(ys) + padding) * h))

    crop = frame[y_min:y_max, x_min:x_max]
    return crop if crop.size > 0 else frame