import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from face_state import FaceState
from helper import get_face_crop

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# --- landmark indices ---
LEFT_BROW  = [46, 52, 55]
RIGHT_BROW = [276, 282, 285]

LEFT_EYE_CENTER  = 159
RIGHT_EYE_CENTER = 386

LEFT_EYE  = [159, 145, 33, 133]
RIGHT_EYE = [386, 374, 362, 263]
MOUTH     = [13, 14, 61, 291]

# --- thresholds ---
EAR_THRESH          = 0.25
MAR_THRESH          = 0.35
EXPR_EVERY_N_FRAMES = 15
EXPR_MIN_CONFIDENCE = 0.4

# --- labels ---
LABELS = ['neutral', 'happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'contempt']

# --- drawing specs ---
TESSELATION_SPEC = mp_drawing.DrawingSpec(
    color=(80, 80, 80), thickness=1, circle_radius=1)
CONTOUR_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=1)
IRIS_SPEC = mp_drawing.DrawingSpec(
    color=(0, 200, 255), thickness=2, circle_radius=1)

# --- normalization ranges ---
EAR_MIN         = 0.05
EAR_MAX         = 0.35
MAR_MIN         = 0.0
MAR_MAX         = 0.6
BROW_MIN        = 20.0
BROW_MAX        = 50.0
MOUTH_WIDTH_MIN = 0.25
MOUTH_WIDTH_MAX = 0.50

# --- load model and scaler ---
print("Loading model and scaler...")
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler_xgb.pkl", "rb") as f:
    scaler = pickle.load(f)
print(f"Scaler expects: {scaler.n_features_in_} features")
print("Model loaded")

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def normalize(value, min_val, max_val):
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0))

def get_pt(landmarks, idx, w, h):
    pt = landmarks.landmark[idx]
    return np.array([pt.x * w, pt.y * h])

def aspect_ratio(landmarks, ids, w, h):
    t = get_pt(landmarks, ids[0], w, h)
    b = get_pt(landmarks, ids[1], w, h)
    l = get_pt(landmarks, ids[2], w, h)
    r = get_pt(landmarks, ids[3], w, h)
    return np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6)

def brow_raise(landmarks, eye_center, ids, w, h):
    brow_pt = get_pt(landmarks, ids[1], w, h)
    eye_pt  = get_pt(landmarks, eye_center, w, h)
    return np.linalg.norm(brow_pt - eye_pt)

def extract_au_features(lm_flat):
    pts   = lm_flat.reshape(-1, 3)
    nose  = pts[1].copy()
    pts  -= nose
    scale = np.max(np.abs(pts)) + 1e-6
    pts  /= scale
    p     = pts

    # reference distance — outer eye corners, stable across expressions
    # use this to normalize all distance features
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


# def predict_expression(face_landmarks):
#     lm      = face_landmarks.landmark
#     # lm_flat = np.array([[p.x, p.y, p.z] for p in lm]).flatten().astype(np.float32)
#     # slice to first 468 landmarks — refine_landmarks=True adds extra
#     # iris landmarks (469-478) which were not used during training
#     lm_flat = np.array([[p.x, p.y, p.z] for p in lm[:468]]).flatten().astype(np.float32)
#     features = extract_au_features(lm_flat)
#     print(f"extract_au_features produces: {len(features)} features")
#     features = scaler.transform([features]).astype(np.float32)
#     dmatrix  = xgb.DMatrix(features)
#     pred_idx = int(model.predict(dmatrix)[0])
#     proba    = model.predict_proba(features)[0]
#     return LABELS[pred_idx], float(np.max(proba))

def predict_expression(face_landmarks):
    lm      = face_landmarks.landmark
    lm_flat = np.array([[p.x, p.y, p.z] for p in lm[:468]]).flatten().astype(np.float32)
    features = extract_au_features(lm_flat)
    features = scaler.transform([features]).astype(np.float32)
    pred_idx = int(model.predict(features)[0])
    proba    = model.predict_proba(features)[0]
    return LABELS[pred_idx], float(np.max(proba))


def build_face_state(face_landmarks, w, h, expression, confidence) -> FaceState:
    ear_l  = aspect_ratio(face_landmarks, LEFT_EYE,  w, h)
    ear_r  = aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
    mar    = aspect_ratio(face_landmarks, MOUTH,     w, h)
    brow_r = brow_raise(face_landmarks, RIGHT_EYE_CENTER, RIGHT_BROW, w, h)
    brow_l = brow_raise(face_landmarks, LEFT_EYE_CENTER,  LEFT_BROW,  w, h)
    lm     = face_landmarks.landmark
    mouth_w = abs(lm[61].x - lm[291].x)

    return FaceState(
        timestamp          = time.time(),
        expression         = expression,
        confidence         = confidence,
        ear_left           = ear_l,
        ear_right          = ear_r,
        eye_open_left      = normalize(ear_l,   EAR_MIN,         EAR_MAX),
        eye_open_right     = normalize(ear_r,   EAR_MIN,         EAR_MAX),
        blinking_left      = ear_l < EAR_THRESH,
        blinking_right     = ear_r < EAR_THRESH,
        brow_raise_left    = brow_l,
        brow_raise_right   = brow_r,
        brow_raise_left_norm  = normalize(brow_l,  BROW_MIN,  BROW_MAX),
        brow_raise_right_norm = normalize(brow_r,  BROW_MIN,  BROW_MAX),
        mar                = mar,
        mouth_open         = mar > MAR_THRESH,
        mouth_open_norm    = normalize(mar,     MAR_MIN,         MAR_MAX),
        mouth_width_norm   = normalize(mouth_w, MOUTH_WIDTH_MIN, MOUTH_WIDTH_MAX),
        face_detected      = True,
    )

def draw_landmark_dots(frame, landmarks, w, h, ids, color, radius=4):
    for idx in ids:
        pt = get_pt(landmarks, idx, w, h)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)

def draw_hud(frame, left_blinking, right_blinking, mouth_open,
             ear, mar, brow_raise_l, brow_raise_r,
             expression, confidence):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    blink_color = (0, 0, 255) if (left_blinking or right_blinking) else (100, 100, 100)
    cv2.putText(frame,
        f"LEFT BLINK: {'YES' if left_blinking else 'no'}  "
        f"RIGHT BLINK: {'YES' if right_blinking else 'no'}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)

    mouth_color = (0, 255, 0) if mouth_open else (100, 100, 100)
    cv2.putText(frame,
        f"MOUTH: {'OPEN' if mouth_open else 'closed'}  MAR: {mar:.2f}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)

    cv2.putText(frame,
        f"L BROW: {brow_raise_l:.1f}  R BROW: {brow_raise_r:.1f}",
        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame,
        f"EXPR: {expression.upper()}  ({confidence:.0%})",
        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.putText(frame, "press Q to quit",
        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

def print_face_state(face_state: FaceState, min_confidence: float = None) -> dict:
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

    conf_bar  = "█" * int(d["confidence"] * 20)
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

    return d

# ---------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('face_tracker.mp4', fourcc, 30.0, (640, 480))

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------

frame_count    = 0
expression     = "neutral"
confidence     = 0.0
face_state     = FaceState()  # default empty state — always valid

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        brow_raise_l  = 0.0
        brow_raise_r  = 0.0
        left_blinking = False
        right_blinking= False
        mouth_open    = False
        ear           = 0.0
        mar           = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # draw mesh layers
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=TESSELATION_SPEC)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=CONTOUR_SPEC)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=IRIS_SPEC)

                # highlight keypoints
                draw_landmark_dots(frame, face_landmarks, w, h,
                    RIGHT_EYE,  color=(0, 0, 255))
                draw_landmark_dots(frame, face_landmarks, w, h,
                    LEFT_EYE,   color=(0, 0, 255))
                draw_landmark_dots(frame, face_landmarks, w, h,
                    MOUTH,      color=(0, 255, 255))
                draw_landmark_dots(frame, face_landmarks, w, h,
                    RIGHT_BROW, color=(255, 0, 0))
                draw_landmark_dots(frame, face_landmarks, w, h,
                    LEFT_BROW,  color=(255, 0, 0))

                # compute metrics
                ear_r = aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
                ear_l = aspect_ratio(face_landmarks, LEFT_EYE,  w, h)
                ear   = (ear_r + ear_l) / 2
                mar   = aspect_ratio(face_landmarks, MOUTH,     w, h)
                brow_raise_r = brow_raise(face_landmarks, RIGHT_EYE_CENTER, RIGHT_BROW, w, h)
                brow_raise_l = brow_raise(face_landmarks, LEFT_EYE_CENTER,  LEFT_BROW,  w, h)

                left_blinking  = ear_l < EAR_THRESH
                right_blinking = ear_r < EAR_THRESH
                mouth_open     = mar   > MAR_THRESH

                # run expression classifier every N frames
                # raw result always used in HUD regardless of confidence
                if frame_count % EXPR_EVERY_N_FRAMES == 0:
                    try:
                        expression, confidence = predict_expression(face_landmarks)
                    except Exception as e:
                        print(f"Expression error: {e}")

                # build face state every frame
                face_state = build_face_state(
                    face_landmarks, w, h,
                    expression, confidence
                )

        # output to next module every N frames
        # confidence filtering happens inside print_face_state
        if frame_count % EXPR_EVERY_N_FRAMES == 0:
            state_dict = print_face_state(face_state, min_confidence=EXPR_MIN_CONFIDENCE)

        # HUD always shows raw expression + confidence unfiltered
        draw_hud(frame, left_blinking, right_blinking, mouth_open,
                 ear, mar, brow_raise_l, brow_raise_r,
                 expression, confidence)

        out.write(frame)
        cv2.imshow("Face Tracker", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()