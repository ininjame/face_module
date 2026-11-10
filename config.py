"""Shared configuration: landmarks, classes, preprocessing, thresholds."""

# ---------------- Model profile ----------------
# Which trained model this config must match. The two are NOT interchangeable:
#   "update1" -> nkf_best.pth + nkf_model_update1.onnx
#                105 landmarks (8 duplicated), RKFA ref = MP 33, fear before disgust
#   "v2"      -> nkf.onnx (+ its matching .pth, not yet located)
#                97 landmarks, RKFA ref = MP 2, disgust before fear
MODEL_PROFILE = "update1"
assert MODEL_PROFILE in ("update1", "v2"), f"Unknown MODEL_PROFILE: {MODEL_PROFILE}"

# ---------------- Classes (8, incl. contempt) ----------------
if MODEL_PROFILE == "update1":
    CLASSES = ["neutral", "happy", "angry", "sad", "surprise", "fear", "disgust", "contempt"]
else:
    CLASSES = ["neutral", "happy", "angry", "sad", "surprise", "disgust", "fear", "contempt"]
N_CLASSES = len(CLASSES)


# ---------------- Landmarks ----------------
BASE_LANDMARKS = list(range(68))  # "first 68 MediaPipe" (old model)

EXTRA_LANDMARKS = {
    "nose":      [102, 129, 331, 358, 49, 131, 279, 360, 4],
    "cheeks":    [50, 187, 207, 280, 411, 427, 234, 454],
    "jaw_chin":  [152, 175, 172, 136, 397, 365, 176, 149, 400, 378],
    "lips":      [39, 37, 269, 267, 17, 84, 76, 306],
    "forehead":  [10, 9],
}

# Deduplicate the extra list while preserving order.
#   update1: deduplicate within the extras only, so 4, 9, 10, 17, 37, 39, 49, 50
#            appear in both base and extra (105 total) — this is how it was trained.
#   v2:      also drop anything already in the base set (97 total).
_seen = set() if MODEL_PROFILE == "update1" else set(BASE_LANDMARKS)
EXTRA_LANDMARKS_DEDUPED = []
for _group in EXTRA_LANDMARKS.values():
    for _i in _group:
        if _i not in _seen:
            _seen.add(_i)
            EXTRA_LANDMARKS_DEDUPED.append(_i)

ALL_LANDMARK_INDICES = BASE_LANDMARKS + EXTRA_LANDMARKS_DEDUPED
N_LANDMARKS = len(ALL_LANDMARK_INDICES)

# RKFA reference landmark (MediaPipe index). The docs say "index 33 = nasal base",
# but that is the dlib-68 numbering. In MediaPipe, 33 is an eye corner; the
# subnasale (nasal base) is 2. update1 was trained with 33, v2 with 2.
RKFA_REF_MP_INDEX = 33 if MODEL_PROFILE == "update1" else 2
if RKFA_REF_MP_INDEX not in ALL_LANDMARK_INDICES:
    ALL_LANDMARK_INDICES.append(RKFA_REF_MP_INDEX)
    N_LANDMARKS = len(ALL_LANDMARK_INDICES)
RKFA_INDEX = ALL_LANDMARK_INDICES.index(RKFA_REF_MP_INDEX)  # position within the model input

# ---------------- Model / training ----------------
IMG_SIZE = 96
FEAT_DIM = 768
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
LR_GAMMA = 0.98
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
# Landmark perturbation in *pixels* of the 96x96 input (divided by IMG_SIZE in code).
# A value of 0.5 in normalised [0,1] units would move points by half the face.
PERTURB_PIXELS = 0.5
ROTATION_DEG = 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------- Tracker ----------------
CAM_INDEX = 4
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
FRAME_MIRRORED = True          # frame is flipped horizontally before processing
CLASSIFY_EVERY_N = 15
CROP_PADDING = 0.10
CROP_SQUARE  = True
MIN_CONFIDENCE = 0.4
EAR_BLINK_THRESHOLD = 0.25
MAR_OPEN_THRESHOLD = 0.35
# Normalisation ranges (tune per operator)
EAR_CLOSED, EAR_OPEN = 0.15, 0.35
MAR_CLOSED, MAR_OPEN = 0.0, 0.8
BROW_MIN, BROW_MAX = 0.30, 0.50      # brow–eye distance / inter-ocular distance
MOUTH_W_MIN, MOUTH_W_MAX = 0.30, 0.55  # mouth width / face width
