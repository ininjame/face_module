import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
# from deepface import DeepFace
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# Key landmark indices
# inner corner, center, outer corner of each eyebrow
RIGHT_BROW = [46, 52, 55]   # inner, center, outer
LEFT_BROW  = [276, 282, 285] # inner, center, outer

# reference point to measure brow raise — use the eye center
RIGHT_EYE_CENTER = 159  # top of right eye
LEFT_EYE_CENTER  = 386  # top of left eye

RIGHT_EYE = [159, 145, 33, 133]
LEFT_EYE  = [386, 374, 362, 263]
MOUTH     = [13, 14, 61, 291]
EAR_THRESH = 0.25
MAR_THRESH = 0.35

# Drawing specs
TESSELATION_SPEC = mp_drawing.DrawingSpec(
    color=(80, 80, 80), thickness=1, circle_radius=1)
CONTOUR_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=1)
IRIS_SPEC = mp_drawing.DrawingSpec(
    color=(0, 200, 255), thickness=2, circle_radius=1)

def get_pt(landmarks, idx, w, h):
    pt = landmarks.landmark[idx]
    return np.array([pt.x * w, pt.y * h])

def brow_raise(landmarks, ids, w, h):
    brow_pt = get_pt(landmarks, ids[1], w, h)  # center point of eyebrow
    eye_pt  = get_pt(landmarks, RIGHT_EYE_CENTER, w, h)  # eye center
    return np.linalg.norm(brow_pt - eye_pt)

def aspect_ratio(landmarks, ids, w, h):
    t = get_pt(landmarks, ids[0], w, h)
    b = get_pt(landmarks, ids[1], w, h)
    l = get_pt(landmarks, ids[2], w, h)
    r = get_pt(landmarks, ids[3], w, h)
    return np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6)

def draw_hud(frame, left_blinking, right_blinking, mouth_open, ear, mar, brow_raise_l=0.0, brow_raise_r=0.0):
    h, w = frame.shape[:2]

    # semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # blink
    blink_color = (0, 0, 255) if (left_blinking or right_blinking) else (100, 100, 100)
    cv2.putText(frame, f"LEFT BLINK: {'YES' if left_blinking else 'no'}  RIGHT BLINK: {'YES' if right_blinking else 'no'}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)

    # mouth
    mouth_color = (0, 255, 0) if mouth_open else (100, 100, 100)
    cv2.putText(frame, f"MOUTH: {'OPEN' if mouth_open else 'closed'}  MAR: {mar:.2f}",
        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)

    # brow raise
    cv2.putText(frame, f"LEFT BROW RAISE: {brow_raise_l:.2f}  RIGHT BROW RAISE: {brow_raise_r:.2f}",
        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # landmark count
    cv2.putText(frame, "468 landmarks  |  press Q to quit",
        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

def draw_landmark_dots(frame, landmarks, w, h, ids, color, radius=4):
    """Highlight specific keypoints with a filled circle."""
    for idx in ids:
        pt = get_pt(landmarks, idx, w, h)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)




with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        frame = cv2.imread("train_data_ref_2013/angry/Training_99770811.jpg")
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        brow_raise_l = 0.0
        brow_raise_r = 0.0
        left_blinking   = False
        right_blinking  = False
        mouth_open = False
        ear        = 0.0
        mar        = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # full mesh tesselation
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=TESSELATION_SPEC
                )

                # face contours + lips + eyebrows
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=CONTOUR_SPEC
                )

                # irises
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=IRIS_SPEC
                )

                # highlight eye and mouth keypoints
                draw_landmark_dots(frame, face_landmarks, w, h,
                    RIGHT_EYE, color=(0, 0, 255))    # red
                draw_landmark_dots(frame, face_landmarks, w, h,
                    LEFT_EYE,  color=(0, 0, 255))    # red
                draw_landmark_dots(frame, face_landmarks, w, h,
                    MOUTH,     color=(0, 255, 255))  # yellow
                draw_landmark_dots(frame, face_landmarks, w, h,
                    RIGHT_BROW, color=(255, 0, 0))    # blue
                draw_landmark_dots(frame, face_landmarks, w, h,
                    LEFT_BROW,  color=(255, 0, 0))    # blue

                # compute metrics
                ear_r = aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
                ear_l = aspect_ratio(face_landmarks, LEFT_EYE,  w, h)
                ear   = (ear_r + ear_l) / 2
                mar   = aspect_ratio(face_landmarks, MOUTH, w, h)
                brow_raise_r = brow_raise(face_landmarks, RIGHT_BROW, w, h)
                brow_raise_l = brow_raise(face_landmarks, LEFT_BROW, w, h)

                left_blinking = ear_l < EAR_THRESH
                right_blinking  = ear_r < EAR_THRESH

                mouth_open = mar > MAR_THRESH

        draw_hud(frame, left_blinking, right_blinking, mouth_open, ear, mar, brow_raise_l, brow_raise_r)

        cv2.imshow("Face Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

# res = DeepFace.analyze('train_data_ref_2013/angry/Training_99770811.jpg', actions=['emotion'], enforce_detection=False, silent=True)
# print(f'Expression: {res[0]["dominant_emotion"]}')
# print(f'Scores: {res[0]["emotion"]}')