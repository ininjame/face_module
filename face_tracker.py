"""Live face tracker: geometric metrics every frame, NKF expression every N frames."""
import argparse
import json
import socket
import time
from dataclasses import dataclass, asdict

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

import config as C
from landmarks import extract_landmarks_from_result

mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# MediaPipe indices, subject's anatomical side in an UNMIRRORED image
EYE_R = [33, 160, 158, 133, 153, 144]
EYE_L = [362, 385, 387, 263, 373, 380]
BROW_R, BROW_L = [70, 63, 105, 66, 107], [300, 293, 334, 296, 336]
MOUTH_TOP, MOUTH_BOT, MOUTH_LC, MOUTH_RC = 13, 14, 78, 308
FACE_L, FACE_R = 234, 454


@dataclass
class FaceState:
    face_detected: bool = False
    expression: str = "neutral"
    confidence: float = 0.0
    eye_open_left: float = 0.0
    eye_open_right: float = 0.0
    blinking_left: bool = False
    blinking_right: bool = False
    brow_raise_left: float = 0.0
    brow_raise_right: float = 0.0
    mouth_open: bool = False
    mouth_open_norm: float = 0.0
    mouth_width_norm: float = 0.0

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def norm01(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def ear(p, idx):
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6)


def geometric_metrics(pts):
    iod = np.linalg.norm(pts[EYE_L].mean(0) - pts[EYE_R].mean(0)) + 1e-6
    ear_l, ear_r = ear(pts, EYE_L), ear(pts, EYE_R)
    mar = np.linalg.norm(pts[MOUTH_TOP] - pts[MOUTH_BOT]) / (np.linalg.norm(pts[MOUTH_LC] - pts[MOUTH_RC]) + 1e-6)
    brow_l = (pts[EYE_L].mean(0)[1] - pts[BROW_L].mean(0)[1]) / iod
    brow_r = (pts[EYE_R].mean(0)[1] - pts[BROW_R].mean(0)[1]) / iod
    width = np.linalg.norm(pts[MOUTH_LC] - pts[MOUTH_RC]) / (np.linalg.norm(pts[FACE_L] - pts[FACE_R]) + 1e-6)
    m = dict(ear_l=ear_l, ear_r=ear_r, mar=mar, brow_l=brow_l, brow_r=brow_r, width=width)
    if C.FRAME_MIRRORED:
        m["ear_l"], m["ear_r"] = m["ear_r"], m["ear_l"]
        m["brow_l"], m["brow_r"] = m["brow_r"], m["brow_l"]
    return m


def get_face_crop(frame, pts):
    h, w = frame.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py = (x1 - x0) * C.CROP_PADDING, (y1 - y0) * C.CROP_PADDING
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame[y0:y1, x0:x1]


class ExpressionClassifier:
    def __init__(self, onnx_path):
        available = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        print(f"[ExpressionClassifier] Available ONNX providers: {available}")
        print(f"[ExpressionClassifier] Using providers: {prov}")
        self.sess = ort.InferenceSession(onnx_path, providers=prov)

        # print model input shapes so we can verify landmark count
        for inp in self.sess.get_inputs():
            print(f"[ExpressionClassifier] Model input: name={inp.name} shape={inp.shape} dtype={inp.type}")

        self.mesh = mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)
        self.mean = np.array(C.IMAGENET_MEAN, np.float32)
        self.std  = np.array(C.IMAGENET_STD,  np.float32)
        self._call_count   = 0
        self._fail_lm      = 0
        self._fail_none    = 0

    def __call__(self, crop_bgr):
        self._call_count += 1
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        lm  = extract_landmarks_from_result(self.mesh.process(rgb))

        if lm is None:
            self._fail_lm += 1
            if self._call_count <= 10 or self._call_count % 30 == 0:
                print(f"[clf call {self._call_count}] landmark extraction FAILED on crop "
                      f"(crop shape={crop_bgr.shape}) — total fails: {self._fail_lm}")
            return None

        img = cv2.resize(rgb, (C.IMG_SIZE, C.IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = ((img - self.mean) / self.std).transpose(2, 0, 1)[None]

        try:
            logits = self.sess.run(None, {"image": img, "landmarks": lm[None]})[0][0]
        except Exception as e:
            print(f"[clf call {self._call_count}] ONNX inference error: {e}")
            return None

        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        k     = int(prob.argmax())
        label = C.CLASSES[k]
        conf  = float(prob[k])

        if self._call_count <= 10 or self._call_count % 30 == 0:
            top3 = sorted(zip(C.CLASSES, prob.tolist()), key=lambda x: -x[1])[:3]
            print(f"[clf call {self._call_count}] raw={label} conf={conf:.3f}  "
                  f"top3={top3}  lm_shape={lm.shape}")

        return label, conf


def draw_hud(frame, m, raw_expr):
    lines = [
        f"Blink L:{m['ear_l'] < C.EAR_BLINK_THRESHOLD}  R:{m['ear_r'] < C.EAR_BLINK_THRESHOLD}",
        f"Mouth open:{m['mar'] > C.MAR_OPEN_THRESHOLD}  MAR:{m['mar']:.2f}",
        f"Brow L:{m['brow_l']:.2f}  R:{m['brow_r']:.2f}",
        f"Expr (raw): {raw_expr[0]} {raw_expr[1]:.2f}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)


def print_face_state(state, sock=None, addr=None):
    d   = state.to_dict()
    msg = json.dumps(d)
    if sock:
        sock.sendto(msg.encode(), addr)
    else:
        print(msg)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default="nkf.onnx")
    ap.add_argument("--udp",          default=None, help="host:port to stream FaceState JSON")
    ap.add_argument("--output-every", type=int, default=1,
                    help="1 = every frame, 15 = on classification")
    ap.add_argument("--no-gui",       action="store_true")
    a = ap.parse_args()

    # --- startup config summary ---
    print("=" * 50)
    print(f"[config] CLASSIFY_EVERY_N  = {C.CLASSIFY_EVERY_N}")
    print(f"[config] MIN_CONFIDENCE    = {C.MIN_CONFIDENCE}")
    print(f"[config] EAR_BLINK_THRESH  = {C.EAR_BLINK_THRESHOLD}")
    print(f"[config] MAR_OPEN_THRESH   = {C.MAR_OPEN_THRESHOLD}")
    print(f"[config] FRAME_MIRRORED    = {C.FRAME_MIRRORED}")
    print(f"[config] IMG_SIZE          = {C.IMG_SIZE}")
    print(f"[config] CLASSES           = {C.CLASSES}")
    print("=" * 50)

    sock, addr = None, None
    if a.udp:
        host, port = a.udp.split(":")
        sock, addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))

    cap = cv2.VideoCapture(C.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          C.CAM_FPS)
    print(f"[camera] opened={cap.isOpened()} "
          f"w={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
          f"h={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"fps={cap.get(cv2.CAP_PROP_FPS):.0f}")

    clf      = ExpressionClassifier(a.model)
    mesh     = mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    reliable = ("neutral", 0.0)
    raw      = ("neutral", 0.0)
    frame_i, t0 = 0, time.time()

    # per-run counters for periodic summary
    n_face       = 0
    n_no_face    = 0
    n_classified = 0
    n_accepted   = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if C.FRAME_MIRRORED:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res   = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        state = FaceState()

        if res.multi_face_landmarks:
            n_face += 1
            face = res.multi_face_landmarks[0]
            pts  = np.array([[p.x * w, p.y * h] for p in face.landmark], np.float32)

            if not a.no_gui:
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_TESSELATION, None,
                                       mp_draw.DrawingSpec((128, 128, 128), 1, 1))
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_CONTOURS, None,
                                       mp_draw.DrawingSpec((0, 255, 0), 1, 1))
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_IRISES, None,
                                       mp_draw.DrawingSpec((255, 255, 0), 1, 1))

            m = geometric_metrics(pts)

            if frame_i % C.CLASSIFY_EVERY_N == 0:
                crop = get_face_crop(frame, pts)
                out  = clf(crop) if crop is not None else None

                if crop is None:
                    print(f"[frame {frame_i}] get_face_crop returned None — pts range "
                          f"x=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
                          f"y=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}]")

                if out is not None:
                    n_classified += 1
                    raw = out
                    if out[1] >= C.MIN_CONFIDENCE:
                        n_accepted += 1
                        reliable = out
                    else:
                        print(f"[frame {frame_i}] confidence {out[1]:.3f} below "
                              f"MIN_CONFIDENCE {C.MIN_CONFIDENCE} — holding '{reliable[0]}'")
        else:
            n_no_face += 1

        state = FaceState(
            face_detected  = res.multi_face_landmarks is not None,
            expression     = reliable[0],
            confidence     = reliable[1],
            eye_open_left  = norm01(m["ear_l"], C.EAR_CLOSED, C.EAR_OPEN)  if res.multi_face_landmarks else 0.0,
            eye_open_right = norm01(m["ear_r"], C.EAR_CLOSED, C.EAR_OPEN)  if res.multi_face_landmarks else 0.0,
            blinking_left  = bool(m["ear_l"] < C.EAR_BLINK_THRESHOLD)      if res.multi_face_landmarks else False,
            blinking_right = bool(m["ear_r"] < C.EAR_BLINK_THRESHOLD)      if res.multi_face_landmarks else False,
            brow_raise_left  = norm01(m["brow_l"], C.BROW_MIN, C.BROW_MAX) if res.multi_face_landmarks else 0.0,
            brow_raise_right = norm01(m["brow_r"], C.BROW_MIN, C.BROW_MAX) if res.multi_face_landmarks else 0.0,
            mouth_open     = bool(m["mar"] > C.MAR_OPEN_THRESHOLD)          if res.multi_face_landmarks else False,
            mouth_open_norm  = norm01(m["mar"],   C.MAR_CLOSED,   C.MAR_OPEN)   if res.multi_face_landmarks else 0.0,
            mouth_width_norm = norm01(m["width"], C.MOUTH_W_MIN, C.MOUTH_W_MAX) if res.multi_face_landmarks else 0.0,
        ) if res.multi_face_landmarks else FaceState()

        if not a.no_gui and res.multi_face_landmarks:
            draw_hud(frame, m, raw)

        if frame_i % a.output_every == 0:
            print_face_state(state, sock, addr)

        if not a.no_gui:
            fps = (frame_i + 1) / (time.time() - t0)
            cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Face tracker", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        # periodic summary every 150 frames
        if frame_i > 0 and frame_i % 150 == 0:
            elapsed = time.time() - t0
            print(f"[summary frame {frame_i}] "
                  f"fps={frame_i/elapsed:.1f}  "
                  f"face={n_face}  no_face={n_no_face}  "
                  f"classified={n_classified}  accepted={n_accepted}  "
                  f"reliable='{reliable[0]}' conf={reliable[1]:.3f}")

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[done] frames={frame_i} classified={n_classified} accepted={n_accepted}")


if __name__ == "__main__":
    main()