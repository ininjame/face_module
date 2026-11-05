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
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def norm01(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def ear(p, idx):
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (
        2 * np.linalg.norm(p1 - p4) + 1e-6)


def geometric_metrics(pts):
    """pts: (478, 2) pixel coordinates."""
    iod   = np.linalg.norm(pts[EYE_L].mean(0) - pts[EYE_R].mean(0)) + 1e-6
    ear_l = ear(pts, EYE_L)
    ear_r = ear(pts, EYE_R)
    mar   = np.linalg.norm(pts[MOUTH_TOP] - pts[MOUTH_BOT]) / (
        np.linalg.norm(pts[MOUTH_LC] - pts[MOUTH_RC]) + 1e-6)
    brow_l = (pts[EYE_L].mean(0)[1] - pts[BROW_L].mean(0)[1]) / iod
    brow_r = (pts[EYE_R].mean(0)[1] - pts[BROW_R].mean(0)[1]) / iod
    width  = np.linalg.norm(pts[MOUTH_LC] - pts[MOUTH_RC]) / (
        np.linalg.norm(pts[FACE_L] - pts[FACE_R]) + 1e-6)
    m = dict(ear_l=ear_l, ear_r=ear_r, mar=mar,
             brow_l=brow_l, brow_r=brow_r, width=width)
    if C.FRAME_MIRRORED:
        m["ear_l"],  m["ear_r"]  = m["ear_r"],  m["ear_l"]
        m["brow_l"], m["brow_r"] = m["brow_r"], m["brow_l"]
    return m


def get_face_crop(frame, pts):
    """Returns (crop_bgr, (x0, y0, x1, y1)) or (None, None)."""
    h, w  = frame.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    px, py  = (x1 - x0) * C.CROP_PADDING, (y1 - y0) * C.CROP_PADDING
    x0, y0  = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1  = int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, None
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def remap_landmarks_to_crop(pts_norm, crop_box, frame_wh):
    """
    Remap full-frame normalized landmarks (N, 2) into crop-normalized space.
    pts_norm  : (N, 2) in [0,1] relative to full frame
    crop_box  : (x0, y0, x1, y1) pixel coords in full frame
    frame_wh  : (w, h) of full frame
    Returns   : (N, 2) in [0,1] relative to crop
    """
    fw, fh       = frame_wh
    x0, y0, x1, y1 = crop_box
    crop_w_n     = (x1 - x0) / fw
    crop_h_n     = (y1 - y0) / fh
    x0_n         = x0 / fw
    y0_n         = y0 / fh
    lm           = pts_norm.copy()
    lm[:, 0]     = (lm[:, 0] - x0_n) / (crop_w_n + 1e-6)
    lm[:, 1]     = (lm[:, 1] - y0_n) / (crop_h_n + 1e-6)
    return np.clip(lm, 0.0, 1.0)


class ExpressionClassifier:
    def __init__(self, onnx_path):
        avail = ort.get_available_providers()
        prov  = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in avail]
        print(f"[ExpressionClassifier] Available ONNX providers: {avail}")
        print(f"[ExpressionClassifier] Using providers: {prov}")
        self.sess = ort.InferenceSession(onnx_path, providers=prov)

        for inp in self.sess.get_inputs():
            print(f"[ExpressionClassifier] Model input: "
                  f"name={inp.name} shape={inp.shape} dtype={inp.type}")

        # lower confidence threshold to improve detection on tight crops
        self.mesh = mp_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        )
        self.mean = np.array(C.IMAGENET_MEAN, np.float32)
        self.std  = np.array(C.IMAGENET_STD,  np.float32)
        self._call_count = 0
        self._fail_lm    = 0
        self._fallback   = 0

    def __call__(self, crop_bgr,
                 full_frame_pts_norm=None,
                 crop_box=None,
                 frame_wh=None):
        """
        crop_bgr            : cropped face image (BGR)
        full_frame_pts_norm : (N, 2) landmarks normalized to full frame — used
                              as fallback if MediaPipe fails on the crop
        crop_box            : (x0, y0, x1, y1) pixel coords of crop in full frame
        frame_wh            : (w, h) of full frame
        """
        self._call_count += 1
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        lm  = extract_landmarks_from_result(self.mesh.process(rgb))

        if lm is None:
            self._fail_lm += 1
            # --- fallback: remap full-frame landmarks to crop space ---
            if (full_frame_pts_norm is not None
                    and crop_box is not None
                    and frame_wh is not None):
                lm = remap_landmarks_to_crop(
                    full_frame_pts_norm, crop_box, frame_wh)
                self._fallback += 1
                if self._call_count <= 10 or self._call_count % 30 == 0:
                    print(f"[clf call {self._call_count}] MediaPipe failed on crop "
                          f"— using remapped full-frame landmarks "
                          f"(fallbacks so far: {self._fallback})")
            else:
                if self._call_count <= 10 or self._call_count % 30 == 0:
                    print(f"[clf call {self._call_count}] landmark extraction FAILED, "
                          f"no fallback available "
                          f"(crop shape={crop_bgr.shape}, fails={self._fail_lm})")
                return None

        # preprocess image
        img = cv2.resize(rgb, (C.IMG_SIZE, C.IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = ((img - self.mean) / self.std).transpose(2, 0, 1)[None]

        try:
            logits = self.sess.run(
                None, {"image": img, "landmarks": lm[None]})[0][0]
        except Exception as e:
            print(f"[clf call {self._call_count}] ONNX inference error: {e}")
            return None

        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        k     = int(prob.argmax())
        label = C.CLASSES[k]
        conf  = float(prob[k])

        if self._call_count <= 10 or self._call_count % 30 == 0:
            top3 = sorted(zip(C.CLASSES, prob.tolist()),
                          key=lambda x: -x[1])[:3]
            print(f"[clf call {self._call_count}] "
                  f"raw={label} conf={conf:.3f}  top3={top3}  "
                  f"lm_shape={lm.shape}  "
                  f"fallback={'yes' if self._fallback == self._call_count - (self._call_count - self._fallback) else 'no'}")

        return label, conf


def draw_hud(frame, m, raw_expr):
    lines = [
        f"Blink L:{m['ear_l'] < C.EAR_BLINK_THRESHOLD}  "
        f"R:{m['ear_r'] < C.EAR_BLINK_THRESHOLD}",
        f"Mouth open:{m['mar'] > C.MAR_OPEN_THRESHOLD}  MAR:{m['mar']:.2f}",
        f"Brow L:{m['brow_l']:.2f}  R:{m['brow_r']:.2f}",
        f"Expr (raw): {raw_expr[0]} {raw_expr[1]:.2f}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 25 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


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
    ap.add_argument("--udp",          default=None,
                    help="host:port to stream FaceState JSON")
    ap.add_argument("--output-every", type=int, default=1,
                    help="1 = every frame, 15 = on classification")
    ap.add_argument("--no-gui",       action="store_true")
    a = ap.parse_args()

    # --- startup summary ---
    print("=" * 50)
    print(f"[config] CLASSIFY_EVERY_N  = {C.CLASSIFY_EVERY_N}")
    print(f"[config] MIN_CONFIDENCE    = {C.MIN_CONFIDENCE}")
    print(f"[config] EAR_BLINK_THRESH  = {C.EAR_BLINK_THRESHOLD}")
    print(f"[config] MAR_OPEN_THRESH   = {C.MAR_OPEN_THRESHOLD}")
    print(f"[config] FRAME_MIRRORED    = {C.FRAME_MIRRORED}")
    print(f"[config] IMG_SIZE          = {C.IMG_SIZE}")
    print(f"[config] N_LANDMARKS       = {C.N_LANDMARKS}")
    print(f"[config] CLASSES           = {C.CLASSES}")
    print("=" * 50)

    sock, addr = None, None
    if a.udp:
        host, port = a.udp.split(":")
        sock, addr = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                      (host, int(port)))

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
    mesh     = mp_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    reliable = ("neutral", 0.0)
    raw      = ("neutral", 0.0)
    frame_i, t0 = 0, time.time()

    n_face, n_no_face   = 0, 0
    n_classified        = 0
    n_accepted          = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if C.FRAME_MIRRORED:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res   = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        state = FaceState()
        m     = {}

        if res.multi_face_landmarks:
            n_face += 1
            face = res.multi_face_landmarks[0]
            pts  = np.array([[p.x * w, p.y * h]
                              for p in face.landmark], np.float32)

            if not a.no_gui:
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_TESSELATION,
                                       None, mp_draw.DrawingSpec((128, 128, 128), 1, 1))
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_CONTOURS,
                                       None, mp_draw.DrawingSpec((0, 255, 0), 1, 1))
                mp_draw.draw_landmarks(frame, face, mp_mesh.FACEMESH_IRISES,
                                       None, mp_draw.DrawingSpec((255, 255, 0), 1, 1))

            m = geometric_metrics(pts)

            if frame_i % C.CLASSIFY_EVERY_N == 0:
                crop, box = get_face_crop(frame, pts)

                if crop is None:
                    print(f"[frame {frame_i}] get_face_crop returned None — "
                          f"pts range x=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
                          f"y=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}]")
                else:
                    # normalized full-frame landmarks for fallback
                    pts_norm = pts.copy()
                    pts_norm[:, 0] /= w
                    pts_norm[:, 1] /= h
                    # extract only the indices the model expects
                    pts_norm_subset = np.array(
                        [[pts_norm[i, 0], pts_norm[i, 1]]
                         for i in C.ALL_LANDMARK_INDICES],
                        dtype=np.float32
                    )

                    out = clf(
                        crop,
                        full_frame_pts_norm=pts_norm_subset,
                        crop_box=box,
                        frame_wh=(w, h),
                    )

                    if out is not None:
                        n_classified += 1
                        raw = out
                        if out[1] >= C.MIN_CONFIDENCE:
                            n_accepted += 1
                            reliable = out
                        else:
                            print(f"[frame {frame_i}] confidence {out[1]:.3f} below "
                                  f"MIN_CONFIDENCE {C.MIN_CONFIDENCE} "
                                  f"— holding '{reliable[0]}'")

            state = FaceState(
                face_detected    = True,
                expression       = reliable[0],
                confidence       = reliable[1],
                eye_open_left    = norm01(m["ear_l"], C.EAR_CLOSED,   C.EAR_OPEN),
                eye_open_right   = norm01(m["ear_r"], C.EAR_CLOSED,   C.EAR_OPEN),
                blinking_left    = bool(m["ear_l"] < C.EAR_BLINK_THRESHOLD),
                blinking_right   = bool(m["ear_r"] < C.EAR_BLINK_THRESHOLD),
                brow_raise_left  = norm01(m["brow_l"], C.BROW_MIN,    C.BROW_MAX),
                brow_raise_right = norm01(m["brow_r"], C.BROW_MIN,    C.BROW_MAX),
                mouth_open       = bool(m["mar"] > C.MAR_OPEN_THRESHOLD),
                mouth_open_norm  = norm01(m["mar"],   C.MAR_CLOSED,   C.MAR_OPEN),
                mouth_width_norm = norm01(m["width"], C.MOUTH_W_MIN,  C.MOUTH_W_MAX),
            )
            if not a.no_gui:
                draw_hud(frame, m, raw)
        else:
            n_no_face += 1

        if frame_i % a.output_every == 0:
            print_face_state(state, sock, addr)

        if not a.no_gui:
            fps = (frame_i + 1) / (time.time() - t0)
            cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Face tracker", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        # periodic summary
        if frame_i > 0 and frame_i % 150 == 0:
            elapsed = time.time() - t0
            print(f"[summary frame {frame_i}] "
                  f"fps={frame_i/elapsed:.1f}  "
                  f"face={n_face}  no_face={n_no_face}  "
                  f"classified={n_classified}  accepted={n_accepted}  "
                  f"fallbacks={clf._fallback}  "
                  f"reliable='{reliable[0]}' conf={reliable[1]:.3f}")

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[done] frames={frame_i}  classified={n_classified}  "
          f"accepted={n_accepted}  fallbacks={clf._fallback}")


if __name__ == "__main__":
    main()